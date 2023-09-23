from enum import Enum
import re
import textwrap
from typing import List, Optional, Set, Tuple

import rift.ir.IR as IR
import rift.ir.parser as parser
import rift.ir.python_typing as python_typing
import logging

logger = logging.getLogger(__name__)


def extract_blocks_from_response(response: str) -> List[IR.Code]:
    """
    Extract code blocks from a response string.

    Args:
        response (str): The response string to be processed.

    Returns:
        List[Code]: A list of code blocks.
    """
    code_blocks_str: List[str] = []
    current_block: str = ""
    inside_code_block = False
    for line in response.splitlines():
        if line.startswith("```"):
            if inside_code_block:
                code_blocks_str.append(current_block)
                current_block = ""
                inside_code_block = False
            else:
                inside_code_block = True
        elif inside_code_block:
            current_block += line + "\n"
    code_blocks = [IR.Code(block.encode("utf-8")) for block in code_blocks_str]
    return code_blocks


def parse_code_blocks(code_blocks: List[IR.Code], language: IR.Language) -> IR.File:
    """
    Parses code blocks and returns intermediate representation (IR).

    Args:
        code_blocks (List[str]): List of code blocks to be parsed.
        language (Language): The programming language of the code blocks.

    Returns:
        IR: The intermediate representation of the parsed code blocks.
    """
    file = IR.File("response")
    for block in code_blocks:
        parser.parse_code_block(file, block, language)
    return file


def get_typing_names_from_types(types: List[IR.Type]) -> Set[str]:
    """
    Get names that need to be imported from "typing" given a list of types.
    """
    names: Set[str] = set()
    for t in types:
        if t.name and python_typing.is_typing_type(t.name):
            names.add(t.name)
        new_names = get_typing_names_from_types(t.arguments)
        names = names.union(new_names)
    return names


Replace = Enum("Replace", ["ALL", "DOC", "SIGNATURE"])


def replace_functions_in_document(
    ir_doc: IR.File,
    ir_blocks: IR.File,
    replace: Replace,
    filter_function_ids: Optional[List[IR.QualifiedId]] = None,
) -> Tuple[List[IR.CodeEdit], List[IR.Symbol]]:
    """
    Replaces functions in the document with corresponding functions from parsed blocks.
    """
    function_declarations_in_document: List[
        IR.Symbol
    ] = ir_doc.get_function_declarations()

    code_edits: List[IR.CodeEdit] = []
    updated_functions: List[IR.Symbol] = []

    for function_declaration in function_declarations_in_document:
        function_in_blocks_ = ir_blocks.search_symbol(function_declaration.name)
        function_in_blocks = None
        if len(function_in_blocks_) == 1:
            f0 = function_in_blocks_[0]
            if isinstance(f0.symbol_kind, IR.FunctionKind):
                function_in_blocks = f0
        if filter_function_ids is None:
            filter = True
        else:
            filter = function_declaration.get_qualified_id() in filter_function_ids
        if filter and function_in_blocks is not None:
            updated_functions.append(function_in_blocks)
            if replace == Replace.ALL:
                substring = function_declaration.substring
                new_bytes = function_in_blocks.get_substring()
            elif replace == Replace.DOC:
                if function_in_blocks.docstring is None:
                    logger.warning(f"No docstring for function {function_declaration.name}")
                    continue
                if function_declaration.docstring_sub is not None:
                    logger.warning(
                        f"Docstring already exists for function {function_declaration.name}"
                    )
                    continue

                # find indent by looking backwards in the bytes until we find a newline
                def find_indent(bytes: bytes, start: int) -> int:
                    for i in range(start, -1, -1):
                        if bytes[i] == 10:
                            return start - i - 1
                    return 0

                if (
                    function_declaration.body_sub is not None
                    and function_in_blocks.body_sub is not None
                ):
                    if function_declaration.language == "python":
                        body_start = function_declaration.body_sub[0]
                        old_indent = find_indent(function_declaration.code.bytes, body_start)
                        new_indent = find_indent(
                            function_in_blocks.code.bytes, function_in_blocks.body_sub[0]
                        )
                        substring = (body_start - old_indent, body_start - old_indent)
                    else:
                        # add the doc comment before the function
                        old_function_start = function_declaration.substring[0]
                        old_indent = find_indent(
                            function_declaration.code.bytes, old_function_start
                        )
                        new_function_start = function_in_blocks.substring[0]
                        new_indent = find_indent(function_in_blocks.code.bytes, new_function_start)
                        substring = (
                            old_function_start - old_indent,
                            old_function_start - old_indent,
                        )
                else:
                    logger.warning(f"No body for function {function_declaration.name}")
                    continue

                docstring = textwrap.dedent(" " * new_indent + function_in_blocks.docstring)
                docstring = textwrap.indent(docstring, " " * old_indent)
                new_bytes = docstring.encode("utf-8") + b"\n"
            elif replace == Replace.SIGNATURE:
                new_function_text = function_in_blocks.get_substring_without_body()
                logger.info(f"{new_function_text=}")
                old_function_text = function_declaration.get_substring_without_body()
                # Get trailing newline and/or whitespace from old text
                old_trailing_whitespace = re.search(rb"\s*$", old_function_text)
                # Add it to new text
                if old_trailing_whitespace is not None:
                    new_function_text = new_function_text.rstrip()
                    logger.info(f"{new_function_text=}")
                    new_function_text += old_trailing_whitespace.group(0)
                    logger.info(f"{new_function_text=}")
                start_replace = function_declaration.substring[0]
                end_replace = start_replace + len(old_function_text)
                substring = (start_replace, end_replace)
                new_bytes = new_function_text
            code_edit = IR.CodeEdit(substring=substring, new_bytes=new_bytes)
            code_edits.append(code_edit)
    return (code_edits, updated_functions)


def update_typing_imports(
    code: IR.Code, language: IR.Language, updated_functions: List[IR.Symbol]
) -> Optional[IR.CodeEdit]:
    file = parse_code_blocks(code_blocks=[code], language=language)
    typing_import = file.search_module_import("typing")
    typing_names: Set[str] = set()
    missing_names: Set[str] = set()
    if typing_import is not None:
        typing_names = set(typing_import.names)
    for f in updated_functions:
        if isinstance(f.symbol_kind, IR.FunctionKind):
            fun_kind = f.symbol_kind
            types_in_function = [p.type for p in fun_kind.parameters if p.type is not None]
            if fun_kind.return_type is not None:
                types_in_function.append(fun_kind.return_type)
            names_from_types = get_typing_names_from_types(types_in_function)
            new_missing_names = names_from_types.difference(typing_names)
            missing_names = missing_names.union(new_missing_names)
    if len(missing_names) > 0:
        all_names = typing_names.union(missing_names)
        import_str = f"from typing import {', '.join(sorted(all_names))}"
        if typing_import is None:
            substring = (0, 0)
            import_str += "\n"
        else:
            substring = typing_import.substring
        code_edit = IR.CodeEdit(substring=substring, new_bytes=import_str.encode("utf-8"))
        return code_edit


def replace_functions_from_code_blocks(
    code_blocks: List[IR.Code],
    document: IR.Code,
    language: IR.Language,
    replace: Replace,
    filter_function_ids: Optional[List[IR.QualifiedId]] = None,
) -> Tuple[List[IR.CodeEdit], List[IR.Symbol]]:
    """
    Generates a new document by replacing functions in the original document with the corresponding functions
    from the code blocks.
    """
    ir_blocks = parse_code_blocks(code_blocks=code_blocks, language=language)
    ir_doc = parse_code_blocks(code_blocks=[document], language=language)
    code_edits, updated_functions = replace_functions_in_document(
        filter_function_ids=filter_function_ids,
        ir_doc=ir_doc,
        ir_blocks=ir_blocks,
        replace=replace,
    )
    return code_edits, updated_functions
