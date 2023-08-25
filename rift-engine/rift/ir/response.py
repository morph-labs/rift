import re
from typing import List, Optional

import rift.ir.IR as IR
import rift.ir.parser as parser


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


def replace_functions_in_document(
    ir_doc: IR.File,
    ir_blocks: IR.File,
    document: IR.Code,
    replace_body: bool,
    filter_function_ids: Optional[List[IR.QualifiedId]] = None,
) -> List[IR.CodeEdit]:
    """
    Replaces functions in the document with corresponding functions from parsed blocks.
    """
    function_declarations_in_document: List[
        IR.ValueDeclaration
    ] = ir_doc.get_function_declarations()

    code_edits: List[IR.CodeEdit] = []

    for function_declaration in function_declarations_in_document:
        function_in_blocks_ = ir_blocks.search_symbol(function_declaration.name)
        function_in_blocks = None
        if len(function_in_blocks_) == 1:
            f0 = function_in_blocks_[0]
            if isinstance(f0, IR.ValueDeclaration) and isinstance(f0.value_kind, IR.FunctionKind):
                function_in_blocks = f0
        if filter_function_ids is None:
            filter = True
        else:
            filter = function_declaration.get_qualified_id() in filter_function_ids
        if filter and function_in_blocks is not None:
            if replace_body:
                substring = function_declaration.substring
                new_bytes = function_in_blocks.get_substring()
            else:
                new_function_text = function_in_blocks.get_substring_without_body()
                old_function_text = function_declaration.get_substring_without_body()
                # Get trailing newline and/or whitespace from old text
                old_trailing_whitespace = re.search(rb"\s*$", old_function_text)
                # Add it to new text
                if old_trailing_whitespace is not None:
                    new_function_text = new_function_text.rstrip()
                    new_function_text += old_trailing_whitespace.group(0)
                start_replace = function_declaration.substring[0]
                end_replace = start_replace + len(old_function_text)
                substring = (start_replace, end_replace)
                new_bytes = new_function_text
            code_edit = IR.CodeEdit(substring=substring, new_bytes=new_bytes)
            code_edits.append(code_edit)
    return code_edits


def replace_functions_from_code_blocks(
    code_blocks: List[IR.Code],
    document: IR.Code,
    language: IR.Language,
    replace_body: bool,
    filter_function_ids: Optional[List[IR.QualifiedId]] = None,
) -> List[IR.CodeEdit]:
    """
    Generates a new document by replacing functions in the original document with the corresponding functions
    from the code blocks.
    """
    ir_blocks = parse_code_blocks(code_blocks=code_blocks, language=language)
    ir_doc = parse_code_blocks(code_blocks=[document], language=language)
    return replace_functions_in_document(
        filter_function_ids=filter_function_ids,
        ir_doc=ir_doc,
        ir_blocks=ir_blocks,
        document=document,
        replace_body=replace_body,
    )
