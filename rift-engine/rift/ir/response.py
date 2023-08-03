import difflib
import os
import re
from textwrap import dedent
from typing import List, Optional
from rift.ir.missing_types import functions_missing_types_in_file

import rift.ir.parser as parser
import rift.ir.IR as IR


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
    function_declarations_in_document: List[IR.FunctionDeclaration] = \
        ir_doc.get_function_declarations()

    code_edits: List[IR.CodeEdit] = []

    for function_declaration in function_declarations_in_document:
        function_in_blocks_ = ir_blocks.search_symbol(
            function_declaration.name)
        if len(function_in_blocks_) == 1 and isinstance(function_in_blocks_[0], IR.FunctionDeclaration):
            function_in_blocks = function_in_blocks_[0]
        else:
            function_in_blocks = None
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
                old_trailing_whitespace = re.search(
                    rb'\s*$', old_function_text)
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
    code_blocks: List[IR.Code], document: IR.Code,
        language: IR.Language, replace_body: bool,
        filter_function_ids: Optional[List[IR.QualifiedId]] = None,
) -> List[IR.CodeEdit]:
    """
    Generates a new document by replacing functions in the original document with the corresponding functions
    from the code blocks.
    """
    ir_blocks = parse_code_blocks(code_blocks=code_blocks, language=language)
    ir_doc = parse_code_blocks(code_blocks=[document], language=language)
    return replace_functions_in_document(filter_function_ids=filter_function_ids, ir_doc=ir_doc, ir_blocks=ir_blocks, document=document, replace_body=replace_body)

############################
#### TESTS FROM HERE ON ####
############################


class Test:
    document = dedent("""
        int aa() {
          return 0;
        }

        void foo(int **x) {
          *x = 0;
        }

        int bb() {
          return 0;
        }

        int main() {
          int *x;
          foo(&x);
          *x = 1;
          return 0;
        }
    """).lstrip().encode("utf-8")

    response1 = dedent("""
        To fix the error reported in the `main` function, you need to make the following changes:

        1. Modify the `foo` function to validate the pointer passed to it and assign it a value only if it is not null.
        2. Update the `main` function to handle the case where the pointer returned by `foo` is null.

        Here are the required changes:

        1. In the `foo` function, add a null check before assigning a value to `*x`:
        ```c
        void foo(int **x) {
          if (x != NULL) {
            *x = 0;
          }
        }
        ```

        2. In the `main` function, add a null check after calling `foo` and before dereferencing `x`:
        ```c
        int main() {
          int *x;
          foo(&x);

          if (x != NULL) {
            *x = 1;
          }

          return 0;
        }
        ```
    """).lstrip()

    response2 = dedent("""
        The bug is caused by dereferencing a potentially null pointer `x` on line 18. To fix this bug, we need to modify the following functions:

        1. `foo()`
        2. `main()`

        Here are the required changes:

        ```...c
        void foo(int **x) {
          *x = (int*) malloc(sizeof(int));
          **x = 0;
        }

        int main() {
          int *x;
          foo(&x);
          *x = 1;
          free(x);
          return 0;
        }
        ```

        In the `foo()` function, we allocate memory for `x` using `malloc()` and then assign a value of 0 to `*x`. This ensures that `x` is not null when it is passed back to `main()`.

        In `main()`, we add a call to `free(x)` to release the allocated memory before the program exits.
        """).lstrip()

    code3 = dedent("""
        def foo() -> None:
            print("Hello world!")

        @cache
        def get_num_tokens(content):
            return len(ENCODER.encode(content))

        @cache
        def get_num_tokens2(content):
            return len(ENCODER.encode(content))
        
        def bar() -> None:
            print("Hello world!")
        """).lstrip().encode("utf-8")

    response3 = dedent("""
        Here are the required changes:

        ```
        @cache
        def get_num_tokens(content: str) -> int:           
        ...
                       
        @cache
        def get_num_tokens2(content: t1) -> t2:           
            return some(imaginary(code))
                       
        def foo() -> string:
            print("This should be ignored as the return type was not missing")
        ```

        Some other thoutghts:
        - this
        
        """).lstrip()


def test_response():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_file = os.path.join(script_dir, 'response_test.txt')
    with open(test_output_file, 'r') as f:
        old_test_output = f.read()
    new_test_output = ""

    language: IR.Language = "c"
    code_blocks1 = extract_blocks_from_response(Test.response1)
    document1 = IR.Code(Test.document)
    edits1 = replace_functions_from_code_blocks(
        code_blocks=code_blocks1, document=document1, language=language, replace_body=True)
    new_document1 = document1.apply_edits(edits1)
    new_test_output += f"\nNew document1:\n```\n{new_document1}```"
    code_blocks2 = extract_blocks_from_response(Test.response2)
    document2 = IR.Code(Test.document)
    edits2 = replace_functions_from_code_blocks(
        code_blocks=code_blocks2, document=document2, language=language, replace_body=True)
    new_document2 = document2.apply_edits(edits2)
    new_test_output += f"\n\nNew document2:\n```\n{new_document2}```"

    language = "python"
    code_blocks3 = extract_blocks_from_response(Test.response3)
    file = IR.File("response3")
    parser.parse_code_block(file, IR.Code(Test.code3), language)
    missing_types = functions_missing_types_in_file(file)
    filter_function_ids = [
        mt.function_declaration.get_qualified_id() for mt in missing_types]
    document3 = IR.Code(Test.code3)
    edits3 = replace_functions_from_code_blocks(
        code_blocks=code_blocks3, document=document3,
        filter_function_ids=filter_function_ids,
        language=language, replace_body=False)
    new_document3 = document3.apply_edits(edits3)
    new_test_output += f"\n\nNew document3:\n```\n{new_document3}```"

    if new_test_output != old_test_output:
        diff = difflib.unified_diff(old_test_output.splitlines(keepends=True),
                                    new_test_output.splitlines(keepends=True))
        diff_output = ''.join(diff)

        # if you want to update the missing types, set this to True
        update_missing_types = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_missing_types:
            print("Updating Missing Types...")
            with open(test_output_file, 'w') as f:
                f.write(new_test_output)

        assert update_missing_types, f"Missing Types have changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"
