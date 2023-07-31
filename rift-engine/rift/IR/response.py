import difflib
import os
import re
from textwrap import dedent
from typing import List, Optional
from rift.IR.ir import FunctionDeclaration, IR, Language
from rift.IR.parser import functions_missing_types_in_ir, parse_code_block


def extract_blocks_from_response(response: bytes) -> List[bytes]:
    """
    Extract code blocks from a response string.

    Args:
        response (str): The response string to be processed.

    Returns:
        List[str]: A list of strings, each string being a block of code from the response.
    """
    code_blocks: List[bytes] = []
    current_block = b""
    inside_code_block = False
    for line in response.splitlines():
        if line.startswith(b"```"):
            if inside_code_block:
                code_blocks.append(current_block)
                current_block = b""
                inside_code_block = False
            else:
                inside_code_block = True
        elif inside_code_block:
            current_block += line + b"\n"
    return code_blocks


def parse_code_blocks(code_blocks: List[bytes], language: Language) -> IR:
    """
    Parses code blocks and returns intermediate representation (IR).

    Args:
        code_blocks (List[str]): List of code blocks to be parsed.
        language (Language): The programming language of the code blocks.

    Returns:
        IR: The intermediate representation of the parsed code blocks.
    """
    ir = IR(symbol_table={})
    for block in code_blocks:
        parse_code_block(ir, block, language)
    return ir


def replace_functions_in_document(
    ir_doc: IR,
    ir_blocks: IR,
    document: bytes,
    replace_body: bool,
    filter_function_names: Optional[List[str]] = None,
) -> bytes:
    """
    Replaces functions in the document with corresponding functions from parsed blocks.

    Args:
        ir_doc (IR): The intermediate representation of the original document.
        ir_blocks (IR): The intermediate representation of the code blocks.
        original_document (str): The original document string.

    Returns:
        str: The modified document with replaced functions.
    """
    function_declarations_in_document: List[FunctionDeclaration] = []
    for symbol_item in ir_doc.symbol_table.values():
        if isinstance(symbol_item, FunctionDeclaration):
            function_declarations_in_document.append(symbol_item)

    # sort the function declarations in descending order of their start position
    function_declarations_in_document.sort(key=lambda x: -x.substring[0])

    modified_document = document
    for function_declaration in function_declarations_in_document:
        function_in_blocks = ir_blocks.symbol_table.get(
            function_declaration.name)
        filter = True if filter_function_names is None else function_declaration.name in filter_function_names
        if filter and isinstance(function_in_blocks, FunctionDeclaration):
            if replace_body:
                new_function_text = function_in_blocks.get_substring()
                start_replace, end_replace = function_declaration.substring
                modified_document = \
                    modified_document[:start_replace] + \
                    new_function_text + \
                    modified_document[end_replace:]
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
                modified_document = \
                    modified_document[:start_replace] + \
                    new_function_text + \
                    modified_document[end_replace:]
    return modified_document


def replace_functions_from_code_blocks(
    code_blocks: List[bytes], document: bytes,
        language: Language, replace_body: bool,
        filter_function_names: Optional[List[str]] = None,
) -> bytes:
    """
    Generates a new document by replacing functions in the original document with the corresponding functions
    from the code blocks.

    Args:
        code_blocks (List[str]): List of code blocks containing updated functions.
        document (str): Original document.
        language (Language): Language of the code blocks and document.

    Returns:
        str: The modified document with the updated functions.
    """
    ir_blocks = parse_code_blocks(code_blocks=code_blocks, language=language)
    ir_doc = parse_code_blocks(code_blocks=[document], language=language)
    return replace_functions_in_document(filter_function_names=filter_function_names, ir_doc=ir_doc, ir_blocks=ir_blocks, document=document, replace_body=replace_body)

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
    """).lstrip().encode("utf-8")

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
        """).lstrip().encode("utf-8")

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
        
        """).lstrip().encode("utf-8")


def test_response():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_file = os.path.join(script_dir, 'response_test.txt')
    with open(test_output_file, 'r') as f:
        old_test_output = f.read()
    new_test_output = ""

    language: Language = "c"
    code_blocks1 = extract_blocks_from_response(Test.response1)
    new_document1 = replace_functions_from_code_blocks(
        code_blocks=code_blocks1, document=Test.document, language=language, replace_body=True)
    new_test_output += f"\nNew document1:\n```\n{new_document1.decode()}```"
    code_blocks2 = extract_blocks_from_response(Test.response2)
    new_document2 = replace_functions_from_code_blocks(
        code_blocks=code_blocks2, document=Test.document, language=language, replace_body=True)
    new_test_output += f"\n\nNew document2:\n```\n{new_document2.decode()}```"

    language = "python"
    code_blocks3 = extract_blocks_from_response(Test.response3)
    ir = IR(symbol_table={})
    parse_code_block(ir, Test.code3, language)
    missing_types = functions_missing_types_in_ir(ir)
    functions_missing_types = [
        mt.function_declaration.name for mt in missing_types]
    new_document3 = replace_functions_from_code_blocks(
        code_blocks=code_blocks3, document=Test.code3,
        filter_function_names=functions_missing_types,
        language=language, replace_body=False)
    new_test_output += f"\n\nNew document3:\n```\n{new_document3.decode()}```"

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
