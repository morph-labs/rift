from textwrap import dedent
from typing import List
from rift.IR.ir import FunctionDeclaration, IR, Language
from rift.IR.parser import parse_code_block


def extract_blocks_from_response(response: str) -> List[str]:
    """
    Extract code blocks from a response string.

    Args:
        response (str): The response string to be processed.

    Returns:
        List[str]: A list of strings, each string being a block of code from the response.
    """
    code_blocks = []
    current_block = ""
    inside_code_block = False
    for line in response.splitlines():
        if line.startswith("```"):
            if inside_code_block:
                code_blocks.append(current_block)
                current_block = ""
                inside_code_block = False
            else:
                inside_code_block = True
        elif inside_code_block:
            current_block += line + "\n"
    return code_blocks


def parse_code_blocks(code_blocks: List[str], language: Language) -> IR:
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


def replace_functions_in_document(ir_doc: IR, ir_blocks: IR, document: str) -> str:
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
        if isinstance(function_in_blocks, FunctionDeclaration):
            new_function_text = function_in_blocks.get_substring()
            modified_document = modified_document[:function_declaration.substring[0]] + \
                new_function_text + \
                modified_document[function_declaration.substring[1]:]
    return modified_document


def replace_functions_from_code_blocks(code_blocks: List[str], document: str, language: Language) -> str:
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
    return replace_functions_in_document(ir_doc, ir_blocks, document)

# your test class and main logic remain unchanged.


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
    """).lstrip()

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


def test_response():
    language: Language = "c"
    code_blocks1 = extract_blocks_from_response(Test.response1)
    new_document1 = replace_functions_from_code_blocks(
        code_blocks=code_blocks1, document=Test.document, language=language)
    print(f"new document1:\n{new_document1}\n")
    code_blocks2 = extract_blocks_from_response(Test.response2)
    new_document2 = replace_functions_from_code_blocks(
        code_blocks=code_blocks2, document=Test.document, language=language)
    print(f"new document2:\n{new_document2}\n")
