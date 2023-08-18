import difflib
import os
from textwrap import dedent

import rift.ir.IR as IR
import rift.ir.parser as parser
import rift.ir.response as response
from rift.ir.missing_types import functions_missing_types_in_file


class Test:
    document = (
        dedent(
            """
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
    """
        )
        .lstrip()
        .encode("utf-8")
    )

    response1 = dedent(
        """
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
    """
    ).lstrip()

    response2 = dedent(
        """
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
        """
    ).lstrip()

    code3 = (
        dedent(
            """
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
        """
        )
        .lstrip()
        .encode("utf-8")
    )

    response3 = dedent(
        """
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
        
        """
    ).lstrip()


def test_response():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_file = os.path.join(script_dir, "response_test.txt")
    with open(test_output_file, "r") as f:
        old_test_output = f.read()
    new_test_output = ""

    language: IR.Language = "c"
    code_blocks1 = response.extract_blocks_from_response(Test.response1)
    document1 = IR.Code(Test.document)
    edits1 = response.replace_functions_from_code_blocks(
        code_blocks=code_blocks1, document=document1, language=language, replace_body=True
    )
    new_document1 = document1.apply_edits(edits1)
    new_test_output += f"\nNew document1:\n```\n{new_document1}```"
    code_blocks2 = response.extract_blocks_from_response(Test.response2)
    document2 = IR.Code(Test.document)
    edits2 = response.replace_functions_from_code_blocks(
        code_blocks=code_blocks2, document=document2, language=language, replace_body=True
    )
    new_document2 = document2.apply_edits(edits2)
    new_test_output += f"\n\nNew document2:\n```\n{new_document2}```"

    language = "python"
    code_blocks3 = response.extract_blocks_from_response(Test.response3)
    file = IR.File("response3")
    parser.parse_code_block(file, IR.Code(Test.code3), language)
    missing_types = functions_missing_types_in_file(file)
    filter_function_ids = [mt.function_declaration.get_qualified_id() for mt in missing_types]
    document3 = IR.Code(Test.code3)
    edits3 = response.replace_functions_from_code_blocks(
        code_blocks=code_blocks3,
        document=document3,
        filter_function_ids=filter_function_ids,
        language=language,
        replace_body=False,
    )
    new_document3 = document3.apply_edits(edits3)
    new_test_output += f"\n\nNew document3:\n```\n{new_document3}```"

    if new_test_output != old_test_output:
        diff = difflib.unified_diff(
            old_test_output.splitlines(keepends=True), new_test_output.splitlines(keepends=True)
        )
        diff_output = "".join(diff)

        # if you want to update the missing types, set this to True
        update_missing_types = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_missing_types:
            print("Updating Missing Types...")
            with open(test_output_file, "w") as f:
                f.write(new_test_output)

        assert (
            update_missing_types
        ), f"Missing Types have changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"
