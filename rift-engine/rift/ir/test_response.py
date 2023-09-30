import difflib
import os
from textwrap import dedent

import rift.ir.IR as IR
from rift.ir.missing_docstrings import functions_missing_docstrings_in_file
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
        from typing import Tuple
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
        def get_num_tokens2(content: t1) -> List[t2]:           
            return some(imaginary(code))
                       
        def foo() -> string:
            print("This should be ignored as the return type was not missing")
        ```

        Some other thoutghts:
        - this
        
        """
    ).lstrip()

    code4 = (
        dedent(
            """
        class TestAddDocs:
            def dump_elements(self, elements: List[str]) -> None:
                def dump_symbol(symbol: SymbolInfo) -> None:
                    decl_without_body = symbol.get_substring_without_body().decode()
                    elements.append(decl_without_body)
                    if isinstance(symbol, ContainerDeclaration):
                        for statement in symbol.body:
                            dump_statement(statement)

                def dump_statement(statement: Statement) -> None:
                    if isinstance(statement, Declaration):
                        for symbol in statement.symbols:
                            dump_symbol(symbol)
                    else:
                        pass

                for statement in self.statements:
                    dump_statement(statement)

                from typing import Tuple
                def foo() -> None:
                    print("Hello world!")
        """
        )
        .lstrip()
        .encode("utf-8")
    )

    response4 = dedent(
        """
        Here are the required changes:

        ```
        def dump_elements():
            \"\"\"
            The doc comment for dump_elements
            Spans multiple lines
            \"\"\"
            ...
        ```

        Some other thoutghts:
        - this
        
        """
    ).lstrip()

    code5 = (
        dedent(
            """

        function add(a: number, b: number) : number {
            return a + b;
        }

        class Employee {
            empCode: number;
            empName: string;

            constructor(code: number, name: string) {
                    this.empName = name;
                    this.empCode = code;
            }

            getSalary() : number {
                return 10000;
            }
        }
        """
        )
        .lstrip()
        .encode("utf-8")
    )

    response5 = dedent(
        """
        Here are the required changes:

        ```
        /**
        * Adds two numbers together.
        * 
        * @param a - The first number to be added.
        * @param b - The second number to be added.
        * @returns The sum of the two numbers.
        */
        function add(a: number, b: number) : number {
            return a + b;
        }
        ```

        ```
        /**
         * Constructor function for creating an instance of a class.
         * @param code - The code of the employee.
         * @param name - The name of the employee.
         */
        constructor(code: number, name: string) {
            this.empName = name;
            this.empCode = code;
        }
        ```

        ```
        /**
         * Returns the salary of an employee.
         * 
         * @returns The salary as a number.
         */
        function getSalary(): number {
            return 10000;
        }
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
    edits1, _ = response.replace_functions_from_code_blocks(
        code_blocks=code_blocks1,
        document=document1,
        language=language,
        replace=response.Replace.ALL,
    )
    new_document1 = document1.apply_edits(edits1)
    new_test_output += f"\nNew document1:\n```\n{new_document1}```"
    code_blocks2 = response.extract_blocks_from_response(Test.response2)
    document2 = IR.Code(Test.document)
    edits2, _ = response.replace_functions_from_code_blocks(
        code_blocks=code_blocks2,
        document=document2,
        language=language,
        replace=response.Replace.ALL,
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
    edits3, updated_functions = response.replace_functions_from_code_blocks(
        code_blocks=code_blocks3,
        document=document3,
        filter_function_ids=filter_function_ids,
        language=language,
        replace=response.Replace.SIGNATURE,
    )
    edit_imports = response.update_typing_imports(
        code=document3, language=language, updated_functions=updated_functions
    )
    if edit_imports is not None:
        edits3.append(edit_imports)

    new_document3 = document3.apply_edits(edits3)
    new_test_output += f"\n\nNew document3:\n```\n{new_document3}```"

    language = "python"
    code_blocks4 = response.extract_blocks_from_response(Test.response4)
    file = IR.File("response4")
    parser.parse_code_block(file, IR.Code(Test.code4), language)
    missing_docs = functions_missing_docstrings_in_file(file)
    filter_function_ids = [md.function_declaration.get_qualified_id() for md in missing_docs]
    document4 = IR.Code(Test.code4)
    edits4, updated_functions = response.replace_functions_from_code_blocks(
        code_blocks=code_blocks4,
        document=document4,
        filter_function_ids=filter_function_ids,
        language=language,
        replace=response.Replace.DOC,
    )
    new_document4 = document4.apply_edits(edits4)
    new_test_output += f"\n\nNew document4:\n```\n{new_document4}```"

    language = "typescript"
    code_blocks5 = response.extract_blocks_from_response(Test.response5)
    file = IR.File("response5")
    parser.parse_code_block(file, IR.Code(Test.code5), language)
    missing_docs = functions_missing_docstrings_in_file(file)
    filter_function_ids = [md.function_declaration.get_qualified_id() for md in missing_docs]
    document5 = IR.Code(Test.code5)
    edits5, updated_functions = response.replace_functions_from_code_blocks(
        code_blocks=code_blocks5,
        document=document5,
        filter_function_ids=filter_function_ids,
        language=language,
        replace=response.Replace.DOC,
    )
    new_document5 = document5.apply_edits(edits5)
    new_test_output += f"\n\nNew document5:\n```\n{new_document5}```"

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
