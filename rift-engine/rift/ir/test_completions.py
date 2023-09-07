import difflib
import os

import rift.ir.completions as completions
import rift.ir.parser as parser
import rift.ir.IR as IR
import rift.ir.test_parser as test_parser


def test_completions_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "completions.txt")
    with open(test_file, "r") as f:
        old_test_data = f.read()

    project = test_parser.get_test_python_project()
    new_test_data = completions.get_symbol_completions(project)

    if new_test_data != old_test_data:
        diff = difflib.unified_diff(
            old_test_data.splitlines(keepends=True),
            new_test_data.splitlines(keepends=True),
        )
        diff_output = "".join(diff)

        update_missing_types = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_missing_types:
            print("Updating Missing Types...")
            with open(test_file, "w") as f:
                f.write(new_test_data)

        assert (
            update_missing_types
        ), f"Completions have changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"

class MyOuterClass:
    class MyInnerClass:
        def some_function(self, x:int, y:int) -> int:
            return x + y

def test_symbol_reference():
    this_file = os.path.abspath(__file__)
    project = parser.parse_files_in_paths([this_file])

    uri_this_file = f"file://{this_file}"
    reference_this_file = IR.Reference.from_uri(uri_this_file)
    res_this_file = project.lookup_reference(reference_this_file)
    assert res_this_file is not None
    assert res_this_file.symbol is None

    uri_some_function = f"file://{this_file}#MyOuterClass.MyInnerClass.some_function"
    reference_some_function = IR.Reference.from_uri(uri_some_function)
    res_some_function = project.lookup_reference(reference_some_function)
    assert res_some_function is not None
    assert res_some_function.symbol is not None
    assert res_some_function.symbol.get_substring_without_body().decode().strip() == "def some_function(self, x:int, y:int) -> int:"

test_symbol_reference()