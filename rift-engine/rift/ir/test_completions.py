import difflib
import os

import rift.ir.completions as completions
import rift.ir.test_parser as test_parser


def test_completions():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "completions.txt")
    with open(test_file, "r") as f:
        old_test_data = f.read()

    project = test_parser.get_test_project()
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
