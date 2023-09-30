import difflib
import os
from typing import List

import rift.ir.parser as parser
import rift.ir.test_parser as test_parser
from rift.ir.missing_types import files_missing_types_in_project, functions_missing_types_in_file


def test_missing_types():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    missing_types_file = os.path.join(script_dir, "missing_types.txt")
    with open(missing_types_file, "r") as f:
        old_missing_types_str = f.read()

    project = test_parser.get_test_project()
    new_missing_types: List[str] = []
    for file in project.get_files():
        missing_types = functions_missing_types_in_file(file)
        new_missing_types += [str(mt) for mt in missing_types]
    new_missing_types_str = "\n".join(new_missing_types)
    if new_missing_types_str != old_missing_types_str:
        diff = difflib.unified_diff(
            old_missing_types_str.splitlines(keepends=True),
            new_missing_types_str.splitlines(keepends=True),
        )
        diff_output = "".join(diff)

        # if you want to update the missing types, set this to True
        update_missing_types = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_missing_types:
            print("Updating Missing Types...")
            with open(missing_types_file, "w") as f:
                f.write(new_missing_types_str)

        assert (
            update_missing_types
        ), f"Missing Types have changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"


def test_missing_types_in_project():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    project = parser.parse_files_in_paths([parent_dir])
    files_missing_types = files_missing_types_in_project(project)
    for fmt in files_missing_types:
        print(f"File: {fmt.file.path}")
        for mt in fmt.missing_types:
            print(f"  {mt}")
        print()
