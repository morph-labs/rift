from dataclasses import dataclass
from typing import List

import rift.ir.IR as IR


@dataclass
class FunctionMissingDocstring:
    function_declaration: IR.ValueDeclaration

    def __str__(self) -> str:
        # let agent generate doc string for function by reading the function code
        return f"Function `{self.function_declaration.name}` is missing a doc string"

    def __repr__(self) -> str:
        return self.__str__()

    def __int__(self) -> int:
        return 1


def functions_missing_docstrings_in_file(file_name: IR.File) -> List[FunctionMissingDocstring]:
    """Find function declarations that are missing doc strings."""
    functions_missing_docstrings: List[FunctionMissingDocstring] = []
    function_declarations = file_name.get_function_declarations()
    for function in function_declarations:
        if function.language not in [
            "javascript",
            "ocaml",
            "python",
            "rescript",
            "tsx",
            "typescript",
        ]:
            continue
        if not function.docstring:
            functions_missing_docstrings.append(FunctionMissingDocstring(function))
    return functions_missing_docstrings


@dataclass
class FileMissingDocstrings:
    ir_code: IR.Code
    ir_name: IR.File
    language: IR.Language
    functions_missing_docstrings: List[FunctionMissingDocstring]


def files_missing_docstrings_in_project(project: IR.Project) -> List[FileMissingDocstrings]:
    """Return a list of files with missing doc strings and the functions missing doc strings in each file."""
    files_with_missing_docstrings: List[FileMissingDocstrings] = []
    for file_name in project.get_files():
        functions_missing_docstrings = functions_missing_docstrings_in_file(file_name)
        if functions_missing_docstrings != []:
            file_decl = functions_missing_docstrings[0].function_declaration
            language = file_decl.language
            file_code = file_decl.code
            files_with_missing_docstrings.append(
                FileMissingDocstrings(file_code, file_name, language, functions_missing_docstrings)
            )
    return files_with_missing_docstrings
