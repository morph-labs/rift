import os
from dataclasses import dataclass
from typing import List

import rift.ir.IR as IR


@dataclass
class FunctionMissingDocString:
    function_declaration: IR.ValueDeclaration

    def __str__(self) -> str:
        # let agent generate doc string for function by reading the function code
        return f"Function `{self.function_declaration.name}` is missing a doc string"

    def __repr__(self) -> str:
        return self.__str__()

    def __int__(self) -> int:
        return 1


def functions_missing_doc_strings_in_file(file_name: IR.File) -> List[FunctionMissingDocString]:
    """Find function declarations that are missing doc strings."""
    functions_missing_doc_strings: List[FunctionMissingDocString] = []
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
            functions_missing_doc_strings.append(FunctionMissingDocString(function))
    return functions_missing_doc_strings


@dataclass
class FileMissingDocStrings:
    ir_code: IR.Code
    ir_name: IR.File
    language: IR.Language
    functions_missing_doc_strings: List[FunctionMissingDocString]


def files_missing_doc_strings_in_project(project: IR.Project) -> List[FileMissingDocStrings]:
    """Return a list of files with missing doc strings and the functions missing doc strings in each file."""
    files_with_missing_doc_strings: List[FileMissingDocStrings] = []
    for file_name in project.get_files():
        functions_missing_doc_strings = functions_missing_doc_strings_in_file(file_name)
        if functions_missing_doc_strings != []:
            file_decl = functions_missing_doc_strings[0].function_declaration
            language = file_decl.language
            file_code = file_decl.code
            files_with_missing_doc_strings.append(
                FileMissingDocStrings(file_code, file_name, language, functions_missing_doc_strings)
            )
    return files_with_missing_doc_strings


if __name__ == "__main__":
    from rift.ir.parser import parse_files_in_paths

    project = parse_files_in_paths([os.path.dirname(os.path.abspath(__file__))])
    files_with_missing_doc_strings = files_missing_doc_strings_in_project(project)
    for f in files_with_missing_doc_strings:
        print(f"file:{f.ir_name.path}")
        for fn in f.functions_missing_doc_strings:
            print(f"  {fn}")
