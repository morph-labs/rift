import os
from dataclasses import dataclass, field
from typing import List, Tuple

import rift.ir.IR as IR
import rift.ir.parser as parser


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
        if function.language not in ["javascript", "ocaml", "python", "rescript", "tsx", "typescript"]:
            continue
        if not function.docstring:
            functions_missing_doc_strings.append(FunctionMissingDocString(function))
    return functions_missing_doc_strings


def functions_missing_doc_strings_in_path(
    root: str, path: str
) -> Tuple[List[FunctionMissingDocString], IR.Code, IR.File]:
    "Given a file path, parse the file and find function declarations that are missing doc strings."
    language = IR.language_from_file_extension(path)
    functions_missing_doc_strings: List[FunctionMissingDocString] = []
    if language is None:
        functions_missing_doc_strings = []
        file_code = IR.Code(b"")
    else:
        full_path = os.path.join(root, path)
        with open(full_path, "r", encoding="utf-8") as f:
            file_code = IR.Code(f.read().encode("utf-8"))
        file_of_path = IR.File(path)
        parser.parse_code_block(file_of_path, file_code, language)
        functions_missing_doc_strings = functions_missing_doc_strings_in_file(file_of_path)
    return (functions_missing_doc_strings, file_code, file_of_path)


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
            files_with_missing_doc_strings.append(FileMissingDocStrings(file_code, file_name, language, functions_missing_doc_strings))
    return files_with_missing_doc_strings


if __name__ == "__main__":
    import json
    from rift.ir.parser import parse_files_in_paths
    project = parse_files_in_paths(["/home/john/rift/"])
    files_with_missing_doc_strings = files_missing_doc_strings_in_project(project)
    for f in files_with_missing_doc_strings:
        print(f.functions_missing_doc_strings[0].function_declaration.name)
