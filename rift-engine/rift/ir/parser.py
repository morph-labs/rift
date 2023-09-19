import os
from typing import Callable, List, Optional

from tree_sitter import Parser
from tree_sitter_languages import get_parser as get_tree_sitter_parser

import rift.ir.custom_parsers as custom_parser
import rift.ir.IR as IR
import rift.ir.parser_core as parser_core


def get_parser(language: IR.Language) -> Parser:
    if language == "rescript" and custom_parser.active:
        parser = custom_parser.parser
        parser.set_language(custom_parser.ReScript)
        return parser
    else:
        return get_tree_sitter_parser(language)


def parse_code_block(file: IR.File, code: IR.Code, language: IR.Language) -> None:
    parser = get_parser(language)
    tree = parser.parse(code.bytes)
    for node in tree.root_node.children:
        statement = parser_core.process_statement(
            code=code, file=file, language=language, node=node, scope=""
        )
        file.statements.append(statement)


def parse_path(
    path: str, project: IR.Project, filter_file: Optional[Callable[[str], bool]] = None
) -> None:
    """
    Parses a single file and adds it to the provided Project instance.
    """
    language = IR.language_from_file_extension(path)
    if language is not None and (filter_file is None or filter_file(path)):
        path_from_root = os.path.relpath(path, project.root_path)
        with open(path, "r", encoding="utf-8") as f:
            code = IR.Code(f.read().encode("utf-8"))
        file_ir = IR.File(path=path_from_root)
        parse_code_block(file=file_ir, code=code, language=language)
        project.add_file(file=file_ir)


def parse_files_in_paths(
    paths: List[str], filter_file: Optional[Callable[[str], bool]] = None
) -> IR.Project:
    """
    Parses all files with known extensions in the provided list of paths.
    """
    if len(paths) == 0:
        raise Exception("No paths provided")
    if len(paths) == 1 and os.path.isfile(paths[0]):
        root_path = os.path.dirname(paths[0])
    else:
        root_path = os.path.commonpath(paths)
    project = IR.Project(root_path=root_path)
    for path in paths:
        if os.path.isfile(path):
            parse_path(path, project, filter_file)
        else:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d != "node_modules"]
                for file in files:
                    full_path = os.path.join(root, file)
                    parse_path(full_path, project, filter_file)
    return project
