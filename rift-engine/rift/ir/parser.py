import logging
import os
from typing import Callable, List, Optional

from tree_sitter import Parser
from tree_sitter_languages import get_parser as get_tree_sitter_parser

from . import IR, custom_parsers, parser_core, parser_lean, parser_ocaml, parser_rescript

logger = logging.getLogger(__name__)


def get_parser(language: IR.Language) -> Parser:
    if language == "rescript" and custom_parsers.active:
        parser = custom_parsers.parser
        parser.set_language(custom_parsers.ReScript)
        return parser
    elif language == "lean" and custom_parsers.active:
        parser = custom_parsers.parser
        parser.set_language(custom_parsers.Lean)
        return parser
    else:
        return get_tree_sitter_parser(language)


def parse_code_block(
    file: IR.File, code: IR.Code, language: IR.Language, metasymbols: bool = False
) -> None:
    parser = get_parser(language)
    tree = parser.parse(code.bytes)
    if language == "ocaml":
        constructor = parser_ocaml.OCamlParser
    elif language == "rescript":
        constructor = parser_rescript.ReScriptParser
    elif language == "lean":
        constructor = parser_lean.LeanParser
    else:
        constructor = parser_core.SymbolParser
    symbol_parser = constructor(
        code=code,
        file=file,
        language=language,
        metasymbols=metasymbols,
        node=tree.root_node,
        parent=file.symbol,
        scope="",
    )
    symbol_parser.parse_block()


def parse_path(
    path: str,
    project: IR.Project,
    filter_file: Optional[Callable[[str], bool]] = None,
    metasymbols: bool = False,
) -> None:
    """
    Parses a single file and adds it to the provided Project instance.
    """
    language = IR.language_from_file_extension(path)
    if language is not None and (filter_file is None or filter_file(path)):
        path_from_root = os.path.relpath(path, project.root_path)
        # check if the file is too large before reading it
        if os.path.getsize(path) > 5000000:
            logger.warning(f"Skipping {path_from_root} because it is too large")
            return
        with open(path, "r", encoding="utf-8") as f:
            code = IR.Code(f.read().encode("utf-8"))
        file_ir = IR.File(code=code, path=path_from_root)
        parse_code_block(file=file_ir, code=code, language=language, metasymbols=metasymbols)
        project.add_file(file=file_ir)


def parse_files_in_paths(
    paths: List[str], filter_file: Optional[Callable[[str], bool]] = None, metasymbols: bool = False
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
            parse_path(path, project, filter_file, metasymbols)
        else:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in ["node_modules", ".git"]]
                for file in files:
                    full_path = os.path.join(root, file)
                    parse_path(full_path, project, filter_file, metasymbols)
    return project
