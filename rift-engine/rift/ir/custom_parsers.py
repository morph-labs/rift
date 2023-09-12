import logging
import os

from tree_sitter import Language, Parser

from rift.util.fs import RIFT_PROJECT_DIR

TREE_SITTER_LANGUAGES_PATH = os.path.join(RIFT_PROJECT_DIR, "build", "tree-sitter-languages.so")
VENDOR_PATH = os.path.join(RIFT_PROJECT_DIR, "vendor")

# Language.build_library(
#     TREE_SITTER_LANGUAGES_PATH,
#     [os.path.join(VENDOR_PATH, 'tree-sitter-rescript')])

# ReScript = Language(TREE_SITTER_LANGUAGES_PATH, 'rescript')

parser: Parser = Parser()
