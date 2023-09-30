import logging
import os

from tree_sitter import Language, Parser

from rift.util.fs import RIFT_PROJECT_DIR

TREE_SITTER_LANGUAGES_PATH = os.path.join(RIFT_PROJECT_DIR, "build", "tree-sitter-languages.so")
VENDOR_PATH = os.path.join(RIFT_PROJECT_DIR, "vendor")

active = True

ReScript = None
Lean = None

logger = logging.getLogger(__name__)


def activate():
    try:
        global active, ReScript, Lean
        Language.build_library(
            TREE_SITTER_LANGUAGES_PATH,
            [
                os.path.join(VENDOR_PATH, "tree-sitter-rescript"),
                os.path.join(VENDOR_PATH, "tree-sitter-lean"),
            ],
        )
        ReScript = Language(TREE_SITTER_LANGUAGES_PATH, "rescript")
        Lean = Language(TREE_SITTER_LANGUAGES_PATH, "lean")
        active = True
    except Exception as e:
        logger.error("Failed to activate custom parsers: %s", e)


if active:
    activate()

parser: Parser = Parser()
