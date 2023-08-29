from tree_sitter import Language, Parser
Language.build_library('build/tree-sitter-languages.so', [ 'vendor/tree-sitter-lean' ] )

LEAN_LANGUAGE = Language('build/tree-sitter-languages.so', 'lean')

parser : Parser = Parser()
parser.set_language(LEAN_LANGUAGE)
