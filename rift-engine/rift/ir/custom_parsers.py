from tree_sitter import Language, Parser
Language.build_library(
    'build/tree-sitter-languages.so',
    ['vendor/tree-sitter-rescript'])

ReScript = Language('build/tree-sitter-languages.so', 'rescript')

parser: Parser = Parser()
