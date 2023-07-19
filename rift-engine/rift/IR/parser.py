from typing import Literal, List, Tuple
from tree_sitter import Language as TreeLanguage, Parser, Tree, Node
from textwrap import dedent
from rift.IR.ir import Document, FunctionDeclaration, Language, IR

TreeLanguage.build_library(
  'build/my-languages.so',
  [ 'vendor/tree-sitter-c', 'vendor/tree-sitter-javascript', 'vendor/tree-sitter-python', 'vendor/tree-sitter-typescript/typescript']
)

def find_function_declarations(code_block: str, language: Language, node: Node) -> List[FunctionDeclaration]:
    document=Document(text=code_block, language=language)
    declarations: List[FunctionDeclaration] = []
    def mk_fun_decl(id: Node, node: Node):
        return FunctionDeclaration(
            document=document,
            name=code_block[id.start_byte:id.end_byte],
            range=(node.start_point, node.end_point),
            substring=(node.start_byte, node.end_byte)
        )
    if node.type in ['function_definition', 'function_declaration']:
        for child in node.children:
            if child.type == 'function_declarator': # in C
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        declarations.append(mk_fun_decl(grandchild, node))
            elif child.type == 'identifier':
                declarations.append(mk_fun_decl(child, node))
                break
    elif node.type in ['lexical_declaration', 'variable_declaration']:
        # arrow functions in js/ts e.g. let foo = x => x+1
        for child in node.children:
            if child.type == 'variable_declarator':
                # look for identifier and arrow_function
                is_arrow_function = False
                id = None
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        id = grandchild
                    elif grandchild.type == 'arrow_function':
                        is_arrow_function = True
                if is_arrow_function and id is not None:
                    declarations.append(mk_fun_decl(id, node))
    return declarations

def get_tree_language(language: Language) -> TreeLanguage:
    if language == 'c':
        return TreeLanguage('build/my-languages.so', 'c')
    elif language == 'javascript':
        return TreeLanguage('build/my-languages.so', 'javascript')
    elif language == 'python':
        return TreeLanguage('build/my-languages.so', 'python')
    elif language == 'typescript':
        return TreeLanguage('build/my-languages.so', 'typescript')
    else:
        raise ValueError(f"Unknown language: {language}")
    
def parse_code_block(ir:IR, code_block: str, language: Language) -> None:
    parser = Parser()
    parser.set_language(get_tree_language(language))
    tree = parser.parse(code_block.encode())
    declarations: List[FunctionDeclaration] = []
    for node in tree.root_node.children:
        declarations += find_function_declarations(
            code_block=code_block, language=language, node=node)
    for declaration in declarations:
        ir.symbol_table[declaration.name] = declaration

if __name__ == "__main__":
    code_c = dedent("""
        int aa() {
          return 0;
        }

        void foo(int **x) {
          *x = 0;
        }

        int bb() {
          return 0;
        }

        int main() {
          int *x;
          foo(&x);
          *x = 1;
          return 0;
        }
    """).lstrip()
    code_js = dedent("""
        function f1() { return 0; }
        let f2 = x => x+1;
    """).lstrip()
    ir = IR(symbol_table={})
    parse_code_block(ir, code_c, 'c')
    parse_code_block(ir, code_js, 'javascript')
    print(f"\nCode:\n-----\n{code_c}\n----")
    print(f"\nCode:\n-----\n{code_js}\n----\n")
    symbol_table = ir.symbol_table
    for id in ir.symbol_table:
        d = symbol_table[id]
        if isinstance(d, FunctionDeclaration):
            print(f"Function: {d.name} language: {d.document.language} range: {d.range} substring: {d.substring}")
