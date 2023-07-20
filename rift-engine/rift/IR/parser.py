from typing import Literal, List, Tuple
from tree_sitter import Language as TreeLanguage, Parser, Tree, Node
from textwrap import dedent
from rift.IR.ir import Document, FunctionDeclaration, Language, IR, Scope

TreeLanguage.build_library(
  'build/my-languages.so',
  [ 'vendor/tree-sitter-c', 'vendor/tree-sitter-javascript', 'vendor/tree-sitter-python', 'vendor/tree-sitter-typescript/tsx']
)

def find_function_declarations(code_block: str, language: Language, node: Node, scope: Scope) -> List[FunctionDeclaration]:
    document=Document(text=code_block, language=language)
    declarations: List[FunctionDeclaration] = []
    def mk_fun_decl(id: Node, node: Node):
        return FunctionDeclaration(
            document=document,
            name=code_block[id.start_byte:id.end_byte],
            range=(node.start_point, node.end_point),
            scope=scope,
            substring=(node.start_byte, node.end_byte)
        )
    if node.type in ['block']:
        print(f"block: {node}")
        pass
    if node.type in ['class_definition']:
        body = node.child_by_field_name('body')
        name = node.child_by_field_name('name')
        if body is not None and name is not None:
            scope = scope + [code_block[name.start_byte:name.end_byte]]
            for child in body.children:
                declarations += find_function_declarations(code_block, language, child, scope)
    if node.type in ['decorated_definition']: # python decorator
        defitinion = node.child_by_field_name('definition')
        if defitinion is not None:
            declarations += find_function_declarations(code_block, language, defitinion, scope)
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
        return TreeLanguage('build/my-languages.so', 'tsx')
    elif language == 'tsx':
        return TreeLanguage('build/my-languages.so', 'tsx')
    else:
        raise ValueError(f"Unknown language: {language}")
    
def parse_code_block(ir:IR, code_block: str, language: Language) -> None:
    parser = Parser()
    parser.set_language(get_tree_language(language))
    tree = parser.parse(code_block.encode())
    declarations: List[FunctionDeclaration] = []
    for node in tree.root_node.children:
        declarations += find_function_declarations(
            code_block=code_block, language=language, node=node, scope=[])
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
    code_ts = dedent("""
        type a = readonly b[][];
        function ts(x:number) { return x }
    """).lstrip()
    code_tsx = dedent("""
        d = <div> "abc" </div>
        function tsx() { return d }
    """).lstrip()
    code_py = dedent("""
        class A:
            def py(x):
                return x
        class B:
            @abstractmethod
            async def insert_code(
                self, document: str, cursor_offset: int, goal: Optional[str] = None
            ) -> InsertCodeResult:
                pass
            async def load(self):
                pass
            class Nested:
                def nested():
                    pass
    """).lstrip()
    ir = IR(symbol_table={})
    parse_code_block(ir, code_c, 'c')
    parse_code_block(ir, code_js, 'javascript')
    parse_code_block(ir, code_ts, 'typescript')
    parse_code_block(ir, code_tsx, 'tsx')
    parse_code_block(ir, code_py, 'python')
    print(f"\nCode:\n-----\n{code_c}\n----")
    print(f"\nCode:\n-----\n{code_js}\n----\n")
    print(f"\nCode:\n-----\n{code_ts}\n----\n")
    print(f"\nCode:\n-----\n{code_tsx}\n----\n")
    print(f"\nCode:\n-----\n{code_py}\n----\n")
    symbol_table = ir.symbol_table
    for id in ir.symbol_table:
        d = symbol_table[id]
        if isinstance(d, FunctionDeclaration):
            print(f"Function: {d.name} language: {d.document.language} range: {d.range} scope: {d.scope} substring: {d.substring}")
