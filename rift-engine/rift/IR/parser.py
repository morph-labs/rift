from typing import Literal, List, Optional, Tuple
from tree_sitter import Node
from tree_sitter_languages import get_parser
from textwrap import dedent
from rift.IR.ir import Document, FunctionDeclaration, Language, IR, Parameter, Scope

def get_parameters(text:str, node: Node)-> List[Parameter]:
    parameters: List[Parameter] = []
    for child in node.children:
        if child.type == 'identifier':
            name = text[child.start_byte:child.end_byte]
            parameters.append(Parameter(name=name))
        elif child.type == 'typed_parameter':
            name = ""
            type = ""
            for grandchild in child.children:
                if grandchild.type == 'identifier':
                    name = text[grandchild.start_byte:grandchild.end_byte]
                elif grandchild.type == 'type':
                    type = text[grandchild.start_byte:grandchild.end_byte]
            parameters.append(Parameter(name=name, type=type))
        elif child.type == 'parameter_declaration':
            type = ""
            type_node = child.child_by_field_name('type')
            if type_node is not None:
                type = text[type_node.start_byte:type_node.end_byte]
            name = text[child.start_byte:child.end_byte]
            parameters.append(Parameter(name=name, type=type))
        elif child.type == 'required_parameter' or child.type == 'optional_parameter':
            name = ""
            pattern_node = child.child_by_field_name('pattern')
            if pattern_node is not None:
                name = text[pattern_node.start_byte:pattern_node.end_byte]
            type = None
            type_node = child.child_by_field_name('type')
            if type_node is not None:
                type = text[type_node.start_byte:type_node.end_byte]
            parameters.append(Parameter(name=name, type=type, optional=child.type == 'optional_parameter'))
    return parameters

def find_function_declarations(code_block: str, language: Language, node: Node, scope: Scope) -> List[FunctionDeclaration]:
    document=Document(text=code_block, language=language)
    declarations: List[FunctionDeclaration] = []
    def mk_fun_decl(id: Node, node: Node, parameters: List[Parameter] = [], return_type: Optional[str] = None):
        return FunctionDeclaration(
            document=document,
            name=code_block[id.start_byte:id.end_byte],
            parameters=parameters,
            range=(node.start_point, node.end_point),
            return_type=return_type,
            scope=scope,
            substring=(node.start_byte, node.end_byte)
        )
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
        id: Optional[Node] = None
        parameters: List[Parameter] = []
        for child in node.children:
            if child.type == 'function_declarator': # in C
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        id = grandchild
                    elif grandchild.type == 'parameter_list':
                        parameters = get_parameters(text=code_block, node=grandchild)
            elif child.type == 'identifier':
                id = child
        parameters_node = node.child_by_field_name('parameters')
        if parameters_node is not None:
            parameters = get_parameters(text=code_block, node=parameters_node)

        return_type: Optional[str] = None
        return_type_node = node.child_by_field_name('return_type')
        if return_type_node is None:
            return_type_node = node.child_by_field_name('type')
        if return_type_node is not None:
            return_type = code_block[return_type_node.start_byte:return_type_node.end_byte]

        if id is not None:
            declarations.append(mk_fun_decl(id=id, node=node, parameters=parameters, return_type=return_type))

    elif node.type in ['lexical_declaration', 'variable_declaration']:
        # arrow functions in js/ts e.g. let foo = x => x+1
        for child in node.children:
            if child.type == 'variable_declarator':
                # look for identifier and arrow_function
                is_arrow_function = False
                id: Optional[Node] = None
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        id = grandchild
                    elif grandchild.type == 'arrow_function':
                        is_arrow_function = True
                if is_arrow_function and id is not None:
                    declarations.append(mk_fun_decl(id, node))
    return declarations
    
def parse_code_block(ir:IR, code_block: str, language: Language) -> None:
    parser = get_parser(language)
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
        function ts(x:number, opt?:string) : number { return x }
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
            print(f"Function: {d.name}\n   language: {d.document.language}\n   range: {d.range}\n   substring: {d.substring}")
            if d.parameters != []:
                print(f"   parameters: {d.parameters}")
            if d.return_type is not None:
                print(f"   return_type: {d.return_type}")
            if d.scope != []:
                print(f"   scope: {d.scope}")
