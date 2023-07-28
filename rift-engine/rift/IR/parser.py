import os
import difflib
from typing import Literal, List, Optional, Tuple
from attr import dataclass
from tree_sitter import Node
from tree_sitter_languages import get_parser
from textwrap import dedent
from rift.IR.ir import Document, FunctionDeclaration, Language, IR, Parameter, Scope, Substring, language_from_file_extension


def get_type(text: str, language: Language, node: Node) -> str:
    if language in ["typescript", "tsx"] and node.type == 'type_annotation' and len(node.children) >= 2:
        # TS: first child should be ":" and second child should be type
        second_child = node.children[1]
        return text[second_child.start_byte:second_child.end_byte]
    return text[node.start_byte:node.end_byte]


def add_c_cpp_declarators_to_type(type: str, declarators: List[str]) -> str:
    for d in declarators:
        if d == 'pointer_declarator':
            type += '*'
        elif d == 'array_declarator':
            type += '[]'
        elif d == 'function_declarator':
            type += '()'
        elif d == 'identifier':
            pass
        else:
            raise Exception(f"Unknown declarator: {d}")
    return type


def extract_c_cpp_declarators(node: Node) -> Tuple[List[str], Node]:
    declarator_node = node.child_by_field_name('declarator')
    if declarator_node is None:
        return [], node
    declarators, final_node = extract_c_cpp_declarators(declarator_node)
    declarators.append(declarator_node.type)
    return declarators, final_node


def get_c_cpp_parameter(text: str, node: Node) -> Parameter:
    declarators, final_node = extract_c_cpp_declarators(node)
    type_node = node.child_by_field_name('type')
    if type_node is None:
        raise Exception(f"Could not find type node in {node}")
    type = text[type_node.start_byte:type_node.end_byte]
    type = add_c_cpp_declarators_to_type(type, declarators)
    name = ""
    if final_node.type == 'identifier':
        name = text[final_node.start_byte:final_node.end_byte]
    return Parameter(name=name, type=type)


def get_parameters(text: str, language: Language, node: Node) -> List[Parameter]:
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
            if language in ['c', 'cpp']:
                parameters.append(get_c_cpp_parameter(text, child))
            else:
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
                type = get_type(text=text, language=language, node=type_node)
            parameters.append(Parameter(name=name, type=type,
                              optional=child.type == 'optional_parameter'))
    return parameters


def find_c_cpp_function_declarator(node: Node) -> Optional[Tuple[List[str], Node]]:
    if node.type == 'function_declarator':
        return [], node
    declarator_node = node.child_by_field_name('declarator')
    if declarator_node is not None:
        res = find_c_cpp_function_declarator(declarator_node)
        if res is None:
            return None
        declarators, fun_node = res
        if declarator_node.type != 'function_declarator':
            declarators.append(declarator_node.type)
        return declarators, fun_node
    else:
        return None


def find_function_declarations(code_block: str, language: Language, node: Node, scope: Scope) -> List[FunctionDeclaration]:
    document = Document(text=code_block, language=language)
    declarations: List[FunctionDeclaration] = []
    docstring = ""
    body_sub = None

    def mk_fun_decl(id: Node, parameters: List[Parameter] = [], return_type: Optional[str] = None):
        return FunctionDeclaration(
            body=body_sub,
            docstring=docstring,
            document=document,
            language=language,
            name=code_block[id.start_byte:id.end_byte],
            parameters=parameters,
            range=(node.start_point, node.end_point),
            return_type=return_type,
            scope=scope,
            substring=(node.start_byte, node.end_byte)
        )

    previous_node = node.prev_sibling
    if previous_node is not None and previous_node.type == 'comment':
        docstring_ = code_block[previous_node.start_byte:previous_node.end_byte]
        if docstring_.startswith('/**'):
            docstring = docstring_

    body_node = node.child_by_field_name('body')
    if body_node is not None:
        body_sub = (body_node.start_byte, body_node.end_byte)

    if node.type in ['class_definition']:
        body_node = node.child_by_field_name('body')
        name = node.child_by_field_name('name')
        if body_node is not None and name is not None:
            scope = scope + [code_block[name.start_byte:name.end_byte]]
            for child in body_node.children:
                declarations += find_function_declarations(
                    code_block, language, child, scope)
    elif node.type in ['decorated_definition']:  # python decorator
        defitinion = node.child_by_field_name('definition')
        if defitinion is not None:
            declarations += find_function_declarations(
                code_block, language, defitinion, scope)
    elif node.type == 'function_definition' and language in ['c', 'cpp']:
        type_node = node.child_by_field_name('type')
        type = None
        if type_node is not None:
            type = get_type(text=code_block, language=language, node=type_node)
        res = find_c_cpp_function_declarator(node)
        if res is None or type is None:
            return []
        declarators, fun_node = res
        type = add_c_cpp_declarators_to_type(type, declarators)
        id: Optional[Node] = None
        parameters: List[Parameter] = []
        for child in fun_node.children:
            if child.type == 'identifier':
                id = child
            elif child.type == 'parameter_list':
                parameters = get_parameters(
                    text=code_block, language=language, node=child)
        if id is None:
            return []
        declarations.append(mk_fun_decl(
            id=id, parameters=parameters, return_type=type))
    elif node.type in ['function_definition', 'function_declaration']:
        id: Optional[Node] = None
        for child in node.children:
            if child.type == 'identifier':
                id = child
        parameters: List[Parameter] = []
        parameters_node = node.child_by_field_name('parameters')
        if parameters_node is not None:
            parameters = get_parameters(
                text=code_block, language=language, node=parameters_node)
        return_type: Optional[str] = None
        return_type_node = node.child_by_field_name('return_type')
        if return_type_node is not None:
            return_type = get_type(
                text=code_block, language=language, node=return_type_node)
        if body_node is not None and len(body_node.children) > 0 and body_node.children[0].type == 'expression_statement':
            stmt = body_node.children[0]
            if len(stmt.children) > 0 and stmt.children[0].type == 'string':
                docstring_node = stmt.children[0]
                docstring = code_block[docstring_node.start_byte:docstring_node.end_byte]
        if id is not None:
            declarations.append(mk_fun_decl(
                id=id, parameters=parameters, return_type=return_type))

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
                    declarations.append(mk_fun_decl(id=id))
    return declarations


def parse_code_block(ir: IR, code_block: str, language: Language) -> None:
    parser = get_parser(language)
    tree = parser.parse(code_block.encode())
    declarations: List[FunctionDeclaration] = []
    for node in tree.root_node.children:
        declarations += find_function_declarations(
            code_block=code_block, language=language, node=node, scope=[])
    for declaration in declarations:
        ir.symbol_table[declaration.name] = declaration


@dataclass
class MissingType:
    function_declaration: FunctionDeclaration
    parameters: List[str] = []
    return_type: bool = False

    def __str__(self) -> str:
        s = f"Function {self.function_declaration.name} is missing type annotations"
        if self.parameters != []:
            if len(self.parameters) == 1:
                s += f" in parameter '{self.parameters[0]}'"
            else:
                s += f" in parameters {self.parameters}"
        if self.return_type:
            if self.parameters != []:
                s += " and"
            s += " in return type"
        return s

    def __repr__(self) -> str:
        return self.__str__()


def functions_missing_types_in_ir(ir: IR) -> List[MissingType]:
    """Given an IR, find function declarations that are missing types in the parameters or the return type."""
    functions_missing_types: List[MissingType] = []
    for id in ir.symbol_table:
        d = ir.symbol_table[id]
        if isinstance(d, FunctionDeclaration):
            missing_parameters = []
            missing_return = False
            if d.parameters != []:
                for p in d.parameters:
                    if p.type is None and not (p.name == "self" and d.language == "python"):
                        missing_parameters.append(p.name)
            if d.return_type is None:
                missing_return = True
            if missing_parameters != [] or missing_return:
                functions_missing_types.append(MissingType(
                    function_declaration=d, parameters=missing_parameters, return_type=missing_return))
    return functions_missing_types


def functions_missing_types_in_file(path: str) -> Tuple[List[MissingType], str]:
    """Given a file path, parse the file and find function declarations that are missing types in the parameters or the return type."""
    ir = IR(symbol_table={})
    language = language_from_file_extension(path)
    if language is None:
        return ([], "")
    with open(path, 'r') as f:
        code_block = f.read()
    parse_code_block(ir, code_block, language)
    return (functions_missing_types_in_ir(ir), code_block)


def files_missing_types_in_project(root_path: str) -> List[Tuple[str, List[MissingType], str]]:
    """"Return a list of files with missing types, and the missing types in each file."""
    files_with_missing_types: List[Tuple[str, List[MissingType], str]] = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            path = os.path.join(root, file)
            (missing_types, code) = functions_missing_types_in_file(path)
            if missing_types != []:
                path_from_root = os.path.relpath(path, root_path)
                files_with_missing_types.append(
                    (path_from_root, missing_types, code))
    return files_with_missing_types


############################
#### TESTS FROM HERE ON ####
############################


class Tests:
    code_c = dedent("""
        int aa() {
          return 0;
        }
        /** This is a docstring */
        int * foo(int **x) {
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
        /** Some docstring */
        function f1() { return 0; }
        /** Some docstring on an arrow function */
        let f2 = x => x+1;
    """).lstrip()
    code_ts = dedent("""
        type a = readonly b[][];
        function ts(x:number, opt?:string) : number { return x }
        function ts2() : array<number> { return [] }
    """).lstrip()
    code_tsx = dedent("""
        d = <div> "abc" </div>
        function tsx() { return d }
    """).lstrip()
    code_py = dedent("""
        class A:
            def py(x, y):
                \"\"\"This is a docstring\"\"\"
                return x
        class B:
            @abstractmethod
            async def insert_code(
                self, document: str, cursor_offset: int, goal: Optional[str] = None
            ) -> InsertCodeResult:
                pass
            async def load(self, v):
                pass
            class Nested:
                def nested():
                    pass
    """).lstrip()


def get_ir():
    ir = IR(symbol_table={})
    parse_code_block(ir, Tests.code_c, 'c')
    parse_code_block(ir, Tests.code_js, 'javascript')
    parse_code_block(ir, Tests.code_ts, 'typescript')
    parse_code_block(ir, Tests.code_tsx, 'tsx')
    parse_code_block(ir, Tests.code_py, 'python')
    return ir


def symbol_table_to_str(symbol_table):
    lines = []
    for id in symbol_table:
        d = symbol_table[id]
        if isinstance(d, FunctionDeclaration):
            lines.append(
                f"Function: {d.name}\n   language: {d.document.language}\n   range: {d.range}\n   substring: {d.substring}")
            if d.parameters != []:
                lines.append(f"   parameters: {d.parameters}")
            if d.return_type is not None:
                lines.append(f"   return_type: {d.return_type}")
            if d.scope != []:
                lines.append(f"   scope: {d.scope}")
            if d.docstring != "":
                lines.append(f"   docstring: {d.docstring}")
            if d.body is not None:
                lines.append(f"   body: {d.body}")
    output = '\n'.join(lines)
    return output


def test_parsing():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    symbol_table_file = os.path.join(script_dir, 'symbol_table.txt')
    with open(symbol_table_file, 'r') as f:
        old_symbol_table = f.read()
    ir = get_ir()
    symbol_table_fixture = symbol_table_to_str(ir.symbol_table)
    if symbol_table_fixture != old_symbol_table:
        diff = difflib.unified_diff(old_symbol_table.splitlines(keepends=True),
                                    symbol_table_fixture.splitlines(keepends=True))
        diff_output = ''.join(diff)

        # if you want to update the symbol table, set this to True
        update_symbol_table = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_symbol_table:
            print("Updating Symbol Table...")
            with open(symbol_table_file, 'w') as f:
                f.write(symbol_table_fixture)

        assert update_symbol_table, f"Symbol Table has changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"

def test_missing_types():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    missing_types_file = os.path.join(script_dir, 'missing_types.txt')
    with open(missing_types_file, 'r') as f:
        old_missing_types = f.read()

    ir = get_ir()
    missing_types = functions_missing_types_in_ir(ir)
    new_missing_types = '\n'.join([str(mt) for mt in missing_types])
    if new_missing_types != old_missing_types:
        diff = difflib.unified_diff(old_missing_types.splitlines(keepends=True),
                                    new_missing_types.splitlines(keepends=True))
        diff_output = ''.join(diff)

        # if you want to update the missing types, set this to True
        update_missing_types = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_missing_types:
            print("Updating Missing Types...")
            with open(missing_types_file, 'w') as f:
                f.write(new_missing_types)

        assert update_missing_types, f"Missing Types have changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"


def test_missing_types_in_project():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    files = files_missing_types_in_project(parent_dir)
    for file, missing_types, code in files:
        print(f"File: {file}")
        for mt in missing_types:
            print(f"  {mt}")
        print()

