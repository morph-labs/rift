import os
import difflib
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from tree_sitter import Node
from tree_sitter_languages import get_parser
from textwrap import dedent
from rift.ir.IR import \
    ClassDeclaration, Code, Declaration, File, FunctionDeclaration, Language, Parameter, Project, Scope, Statement, SymbolInfo,\
    language_from_file_extension


def get_type(code: Code, language: Language, node: Node) -> str:
    if language in ["typescript", "tsx"] and node.type == 'type_annotation' and len(node.children) >= 2:
        # TS: first child should be ":" and second child should be type
        second_child = node.children[1]
        return code.bytes[second_child.start_byte:second_child.end_byte].decode()
    return code.bytes[node.start_byte:node.end_byte].decode()


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


def get_c_cpp_parameter(code: Code, node: Node) -> Parameter:
    declarators, final_node = extract_c_cpp_declarators(node)
    type_node = node.child_by_field_name('type')
    if type_node is None:
        raise Exception(f"Could not find type node in {node}")
    type = code.bytes[type_node.start_byte:type_node.end_byte].decode()
    type = add_c_cpp_declarators_to_type(type, declarators)
    name = ""
    if final_node.type == 'identifier':
        name = code.bytes[final_node.start_byte:final_node.end_byte].decode()
    return Parameter(name=name, type=type)


def get_parameters(code: Code, language: Language, node: Node) -> List[Parameter]:
    parameters: List[Parameter] = []
    for child in node.children:
        if child.type == 'identifier':
            name = code.bytes[child.start_byte:child.end_byte].decode()
            parameters.append(Parameter(name=name))
        elif child.type == 'typed_parameter':
            name = ""
            type = ""
            for grandchild in child.children:
                if grandchild.type == 'identifier':
                    name = code.bytes[grandchild.start_byte:grandchild.end_byte].decode(
                    )
                elif grandchild.type == 'type':
                    type = code.bytes[grandchild.start_byte:grandchild.end_byte].decode(
                    )
            parameters.append(Parameter(name=name, type=type))
        elif child.type == 'parameter_declaration':
            if language in ['c', 'cpp']:
                parameters.append(get_c_cpp_parameter(code, child))
            else:
                type = ""
                type_node = child.child_by_field_name('type')
                if type_node is not None:
                    type = code.bytes[type_node.start_byte:type_node.end_byte].decode(
                    )
                name = code.bytes[child.start_byte:child.end_byte].decode()
                parameters.append(Parameter(name=name, type=type))
        elif child.type == 'required_parameter' or child.type == 'optional_parameter':
            name = ""
            pattern_node = child.child_by_field_name('pattern')
            if pattern_node is not None:
                name = code.bytes[pattern_node.start_byte:pattern_node.end_byte].decode(
                )
            type = None
            type_node = child.child_by_field_name('type')
            if type_node is not None:
                type = get_type(code=code, language=language, node=type_node)
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


def find_declaration(code: Code, file: File, language: Language, node: Node, scope: Scope) -> Optional[SymbolInfo]:
    docstring: str = ""
    body_sub = None

    def mk_fun_decl(id: Node, parameters: List[Parameter] = [], return_type: Optional[str] = None):
        return FunctionDeclaration(
            body_sub=body_sub,
            docstring=docstring,
            code=code,
            language=language,
            name=code.bytes[id.start_byte:id.end_byte].decode(),
            parameters=parameters,
            range=(node.start_point, node.end_point),
            return_type=return_type,
            scope=scope,
            substring=(node.start_byte, node.end_byte)
        )

    def mk_class_decl(id: Node, body: List[Statement]):
        return ClassDeclaration(
            body=body,
            code=code,
            docstring=docstring,
            language=language,
            name=code.bytes[id.start_byte:id.end_byte].decode(),
            range=(node.start_point, node.end_point),
            scope=scope,
            substring=(node.start_byte, node.end_byte)
        )

    previous_node = node.prev_sibling
    if previous_node is not None and previous_node.type == 'comment':
        docstring_ = code.bytes[previous_node.start_byte:previous_node.end_byte].decode(
        )
        if docstring_.startswith('/**'):
            docstring = docstring_

    body_node = node.child_by_field_name('body')
    if body_node is not None:
        body_sub = (body_node.start_byte, body_node.end_byte)

    if node.type in ['class_definition']:
        body_node = node.child_by_field_name('body')
        name = node.child_by_field_name('name')
        if body_node is not None and name is not None:
            scope = scope + \
                [code.bytes[name.start_byte:name.end_byte].decode()]
            body = process_body(
                code=code, file=file, language=language, node=body_node, scope=scope)
            docstring = ""
            # see if the first child is a string expression statemetns, and if so, use it as the docstring
            if body_node.child_count > 0 and body_node.children[0].type == 'expression_statement':
                stmt = body_node.children[0]
                if len(stmt.children) > 0 and stmt.children[0].type == 'string':
                    docstring_node = stmt.children[0]
                    docstring = code.bytes[docstring_node.start_byte:docstring_node.end_byte].decode(
                    )
            declaration = mk_class_decl(id=name, body=body)
            file.add_symbol(declaration)
            return declaration
    elif node.type in ['decorated_definition']:  # python decorator
        defitinion = node.child_by_field_name('definition')
        if defitinion is not None:
            return find_declaration(
                code, file, language, defitinion, scope)
    elif node.type == 'function_definition' and language in ['c', 'cpp']:
        type_node = node.child_by_field_name('type')
        type = None
        if type_node is not None:
            type = get_type(code=code, language=language, node=type_node)
        res = find_c_cpp_function_declarator(node)
        if res is None or type is None:
            return None
        declarators, fun_node = res
        type = add_c_cpp_declarators_to_type(type, declarators)
        id: Optional[Node] = None
        parameters: List[Parameter] = []
        for child in fun_node.children:
            if child.type == 'identifier':
                id = child
            elif child.type == 'parameter_list':
                parameters = get_parameters(
                    code=code, language=language, node=child)
        if id is None:
            return None
        declaration = mk_fun_decl(
            id=id, parameters=parameters, return_type=type)
        file.add_symbol(declaration)
        return declaration
    elif node.type in ['function_definition', 'function_declaration']:
        id: Optional[Node] = None
        for child in node.children:
            if child.type == 'identifier':
                id = child
        parameters: List[Parameter] = []
        parameters_node = node.child_by_field_name('parameters')
        if parameters_node is not None:
            parameters = get_parameters(
                code=code, language=language, node=parameters_node)
        return_type: Optional[str] = None
        return_type_node = node.child_by_field_name('return_type')
        if return_type_node is not None:
            return_type = get_type(
                code=code, language=language, node=return_type_node)
        if body_node is not None and len(body_node.children) > 0 and body_node.children[0].type == 'expression_statement':
            stmt = body_node.children[0]
            if len(stmt.children) > 0 and stmt.children[0].type == 'string':
                docstring_node = stmt.children[0]
                docstring = code.bytes[docstring_node.start_byte:docstring_node.end_byte].decode(
                )
        if id is not None:
            declaration = (mk_fun_decl(
                id=id, parameters=parameters, return_type=return_type))
            file.add_symbol(declaration)
            return declaration

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
                    declaration = mk_fun_decl(id=id)
                    file.add_symbol(declaration)
                    return declaration


def process_statement(code: Code, file: File, language: Language, node: Node, scope: Scope) -> Statement:
    declaration = find_declaration(
        code=code, file=file, language=language, node=node, scope=scope)
    if declaration is not None:
        return Declaration(type=node.type, symbol=declaration)
    else:
        return Statement(type=node.type)


def process_body(code: Code, file: File, language: Language, node: Node, scope: Scope) -> List[Statement]:
    return [process_statement(code=code, file=file, language=language, node=child, scope=scope)
            for child in node.children]


def parse_code_block(file: File, code: Code, language: Language) -> None:
    parser = get_parser(language)
    tree = parser.parse(code.bytes)
    for node in tree.root_node.children:
        statement = process_statement(
            code=code, file=file, language=language, node=node, scope=[])
        file.statements.append(statement)


@dataclass
class MissingType:
    function_declaration: FunctionDeclaration
    parameters: List[str] = field(default_factory=list)
    return_type: bool = False

    def __str__(self) -> str:
        s = f"Function `{self.function_declaration.name}` is missing type annotations"
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

    def __int__(self) -> int:
        return len(self.parameters) + int(self.return_type)


def functions_missing_types_in_file(file: File) -> List[MissingType]:
    """Given an IR, find function declarations that are missing types in the parameters or the return type."""
    functions_missing_types: List[MissingType] = []
    function_declarations = file.get_function_declarations()
    for d in function_declarations:
        missing_parameters = []
        missing_return = False
        parameters = d.parameters
        if parameters != []:
            if (parameters[0].name == "self" or parameters[0].name == "cls") and d.language == "python" and d.scope != []:
                parameters = parameters[1:]
            for p in parameters:
                if p.type is None:
                    missing_parameters.append(p.name)
        if d.return_type is None:
            missing_return = True
        if missing_parameters != [] or missing_return:
            functions_missing_types.append(MissingType(
                function_declaration=d, parameters=missing_parameters, return_type=missing_return))
    return functions_missing_types


def functions_missing_types_in_path(path: str) -> Tuple[List[MissingType], Code, File]:
    """Given a file path, parse the file and find function declarations that are missing types in the parameters or the return type."""
    file = File(path)
    language = language_from_file_extension(path)
    missing_types: List[MissingType] = []
    if language is None:
        missing_types = []
        code = Code(b"")
    else:
        with open(path, 'r', encoding='utf-8') as f:
            code = Code(f.read().encode('utf-8'))
        parse_code_block(file, code, language)
        missing_types = functions_missing_types_in_file(file)
    return (missing_types, code, file)


@dataclass
class FileMissingTypes:
    code: Code  # code of the file
    ir: File  # IR of the file
    language: Language  # language of the file
    missing_types: List[MissingType]  # list of missing types in the file
    path_from_root: str  # path of the file relative to the root directory


def files_missing_types_in_project(root_path: str) -> List[FileMissingTypes]:
    """"Return a list of files with missing types, and the missing types in each file."""
    files_with_missing_types: List[FileMissingTypes] = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            language = language_from_file_extension(file)
            if language is not None:
                path = os.path.join(root, file)
                (missing_types, code, ir) = functions_missing_types_in_path(path)
                if missing_types != []:
                    path_from_root = os.path.relpath(path, root_path)
                    files_with_missing_types.append(
                        FileMissingTypes(code, ir, language, missing_types, path_from_root))
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
    """).lstrip().encode('utf-8')
    code_js = dedent("""
        /** Some docstring */
        function f1() { return 0; }
        /** Some docstring on an arrow function */
        let f2 = x => x+1;
    """).lstrip().encode('utf-8')
    code_ts = dedent("""
        type a = readonly b[][];
        function ts(x:number, opt?:string) : number { return x }
        function ts2() : array<number> { return [] }
    """).lstrip().encode('utf-8')
    code_tsx = dedent("""
        d = <div> "abc" </div>
        function tsx() { return d }
    """).lstrip().encode("utf-8")
    code_py = dedent("""
        class A:
            \"\"\"
            This is a docstring
            for class A
            \"\"\"

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
    """).lstrip().encode("utf-8")


def get_test_project():
    project = Project()
    def new_file(path):
        file = File(path)
        project.add_file(file)
        return file
    file_c = new_file("test.c")
    parse_code_block(file_c, Code(Tests.code_c), 'c')
    file_js = new_file("test.js")
    parse_code_block(file_js, Code(Tests.code_js), 'javascript')
    file_ts = new_file("test.ts")
    parse_code_block(file_ts, Code(Tests.code_ts), 'typescript')
    file_tsx = new_file("test.tsx")
    parse_code_block(file_tsx, Code(Tests.code_tsx), 'tsx')
    file_py = new_file("test.py")
    parse_code_block(file_py, Code(Tests.code_py), 'python')
    return project


def test_parsing():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    symbol_table_file = os.path.join(script_dir, 'symbol_table.txt')
    with open(symbol_table_file, 'r') as f:
        old_symbol_table = f.read()
    project = get_test_project()

    lines = []
    for file in project.get_files():
        lines.append(f"=== Symbol Table for {file.path} ===")
        lines.append(file.dump_symbol_table())
    symbol_table_str = '\n'.join(lines)
    ir_map_str = project.dump_map(indent=0)
    symbol_table_str += "\n\n=== Project Map ===\n" + ir_map_str
    if symbol_table_str != old_symbol_table:
        diff = difflib.unified_diff(old_symbol_table.splitlines(keepends=True),
                                    symbol_table_str.splitlines(keepends=True))
        diff_output = ''.join(diff)

        # if you want to update the symbol table, set this to True
        update_symbol_table = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_symbol_table:
            print("Updating Symbol Table...")
            with open(symbol_table_file, 'w') as f:
                f.write(symbol_table_str)

        assert update_symbol_table, f"Symbol Table has changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"


def test_missing_types():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    missing_types_file = os.path.join(script_dir, 'missing_types.txt')
    with open(missing_types_file, 'r') as f:
        old_missing_types_str = f.read()

    project = get_test_project()
    new_missing_types = []
    for file in project.get_files():
        missing_types = functions_missing_types_in_file(file)
        new_missing_types += ([str(mt) for mt in missing_types])
    new_missing_types_str = '\n'.join(new_missing_types)
    if new_missing_types_str != old_missing_types_str:
        diff = difflib.unified_diff(old_missing_types_str.splitlines(keepends=True),
                                    new_missing_types_str.splitlines(keepends=True))
        diff_output = ''.join(diff)

        # if you want to update the missing types, set this to True
        update_missing_types = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_missing_types:
            print("Updating Missing Types...")
            with open(missing_types_file, 'w') as f:
                f.write(new_missing_types_str)

        assert update_missing_types, f"Missing Types have changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"


def test_missing_types_in_project():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    files_missing_types = files_missing_types_in_project(parent_dir)
    for fmt in files_missing_types:
        print(f"File: {fmt.path_from_root}")
        for mt in fmt.missing_types:
            print(f"  {mt}")
        print()
