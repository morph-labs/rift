import os
from typing import Callable, List, Optional, Tuple

from rift.ir.IR import (
    ClassKind,
    Code,
    ContainerDeclaration,
    ContainerKind,
    Declaration,
    File,
    FunctionKind,
    InterfaceKind,
    Language,
    ModuleKind,
    NamespaceKind,
    Parameter,
    Project,
    Scope,
    Statement,
    SymbolInfo,
    TypeKind,
    ValKind,
    ValueDeclaration,
    ValueKind,
    language_from_file_extension,
)
import rift.ir.custom_parsers as custom_parser
from tree_sitter import Node, Parser
from tree_sitter_languages import get_parser as get_tree_sitter_parser

def get_type(code: Code, language: Language, node: Node) -> str:
    if (
        language in ["typescript", "tsx"]
        and node.type == "type_annotation"
        and len(node.children) >= 2
    ):
        # TS: first child should be ":" and second child should be type
        second_child = node.children[1]
        return code.bytes[second_child.start_byte : second_child.end_byte].decode()
    return code.bytes[node.start_byte : node.end_byte].decode()


def add_c_cpp_declarators_to_type(type: str, declarators: List[str]) -> str:
    for d in declarators:
        if d == "pointer_declarator":
            type += "*"
        elif d == "array_declarator":
            type += "[]"
        elif d == "function_declarator":
            type += "()"
        elif d == "identifier":
            pass
        else:
            raise Exception(f"Unknown declarator: {d}")
    return type


def extract_c_cpp_declarators(node: Node) -> Tuple[List[str], Node]:
    declarator_node = node.child_by_field_name("declarator")
    if declarator_node is None:
        return [], node
    declarators, final_node = extract_c_cpp_declarators(declarator_node)
    declarators.append(declarator_node.type)
    return declarators, final_node


def get_c_cpp_parameter(code: Code, node: Node) -> Parameter:
    declarators, final_node = extract_c_cpp_declarators(node)
    type_node = node.child_by_field_name("type")
    if type_node is None:
        raise Exception(f"Could not find type node in {node}")
    type = code.bytes[type_node.start_byte : type_node.end_byte].decode()
    type = add_c_cpp_declarators_to_type(type, declarators)
    name = ""
    if final_node.type == "identifier":
        name = code.bytes[final_node.start_byte : final_node.end_byte].decode()
    return Parameter(name=name, type=type)


def get_parameters(code: Code, language: Language, node: Node) -> List[Parameter]:
    parameters: List[Parameter] = []
    for child in node.children:
        if child.type == "identifier":
            name = code.bytes[child.start_byte : child.end_byte].decode()
            parameters.append(Parameter(name=name))
        elif child.type == "typed_parameter":
            name = ""
            type = ""
            for grandchild in child.children:
                if grandchild.type == "identifier":
                    name = code.bytes[grandchild.start_byte : grandchild.end_byte].decode()
                elif grandchild.type == "type":
                    type = code.bytes[grandchild.start_byte : grandchild.end_byte].decode()
            parameters.append(Parameter(name=name, type=type))
        elif child.type == "parameter_declaration":
            if language in ["c", "cpp"]:
                parameters.append(get_c_cpp_parameter(code, child))
            else:
                type = ""
                type_node = child.child_by_field_name("type")
                if type_node is not None:
                    type = code.bytes[type_node.start_byte : type_node.end_byte].decode()
                name = code.bytes[child.start_byte : child.end_byte].decode()
                parameters.append(Parameter(name=name, type=type))
        elif child.type == "required_parameter" or child.type == "optional_parameter":
            name = ""
            pattern_node = child.child_by_field_name("pattern")
            if pattern_node is not None:
                name = code.bytes[pattern_node.start_byte : pattern_node.end_byte].decode()
            type = None
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                type = get_type(code=code, language=language, node=type_node)
            parameters.append(
                Parameter(name=name, type=type, optional=child.type == "optional_parameter")
            )
    return parameters


def find_c_cpp_function_declarator(node: Node) -> Optional[Tuple[List[str], Node]]:
    if node.type == "function_declarator":
        return [], node
    declarator_node = node.child_by_field_name("declarator")
    if declarator_node is not None:
        res = find_c_cpp_function_declarator(declarator_node)
        if res is None:
            return None
        declarators, fun_node = res
        if declarator_node.type != "function_declarator":
            declarators.append(declarator_node.type)
        return declarators, fun_node
    else:
        return None


def contains_direct_return(body: Node):
    """
    Recursively check if the function body contains a direct return statement.
    """
    for child in body.children:
        # If the child is a function or method, skip it.
        if child.type in [
            "arrow_function",
            "class_definition",
            "class_declaration",
            "function_declaration",
            "function_definition",
            "method_definition",
        ]:
            continue
        # If the child is a return statement, return True.
        if child.type == "return_statement":
            return True
        # If the child has its own children, recursively check them.
        if contains_direct_return(child):
            return True
    return False


def find_declarations(
    code: Code, file: File, language: Language, node: Node, scope: Scope
) -> List[SymbolInfo]:
    body_sub = None
    docstring: str = ""
    exported = False
    has_return = False

    def dump_node(node: Node) -> str:
        """ Dump a node for debugging purposes. """
        return f"  type:{node.type} children:{node.child_count}\n  code:{code.bytes[node.start_byte: node.end_byte].decode()}\n  sexp:{node.sexp()}"

    def mk_value_decl(id: Node, parents: List[Node], value_kind: ValueKind):
        return ValueDeclaration(
            body_sub=body_sub,
            code=code,
            docstring=docstring,
            exported=exported,
            language=language,
            name=code.bytes[id.start_byte : id.end_byte].decode(),
            range=(parents[0].start_point, parents[-1].end_point),
            scope=scope,
            substring=(parents[0].start_byte, parents[-1].end_byte),
            value_kind=value_kind,
        )

    def mk_fun_decl(id: Node, parents: List[Node], parameters: List[Parameter] = [], return_type: Optional[str] = None):
        value_kind = FunctionKind(has_return=has_return, parameters=parameters, return_type=return_type)
        return mk_value_decl(id=id, parents=parents, value_kind=value_kind)

    def mk_val_decl(id: Node, parents: List[Node], type: Optional[str] = None):
        value_kind = ValKind(type=type)
        return mk_value_decl(id=id, parents=parents, value_kind=value_kind)

    def mk_type_decl(id: Node, parents: List[Node]):
        value_kind = TypeKind()
        return mk_value_decl(id=id, parents=parents, value_kind=value_kind)

    def mk_interface_decl(id: Node, parents: List[Node]):
        value_kind = InterfaceKind()
        return mk_value_decl(id=id, parents=parents, value_kind=value_kind)

    def mk_container_decl(id: Node, parents: List[Node], body: List[Statement], container_kind: ContainerKind):
        return ContainerDeclaration(
            container_kind=container_kind,
            body=body,
            body_sub=body_sub,
            code=code,
            docstring=docstring,
            exported=exported,
            language=language,
            name=code.bytes[id.start_byte : id.end_byte].decode(),
            range=(parents[0].start_point, parents[-1].end_point),
            scope=scope,
            substring=(parents[0].start_byte, parents[-1].end_byte),
        )

    def mk_class_decl(id: Node, body: List[Statement], parents: List[Node], superclasses: Optional[str]):
        container_kind = ClassKind(superclasses=superclasses)
        return mk_container_decl(id=id, body=body, container_kind=container_kind, parents=parents)

    def mk_namespace_decl(id: Node, body: List[Statement], parents: List[Node]):
        container_kind = NamespaceKind()
        return mk_container_decl(id=id, body=body, container_kind=container_kind, parents=parents)

    def mk_module_decl(id: Node, body: List[Statement], parents: List[Node]):
        container_kind = ModuleKind()
        return mk_container_decl(id=id, body=body, container_kind=container_kind, parents=parents)

    previous_node = node.prev_sibling
    if previous_node is not None and previous_node.type == "comment":
        docstring_ = code.bytes[previous_node.start_byte : previous_node.end_byte].decode()
        if docstring_.startswith("/**"):
            docstring = docstring_

    body_node = node.child_by_field_name("body")
    if body_node is not None:
        body_sub = (body_node.start_byte, body_node.end_byte)
    def process_ocaml_body(n: Node) -> None:
        nonlocal body_node, body_sub
        body_node = n.child_by_field_name("body")
        if body_node is not None:
            node_before = body_node.prev_sibling
            if node_before is not None and node_before.type == "=":
                # consider "=" part of the body
                body_sub = (node_before.start_byte, body_node.end_byte)
            else:
                body_sub = (body_node.start_byte, body_node.end_byte)


    if node.type in [
        "class_definition",
        "class_declaration",
        "class_specifier",
        "namespace_definition",
    ]:
        is_namespace = node.type == "namespace_definition"
        superclasses_node = node.child_by_field_name("superclasses")
        superclasses = None
        if superclasses_node is not None:
            superclasses = code.bytes[
                superclasses_node.start_byte : superclasses_node.end_byte
            ].decode()
        body_node = node.child_by_field_name("body")
        name = node.child_by_field_name("name")
        if body_node is not None and name is not None:
            if is_namespace:
                separator = "::"
            else:
                separator = "."
            scope = scope + code.bytes[name.start_byte : name.end_byte].decode() + separator
            body = process_body(
                code=code, file=file, language=language, node=body_node, scope=scope
            )
            docstring = ""
            # see if the first child is a string expression statemetns, and if so, use it as the docstring
            if body_node.child_count > 0 and body_node.children[0].type == "expression_statement":
                stmt = body_node.children[0]
                if len(stmt.children) > 0 and stmt.children[0].type == "string":
                    docstring_node = stmt.children[0]
                    docstring = code.bytes[
                        docstring_node.start_byte : docstring_node.end_byte
                    ].decode()
            if is_namespace:
                declaration = mk_namespace_decl(id=name, body=body, parents=[node])
            else:
                declaration = mk_class_decl(id=name, body=body, parents=[node], superclasses=superclasses)
            file.add_symbol(declaration)
            return [declaration]

    elif node.type in ["decorated_definition"]:  # python decorator
        defitinion = node.child_by_field_name("definition")
        if defitinion is not None:
            return find_declarations(code, file, language, defitinion, scope)

    elif node.type in ["field_declaration", "function_definition"] and language in ["c", "cpp"]:
        type_node = node.child_by_field_name("type")
        type = None
        if type_node is not None:
            type = get_type(code=code, language=language, node=type_node)
        res = find_c_cpp_function_declarator(node)
        if res is None or type is None:
            return []
        declarators, fun_node = res
        type = add_c_cpp_declarators_to_type(type, declarators)
        id: Optional[Node] = None
        parameters: List[Parameter] = []
        for child in fun_node.children:
            if child.type in ["field_identifier", "identifier"]:
                id = child
            elif child.type == "parameter_list":
                parameters = get_parameters(code=code, language=language, node=child)
        if id is None:
            return []
        declaration = mk_fun_decl(id=id, parameters=parameters, return_type=type, parents=[node])
        file.add_symbol(declaration)
        return [declaration]

    elif node.type in ["function_definition", "function_declaration", "method_definition"]:
        id: Optional[Node] = None
        for child in node.children:
            if child.type in ["identifier", "property_identifier"]:
                id = child
        parameters: List[Parameter] = []
        parameters_node = node.child_by_field_name("parameters")
        if parameters_node is not None:
            parameters = get_parameters(code=code, language=language, node=parameters_node)
        return_type: Optional[str] = None
        return_type_node = node.child_by_field_name("return_type")
        if return_type_node is not None:
            return_type = get_type(code=code, language=language, node=return_type_node)
        if (
            body_node is not None
            and len(body_node.children) > 0
            and body_node.children[0].type == "expression_statement"
        ):
            stmt = body_node.children[0]
            if len(stmt.children) > 0 and stmt.children[0].type == "string":
                docstring_node = stmt.children[0]
                docstring = code.bytes[docstring_node.start_byte : docstring_node.end_byte].decode()
        if body_node is not None:
            has_return = contains_direct_return(body_node)
        if id is not None:
            declaration = mk_fun_decl(id=id, parents=[node], parameters=parameters, return_type=return_type)
            file.add_symbol(declaration)
            return [declaration]

    elif node.type in ["lexical_declaration", "variable_declaration"]:
        # arrow functions in js/ts e.g. let foo = x => x+1
        for child in node.children:
            if child.type == "variable_declarator":
                # look for identifier and arrow_function
                is_arrow_function = False
                id: Optional[Node] = None
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        id = grandchild
                    elif grandchild.type == "arrow_function":
                        is_arrow_function = True
                if is_arrow_function and id is not None:
                    declaration = mk_fun_decl(id=id, parents=[node])
                    file.add_symbol(declaration)
                    return [declaration]

    elif node.type == "export_statement" and language in ["js", "typescript", "tsx"]:
        if len(node.children) >= 2:
            exported = True
            return find_declarations(
                code=code, file=file, language=language, node=node.children[1], scope=scope
            )

    elif node.type in ["interface_declaration", "type_alias_declaration"]:
        id: Optional[Node] = node.child_by_field_name("name")
        if id is not None:
            if node.type == "interface_declaration":
                declaration = mk_interface_decl(id=id, parents=[node])
            else:
                declaration = mk_type_decl(id=id, parents=[node])
            file.add_symbol(declaration)
            return [declaration]

    elif node.type == "value_definition" and language == "ocaml":
        parameters = []
        def extract_type(node: Node) -> str:
            return code.bytes[node.start_byte: node.end_byte].decode()
        def parse_inner_parameter(inner: Node) -> Optional[Parameter]:
            if inner.type in ["label_name", "value_pattern"]:
                name = code.bytes[inner.start_byte: inner.end_byte].decode()
                return Parameter(name=name)
            elif inner.type == "typed_pattern" and inner.child_count == 5 and inner.children[2].type == ":":
                # "(", par, ":", typ, ")"
                id = inner.children[1]
                tp = inner.children[3]
                if id.type == "value_pattern":
                    name = code.bytes[id.start_byte: id.end_byte].decode()
                    type = extract_type(tp)
                    return Parameter(name=name, type=type)
            elif inner.type == "unit":
                name = "()"
                type = "unit"
                return Parameter(name=name, type=type)
        def parse_ocaml_parameter(parameter: Node) -> None:
            if parameter.child_count == 1:
                inner_parameter = parse_inner_parameter(parameter.children[0])
                if inner_parameter is not None:
                    parameters.append(inner_parameter)
            elif parameter.child_count == 2 and parameter.children[0].type in ["~", "?"]:
                inner_parameter = parse_inner_parameter(parameter.children[1])
                if inner_parameter is not None:
                    inner_parameter.name = parameter.children[0].type + inner_parameter.name
                    parameters.append(inner_parameter)
            elif parameter.child_count == 4 and parameter.children[0].type in ["~", "?"] and parameter.children[2].type == ":":
                # "~", par, ":", name
                inner_parameter = parse_inner_parameter(parameter.children[1])
                if inner_parameter is not None:
                    inner_parameter.name = parameter.children[0].type + inner_parameter.name
                    parameters.append(inner_parameter)
            elif parameter.child_count == 6 and parameter.children[0].type in ["~", "?"] and parameter.children[3].type == ":":
                # "~", "(", par, ":", typ, ")"
                inner_parameter = parse_inner_parameter(parameter.children[2])
                if inner_parameter is not None:
                    inner_parameter.name = parameter.children[0].type + inner_parameter.name
                    type = extract_type(parameter.children[4])
                    inner_parameter.type = type
                    parameters.append(inner_parameter)
            elif parameter.child_count == 6 and parameter.children[0].type == "?" and parameter.children[3].type == "=":
                # "?", "(", par, "=", val, ")"
                inner_parameter = parse_inner_parameter(parameter.children[2])
                if inner_parameter is not None:
                    inner_parameter.name = parameter.children[0].type + inner_parameter.name
                    type = "type of " + extract_type(parameter.children[4])
                    inner_parameter.type = type
                    parameters.append(inner_parameter)
        for child in node.children:
            if child.type == "let_binding":
                process_ocaml_body(child)
                pattern_node = child.child_by_field_name("pattern")
                return_type = None
                if pattern_node is not None and pattern_node.type == "value_name":
                    for grandchild in child.children:
                        if grandchild.type == "parameter":
                            parse_ocaml_parameter(grandchild)
                    if parameters != []:
                        parents = [n for n in (child.prev_sibling, child) if n]
                        declaration = mk_fun_decl(
                            id=pattern_node, parents=parents, parameters=parameters, return_type=return_type)
                        file.add_symbol(declaration)
                        return [declaration]

    elif node.type == "module_definition" and language == "ocaml":
        for child in node.children:
            if child.type == "module_binding":
                process_ocaml_body(child)
                name = child.child_by_field_name("name")
                if name is not None:
                    scope = scope + code.bytes[name.start_byte : name.end_byte].decode() + "."
                    if body_node is not None:
                        body = process_body(code=code, file=file, language=language, node=body_node, scope=scope)
                    else:
                        body = []
                    declaration = mk_module_decl(id=name, body=body, parents=[node])
                    file.add_symbol(declaration)
                    return [declaration]
    
    elif node.type == "let_declaration" and language == "rescript":
        return_type = None
        def parse_res_parameter(par: Node, parameters: List[Parameter]) -> None:
            if par.type in ["(", ")", ","]:
                pass
            elif par.type == "parameter" and par.child_count >= 1:
                nodes = par.children
                type = None
                if len(nodes) >= 2 and nodes[1].type == "type_annotation" and len(nodes[1].children) >= 2:
                    type = nodes[1].children[1].text.decode()
                default_value = None
                if nodes[0].type == "labeled_parameter":
                    children = nodes[0].children
                    default_value_node = nodes[0].child_by_field_name("default_value")
                    if default_value_node is not None:
                        next = default_value_node.next_sibling
                        if next is not None:
                            default_value = next.text.decode()
                    for child in children:
                        if child.type == "type_annotation" and len(child.children) >= 2:
                            type = child.children[1].text.decode()
                    name = "~" + children[1].text.decode()
                else:
                    name = nodes[0].text.decode()
                parameters.append(Parameter(default_value=default_value, name=name, type=type))
            else:
                raise Exception(f"Unexpected parameter type: {par.type}")
        def parse_res_parameters(exp: Node, parameters: List[Parameter]) -> None:
            nonlocal body_sub, return_type
            if exp.type == "function":
                nodes = exp.children
                if len(nodes) >= 2:
                    if nodes[0].type == "formal_parameters":
                        for par in nodes[0].children:
                            parse_res_parameter(par, parameters)
                    if nodes[1].type == "type_annotation" and nodes[1].child_count >= 2:
                        return_type = nodes[1].children[1].text.decode()
                    if body_sub is not None:
                            body_sub = (nodes[-2].start_byte, body_sub[1])
        def parse_res_let_binding(nodes: List[Node], parents: List[Node]) -> Optional[ValueDeclaration]:
            nonlocal body_sub
            id = None
            exp = None
            typ = None
            if len(nodes) == 0:
                pass
            elif len(nodes) == 3 and nodes[0].type == "value_identifier" and nodes[1].text == b"=":
                id = nodes[0]
                exp = nodes[2]
                body_sub = (nodes[1].start_byte, exp.end_byte)
            elif len(nodes) > 2 and nodes[0].type == "parenthesized_pattern" and nodes[1].type == "=":
                pat = nodes[0].children[1:-1] # remove ( and )
                return parse_res_let_binding(pat + nodes[1:], parents)
            elif len(nodes) == 4 and nodes[1].type == "type_annotation" and nodes[2].type == "=":
                id = nodes[0]
                typ = nodes[1]
                exp = nodes[3]
                body_sub = (nodes[2].start_byte, exp.end_byte)
            elif len(nodes) == 4 and nodes[1].type == "as_aliasing" and nodes[2].type == "=":
                id = nodes[1].children[1]
                exp = nodes[3]
                body_sub = (nodes[2].start_byte, exp.end_byte)
            elif nodes[0].type in ["tuple_pattern", "unit"]:
                pass
            else:
                print(f"Unexpected let_binding nodes:{nodes}")
            if id is not None and id.text != b"_":
                parameters: List[Parameter] = []
                if exp is not None:
                    parse_res_parameters(exp, parameters)
                if parameters == []:
                    type = None
                    if typ is not None and typ.child_count >= 2:
                        type = typ.children[1].text.decode()
                    declaration = mk_val_decl(id=id, parents=parents, type=type)
                else:
                    declaration = mk_fun_decl(id=id, parents=parents, parameters=parameters, return_type=return_type)
                file.add_symbol(declaration)
                return declaration
        declarations: List[SymbolInfo] = []
        for child in node.children:
            if child.type == "let_binding":
                parents = [n for n in (child.prev_sibling, child) if n]
                # let rec: add node of type "let" if present before the first parent
                if len(parents) > 0 and parents[0].prev_sibling is not None and parents[0].prev_sibling.type == "let":
                    parents = [parents[0].prev_sibling] + parents
                decl = parse_res_let_binding(nodes=child.children, parents=parents)
                if decl is not None:
                    declarations.append(decl)
        return declarations
    elif node.type == "module_declaration" and language == "rescript":
        def parse_module_binding(nodes: List[Node]) -> List[SymbolInfo]:
            nonlocal body_sub, scope
            id = None
            body = None
            if len(nodes) == 3 and nodes[0].type == "module_identifier" and nodes[1].type == "=":
                id = nodes[0]
                body = nodes[2]
                body_sub = (nodes[0].end_byte, nodes[2].end_byte)
            elif len(nodes) == 5 and nodes[0].type == "module_identifier" and nodes[1].type == ":" and nodes[3].type == "=":
                id = nodes[0]
                body = nodes[4]
                body_sub = (nodes[0].end_byte, nodes[4].end_byte)
            else:
                print(f"Unexpected module_binding nodes:{len(nodes)}")
            if id is not None and body is not None:
                scope = scope + code.bytes[id.start_byte : id.end_byte].decode() + "."
                body = process_body(code=code, file=file, language=language, node=body, scope=scope)
                declaration = mk_module_decl(id=id, body=body, parents=[node])
                file.add_symbol(declaration)
                return [declaration]
            else:
                return []
        if len (node.children) == 2:
            m1 = node.children[1]
            if m1.type == "module_binding":
                nodes = m1.children
                return parse_module_binding(nodes)
            else:
                raise Exception(f"Unexpected node type in module_declaration: {m1.type}")

    # if not returned earlier
    return []


def process_statement(
    code: Code, file: File, language: Language, node: Node, scope: Scope
) -> Statement:
    declarations = find_declarations(code=code, file=file, language=language, node=node, scope=scope)
    if declarations != []:
        return Declaration(type=node.type, symbols=declarations)
    else:
        return Statement(type=node.type)


def process_body(
    code: Code, file: File, language: Language, node: Node, scope: Scope
) -> List[Statement]:
    return [
        process_statement(code=code, file=file, language=language, node=child, scope=scope)
        for child in node.children
    ]

def get_parser(language: Language) -> Parser:
    if language == "rescript":
        parser = custom_parser.parser
        parser.set_language(custom_parser.ReScript)
        return parser
    else:
        return get_tree_sitter_parser(language)

def parse_code_block(file: File, code: Code, language: Language) -> None:
    parser = get_parser(language)
    tree = parser.parse(code.bytes)
    for node in tree.root_node.children:
        statement = process_statement(code=code, file=file, language=language, node=node, scope="")
        file.statements.append(statement)


def parse_files_in_project(
    root_path: str, filter: Optional[Callable[[str], bool]] = None
) -> Project:
    """
    Parses all files with known extensions in a directory and its subdirectories, starting from the provided root path.
    Returns a Project instance containing all parsed files.
    If a filter function is provided, it is used to decide which files should be included in the Project.
    """
    project = Project(root_path=root_path)
    for root, dirs, files in os.walk(root_path):
        for file in files:
            language = language_from_file_extension(file)
            if language is not None:
                full_path = os.path.join(root, file)
                path_from_root = os.path.relpath(full_path, root_path)
                if filter is None or filter(path_from_root):
                    with open(os.path.join(root_path, full_path), "r", encoding="utf-8") as f:
                        code = Code(f.read().encode("utf-8"))
                    file_ir = File(path=path_from_root)
                    parse_code_block(file=file_ir, code=code, language=language)
                    project.add_file(file=file_ir)
    return project


def parse_files_in_paths(paths: List[str]) -> Project:
    """
    Parses all files with known extensions in the provided list of paths.
    """
    if len(paths) == 0:
        raise Exception("No paths provided")
    if len(paths) == 1 and os.path.isfile(paths[0]):
        root_path = os.path.dirname(paths[0])
    else:
        root_path = os.path.commonpath(paths)
    project = Project(root_path=root_path)
    for path in paths:
        if os.path.isfile(path):
            language = language_from_file_extension(path)
            if language is not None:
                path_from_root = os.path.relpath(path, root_path)
                with open(path, "r", encoding="utf-8") as f:
                    code = Code(f.read().encode("utf-8"))
                file_ir = File(path=path_from_root)
                parse_code_block(file=file_ir, code=code, language=language)
                project.add_file(file=file_ir)
        else:
            for root, dirs, files in os.walk(path):
                for file in files:
                    language = language_from_file_extension(file)
                    if language is not None:
                        full_path = os.path.join(root, file)
                        path_from_root = os.path.relpath(full_path, root_path)
                        with open(os.path.join(root_path, full_path), "r", encoding="utf-8") as f:
                            code = Code(f.read().encode("utf-8"))
                        file_ir = File(path=path_from_root)
                        parse_code_block(file=file_ir, code=code, language=language)
                        project.add_file(file=file_ir)
    return project
