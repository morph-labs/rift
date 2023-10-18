import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tree_sitter import Node

from .IR import (
    Block,
    BodyKind,
    CallKind,
    Case,
    ClassKind,
    Code,
    Expression,
    ExpressionKind,
    File,
    ForKind,
    FunctionKind,
    GuardKind,
    IfKind,
    Import,
    InterfaceKind,
    Language,
    ModuleKind,
    NamespaceKind,
    Parameter,
    Scope,
    Substring,
    Symbol,
    SymbolKind,
    Type,
    TypeDefinitionKind,
    UnknownKind,
    ValueKind,
)

logger = logging.getLogger(__name__)


def dump_node(node: Node) -> str:
    """Dump a node for debugging purposes."""
    return f"  type:{node.type} children:{node.child_count}\n  code:{node.text.decode()}\n  sexp:{node.sexp()}"


jslangs = ["javascript", "typescript", "tsx"]


def parse_type(language: Language, node: Node, parent: Symbol) -> Type:
    if language in jslangs and node.type == "type_annotation" and len(node.children) >= 2:
        # TS: first child should be ":" and second child should be type
        second_child = node.children[1]
        return Type.unknown(second_child.text.decode(), parent=parent)
    elif language == "python" and node.type == "type" and node.child_count >= 1:
        child = node.children[0]
        if child.type == "subscript":
            node_value = child.child_by_field_name("value")
            if node_value is not None:
                subscripts = child.children_by_field_name("subscript")
                arguments = [parse_type(language, n, parent) for n in subscripts]
                name = node_value.text.decode()
                return Type.constructor(name=name, arguments=arguments, parent=parent)
        elif child.type == "identifier":
            name = child.text.decode()
            return Type.constructor(name=name, parent=parent)

    return Type.unknown(node.text.decode(), parent)


def add_c_cpp_declarators_to_type(type: Type, declarators: List[str], parent: Symbol) -> Type:
    t = type
    for d in declarators:
        if d == "pointer_declarator":
            t = t.pointer(parent)
        elif d == "array_declarator":
            t = t.array(parent)
        elif d == "function_declarator":
            t = t.function(parent)
        elif d == "reference_declarator":
            t = t.reference(parent)
        elif d == "identifier":
            pass
        else:
            logger.warning(f"Unknown declarator: {d}")
    return t


def extract_c_cpp_declarators(node: Node) -> Tuple[List[str], Node]:
    declarator_node = node.child_by_field_name("declarator")
    if declarator_node is None:
        if node.type == "reference_declarator" and node.child_count >= 2:
            return [], node.children[1]
        else:
            return [], node
    declarators, final_node = extract_c_cpp_declarators(declarator_node)
    declarators.append(declarator_node.type)
    return declarators, final_node


def get_c_cpp_parameter(language: Language, node: Node, parent: Symbol) -> Parameter:
    declarators, final_node = extract_c_cpp_declarators(node)
    type_node = node.child_by_field_name("type")
    if type_node is None:
        logger.warning(f"Could not find type node in {node}")
        type = Type.unknown("unknown", parent)
    else:
        type = parse_type(language=language, node=type_node, parent=parent)
        type = add_c_cpp_declarators_to_type(type, declarators, parent)
    name = ""
    if final_node.type == "identifier":
        name = final_node.text.decode()
    return Parameter(name=name, type=type, parent=parent)


def get_parameters(language: Language, node: Node, parent: Symbol) -> List[Parameter]:
    parameters: List[Parameter] = []
    for child in node.children:
        if child.type == "identifier":
            name = child.text.decode()
            parameters.append(Parameter(name=name, parent=parent))
        elif child.type in ["typed_parameter", "typed_default_parameter"]:
            name = ""
            type: Optional[Type] = None
            for grandchild in child.children:
                if grandchild.type == "identifier":
                    name = grandchild.text.decode()
                elif grandchild.type == "type":
                    type = parse_type(language, grandchild, parent)
            value_node = child.child_by_field_name("value")

            default_value = value_node.text.decode() if value_node is not None else None
            parameters.append(
                Parameter(name=name, default_value=default_value, type=type, parent=parent)
            )
        elif child.type == "parameter_declaration":
            if language in ["c", "cpp"]:
                parameters.append(get_c_cpp_parameter(language, child, parent))
            else:
                type: Optional[Type] = None
                type_node = child.child_by_field_name("type")
                if type_node is not None:
                    type = parse_type(language, type_node, parent)
                name = child.text.decode()
                parameters.append(Parameter(name=name, type=type, parent=parent))
        elif child.type in ["required_parameter", "optional_parameter"]:
            name = ""
            pattern_node = child.child_by_field_name("pattern")
            if pattern_node is not None:
                name = pattern_node.text.decode()
            type: Optional[Type] = None
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                type = parse_type(language, type_node, parent)
            parameters.append(
                Parameter(
                    name=name, type=type, optional=child.type == "optional_parameter", parent=parent
                )
            )
        elif child.type in ["formal_parameter", "parameter"]:
            type: Optional[Type] = None
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                type = parse_type(language, type_node, parent)
            name = child.text.decode()
            parameters.append(Parameter(name=name, type=type, parent=parent))
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
            "method",
        ]:
            continue
        # If the child is a return statement, return True.
        if child.type == "return_statement":
            return True
        # If the child has its own children, recursively check them.
        if contains_direct_return(child):
            return True
    return False


def parse_import(node: Node) -> Optional[Import]:
    substring = (node.start_byte, node.end_byte)
    if node.type == "import_statement":
        names = [n.text.decode() for n in node.children_by_field_name("name")]
        return Import(names=names, substring=substring)
    elif node.type == "import_from_statement":
        names = [n.text.decode() for n in node.children_by_field_name("name")]
        module_name_node = node.child_by_field_name("module_name")
        if module_name_node is not None:
            module_name = module_name_node.text.decode()
        else:
            module_name = None
        return Import(names=names, module_name=module_name, substring=substring)


@dataclass
class Counter:
    """
    Counter class that maintains a count for unique names.
    """

    dict: Dict[str, int] = field(default_factory=dict)

    def next(self, name: str) -> int:
        """
        Increment the count of the given name and return the previous count.

        """
        count = self.dict.get(name, 0)
        self.dict[name] = count + 1
        return count


class SymbolParser:
    def __init__(
        self,
        code: Code,
        file: File,
        language: Language,
        node: Node,
        parent: Optional[Symbol],
        scope: Scope,
        metasymbols: bool,
    ) -> None:
        self.code = code
        self.file = file
        self.language: Language = language
        self.metasymbols = metasymbols
        self.node = node
        self.parent = parent
        self.scope = scope

        self.body_sub: Optional[Substring] = None
        self.docstring_sub: Optional[Substring] = None
        self.exported = False
        self.has_return = False

    def recurse(self, node: Node, scope: Scope, parent: Optional[Symbol]) -> "SymbolParser":
        return self.__class__(
            code=self.code,
            file=self.file,
            language=self.language,
            metasymbols=self.metasymbols,
            node=node,
            parent=parent,
            scope=scope,
        )

    def mk_symbol_decl(
        self,
        id: Node | str,
        parents: List[Node],
        symbol_kind: SymbolKind,
        body: Optional[Block] = None,
    ) -> Symbol:
        if isinstance(id, str):
            name: str = id
        else:
            name = id.text.decode()
        if body is None:
            body = Block()
        return Symbol(
            body=body,
            body_sub=self.body_sub,
            code=self.code,
            docstring_sub=self.docstring_sub,
            exported=self.exported,
            language=self.language,
            id=name,
            parent=self.parent,
            range=(parents[0].start_point, parents[-1].end_point),
            scope=self.scope,
            substring_=(parents[0].start_byte, parents[-1].end_byte),
            symbol_kind=symbol_kind,
        )

    def mk_dummy_symbol(self, id: Node | str, parents: List[Node]) -> Symbol:
        dummy_kind: SymbolKind = None  # type: ignore
        symbol = self.mk_symbol_decl(id=id, parents=parents, symbol_kind=dummy_kind)
        symbol.symbol_kind = ValueKind(symbol)
        return symbol

    def mk_dummy_metasymbol(self, counter: Counter, name: str) -> Symbol:
        count = counter.next(name)
        id = f"{name}${count}"
        dummy = self.mk_dummy_symbol(id=id, parents=[self.node])
        self.scope = self.scope + f"{id}."
        return dummy

    def update_dummy_symbol(self, symbol: Symbol, symbol_kind: SymbolKind) -> None:
        symbol.body_sub = self.body_sub
        if symbol.docstring_sub is None:
            symbol.docstring_sub = self.docstring_sub
        symbol.symbol_kind = symbol_kind

    def mk_fun_decl(
        self,
        id: Node,
        parents: List[Node],
        parameters: Optional[List[Parameter]] = None,
        return_type: Optional[Type] = None,
    ) -> Symbol:
        if parameters is None:
            parameters = []
        symbol = self.mk_dummy_symbol(id=id, parents=parents)
        symbol_kind = FunctionKind(
            has_return=self.has_return,
            parameters=parameters,
            return_type=return_type,
            symbol=symbol,
        )
        self.update_dummy_symbol(symbol, symbol_kind)
        return symbol

    def mk_val_decl(self, id: Node, parents: List[Node], type: Optional[Type] = None) -> Symbol:
        symbol = self.mk_dummy_symbol(id=id, parents=parents)
        symbol_kind = ValueKind(type=type, symbol=symbol)
        self.update_dummy_symbol(symbol, symbol_kind)
        return symbol

    def mk_type_decl(self, id: Node, parents: List[Node], type: Optional[Type] = None) -> Symbol:
        symbol = self.mk_dummy_symbol(id=id, parents=parents)
        symbol_kind = TypeDefinitionKind(type=type, symbol=symbol)
        self.update_dummy_symbol(symbol, symbol_kind)
        return symbol

    def mk_interface_decl(self, id: Node, parents: List[Node]) -> Symbol:
        symbol = self.mk_dummy_symbol(id=id, parents=parents)
        symbol_kind = InterfaceKind(symbol)
        self.update_dummy_symbol(symbol, symbol_kind)
        return symbol

    def process_ocaml_body(self, n: Node, parent: Symbol) -> Tuple[Optional[Type], Optional[Node]]:
        type = None
        body_node = n.child_by_field_name("body")
        if body_node is not None:
            node_before = body_node.prev_sibling
            if node_before is not None and node_before.type == "=":
                # consider "=" part of the body
                self.body_sub = (node_before.start_byte, body_node.end_byte)
                n2 = node_before.prev_sibling
                if n2:
                    n3 = n2.prev_sibling
                    if n3 and n3.type == ":":
                        type = parse_type(self.language, n2, parent)
            else:
                self.body_sub = (body_node.start_byte, body_node.end_byte)
        return type, body_node

    def process_ruby_body(self) -> Node:
        method_name_node = self.node.child_by_field_name("name")
        if method_name_node is not None:
            start_node = method_name_node
            parameters_node = self.node.child_by_field_name("parameters")
            if parameters_node is not None:
                start_node = parameters_node

            if start_node.next_sibling is not None:
                start_node = start_node.next_sibling
            self.body_sub = (start_node.start_byte, self.node.end_byte)
        return self.node

    def process_body(self) -> Optional[Node]:
        if self.language == "ruby":
            return self.process_ruby_body()
        else:
            body_node = self.node.child_by_field_name("body")
            if body_node is not None:
                self.body_sub = (body_node.start_byte, body_node.end_byte)
            return body_node

    def parse_symbols(self, counter: Counter) -> List[Symbol]:
        """
        Parse the node specified by the index to extract recognized symbols, such as classes,
        functions, methods, modules, namespaces, etc., across various languages.

        This function processes symbols from languages like C, C++, Python, JavaScript, TypeScript,
        Ruby, C#, Java, ReScript, and potentially others as the function evolves. It also captures
        associated documentation comments and appends them to the recognized symbol.

        Returns:
            List[Symbol]: A list of identified symbols, which can be empty if no symbol is recognized
            or can contain one or more symbols as the parsing process can identify multiple symbols
            from a single node (e.g., ReScript let bindings).
        """

        previous_node = self.node.prev_sibling
        if previous_node is not None and previous_node.type == "comment":
            docstring = previous_node.text.decode()
            if docstring.startswith("/**"):
                self.docstring_sub = (previous_node.start_byte, previous_node.end_byte)

        node = self.node
        language = self.language
        body_node = self.process_body()

        if (
            (node.type in ["class_specifier"] and language in ["c", "cpp"])
            or (node.type in ["class_declaration"] and language in jslangs + ["c_sharp", "java"])
            or (node.type in ["class_definition"] and language in ["python"])
            or (node.type in ["namespace_definition"] and language in ["cpp"])
            or (node.type in ["namespace_declaration"] and language in ["c_sharp"])
            or (node.type in ["class", "module"] and language == "ruby")
        ):
            is_namespace = node.type in ["namespace_definition", "namespace_declaration"]
            is_module = node.type == "module"
            superclasses_node = node.child_by_field_name("superclasses")
            superclasses = None
            if superclasses_node is not None:
                superclasses = superclasses_node.text.decode()
            name = node.child_by_field_name("name")

            if body_node is not None and name is not None:
                if is_namespace or language == "ruby":
                    separator = "::"
                else:
                    separator = "."
                new_scope = self.scope + name.text.decode() + separator
                symbol = self.mk_dummy_symbol(id=name, parents=[node])
                self.recurse(body_node, new_scope, parent=symbol).parse_block()
                # see if the first child is a string expression statements, and if so, use it as the docstring
                if (
                    body_node.child_count > 0
                    and body_node.children[0].type == "expression_statement"
                ):
                    stmt = body_node.children[0]
                    if len(stmt.children) > 0 and stmt.children[0].type == "string":
                        docstring_node = stmt.children[0]
                        symbol.docstring_sub = (docstring_node.start_byte, docstring_node.end_byte)
                elif node.prev_sibling is not None and node.prev_sibling.type in [
                    "comment",
                    "line_comment",
                    "block_comment",
                ]:
                    # parse class comments before class definition
                    docstring_node = node.prev_sibling
                    symbol.docstring_sub = (docstring_node.start_byte, docstring_node.end_byte)

                if is_namespace:
                    self.update_dummy_symbol(symbol, NamespaceKind(symbol))
                elif is_module:
                    self.update_dummy_symbol(symbol, ModuleKind(symbol))
                else:
                    self.update_dummy_symbol(symbol, ClassKind(symbol, superclasses))
                self.file.add_symbol(symbol)
                return [symbol]

        elif node.type in ["decorated_definition"] and language == "python":  # python decorator
            definition = node.child_by_field_name("definition")
            if definition is not None:
                return self.recurse(definition, self.scope, parent=self.parent).parse_symbols(
                    counter
                )

        elif node.type in ["field_declaration", "function_definition"] and language in ["c", "cpp"]:
            type_node = node.child_by_field_name("type")
            type = None
            symbol = self.mk_dummy_symbol(id="TODO", parents=[node])
            if type_node is not None:
                type = parse_type(language=language, node=type_node, parent=symbol)
            res = find_c_cpp_function_declarator(node)
            if res is None or type is None:
                return []
            declarators, fun_node = res
            type = add_c_cpp_declarators_to_type(type, declarators, parent=symbol)
            id: Optional[Node] = None
            parameters: List[Parameter] = []
            for child in fun_node.children:
                if child.type in ["field_identifier", "identifier"]:
                    id = child
                elif child.type == "parameter_list":
                    parameters = get_parameters(language=language, node=child, parent=symbol)
            if id is None:
                return []
            symbol.id = id.text.decode()
            symbol_kind = symbol_kind = FunctionKind(
                has_return=self.has_return,
                parameters=parameters,
                return_type=type,
                symbol=symbol,
            )
            self.update_dummy_symbol(symbol, symbol_kind)
            self.file.add_symbol(symbol)
            return [symbol]

        elif (
            (node.type in ["function_declaration", "method_definition"] and language in jslangs)
            or (node.type in ["function_definition"] and language in ["python"])
            or (node.type in ["method_declaration"] and language in ["c_sharp", "java"])
            or (node.type in ["method"] and language in ["ruby"])
        ):
            id: Optional[Node] = None
            symbol = self.mk_dummy_symbol(id="TODO", parents=[node])
            is_async = node.text.startswith(b"async ")
            for child in node.children:
                if child.type in ["identifier", "property_identifier"]:
                    id = child
            parameters: List[Parameter] = []
            parameters_node = node.child_by_field_name("parameters")
            if parameters_node is not None:
                parameters = get_parameters(language=language, node=parameters_node, parent=symbol)
            return_type: Optional[Type] = None
            if language in ["c_sharp", "java"]:
                return_type_node = node.child_by_field_name("type")
            else:
                return_type_node = node.child_by_field_name("return_type")
            if return_type_node is not None:
                return_type = parse_type(language=language, node=return_type_node, parent=symbol)
            if (
                body_node is not None
                and len(body_node.children) > 0
                and body_node.children[0].type == "expression_statement"
            ):
                stmt = body_node.children[0]
                if len(stmt.children) > 0 and stmt.children[0].type == "string":
                    docstring_node = stmt.children[0]
                    self.docstring_sub = (docstring_node.start_byte, docstring_node.end_byte)
            if body_node is not None:
                self.has_return = contains_direct_return(body_node)

            if id is None:
                return []
            symbol.id = id.text.decode()

            if body_node is not None and language == "python" and self.metasymbols:
                scope_body = self.scope + f"{id.text.decode()}."
                self.recurse(body_node, scope_body, parent=symbol).parse_block()

            self.update_dummy_symbol(
                symbol,
                FunctionKind(
                    symbol,
                    has_return=self.has_return,
                    is_async=is_async,
                    parameters=parameters,
                    return_type=return_type,
                ),
            )
            self.file.add_symbol(symbol)
            return [symbol]

        elif node.type in ["lexical_declaration", "variable_declaration"] and language in jslangs:
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
                        declaration = self.mk_fun_decl(id=id, parents=[node])
                        self.file.add_symbol(declaration)
                        return [declaration]

        elif node.type == "export_statement" and language in jslangs:
            if len(node.children) >= 2:
                self.node = node.children[1]
                self.exported = True
                return self.parse_symbols(counter)

        elif node.type in [
            "interface_declaration",
            "type_alias_declaration",
        ] and language in jslangs + [
            "c_sharp",
            "java",
        ]:
            id: Optional[Node] = node.child_by_field_name("name")
            if id is not None:
                if node.type == "interface_declaration":
                    declaration = self.mk_interface_decl(id=id, parents=[node])
                else:
                    declaration = self.mk_type_decl(id=id, parents=[node])
                self.file.add_symbol(declaration)
                return [declaration]

        if self.metasymbols:
            metasymbol = self.parse_metasymbol(counter)
            if metasymbol is not None:
                return [metasymbol]
            else:
                return []
        else:
            return []

    def parse_guard(self, counter: Counter) -> Symbol:
        """Parse the guard of a conditional"""
        guard_symbol = self.mk_dummy_metasymbol(counter, "guard")
        guard_counter = Counter()
        condition = self.parse_expression(guard_counter)
        self.update_dummy_symbol(
            symbol=guard_symbol, symbol_kind=GuardKind(guard_symbol, condition)
        )
        self.file.add_symbol(guard_symbol)
        return guard_symbol

    def parse_body(self, counter: Counter) -> Symbol:
        """Parse the body of a conditional branch"""
        body_symbol = self.mk_dummy_metasymbol(counter, "body")
        self.recurse(self.node, self.scope, parent=body_symbol).parse_block()
        self.update_dummy_symbol(
            symbol=body_symbol, symbol_kind=BodyKind(body_symbol, body_symbol.body)
        )
        self.file.add_symbol(body_symbol)
        return body_symbol

    def parse_metasymbol(self, counter: Counter) -> Optional[Symbol]:
        node = self.node
        language = self.language

        if node.type == "if_statement" and language in ["python"] + jslangs:
            guard_n = node.child_by_field_name("condition")
            body_n = node.child_by_field_name("consequence")
            if guard_n is not None and body_n is not None:
                if_symbol = self.mk_dummy_metasymbol(counter, "if")
                scope = self.scope

                if_guard = self.recurse(guard_n, scope, parent=if_symbol).parse_guard(counter)
                if_body = self.recurse(body_n, scope, parent=if_symbol).parse_body(counter)

                if_case = Case(guard=if_guard, body=if_body)
                alternative_nodes = node.children_by_field_name("alternative")
                elif_cases: List[Case] = []
                else_body: Optional[Symbol] = None
                if language == "python":
                    for an in alternative_nodes:
                        if an.type == "elif_clause":
                            guard_n = an.child_by_field_name("condition")
                            body_n = an.child_by_field_name("consequence")
                            if guard_n is None or body_n is None:
                                continue
                            guard = self.recurse(guard_n, scope, parent=if_symbol).parse_guard(
                                counter
                            )
                            body = self.recurse(body_n, scope, parent=if_symbol).parse_body(counter)
                            elif_cases.append(Case(guard=guard, body=body))
                        elif an.type == "else_clause":
                            else_n = an.child_by_field_name("body")
                            # TODO: there can be comments in the else clause before the body
                            if else_n is not None:
                                else_body = self.recurse(
                                    else_n, scope, parent=if_symbol
                                ).parse_body(counter)
                elif len(alternative_nodes) == 1:
                    an = alternative_nodes[0]
                    if len(an.children) >= 2:  # else, body
                        else_body = self.recurse(
                            an.children[1], scope, parent=if_symbol
                        ).parse_body(counter)

                if_kind = IfKind(
                    if_symbol, if_case=if_case, elif_cases=elif_cases, else_body=else_body
                )
                self.update_dummy_symbol(symbol=if_symbol, symbol_kind=if_kind)
                self.file.add_symbol(if_symbol)
                return if_symbol

        elif node.type == "for_statement" and language == "python":
            left_node = node.child_by_field_name("left")
            right_node = node.child_by_field_name("right")
            body_node = node.child_by_field_name("body")
            if left_node is not None and right_node is not None and body_node is not None:
                for_symbol = self.mk_dummy_metasymbol(counter, "for")
                scope = self.scope
                left = left_node.text.decode()
                right = self.recurse(right_node, scope, parent=for_symbol).parse_expression(counter)
                body = self.recurse(body_node, scope, parent=for_symbol).parse_body(counter)
                for_kind = ForKind(for_symbol, left=left, right=right, body=body)
                self.update_dummy_symbol(symbol=for_symbol, symbol_kind=for_kind)
                self.file.add_symbol(for_symbol)
                return for_symbol

        elif node.type == "expression_statement" and node.child_count == 1:
            child = node.children[0]
            if self.expression_requires_node(child) and self.parent:
                # Don't need to create a sybmol as parsing the expression will create one
                _code = self.recurse(child, self.scope, parent=self.parent).parse_expression(
                    counter
                )
                # return the last item in the body of the parent
                return self.parent.body[-1]

            else:
                symbol = self.mk_dummy_metasymbol(counter, "expression")
                code = self.recurse(child, self.scope, parent=symbol).parse_expression(counter)
                self.update_dummy_symbol(symbol=symbol, symbol_kind=ExpressionKind(symbol, code))
                self.file.add_symbol(symbol)
                return symbol

    def parse_expression(self, counter: Counter) -> Expression:
        """
        Parse an expression to generate its corresponding code with symbols replaced by their names.
        """

        self.recurse(self.node, self.scope, parent=self.parent).walk_expression(counter)

        code = self.node.text.decode()

        if self.parent is None:
            return code

        # Get a list of symbols from the parent's body
        symbols = self.parent.body

        # Sort the symbols based on their starting substring index in descending order for accurate replacement
        sorted_symbols = sorted(symbols, key=lambda s: s.substring_[0], reverse=True)

        # Replace each symbol in the code with its name
        for symbol in sorted_symbols:
            start, end = symbol.substring_
            # Adjust start and end based on the node's starting byte
            start -= self.node.start_byte
            end -= self.node.start_byte
            # Ensure the symbol's start and end are within bounds of the code's length
            if start >= 0 and end <= len(code):
                code = code[:start] + symbol.id + code[end:]

        return code

    @classmethod
    def expression_requires_node(cls, node: Node) -> bool:
        if node.type in ["call", "call_expression"]:
            return True
        else:
            return False

    def walk_expression(self, counter: Counter) -> None:
        node = self.node
        if self.expression_requires_node(node):
            if node.type in ["call", "call_expression"]:
                function_node = node.child_by_field_name("function")
                if function_node is None:
                    logger.warning(f"Unexpected call node structure: {node.text.decode()}")
                else:
                    function_name = function_node.text.decode()

                    count = counter.next("call")
                    symbol = self.mk_dummy_symbol(id=f"call${count}", parents=[node])
                    self.scope = self.scope + "call."

                    arguments: List[Expression] = []
                    arguments_node = node.child_by_field_name("arguments")
                    if arguments_node is not None:
                        arg_counter = Counter()
                        for arg in arguments_node.children:
                            if arg.type in ["(", ")"]:
                                continue
                            expression = self.recurse(
                                arg, self.scope, parent=symbol
                            ).parse_expression(arg_counter)
                            arguments.append(expression)
                    self.update_dummy_symbol(symbol, CallKind(symbol, function_name, arguments))
                    self.file.add_symbol(symbol)
            else:
                logger.warning(f"Unexpected expression: {node.type}")
        elif node.type in ["await", "assignment", "binary_operator", "parenthesized_expression"]:
            for child in node.children:
                self.node = child
                self.walk_expression(counter)

    def parse_block(self) -> None:
        counter = Counter()
        self_ = self
        for child in self.node.children:
            if child.type in ["{", "}"]:
                continue
            self = self_.recurse(child, self_.scope, parent=self_.parent)
            symbols = self.recurse(self.node, self.scope, parent=self.parent).parse_symbols(counter)
            import_ = parse_import(self.node)
            if import_ is not None:
                self.file.add_import(import_)
            if symbols == []:
                symbol = self.mk_dummy_symbol(id=self.node.type, parents=[self.node])
                self.update_dummy_symbol(symbol, UnknownKind(symbol))
                self.file.add_symbol(symbol)
