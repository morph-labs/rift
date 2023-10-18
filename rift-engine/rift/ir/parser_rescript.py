import logging
from typing import List, Optional

from tree_sitter import Node

from rift.ir.parser_core import Counter, SymbolParser

from .IR import (
    Case,
    Field,
    FunctionKind,
    IfKind,
    Language,
    ModuleKind,
    Parameter,
    SwitchKind,
    Symbol,
    Type,
    TypeDefinitionKind,
    ValueKind,
)

logger = logging.getLogger(__name__)


def parse_type(language: Language, node: Node, parent: Symbol) -> Type:
    if node.type == "type_identifier":
        name = node.text.decode()
        return Type.constructor(name=name, parent=parent)
    elif node.type == "generic_type" and node.child_count == 2:
        name = node.children[0].text.decode()
        arguments_node = node.children[1]
        if arguments_node.type == "type_arguments":
            # remove first and last argument: < and >
            arguments = arguments_node.children[1:-1]
            arguments = [parse_type(language, n, parent) for n in arguments]
            t = Type.constructor(name=name, arguments=arguments, parent=parent)
            return t
        else:
            logger.warning(f"Unknown arguments_node type node: {arguments_node}")
    else:
        logger.warning(f"Unknown type node: {node}")

    return Type.unknown(node.text.decode(), parent)


class ReScriptParser(SymbolParser):
    def parse_symbols(self, counter: Counter) -> List[Symbol]:
        if self.node.type == "let_declaration":
            return_type = None

            def parse_res_parameter(par: Node, parameters: List[Parameter], parent: Symbol) -> None:
                if par.type in ["(", ")", ","]:
                    pass
                elif par.type == "parameter" and par.child_count >= 1:
                    nodes = par.children
                    type: Optional[Type] = None
                    if (
                        len(nodes) >= 2
                        and nodes[1].type == "type_annotation"
                        and len(nodes[1].children) >= 2
                    ):
                        type = parse_type(self.language, nodes[1].children[1], parent)
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
                                type = parse_type(self.language, child.children[1], parent)
                        name = "~" + children[1].text.decode()
                    else:
                        name = nodes[0].text.decode()
                    parameters.append(
                        Parameter(default_value=default_value, name=name, type=type, parent=parent)
                    )
                else:
                    logger.warning(f"Unexpected parameter type: {par.type}")

            def parse_res_parameters(
                exp: Node, parameters: List[Parameter], parent: Symbol
            ) -> None:
                nonlocal return_type
                if exp.type == "function":
                    parameters_node = exp.child_by_field_name("parameters")
                    if parameters_node is not None:
                        for par in parameters_node.children:
                            parse_res_parameter(par, parameters, parent)
                    parameter_node = exp.child_by_field_name("parameter")
                    if parameter_node is not None:
                        parameters.append(
                            Parameter(
                                default_value=None, name=parameter_node.text.decode(), parent=parent
                            )
                        )
                    nodes = exp.children
                    if len(nodes) >= 2:
                        if nodes[1].type == "type_annotation" and nodes[1].child_count >= 2:
                            return_type = parse_type(self.language, nodes[1].children[1], parent)
                        if self.body_sub is not None:
                            self.body_sub = (nodes[-2].start_byte, self.body_sub[1])

            def parse_res_body(exp: Node, parent: Symbol, id_name: str) -> None:
                if exp.type == "function":
                    body_node = exp.child_by_field_name("body")
                    if body_node is not None:
                        counter = Counter()
                        scope = self.scope + id_name + "."
                        self.recurse(body_node, scope, parent=parent).parse_expression(counter)

            def parse_res_let_binding(nodes: List[Node], parents: List[Node]) -> Optional[Symbol]:
                id = None
                exp = None
                typ = None
                if len(nodes) == 0:
                    pass
                elif (
                    len(nodes) == 3
                    and nodes[0].type == "value_identifier"
                    and nodes[1].text == b"="
                ):
                    id = nodes[0]
                    exp = nodes[2]
                    self.body_sub = (nodes[1].start_byte, exp.end_byte)
                elif (
                    len(nodes) > 2
                    and nodes[0].type == "parenthesized_pattern"
                    and nodes[1].type == "="
                ):
                    pat = nodes[0].children[1:-1]  # remove ( and )
                    return parse_res_let_binding(pat + nodes[1:], parents)
                elif (
                    len(nodes) == 4 and nodes[1].type == "type_annotation" and nodes[2].type == "="
                ):
                    id = nodes[0]
                    typ = nodes[1]
                    exp = nodes[3]
                    self.body_sub = (nodes[2].start_byte, exp.end_byte)
                elif len(nodes) == 4 and nodes[1].type == "as_aliasing" and nodes[2].type == "=":
                    id = nodes[1].children[1]
                    exp = nodes[3]
                    self.body_sub = (nodes[2].start_byte, exp.end_byte)
                elif nodes[0].type in ["tuple_pattern", "unit"]:
                    pass
                else:
                    print(f"Unexpected let_binding nodes:{nodes}")
                if id is not None and id.text != b"_":
                    parameters: List[Parameter] = []
                    symbol = self.mk_dummy_symbol(id=id, parents=parents)
                    if exp is not None:
                        parse_res_parameters(exp, parameters, parent=symbol)
                    if parameters == []:
                        type: Optional[Type] = None
                        if typ is not None and typ.child_count >= 2:
                            type = parse_type(self.language, typ.children[1], symbol)
                        symbol_kind = ValueKind(type=type, symbol=symbol)
                        self.update_dummy_symbol(symbol, symbol_kind)
                    else:
                        symbol_kind = FunctionKind(
                            has_return=self.has_return,
                            parameters=parameters,
                            return_type=return_type,
                            symbol=symbol,
                        )
                        self.update_dummy_symbol(symbol, symbol_kind)
                    if exp is not None:
                        parse_res_body(exp, symbol, id.text.decode())
                    self.file.add_symbol(symbol)
                    return symbol

            declarations: List[Symbol] = []
            for child in self.node.children:
                if child.type == "let_binding":
                    parents = [n for n in (child.prev_sibling, child) if n]
                    # let rec: add node of type "let" if present before the first parent
                    if (
                        len(parents) > 0
                        and parents[0].prev_sibling is not None
                        and parents[0].prev_sibling.type == "let"
                    ):
                        parents = [parents[0].prev_sibling] + parents
                    decl = parse_res_let_binding(nodes=child.children, parents=parents)
                    if decl is not None:
                        declarations.append(decl)
            return declarations

        elif self.node.type == "module_declaration":

            def parse_module_binding(nodes: List[Node]) -> List[Symbol]:
                id = None
                body = None
                if (
                    len(nodes) == 3
                    and nodes[0].type == "module_identifier"
                    and nodes[1].type == "="
                ):
                    id = nodes[0]
                    body = nodes[2]
                    self.body_sub = (nodes[0].end_byte, nodes[2].end_byte)
                elif (
                    len(nodes) == 5
                    and nodes[0].type == "module_identifier"
                    and nodes[1].type == ":"
                    and nodes[3].type == "="
                ):
                    id = nodes[0]
                    body = nodes[4]
                    self.body_sub = (nodes[0].end_byte, nodes[4].end_byte)
                else:
                    print(f"Unexpected module_binding nodes:{len(nodes)}")
                if id is not None and body is not None:
                    new_scope = self.scope + id.text.decode() + "."
                    symbol = self.mk_dummy_symbol(id=id, parents=[self.node])
                    self.recurse(body, new_scope, parent=symbol).parse_block()
                    self.update_dummy_symbol(symbol, ModuleKind(symbol))
                    self.file.add_symbol(symbol)
                    return [symbol]
                else:
                    return []

            if len(self.node.children) == 2:
                m1 = self.node.children[1]
                if m1.type == "module_binding":
                    nodes = m1.children
                    return parse_module_binding(nodes)
                else:
                    logger.warning(f"Unexpected node type in module_declaration: {m1.type}")

        elif self.node.type == "type_declaration":

            def parse_type_body(body: Node, parent: Symbol) -> Optional[Type]:
                if body.type == "record_type":
                    fields: List[Field] = []
                    for f in body.children:
                        if f.type == "record_type_field":
                            children = f.children
                            field = None
                            if (
                                len(children) == 3
                                and children[0].type == "property_identifier"
                                and children[1].type == "?"
                                and children[2].type == "type_annotation"
                            ):
                                fname = children[0].text.decode()
                                optional = True
                                type = parse_type(self.language, children[2].children[1], parent)
                                field = Field(fname, optional, parent, type)
                            elif (
                                len(children) == 2
                                and children[0].type == "property_identifier"
                                and children[1].type == "type_annotation"
                            ):
                                fname = children[0].text.decode()
                                optional = False
                                type = parse_type(self.language, children[1].children[1], parent)
                                field = Field(fname, optional, parent, type)
                            else:
                                logger.warning(
                                    f"Unexpected node structure in record_type_field: {f.text.decode()}"
                                )
                            if field is not None:
                                fields.append(field)
                    return Type.record(fields, parent=parent)
                else:
                    logger.warning(f"Unexpected node type in type_declaration: {body.type}")
                    return None

            if len(self.node.children) == 2:
                t1 = self.node.children[1]
                node_name = t1.child_by_field_name("name")
                node_body = t1.child_by_field_name("body")
                if t1.type == "type_binding" and node_name is not None:
                    symbol = self.mk_dummy_symbol(id=node_name, parents=[self.node])
                    type = None
                    if node_body is not None:
                        type = parse_type_body(node_body, symbol)
                    elif len(t1.children) == 3 and t1.children[1].type == "=":
                        type = parse_type(self.language, t1.children[2], symbol)
                    else:
                        logger.warning(
                            f"Unexpected node structure in type_binding: {t1.text.decode()}"
                        )
                    if type is not None:
                        symbol_kind = TypeDefinitionKind(symbol, type)
                        self.update_dummy_symbol(symbol, symbol_kind)
                        self.file.add_symbol(symbol)
                        return [symbol]
                else:
                    logger.warning(f"Unexpected node type in type_declaration: {t1.type}")
                return []

        return super().parse_symbols(counter)

    def parse_metasymbol(self, counter: Counter) -> Optional[Symbol]:
        node = self.node
        if node.type == "if_expression" and node.child_count >= 3:
            guard_node = node.children[1]
            body_node = node.children[2]
            else_node = None
            if node.child_count >= 4:
                else_node = node.children[3]
            if_symbol = self.mk_dummy_metasymbol(counter, "if")
            scope = self.scope
            if_guard = self.recurse(guard_node, scope, parent=if_symbol).parse_guard(counter)
            if_body = self.recurse(body_node, scope, parent=if_symbol).parse_body(counter)
            if_case = Case(guard=if_guard, body=if_body)
            else_body = None
            if else_node is not None and else_node.child_count >= 2:
                else_body = self.recurse(else_node.children[1], scope, parent=if_symbol).parse_body(
                    counter
                )
            if_kind = IfKind(if_symbol, if_case=if_case, elif_cases=[], else_body=else_body)
            self.update_dummy_symbol(symbol=if_symbol, symbol_kind=if_kind)
            self.file.add_symbol(if_symbol)
            return if_symbol

        elif node.type == "switch_expression":
            if node.child_count >= 3:
                switch_symbol = self.mk_dummy_metasymbol(counter, "switch")
                scope = self.scope
                expression_node = node.children[1]
                expression = self.recurse(expression_node, scope, parent=switch_symbol).parse_guard(
                    counter
                )
                cases: List[Case] = []
                for case_node in node.children[2:]:
                    pattern_node = case_node.child_by_field_name("pattern")
                    body_node = case_node.child_by_field_name("body")
                    if pattern_node is not None and body_node is not None:
                        pattern = self.recurse(
                            pattern_node, scope, parent=switch_symbol
                        ).parse_guard(counter)
                        body = self.recurse(body_node, scope, parent=switch_symbol).parse_body(
                            counter
                        )
                        cases.append(Case(guard=pattern, body=body))
                switch_kind = SwitchKind(
                    switch_symbol, expression=expression, cases=cases, default=None
                )
                self.update_dummy_symbol(symbol=switch_symbol, symbol_kind=switch_kind)
                self.file.add_symbol(switch_symbol)
                return switch_symbol

        return super().parse_metasymbol(counter)

    def walk_expression(self, counter: Counter) -> None:
        if self.node.type in ["if_expression", "switch_expression"]:
            self.parse_symbols(counter)
        elif self.node.type == "block":
            self.parse_block()

        super().walk_expression(counter)
