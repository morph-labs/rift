from typing import List, Optional

from tree_sitter import Node

from . import parser_core
from .IR import FunctionKind, ModuleKind, Parameter, Symbol, Type, ValueKind


class OCamlParser(parser_core.SymbolParser):
    def process_body(self) -> Optional[Node]:
        pass  # handled for each declaration in a let binding

    def parse_symbols(self, counter: parser_core.Counter) -> List[Symbol]:
        if self.node.type == "value_definition":
            parameters: List[Parameter] = []

            def extract_type(node: Node, parent: Symbol) -> Type:
                return Type.unknown(node.text.decode(), parent)

            def parse_inner_parameter(inner: Node, parent: Symbol) -> Optional[Parameter]:
                if inner.type in ["label_name", "value_pattern"]:
                    name = inner.text.decode()
                    return Parameter(name=name, parent=parent)
                elif (
                    inner.type == "typed_pattern"
                    and inner.child_count == 5
                    and inner.children[2].type == ":"
                ):
                    # "(", par, ":", typ, ")"
                    id = inner.children[1]
                    tp = inner.children[3]
                    if id.type == "value_pattern":
                        name = id.text.decode()
                        type = extract_type(tp, parent)
                        return Parameter(name=name, type=type, parent=parent)
                elif inner.type == "unit":
                    name = "()"
                    type = Type.constructor(name="unit", parent=parent)
                    return Parameter(name=name, type=type, parent=parent)

            def parse_ocaml_parameter(parameter: Node, parent: Symbol) -> None:
                if parameter.child_count == 1:
                    inner_parameter = parse_inner_parameter(parameter.children[0], parent)
                    if inner_parameter is not None:
                        parameters.append(inner_parameter)
                elif parameter.child_count == 2 and parameter.children[0].type in ["~", "?"]:
                    inner_parameter = parse_inner_parameter(parameter.children[1], parent)
                    if inner_parameter is not None:
                        inner_parameter.name = parameter.children[0].type + inner_parameter.name
                        parameters.append(inner_parameter)
                elif (
                    parameter.child_count == 4
                    and parameter.children[0].type in ["~", "?"]
                    and parameter.children[2].type == ":"
                ):
                    # "~", par, ":", name
                    inner_parameter = parse_inner_parameter(parameter.children[1], parent)
                    if inner_parameter is not None:
                        inner_parameter.name = parameter.children[0].type + inner_parameter.name
                        parameters.append(inner_parameter)
                elif (
                    parameter.child_count == 6
                    and parameter.children[0].type in ["~", "?"]
                    and parameter.children[3].type == ":"
                ):
                    # "~", "(", par, ":", typ, ")"
                    inner_parameter = parse_inner_parameter(parameter.children[2], parent)
                    if inner_parameter is not None:
                        inner_parameter.name = parameter.children[0].type + inner_parameter.name
                        type = extract_type(parameter.children[4], parent)
                        inner_parameter.type = type
                        parameters.append(inner_parameter)
                elif (
                    parameter.child_count == 6
                    and parameter.children[0].type == "?"
                    and parameter.children[3].type == "="
                ):
                    # "?", "(", par, "=", val, ")"
                    inner_parameter = parse_inner_parameter(parameter.children[2], parent)
                    if inner_parameter is not None:
                        inner_parameter.name = parameter.children[0].type + inner_parameter.name
                        type = extract_type(parameter.children[4], parent).type_of(parent)
                        inner_parameter.type = type
                        parameters.append(inner_parameter)

            declarations: List[Symbol] = []
            for child in self.node.children:
                if child.type == "let_binding":
                    pattern_node = child.child_by_field_name("pattern")
                    if pattern_node is not None and pattern_node.type == "value_name":
                        parents = [n for n in (child.prev_sibling, child) if n]
                        # let rec: add node of type "let" if present before the first parent
                        if (
                            len(parents) > 0
                            and parents[0].prev_sibling is not None
                            and parents[0].prev_sibling.type == "let"
                        ):
                            parents = [parents[0].prev_sibling] + parents
                        symbol = self.mk_dummy_symbol(id=pattern_node, parents=parents)
                        for grandchild in child.children:
                            if grandchild.type == "parameter":
                                parse_ocaml_parameter(grandchild, parent=symbol)
                        return_type, _ = self.process_ocaml_body(child, parent=symbol)
                        if parameters != []:
                            symbol_kind = FunctionKind(
                                has_return=self.has_return,
                                parameters=parameters,
                                return_type=return_type,
                                symbol=symbol,
                            )
                            self.update_dummy_symbol(symbol, symbol_kind)
                        else:
                            symbol_kind = ValueKind(type=return_type, symbol=symbol)
                            self.update_dummy_symbol(symbol, symbol_kind)
                        self.file.add_symbol(symbol)
                        declarations.append(symbol)
            return declarations

        elif self.node.type == "module_definition":
            for child in self.node.children:
                if child.type == "module_binding":
                    name = child.child_by_field_name("name")
                    if name is not None:
                        new_scope = self.scope + name.text.decode() + "."
                        symbol = self.mk_dummy_symbol(id=name, parents=[self.node])
                        _, body_node = self.process_ocaml_body(child, parent=symbol)
                        if body_node is not None:
                            self.recurse(body_node, new_scope, parent=symbol).parse_block()
                        self.update_dummy_symbol(symbol, ModuleKind(symbol))
                        self.file.add_symbol(symbol)
                        return [symbol]

        return super().parse_symbols(counter)
