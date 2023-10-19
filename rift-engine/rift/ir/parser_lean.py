import logging
from typing import List

from rift.ir.parser_core import Counter, SymbolParser

from .IR import DefKind, StructureKind, Symbol, TheoremKind

logger = logging.getLogger(__name__)


class LeanParser(SymbolParser):
    def parse_symbols(self, counter: Counter) -> List[Symbol]:
        node = self.node
        if node.type == "declaration" and len(node.children) >= 1:
            node0 = node.children[-1]
            body_node = node0.child_by_field_name("body")
            if body_node is not None:
                self.body_sub = (body_node.start_byte, body_node.end_byte)
            name_node = node0.child_by_field_name("name")
            if name_node is not None:
                symbol = None
                if node0.type == "def":
                    symbol = self.mk_dummy_symbol(id=name_node, parents=[node])
                    self.update_dummy_symbol(symbol, DefKind(symbol))
                elif node0.type == "structure":
                    # a structure/class does not have a body
                    node_after_name = name_node.next_sibling
                    if node_after_name is not None:
                        self.body_sub = (node_after_name.start_byte, node.end_byte)
                    symbol = self.mk_dummy_symbol(id=name_node, parents=[node])
                    self.update_dummy_symbol(symbol, StructureKind(symbol))
                elif node0.type == "theorem":
                    symbol = self.mk_dummy_symbol(id=name_node, parents=[node])
                    self.update_dummy_symbol(symbol, TheoremKind(symbol))
                else:
                    logger.warning(f"Lean: Unknown node0 type: {node0.type}")
                if symbol is not None:
                    self.file.add_symbol(symbol)
                    return [symbol]

        return super().parse_symbols(counter)
