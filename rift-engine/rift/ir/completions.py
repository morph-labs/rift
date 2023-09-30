import json
from dataclasses import asdict, dataclass
from typing import Dict, List

import rift.ir.IR as IR


@dataclass
class Symbol:
    name: str
    scope: str
    kind: str
    range: IR.Range


@dataclass
class File:
    path: str
    symbols: List[Symbol]


def get_symbol_completions(project: IR.Project) -> str:
    return json.dumps(get_symbol_completions_raw(project), indent=4)


def get_symbol_completions_raw(project: IR.Project) -> List[Dict[str, Symbol]]:
    files: List[File] = []
    for file_ir in project.get_files():
        symbols: List[Symbol] = []
        for symbol in file_ir.search_symbol(lambda _: True):
            if isinstance(symbol.symbol_kind, IR.MetaSymbolKind):
                continue # don't emit completions for statements inside bodies
            symbol = Symbol(
                symbol.name, symbol.scope, symbol.kind(), symbol.range
            )
            symbols.append(symbol)
        file = File(file_ir.path, symbols)
        files.append(file)
    return [asdict(symbol) for symbol in files]
