from dataclasses import asdict, dataclass
import json
from typing import List

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

def get_symbol_completions_raw(project: IR.Project) -> str:
    files: List[File] = []
    for file_ir in project.get_files():
        symbols: List[Symbol] = []
        for symbol_info in file_ir.search_symbol(lambda _: True):
            symbol= Symbol(symbol_info.name, symbol_info.scope, symbol_info.kind(), symbol_info.range)
            symbols.append(symbol)
        file = File(file_ir.path, symbols)
        files.append(file)
    return [asdict(symbol) for symbol in files]