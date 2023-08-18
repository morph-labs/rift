from dataclasses import asdict, dataclass
import json
from typing import List

import rift.ir.IR as IR

@dataclass
class SymbolInfo:
    name: str
    scope: str
    kind: str

def get_symbol_completions(project: IR.Project) -> str:
    symbol_infos: List[SymbolInfo] = []
    for file in project.get_files():
        for symbol in file.search_symbol(""):
            symbol_info = SymbolInfo(symbol.name, symbol.scope, symbol.kind())
            symbol_infos.append(symbol_info)
    json_data = json.dumps([asdict(symbol) for symbol in symbol_infos], indent=4)
    return json_data