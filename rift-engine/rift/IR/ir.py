from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

Language = Literal["c", "cpp", "javascript", "python", "typescript", "tsx"]
QualifiedId = Tuple[str, ...] # e.g. ("A", "B", "foo") for function foo inside class B inside class A
Pos = Tuple[int, int]  # (line, column)
Range = Tuple[Pos, Pos]  # ((start_line, start_column), (end_line, end_column))
Substring = Tuple[int, int]  # (start_byte, end_byte)
Scope = List[str]  # e.g. ["A", "B"] for class B inside class A


@dataclass
class Document:
    text: bytes
    language: Language


@dataclass
class SymbolInfo(ABC):
    """Abstract class for symbol information."""
    document: Document
    language: Language
    name: str
    range: Range
    scope: Scope
    substring: Substring

    # return the substring of the document that corresponds to this symbol info
    def get_substring(self) -> bytes:
        start, end = self.substring
        return self.document.text[start:end]
    
    def get_qualified_id(self) -> QualifiedId:
        return tuple(self.scope + [self.name])


@dataclass
class Parameter:
    name: str
    type: Optional[str] = None
    optional: bool = False

    def __str__(self) -> str:
        name = self.name
        if self.optional:
            name += "?"
        if self.type is None:
            return name
        else:
            return f"{name}:{self.type}"
    __repr__ = __str__

@dataclass
class FunctionDeclaration(SymbolInfo):
    body: Optional[Substring]
    docstring: str
    parameters: List[Parameter]
    return_type: Optional[str] = None

    def get_substring_without_body(self) -> bytes:
        if self.body is None:
            return self.get_substring()
        else:
            start, end = self.substring
            body_start, body_end = self.body
            return self.document.text[start:body_start]


@dataclass
class Statement:
    type: str

    def __str__(self):
        return self.type

    __repr__ = __str__

@dataclass
class IR:
    _symbol_table: Dict[QualifiedId, SymbolInfo] = field(default_factory=dict)
    statements: List[Statement] = field(default_factory=list)

    def lookup_symbol(self, qid: QualifiedId) -> Optional[SymbolInfo]:
        return self._symbol_table.get(qid)
    
    def search_symbol(self, name: str) -> List[SymbolInfo]:
        return [symbol for symbol in self._symbol_table.values() if symbol.name == name]

    def add_symbol(self, symbol: SymbolInfo) -> None:
        self._symbol_table[symbol.get_qualified_id()] = symbol

    def get_function_declarations(self) -> List[FunctionDeclaration]:
        return [symbol for symbol in self._symbol_table.values() if isinstance(symbol, FunctionDeclaration)]

    def dump_symbol_table(self) -> str:
        lines = []
        for id in self._symbol_table:
            d = self._symbol_table[id]
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


def language_from_file_extension(file_path: str) -> Optional[Language]:
    if file_path.endswith(".c"):
        return "c"
    elif file_path.endswith(".cpp") or file_path.endswith(".cc") or file_path.endswith(".cxx") or file_path.endswith(".c++"):
        return "cpp"
    elif file_path.endswith(".js"):
        return "javascript"
    elif file_path.endswith(".py"):
        return "python"
    elif file_path.endswith(".ts"):
        return "typescript"
    elif file_path.endswith(".tsx"):
        return "tsx"
    else:
        return None