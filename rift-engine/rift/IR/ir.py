from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

Language = Literal["c", "cpp", "javascript", "python", "typescript", "tsx"]
Identifier = str
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
    symbol_table: Dict[Identifier, SymbolInfo]
    statements: List[Statement] = field(default_factory=list)


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
