from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

Language = Literal["c", "javascript", "python", "typescript", "tsx"]
Identifier = str
Pos = Tuple[int, int] # (line, column)
Range = Tuple[Pos, Pos] # ((start_line, start_column), (end_line, end_column))
Substring = Tuple[int, int] # (start_byte, end_byte)
Scope = List[str] # e.g. ["A", "B"] for class B inside class A

@dataclass
class Document:
    text: str
    language: Language

@dataclass
class SymbolInfo(ABC):
    """Abstract class for symbol information."""
    document: Document
    name: str
    range: Range
    scope: Scope
    substring: Substring 

    # return the substring of the document that corresponds to this symbol info
    def get_substring(self) -> str:
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
    parameters: List[Parameter]
    return_type: Optional[str] = None
    pass


@dataclass
class IR:
    symbol_table: Dict[Identifier, SymbolInfo]


def language_from_file_extension(file_path: str) -> Optional[Language]:
    if file_path.endswith(".c"):
        return "c"
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