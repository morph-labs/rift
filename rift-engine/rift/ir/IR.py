from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

Language = Literal["c", "cpp", "javascript", "python", "typescript", "tsx"]
# e.g. ("A", "B", "foo") for function foo inside class B inside class A
QualifiedId = str
Pos = Tuple[int, int]  # (line, column)
Range = Tuple[Pos, Pos]  # ((start_line, start_column), (end_line, end_column))
Substring = Tuple[int, int]  # (start_byte, end_byte)
Scope = str  # e.g. "A.B." for class B inside class A


@dataclass
class Code:
    bytes: bytes

    def __str__(self):
        return self.bytes.decode()

    __repr__ = __str__

    def apply_edit(self, edit: "CodeEdit") -> "Code":
        return edit.apply(self)

    def apply_edits(self, edits: List["CodeEdit"]) -> "Code":
        code = self
        # sort the edits in descending order of their start position
        edits.sort(key=lambda x: -x.substring[0])
        for edit in edits:
            code = code.apply_edit(edit)
        return code


@dataclass
class CodeEdit:
    substring: Substring
    new_bytes: bytes

    def apply(self, code: Code) -> Code:
        start, end = self.substring
        return Code(code.bytes[:start] + self.new_bytes + code.bytes[end:])


@dataclass
class Statement:
    type: str

    def __str__(self):
        return self.type

    __repr__ = __str__


@dataclass
class Declaration(Statement):
    symbol: "SymbolInfo"


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
class SymbolInfo(ABC):
    """Abstract class for symbol information."""

    body_sub: Optional[Substring]
    code: Code
    docstring: str
    exported: bool
    language: Language
    name: str
    range: Range
    scope: Scope
    substring: Substring

    # return the substring of the document that corresponds to this symbol info
    def get_substring(self) -> bytes:
        start, end = self.substring
        return self.code.bytes[start:end]

    def get_qualified_id(self) -> QualifiedId:
        return self.scope + self.name

    def get_substring_without_body(self) -> bytes:
        if self.body_sub is None:
            return self.get_substring()
        else:
            start, end = self.substring
            body_start, body_end = self.body_sub
            return self.code.bytes[start:body_start]

    @abstractmethod
    def dump(self, lines: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def kind(self) -> str:
        raise NotImplementedError

@dataclass
class FunctionDeclaration(SymbolInfo):
    has_return: bool
    parameters: List[Parameter]
    return_type: Optional[str] = None

    def kind(self) -> str:
        return "Function"

    def dump(self, lines: List[str]) -> None:
        lines.append(
            f"{self.kind()}: {self.name}\n   language: {self.language}\n   range: {self.range}\n   substring: {self.substring}"
        )
        if self.parameters != []:
            lines.append(f"   parameters: {self.parameters}")
        if self.return_type is not None:
            lines.append(f"   return_type: {self.return_type}")
        if self.scope != "":
            lines.append(f"   scope: {self.scope}")
        if self.docstring != "":
            lines.append(f"   docstring: {self.docstring}")
        if self.body_sub is not None:
            lines.append(f"   body: {self.body_sub}")
        if self.has_return:
            lines.append(f"   has_return: {self.has_return}")


@dataclass
class ClassDeclaration(SymbolInfo):
    body: List[Statement]
    superclasses: Optional[str]

    def kind(self) -> str:
        return "Class"

    def dump(self, lines: List[str]) -> None:
        if self.superclasses is not None:
            id = self.name + self.superclasses
        else:
            id = self.name
        lines.append(
            f"{self.kind()}: {id}\n   language: {self.language}\n   range: {self.range}\n   substring: {self.substring}"
        )
        if self.docstring != "":
            lines.append(f"   docstring: {self.docstring}")

@dataclass
class NamespaceDeclaration(SymbolInfo):
    body: List[Statement]

    def kind(self) -> str:
        return "Namespace"

    def dump(self, lines: List[str]) -> None:
        id = self.name
        lines.append(
            f"{self.kind()}: {id}\n   language: {self.language}\n   range: {self.range}\n   substring: {self.substring}"
        )
        if self.docstring != "":
            lines.append(f"   docstring: {self.docstring}")


@dataclass
class TypeDeclaration(SymbolInfo):
    is_interface: bool

    def kind(self) -> str:
        return "Interface" if self.is_interface else "Type"

    def dump(self, lines: List[str]) -> None:
        lines.append(
            f"{self.kind()}: {self.name}\n   language: {self.language}\n   range: {self.range}\n   substring: {self.substring}"
        )
        if self.docstring != "":
            lines.append(f"   docstring: {self.docstring}")


@dataclass
class File:
    path: str  # path of the file relative to the root directory
    statements: List[Statement] = field(default_factory=list)
    _symbol_table: Dict[QualifiedId, SymbolInfo] = field(default_factory=dict)

    def lookup_symbol(self, qid: QualifiedId) -> Optional[SymbolInfo]:
        return self._symbol_table.get(qid)

    def search_symbol(self, name: str) -> List[SymbolInfo]:
        return [symbol for symbol in self._symbol_table.values() if name == "" or symbol.name == name]

    def add_symbol(self, symbol: SymbolInfo) -> None:
        self._symbol_table[symbol.get_qualified_id()] = symbol

    def get_function_declarations(self) -> List[FunctionDeclaration]:
        return [
            symbol
            for symbol in self._symbol_table.values()
            if isinstance(symbol, FunctionDeclaration)
        ]

    def dump_symbol_table(self, lines: List[str]) -> None:
        for id in self._symbol_table:
            d = self._symbol_table[id]
            d.dump(lines)

    def dump_map(self, indent: int, lines: List[str]) -> None:
        def dump_symbol(symbol: SymbolInfo, indent: int) -> None:
            decl_without_body = symbol.get_substring_without_body().decode()
            lines.append(f"{' ' * indent}{decl_without_body}")
            if isinstance(symbol, ClassDeclaration) or isinstance(symbol, NamespaceDeclaration):
                for statement in symbol.body:
                    dump_statement(statement, indent + 2)

        def dump_statement(statement: Statement, indent: int) -> None:
            if isinstance(statement, Declaration):
                dump_symbol(statement.symbol, indent)
            else:
                pass

        for statement in self.statements:
            dump_statement(statement, indent)

    def dump_elements(self, elements: List[str]) -> None:
        def dump_symbol(symbol: SymbolInfo) -> None:
            decl_without_body = symbol.get_substring_without_body().decode()
            elements.append(decl_without_body)
            if isinstance(symbol, ClassDeclaration):
                for statement in symbol.body:
                    dump_statement(statement)

        def dump_statement(statement: Statement) -> None:
            if isinstance(statement, Declaration):
                dump_symbol(statement.symbol)
            else:
                pass

        for statement in self.statements:
            dump_statement(statement)


@dataclass
class Project:
    root_path: str
    _files: List[File] = field(default_factory=list)

    def add_file(self, file: File):
        self._files.append(file)

    def get_files(self) -> List[File]:
        return self._files

    def dump_map(self, indent: int = 0) -> str:
        lines = []
        for file in self.get_files():
            lines.append(f"{' ' * indent}File: {file.path}")
            file.dump_map(indent + 2, lines)
        return "\n".join(lines)

    def dump_elements(self) -> List[str]:
        elements: List[str] = []
        for file in self.get_files():
            file.dump_elements(elements)
        return elements


def language_from_file_extension(file_path: str) -> Optional[Language]:
    if file_path.endswith(".c"):
        return "c"
    elif (
        file_path.endswith(".cpp")
        or file_path.endswith(".cc")
        or file_path.endswith(".cxx")
        or file_path.endswith(".c++")
    ):
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
