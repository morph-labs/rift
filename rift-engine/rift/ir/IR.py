from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

Language = Literal["c", "cpp", "javascript", "ocaml", "python", "rescript", "typescript", "tsx"]
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
    symbols: List["SymbolInfo"]

@dataclass
class Import:
    names: List[str] # import foo, bar, baz
    substring: Substring # the substring of the document that corresponds to this import
    module_name: Optional[str] = None # from module_name import ...


@dataclass
class Type:
    _str: str
    name: Optional[str] = None
    arguments: List["Type"] = field(default_factory=list)

    def create_pointer(self) -> "Type":
        return Type(f"{self._str}*")
    
    def create_array(self) -> "Type":
        return Type(f"{self._str}[]")
    
    def create_function(self) -> "Type":
        return Type(f"{self._str}()")
    
    def create_type_of(self) -> "Type":
        return Type(f"typeof({self._str})")

    def __str__(self):
        return self._str

    __repr__ = __str__


@dataclass
class Parameter:
    name: str
    default_value: Optional[str] = None
    type: Optional[Type] = None
    optional: bool = False

    def __str__(self) -> str:
        res = self.name
        if self.optional:
            res += "?"
        if self.type is not None:
            res = res + f":{self.type}"
        if self.default_value is not None:
            res = res + f"={self.default_value}"
        return res

    __repr__ = __str__


@dataclass
class ValueKind(ABC):
    """Abstract class for value kinds."""
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def dump(self, lines: List[str]) -> None:
        pass


@dataclass
class FunctionKind(ValueKind):
    has_return: bool
    parameters: List[Parameter]
    return_type: Optional[Type] = None

    def name(self) -> str:
        return "Function"
    
    def dump(self, lines: List[str]) -> None:
        if self.parameters != []:
            lines.append(f"   parameters: {self.parameters}")
        if self.return_type is not None:
            lines.append(f"   return_type: {self.return_type}")
        if self.has_return:
            lines.append(f"   has_return: {self.has_return}")


@dataclass
class ValKind(ValueKind):
    type: Optional[Type] = None

    def name(self) -> str:
        return "Value"
    
    def dump(self, lines: List[str]) -> None:
        if self.type is not None:
            lines.append(f"   type: {self.type}")


@dataclass
class TypeKind(ValueKind):
    def name(self) -> str:
        return "Type"


@dataclass
class InterfaceKind(ValueKind):
    def name(self) -> str:
        return "Interface"


@dataclass
class ContainerKind(ABC):
    """Abstract class for container kinds."""
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def signature(self) -> Optional[str]:
        return None


@dataclass
class ClassKind(ContainerKind):
    superclasses: Optional[str]

    def name(self) -> str:
        return "Class"

    def signature(self) -> Optional[str]:
        if self.superclasses is not None:
            return self.superclasses


@dataclass
class NamespaceKind(ContainerKind):
    def name(self) -> str:
        return "Namespace"


@dataclass
class ModuleKind(ContainerKind):
    def name(self) -> str:
        return "Module"


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
            start, _end = self.substring
            body_start, _body_end = self.body_sub
            return self.code.bytes[start:body_start]

    @abstractmethod
    def dump(self, lines: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def kind(self) -> str:
        raise NotImplementedError


@dataclass
class ValueDeclaration(SymbolInfo):
    value_kind: ValueKind

    def kind(self) -> str:
        return self.value_kind.name()

    def dump(self, lines: List[str]) -> None:
        lines.append(
            f"{self.kind()}: {self.name}\n   language: {self.language}\n   range: {self.range}\n   substring: {self.substring}"
        )
        if self.scope != "":
            lines.append(f"   scope: {self.scope}")
        if self.docstring != "":
            lines.append(f"   docstring: {self.docstring}")
        if self.body_sub is not None:
            lines.append(f"   body: {self.body_sub}")
        self.value_kind.dump(lines)
            

@dataclass
class ContainerDeclaration(SymbolInfo):
    body: List[Statement]
    container_kind: ContainerKind

    def kind(self) -> str:
        return self.container_kind.name()

    def dump(self, lines: List[str]) -> None:
        signature = self.container_kind.signature()
        if signature is not None:
            id = self.name + signature
        else:
            id = self.name
        lines.append(
            f"{self.kind()}: {id}\n   language: {self.language}\n   range: {self.range}\n   substring: {self.substring}"
        )
        if self.docstring != "":
            lines.append(f"   docstring: {self.docstring}")

@dataclass
class File:
    path: str  # path of the file relative to the root directory
    statements: List[Statement] = field(default_factory=list)
    _imports: List[Import] = field(default_factory=list)
    _symbol_table: Dict[QualifiedId, SymbolInfo] = field(default_factory=dict)

    def lookup_symbol(self, qid: QualifiedId) -> Optional[SymbolInfo]:
        return self._symbol_table.get(qid)

    def search_symbol(self, name: Union[str, Callable[[str], bool]]) -> List[SymbolInfo]:
        if callable(name):
            name_filter = name
            return [symbol for symbol in self._symbol_table.values() if name_filter(symbol.name)]
        else:
            return [symbol for symbol in self._symbol_table.values() if symbol.name == name]

    def search_module_import(self, module_name: str) -> Optional[Import]:
        for import_ in self._imports:
            if import_.module_name == module_name:
                return import_
        return None

    def add_symbol(self, symbol: SymbolInfo) -> None:
        self._symbol_table[symbol.get_qualified_id()] = symbol

    def add_import(self, import_: Import) -> None:
        self._imports.append(import_)

    def get_function_declarations(self) -> List[ValueDeclaration]:
        return [
            symbol
            for symbol in self._symbol_table.values()
            if isinstance(symbol, ValueDeclaration) and isinstance(symbol.value_kind, FunctionKind)
        ]

    def dump_symbol_table(self, lines: List[str]) -> None:
        for id in self._symbol_table:
            d = self._symbol_table[id]
            d.dump(lines)

    def dump_map(self, indent: int, lines: List[str]) -> None:
        def dump_symbol(symbol: SymbolInfo, indent: int) -> None:
            decl_without_body = symbol.get_substring_without_body().decode().strip()
            # indent the declaration
            decl_without_body = decl_without_body.replace("\n", "\n" + " " * indent)
            lines.append(f"{' ' * indent}{decl_without_body}")
            if isinstance(symbol, ContainerDeclaration):
                for statement in symbol.body:
                    dump_statement(statement, indent + 2)

        def dump_statement(statement: Statement, indent: int) -> None:
            if isinstance(statement, Declaration):
                for symbol in statement.symbols:
                   dump_symbol(symbol, indent)
            else:
                pass

        for statement in self.statements:
            dump_statement(statement, indent)

    def dump_elements(self, elements: List[str]) -> None:
        def dump_symbol(symbol: SymbolInfo) -> None:
            decl_without_body = symbol.get_substring_without_body().decode()
            elements.append(decl_without_body)
            if isinstance(symbol, ContainerDeclaration):
                for statement in symbol.body:
                    dump_statement(statement)

        def dump_statement(statement: Statement) -> None:
            if isinstance(statement, Declaration):
                for symbol in statement.symbols:
                    dump_symbol(symbol)
            else:
                pass

        for statement in self.statements:
            dump_statement(statement)


@dataclass
class Reference:
    """
    A reference to a file, and optionally a symbol inside that file.
    
    The file path is the path given to the os for reading. A reference can be converted
    to a URI, which is a string that can be used to uniquely identify a reference.
    
    Examples:
    - file_path: "home/user/project/src/main.py", qualified_id: None
    - file_path: "home/user/project/src/main.py", qualified_id: "MyClass"
    - file_path: "home/user/project/src/main.py", qualified_id: "MyClass.my_function"
    
    The URI is of the form "file://<file_path>#<qualified_id>" or "file://<file_path>" 
    if qualified_id is None.
    """
    file_path: str
    qualified_id: Optional[QualifiedId] = None

    def to_uri(self) -> str:
        return self.file_path + (f"#{self.qualified_id}" if self.qualified_id is not None else "")

    @staticmethod
    def from_uri(uri: str) -> "Reference":
        parsed = urlparse(uri)
        qualified_id = parsed.fragment if parsed.fragment != "" else None
        return Reference(file_path=parsed.path, qualified_id=qualified_id)


@dataclass
class ResolvedReference:
    file: File
    symbol: Optional[SymbolInfo] = None


@dataclass
class Project:
    root_path: str
    _files: List[File] = field(default_factory=list)

    def add_file(self, file: File):
        self._files.append(file)

    def lookup_file(self, path: str) -> Optional[File]:
        for file in self._files:
            if os.path.join(self.root_path, file.path) == path:
                return file
        return None

    def lookup_reference(self, reference: Reference) -> Optional[ResolvedReference]:
        file = self.lookup_file(reference.file_path)
        if file:
            if reference.qualified_id is None:
                symbol = None
            else:
                symbol = file.lookup_symbol(reference.qualified_id)
            return ResolvedReference(file=file, symbol=symbol)

    def get_files(self) -> List[File]:
        return self._files

    def dump_map(self, indent: int = 0) -> str:
        lines: List[str] = []
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
    elif file_path.endswith(".ml"):
        return "ocaml"
    elif file_path.endswith(".py"):
        return "python"
    elif file_path.endswith(".res"):
        return "rescript"
    elif file_path.endswith(".ts"):
        return "typescript"
    elif file_path.endswith(".tsx"):
        return "tsx"
    else:
        return None
