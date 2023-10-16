import logging
import os
import re
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Callable, Dict, List, Union

from . import IR, parser

logger = logging.getLogger(__name__)


@dataclass
class MetaLanguage:
    project: IR.Project
    raw_code: str
    report_check_failed: Callable[[str], None]
    meta_variables: List[str] = field(default_factory=list)
    locals: Dict[str, Any] = field(default_factory=dict)
    globals: Dict[str, Any] = field(default_factory=dict)
    code: str = ""
    _all_symbols: List[IR.Symbol] = field(default_factory=list)

    SymbolicType = Union[
        IR.FunctionKind,
        IR.ClassKind,
        IR.Field,
        IR.Parameter,
        IR.Symbol,
        IR.TypeDefinitionKind,
    ]

    @classmethod
    def set_file_path(
        cls,
        x: SymbolicType,
        path: str,
    ) -> None:
        x._file_path = path  # type: ignore

    @classmethod
    def get_file_path(cls, x: SymbolicType) -> str:
        return x._file_path  # type: ignore

    def process_meta_variable(self, mv: str) -> None:
        if mv == "Class":
            if "Class" not in self.locals:
                classes: List[IR.Symbol] = []
                for symbol in self._all_symbols:
                    if isinstance(symbol.kind, IR.ClassKind):
                        classes.append(symbol)
                self.locals["Class"] = classes
        elif mv == "Function":
            if "Function" not in self.locals:
                functions: List[IR.FunctionKind] = []
                for symbol in self._all_symbols:
                    if isinstance(symbol.kind, IR.FunctionKind):
                        f = symbol.kind
                        self.set_file_path(f, self.get_file_path(symbol))
                        for p in f.parameters:
                            self.set_file_path(p, self.get_file_path(symbol))
                        functions.append(symbol.kind)
                self.locals["Function"] = functions
        elif mv == "TypeDefinition":
            if "TypeDefinition" not in self.locals:
                type_definitions: List[IR.TypeDefinitionKind] = []
                for symbol in self._all_symbols:
                    if isinstance(symbol.kind, IR.TypeDefinitionKind):
                        td = symbol.kind
                        self.set_file_path(td, self.get_file_path(symbol))
                        type_definitions.append(symbol.kind)
                self.locals["TypeDefinition"] = type_definitions
        elif mv == "check":

            def check(x: Any, b: bool) -> None:
                if isinstance(x, IR.Symbol):
                    if not b:
                        self.report_check_failed(
                            f"Check failed on {x.get_qualified_id()} in {self.get_file_path(x)}"
                        )
                elif isinstance(
                    x, (IR.Field, IR.FunctionKind, IR.Parameter, IR.TypeDefinitionKind)
                ):
                    if not b:
                        self.report_check_failed(f"Check failed on: {x} in {self.get_file_path(x)}")
                else:
                    raise Exception(f"Unknown type: {type(x)}")

            self.locals["check"] = check
        else:
            raise Exception(f"Unknown meta variable: {mv}")
        self.meta_variables.append(mv)

    def _process_dollar_variables(self) -> None:
        # find all dollar variables and put them in a set, and replace them witn non-dollar variables
        def replace_dollar_variables(match: Any) -> str:
            self.process_meta_variable(match.group(1))
            return match.group(1)

        self.code = re.sub(r"\$(\w+)", replace_dollar_variables, self.raw_code)

    def _populate_symbols(self) -> None:
        self._all_symbols = []
        for file_ir in self.project.get_files():
            for symbol_info in file_ir.search_symbol(lambda _: True):
                self.set_file_path(symbol_info, file_ir.path)
                self._all_symbols.append(symbol_info)

    def eval(self) -> None:
        self._populate_symbols()
        self._process_dollar_variables()
        logging.info(f"Code: {self.code}")
        logging.info(f"Locals: {self.locals.keys()}")
        logging.info(f"Globals: {self.globals.keys()}")
        exec(self.code, self.globals, self.locals)


def test_meta_language():
    code1 = dedent(
        """
        for x in $TypeDefinition:
            if x.type.kind == 'record':
                for f in x.type.fields:
                    if f.optional:
                        $check(x, f.type.name != 'option')
        """
    ).lstrip()
    code2 = dedent(
        """
        for x in $Function: 
            for p in x.parameters:
                $check(p, p.default_value != "[]")
        """
    ).lstrip()

    code = [code1, code2]
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    project = parser.parse_files_in_paths([project_root])
    failures: List[str] = []

    def report_check_failed(msg: str) -> None:
        failures.append(msg)

    ml = MetaLanguage(
        project=project,
        raw_code=code[1],
        report_check_failed=report_check_failed,
    )
    ml.eval()
    for f in failures:
        print(f)
    # assert len(failures) == 1
