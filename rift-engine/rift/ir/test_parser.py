import difflib
import os
from textwrap import dedent

from tree_sitter import Tree

import rift.ir.IR as IR
import rift.ir.parser as parser


class Tests:
    code_c = (
        dedent(
            """
        int aa() {
          return 0;
        }
        /** This is a docstring */
        int * foo(int **x) {
          *x = 0;
        }

        int bb() {
          return 0;
        }

        int main() {
          int *x;
          foo(&x);
          *x = 1;
          return 0;
        }
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_js = (
        dedent(
            """
        /** Some docstring */
        function f1() { return 0; }
        /** Some docstring on an arrow function */
        let f2 = x => x+1;
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_ts = (
        dedent(
            """
        type a = readonly b[][];
        function ts(x:number, opt?:string) : number { return x }
        export function ts2() : array<number> { return [] }
        export class A {
            constructor() {}
            async load(v: number) {
                return v
            }
        }
        interface RunHelperSyncResult {
            id: number
            text: string
        }
        type HelperStatus = 'running' | 'done' | 'error' | 'accepted' | 'rejected'
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_tsx = (
        dedent(
            """
        d = <div> "abc" </div>
        function tsx() { return d }
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_py = (
        dedent(
            """
        class A(C,D):
            \"\"\"
            This is a docstring
            for class A
            \"\"\"

            def py(x, y):
                \"\"\"This is a docstring\"\"\"
                return x
        class B:
            @abstractmethod
            async def insert_code(
                self, document: str, cursor_offset: int, goal: Optional[str] = None
            ) -> InsertCodeResult:
                pass
            async def load(self, v):
                pass
            class Nested:
                def nested():
                    pass
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_cpp = (
        dedent(
            """
        namespace namespace_name 
        {
            void add() {}
            class student {
                public:
                    void print();
            };
        }
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_ocaml = (
        dedent(
            """
        let divide (x:int) y = x / y
        let callback () : unit = ()
        module M = struct
            let bump ?(step = 1) x = x + step
            let hline ~x:x1 ~x:x2 ~y = (x1, x2, y)
        end
        module N = struct
            let with_named_args ~(named_arg1 : int) ?named_arg2 = named_arg1 + named_arg2
        end
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_lean = (
        dedent(
            """
        -- Defines a function that takes a name and produces a greeting.
        def getGreeting (name : String) := s!"Hello, {name}! Isn't Lean great?"

        -- The `main` function is the entry point of your program.
        -- Its type is `IO Unit` because it can perform `IO` operations (side effects).
        def main : IO Unit :=
            -- Define a list of names
            let names := ["Sebastian", "Leo", "Daniel"]

            -- Map each name to a greeting
            let greetings := names.map getGreeting

            -- Print the list of greetings
            for greeting in greetings do
                IO.println greeting

        section Prio
        -- We set this priority to 0 later in this file
        -- porting note: unsupported set_option extends_priority 200

        /- control priority of
        `instance [Algebra R A] : SMul R A` -/
        /-- An associative unital `R`-algebra is a semiring `A` equipped with a map into its center `R → A`.

        See the implementation notes in this file for discussion of the details of this definition.
        -/
        -- porting note: unsupported @[nolint has_nonempty_instance]
        class Algebra (R : Type u) (A : Type v) [CommSemiring R] [Semiring A] extends SMul R A,
            R →+* A where
            commutes' : ∀ r x, toRingHom r * x = x * toRingHom r
            smul_def' : ∀ r x, r • x = toRingHom r * x
        #align algebra Algebra

        end Prio

        @[simp]
        theorem constAlgHom_eq_algebra_ofId : constAlgHom R A R = Algebra.ofId R (A → R) :=
          rfl
        #align pi.const_alg_hom_eq_algebra_of_id Pi.constAlgHom_eq_algebra_ofId
        
    """
        )
        .lstrip()
        .encode("utf-8")
    )


def get_test_project():
    project = IR.Project(root_path="dummy_path")

    def new_file(code: IR.Code, path: str, language: IR.Language) -> None:
        file = IR.File(path)
        parser.parse_code_block(file, code, language)
        project.add_file(file)

    new_file(IR.Code(Tests.code_c), "test.c", "c")
    new_file(IR.Code(Tests.code_js), "test.js", "javascript")
    new_file(IR.Code(Tests.code_ts), "test.ts", "typescript")
    new_file(IR.Code(Tests.code_tsx), "test.tsx", "tsx")
    new_file(IR.Code(Tests.code_py), "test.py", "python")
    new_file(IR.Code(Tests.code_cpp), "test.cpp", "cpp")
    new_file(IR.Code(Tests.code_ocaml), "test.ml", "ocaml")
    new_file(IR.Code(Tests.code_lean), "test.lean", "lean")
    return project


def test_parsing():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    symbol_table_file = os.path.join(script_dir, "symbol_table.txt")
    with open(symbol_table_file, "r") as f:
        old_symbol_table = f.read()
    project = get_test_project()

    lines = []
    for file in project.get_files():
        lines.append(f"=== Symbol Table for {file.path} ===")
        file.dump_symbol_table(lines=lines)
    symbol_table_str = "\n".join(lines)
    ir_map_str = project.dump_map(indent=0)
    symbol_table_str += "\n\n=== Project Map ===\n" + ir_map_str
    if symbol_table_str != old_symbol_table:
        diff = difflib.unified_diff(
            old_symbol_table.splitlines(keepends=True), symbol_table_str.splitlines(keepends=True)
        )
        diff_output = "".join(diff)

        # if you want to update the symbol table, set this to True
        update_symbol_table = os.getenv("UPDATE_TESTS", "False") == "True"
        if update_symbol_table:
            print("Updating Symbol Table...")
            with open(symbol_table_file, "w") as f:
                f.write(symbol_table_str)

        assert (
            update_symbol_table
        ), f"Symbol Table has changed (to update set `UPDATE_TESTS=True`):\n\n{diff_output}"
