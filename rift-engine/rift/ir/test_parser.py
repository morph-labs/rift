import difflib
import os
from textwrap import dedent
from typing import List

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
        
        import foo, bar
        import foo.bar.baz
        import foo.bar.baz as fbb
        from foo.bar import baz
        from typing import Iterable, Union
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
    code_cs = (
        dedent(
            """
        // This is docstring
        namespace SampleNamespace
        {
            // This is docstring
            class SampleClass
            {
                public int sum(int a, int b)  
                {  
                    int sum = a + b;
                    return sq;
                }
            }
        }

        interface IEquatable<T>
        {
            bool Equals(T obj);
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

            let rec f1 (x:int) : int = x+1
            and f2 (x:int) : int = x+2

            let v1 = 1
            let v2:int = 2
        end
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_rescript = (
        dedent(
            """
        let (z0) = 0
        let z1:int = 0
        let (z2:int) = 0
        let z3 as z = 0
        let (z4:int) as zz = 0
        let ((z5:int) as zzz) = 0

        let mulWithDefault = (~def: int, x, y) => {
            switch (x * y) {
            | 0 => def
            | z => z
            }
        }
        
        module SomeRSModule = {
          let annot = (x:int) : int => x+1
          let paramsWithDefault = (~x: int=3, ~y=4.0, ~z: option<int>=?, ~w=?, ()) => 5
        }

        let rec multiple = (x:int, y:int) : int => bindings(x+y)
        and bindings = (z:int) => multiple(z, z)

        let _  = "not in the symbol table"
    """
        )
        .lstrip()
        .encode("utf-8")
    )
    code_ruby = (
        dedent(
            """
        def sum(a, b)
            # This is a docstring
            a + b
        end

        def output
            puts 'hello'
        end

        def greetings(a)
			return 'ciao' if a == 1
            
            'hello'
        end

        def swap(a, b)
            temp = a
            a = b
            b = temp
            return a, b
        end

        # This is a docstring for class Person
        class Person
            attr_accessor :name, :age
            
            def initialize(name, age)
                @name = name
                @age = age
            end
            
            def introduce
                puts "Hi, I'm #{@name} and I'm #{@age} years old."
            end
        end

        module Cream
            def cream?
                true
            end
        end

        module Foo
            class Bar
                def pour(container, liquid)
                    for liquid in container do
                        puts liquid
                    end
                end
            end
        end
    """
        )
        .lstrip()
        .encode("utf-8")
    )

def new_file(code: IR.Code, path: str, language: IR.Language, project:IR.Project) -> None:
    file = IR.File(path)
    parser.parse_code_block(file, code, language)
    project.add_file(file)


def get_test_project():
    project = IR.Project(root_path="dummy_path")
    new_file(IR.Code(Tests.code_c), "test.c", "c", project)
    new_file(IR.Code(Tests.code_js), "test.js", "javascript", project)
    new_file(IR.Code(Tests.code_ts), "test.ts", "typescript", project)
    new_file(IR.Code(Tests.code_tsx), "test.tsx", "tsx", project)
    new_file(IR.Code(Tests.code_py), "test.py", "python", project)
    new_file(IR.Code(Tests.code_cpp), "test.cpp", "cpp", project)
    new_file(IR.Code(Tests.code_cs), "test.cs", "c_sharp", project)
    new_file(IR.Code(Tests.code_ocaml), "test.ml", "ocaml", project)
    new_file(IR.Code(Tests.code_rescript), "test.res", "rescript", project)
    new_file(IR.Code(Tests.code_ruby), "test.rb", "ruby", project)
    return project

def get_test_python_project():
    project = IR.Project(root_path="dummy_path")
    new_file(IR.Code(Tests.code_py), "test.py", "python", project)
    return project

def test_parsing():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    symbol_table_file = os.path.join(script_dir, "symbol_table.txt")
    with open(symbol_table_file, "r") as f:
        old_symbol_table = f.read()
    project = get_test_project()

    lines: List[str] = []
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
