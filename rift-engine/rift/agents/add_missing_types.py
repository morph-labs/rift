import asyncio
from dataclasses import dataclass, field
import functools
import openai
import os
from textwrap import dedent
from typing import AsyncIterable, ClassVar, List, Optional, Type, Dict

from rift.agents.cli_agent import Agent, ClientParams, launcher
from rift.agents.util import ainput
import rift.ir.IR as IR
from rift.ir.missing_types import FileMissingTypes, MissingType, files_missing_types_in_project, functions_missing_types_in_file
import rift.ir.parser as parser
from rift.ir.response import extract_blocks_from_response, replace_functions_from_code_blocks
import rift.util.file_diff as file_diff


class Config:
    debug = False
    max_size_group_missing_types = 10  # maximum size for a group of missing types
    model = "gpt-3.5-turbo-0613"  # ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k"]
    temperature = 0

    @classmethod
    def root_dir(cls) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(script_dir) + "/llm"


@dataclass
class MissingTypesParams(ClientParams):
    pass


Message = Dict[str, str]
Prompt = List[Message]


class MissingTypePrompt:
    @staticmethod
    def mk_user_msg(missing_types: List[MissingType], code: IR.Code) -> str:
        missing_str = ""
        n = 0
        for mt in missing_types:
            n += 1
            missing_str += f"{n}. {mt}\n"
        return dedent(f"""
        Add missing types for the following functions:
        {missing_str}

        The code is:
        ```
        {code}
        ```
        """).lstrip()

    @staticmethod
    def code_for_missing_types(missing_types: List[MissingType]) -> IR.Code:
        bytes = b""
        for mt in missing_types:
            bytes += mt.function_declaration.get_substring()
            bytes += b"\n"
        return IR.Code(bytes)

    @staticmethod
    def create_prompt_for_file(missing_types: List[MissingType]) -> Prompt:
        code = MissingTypePrompt.code_for_missing_types(missing_types)
        system_msg = dedent("""
            Act as an expert software developer.
            For each function to modify, give an *edit block* per the example below.

            You MUST format EVERY code change with an *edit block* like this:

            ```python
                def mul(a: t1, b : t2) -> t3
                    ...
            ```

            Every *edit block* must be fenced with ```...``` with the correct code language.
            Edits to different functions each need their own *edit block*.
            Give all the required changes at once in the reply.
            """).lstrip()
        return [
            dict(role="system", content=system_msg),
            dict(role="user", content=MissingTypePrompt.mk_user_msg(
                missing_types=missing_types, code=code)),
        ]


@dataclass
class FileProcess:
    file_missing_types: FileMissingTypes
    edits: List[IR.CodeEdit] = field(default_factory=list)
    file_change: Optional[file_diff.FileChange] = None
    new_num_missing: Optional[int] = None


def count_missing(missing_types: List[MissingType]) -> int:
    return sum([int(mt) for mt in missing_types])


def get_num_missing_in_code(code: IR.Code, language: IR.Language) -> int:
    file = IR.File("dummy")
    parser.parse_code_block(file, code, language)
    return count_missing(functions_missing_types_in_file(file))


@dataclass
class MissingTypesAgent(Agent):
    name: ClassVar[str] = "missing_types"
    run_params: Type[MissingTypesParams] = MissingTypesParams
    splash: Optional[
        str
    ] = "@@@ Add Missing Types @@@\n"
    debug = Config.debug
    root_dir: str = Config.root_dir()

    def process_response(self, document: IR.Code, language: IR.Language,  missing_types: List[MissingType], response: str) -> List[IR.CodeEdit]:
        if self.debug:
            self.console.print(f"response:\n{response}\n")
        code_blocks = extract_blocks_from_response(response)
        if self.debug:
            self.console.print(f"code_blocks:\n{code_blocks}\n")
        filter_function_ids = [
            mt.function_declaration.get_qualified_id() for mt in missing_types]
        edits = replace_functions_from_code_blocks(
            code_blocks=code_blocks, document=document,
            filter_function_ids=filter_function_ids, language=language, replace_body=False)
        return edits

    async def code_edits_for_missing_files(self, document: IR.Code, language: IR.Language,  missing_types: List[MissingType]) -> List[IR.CodeEdit]:
        loop = asyncio.get_event_loop()
        prompt = MissingTypePrompt.create_prompt_for_file(
            missing_types=missing_types)
        # Partially apply parameters to ChatCompletion.create for later execution
        func = functools.partial(openai.ChatCompletion.create, model=Config.model,
                                 messages=prompt, temperature=Config.temperature)
        # Run OpenAI API call concurrently to avoid blocking event loop
        completion = await loop.run_in_executor(None, func)
        if not isinstance(completion, dict):
            raise Exception(
                f"Unexpected type for completion: {type(completion)}")
        response: str = completion['choices'][0]['message']['content']
        edits = self.process_response(
            document=document, language=language, missing_types=missing_types, response=response)
        return edits

    def split_missing_types_in_groups(self, missing_types: List[MissingType]) -> List[List[MissingType]]:
        """Split the missing types in groups of at most Config.max_size_group_missing_types,
        and that don't contain functions with the same name."""
        groups_of_missing_types: List[List[MissingType]] = []
        group: List[MissingType] = []
        for mt in missing_types:
            group.append(mt)
            do_split = len(group) == Config.max_size_group_missing_types

            # also split if a function with the same name is in the current group (e.g. from another class)
            for mt2 in group:
                if mt.function_declaration.name == mt2.function_declaration.name:
                    do_split = True
                    break

            if do_split:
                groups_of_missing_types.append(group)
                group = []
        if len(group) > 0:
            groups_of_missing_types.append(group)
        return groups_of_missing_types

    async def process_file(self, file_process: FileProcess) -> None:
        fmt = file_process.file_missing_types
        self.console.print(
            f"Fetching types for `{fmt.file.path}`")
        api_key = os.environ["OPENAI_API_KEY"]
        if api_key is None:
            raise Exception("OPENAI_API_KEY environment variable not set")
        openai.api_key = api_key

        language = fmt.language
        document = fmt.code
        groups_of_missing_types = self.split_missing_types_in_groups(
            fmt.missing_types)

        for missing_types in groups_of_missing_types:
            new_edits = await self.code_edits_for_missing_files(document, language, missing_types)
            file_process.edits += new_edits
        new_document = fmt.code.apply_edits(file_process.edits)
        old_num_missing = count_missing(
            file_process.file_missing_types.missing_types)
        new_num_missing = get_num_missing_in_code(new_document, fmt.language)
        self.console.print(
            f"Received types for `{fmt.file.path}` ({new_num_missing}/{old_num_missing} missing)")
        if self.debug:
            self.console.print(f"new_document:\n{new_document}\n")
        path = os.path.join(self.root_dir, fmt.file.path)
        file_change = file_diff.get_file_change(
            path=path, new_content=str(new_document))
        if self.debug:
            self.console.print(f"file_change:\n{file_change}\n")
        file_process.file_change = file_change
        file_process.new_num_missing = new_num_missing

    async def run(self) -> AsyncIterable[List[file_diff.FileChange]]:
        def print_missing(fmt: FileMissingTypes) -> None:
            self.console.print(f"File: {fmt.file.path}")
            for mt in fmt.missing_types:
                self.console.print(f"  {mt}")
            self.console.print()

        file_processes: List[FileProcess] = []
        tot_num_missing = 0
        project = parser.parse_files_in_project(self.root_dir)
        if self.debug:
            self.console.print(f"\n=== Project Map ===\n{project.dump_map()}\n")
        files_missing_types = files_missing_types_in_project(project)
        self.console.print(f"\n=== Missing Types ===\n")
        for fmt in files_missing_types:
            print_missing(fmt)
            tot_num_missing += count_missing(fmt.missing_types)
            file_processes.append(FileProcess(
                file_missing_types=fmt))

        await ainput("\n> Press any key to continue.\n")

        tasks: List[asyncio.Task] = [
            asyncio.create_task(self.process_file(file_processes[i]))
            for i in range(len(files_missing_types))
        ]
        await asyncio.gather(*tasks)

        file_changes: List[file_diff.FileChange] = []
        tot_new_missing = 0
        for fp in file_processes:
            if fp.file_change is not None:
                file_changes.append(fp.file_change)
            if fp.new_num_missing is not None:
                tot_new_missing += fp.new_num_missing
            else:
                tot_new_missing += count_missing(
                    fp.file_missing_types.missing_types)
        self.console.print(
            f"Missing types: {tot_new_missing}/{tot_num_missing} ({tot_new_missing/tot_num_missing*100:.2f}%)")
        yield file_changes


if __name__ == "__main__":
    launcher(MissingTypesAgent, MissingTypesParams)
