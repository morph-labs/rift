import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, ClassVar, Coroutine, Dict, List, Optional, Set, Tuple, cast
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import openai

import rift.agents.abstract as agent
import rift.agents.registry as registry
import rift.ir.IR as IR
import rift.ir.parser as parser
import rift.llm.openai_types as openai_types
import rift.lsp.types as lsp
import rift.util.file_diff as file_diff
from rift.agents.agenttask import AgentTask
from rift.ir.missing_docstrings import (
    FunctionMissingDocstring,
    FileMissingDocstrings,
    functions_missing_docstrings_in_file,
    files_missing_docstrings_in_project,
)
from rift.ir.response import (
    Replace,
    extract_blocks_from_response,
    replace_functions_from_code_blocks,
    update_typing_imports,
)
from rift.lsp import LspServer
from rift.util.TextStream import TextStream

logger = logging.getLogger(__name__)
Message = Dict[str, Any]
Prompt = List[Message]


@dataclass
class Params(agent.AgentParams):
    ...


@dataclass
class Result(agent.AgentRunResult):
    ...


@dataclass
class State(agent.AgentState):
    params: Params
    messages: List[openai_types.Message]
    response_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Config:
    debug = False
    model = "gpt-3.5-turbo"  # ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k"]
    temperature = 0.0
    max_size_group_missing_docstrings = 10  # Max number of functions to process at once


class MissingDocStringPrompt:
    @staticmethod
    def mk_user_msg(
        functions_missing_docstrings: List[FunctionMissingDocstring], code: IR.Code
    ) -> str:
        missing_str = ""
        n = 0
        for function in functions_missing_docstrings:
            n += 1
            missing_str += f"{n}. {function.function_declaration.name}\n"

        return dedent(
            f"""
        Write doc strings for the following functions:
        {missing_str}

        The code is:
        ```
        {code}
        ```
        """
        ).lstrip()

    @staticmethod
    def code_for_missing_docstring_functions(
        functions_missing_docstrings: List[FunctionMissingDocstring],
    ) -> IR.Code:
        bytes = b""
        for function in functions_missing_docstrings:
            bytes += function.function_declaration.get_substring()
            bytes += b"\n"
        return IR.Code(bytes)

    @staticmethod
    def create_prompt_for_file(
        language: IR.Language,
        functions_missing_docstrings: List[FunctionMissingDocstring],
    ) -> Prompt:
        example_py = '''
            ```python
                def foo(a: t1, b : t2) -> t3
                    """
                    Adds two numbers together.

                    :param a: The first number to add.
                    :param b: The second number to add.
                    :return: The sum of a and b.
                    """
            ```
        '''
        example_js = """
            ```javascript
                /**
                * Adds two numbers together.
                * 
                * @param {t1} a - The first number to add.
                * @param {t2} b - The second number to add.
                * @returns {t3} The sum of a and b.
                */
                function foo(a: t1, b : t2) : t3 {
            ```
        """
        example_ts = """
            ```typescript
                /**
                * Adds two numbers together.
                * 
                * @param a - The first number to add.
                * @param b - The second number to add.
                * @returns The sum of a and b.
                */
                function foo(a: t1, b : t2): t3 {
            ```
        """
        example_ocaml = """
            ```ocaml
                (** Adds two numbers together.
                @param a The first number to add.
                @param b The second number to add.
                @return The sum of a and b. *)
                let foo (a: t1) (b : t2) : t3 =
            ```
        """
        example = ""
        if language in ["typescript", "tsx"]:
            example = example_ts
        elif language == "javascript":
            example = example_js
        elif language == "ocaml":
            example = example_ocaml
        else:
            example = example_py
        system_msg = dedent(
            """
            Act as an expert software developer.
            For each function to modify, give an *edit block* per the example below.

            You MUST format EVERY code change with an *edit block* like this:
            """
            + example
            + """
            Every *edit block* must be fenced with ```...``` with the correct code language.
            Edits to different functions each need their own *edit block*.
            Give all the required changes at once in the reply.
            """
        ).lstrip()

        code = MissingDocStringPrompt.code_for_missing_docstring_functions(
            functions_missing_docstrings
        )
        return [
            dict(role="system", content=system_msg),
            dict(
                role="user",
                content=MissingDocStringPrompt.mk_user_msg(
                    functions_missing_docstrings=functions_missing_docstrings,
                    code=code,
                ),
            ),
        ]


@dataclass
class FileProcess:
    file_missing_docstrings: FileMissingDocstrings
    edits: List[IR.CodeEdit] = field(default_factory=list)
    updated_functions: List[IR.ValueDeclaration] = field(default_factory=list)
    file_change: Optional[file_diff.FileChange] = None
    new_num_missing: Optional[int] = None


@registry.agent(
    agent_description="Generate missing docstrings for functions",
    display_name="Auto Doc",
    agent_icon="""\
<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M3.78328 7.38333V6.95H3.34995V7.38333H3.78328ZM3.78328 10.85H3.34995V11.2833H3.78328V10.85ZM12.45 7.38333H12.8833V6.95H12.45V7.38333ZM12.45 10.85V11.2833H12.8833V10.85H12.45ZM13.3167 4.78333H13.75V4.60393L13.6235 4.47653L13.3167 4.78333ZM10.7166 2.18333L11.0234 1.87653L10.896 1.75H10.7166V2.18333ZM3.34995 7.38333V10.85H4.21662V7.38333H3.34995ZM3.78328 11.2833H4.64995V10.4167H3.78328V11.2833ZM5.94996 9.98333V8.25H5.08329V9.98333H5.94996ZM4.64995 6.95H3.78328V7.81667H4.64995V6.95ZM5.94996 8.25C5.94996 7.90522 5.81299 7.57456 5.56919 7.33076C5.3254 7.08696 4.99473 6.95 4.64995 6.95V7.81667C4.76488 7.81667 4.8751 7.86232 4.95637 7.94359C5.03763 8.02485 5.08329 8.13507 5.08329 8.25H5.94996ZM4.64995 11.2833C4.99473 11.2833 5.3254 11.1464 5.56919 10.9026C5.81299 10.6588 5.94996 10.3281 5.94996 9.98333H5.08329C5.08329 10.0983 5.03763 10.2085 4.95637 10.2897C4.8751 10.371 4.76488 10.4167 4.64995 10.4167V11.2833ZM6.81663 8.25V9.98333H7.6833V8.25H6.81663ZM9.41664 9.98333V8.25H8.54997V9.98333H9.41664ZM9.41664 8.25C9.41664 7.90522 9.27967 7.57456 9.03587 7.33076C8.79208 7.08696 8.46141 6.95 8.11663 6.95V7.81667C8.23156 7.81667 8.34178 7.86232 8.42305 7.94359C8.50431 8.02485 8.54997 8.13507 8.54997 8.25H9.41664ZM8.11663 11.2833C8.46141 11.2833 8.79208 11.1464 9.03587 10.9026C9.27967 10.6588 9.41664 10.3281 9.41664 9.98333H8.54997C8.54997 10.0983 8.50431 10.2085 8.42305 10.2897C8.34178 10.371 8.23156 10.4167 8.11663 10.4167V11.2833ZM6.81663 9.98333C6.81663 10.3281 6.95359 10.6588 7.19739 10.9026C7.44119 11.1464 7.77185 11.2833 8.11663 11.2833V10.4167C8.0017 10.4167 7.89148 10.371 7.81022 10.2897C7.72895 10.2085 7.6833 10.0983 7.6833 9.98333H6.81663ZM7.6833 8.25C7.6833 8.13507 7.72895 8.02485 7.81022 7.94359C7.89148 7.86232 8.0017 7.81667 8.11663 7.81667V6.95C7.77185 6.95 7.44119 7.08696 7.19739 7.33076C6.95359 7.57456 6.81663 7.90522 6.81663 8.25H7.6833ZM10.2833 6.95V11.2833H11.15V6.95H10.2833ZM10.7166 7.81667H12.45V6.95H10.7166V7.81667ZM12.0166 7.38333V8.68333H12.8833V7.38333H12.0166ZM10.7166 11.2833H12.45V10.4167H10.7166V11.2833ZM12.8833 10.85V9.55H12.0166V10.85H12.8833ZM3.34995 6.08333V3.05H2.48328V6.08333H3.34995ZM12.8833 4.78333V6.08333H13.75V4.78333H12.8833ZM3.78328 2.61667H10.7166V1.75H3.78328V2.61667ZM10.4098 2.49013L13.0098 5.09013L13.6235 4.47653L11.0234 1.87653L10.4098 2.49013ZM3.34995 3.05C3.34995 2.93507 3.3956 2.82485 3.47687 2.74359C3.55813 2.66232 3.66835 2.61667 3.78328 2.61667V1.75C3.4385 1.75 3.10784 1.88696 2.86404 2.13076C2.62024 2.37456 2.48328 2.70522 2.48328 3.05H3.34995ZM2.48328 12.15V13.45H3.34995V12.15H2.48328ZM3.78328 14.75H12.45V13.8833H3.78328V14.75ZM13.75 13.45V12.15H12.8833V13.45H13.75ZM12.45 14.75C12.7948 14.75 13.1254 14.613 13.3692 14.3692C13.613 14.1254 13.75 13.7948 13.75 13.45H12.8833C12.8833 13.5649 12.8377 13.6751 12.7564 13.7564C12.6751 13.8377 12.5649 13.8833 12.45 13.8833V14.75ZM2.48328 13.45C2.48328 13.7948 2.62024 14.1254 2.86404 14.3692C3.10784 14.613 3.4385 14.75 3.78328 14.75V13.8833C3.66835 13.8833 3.55813 13.8377 3.47687 13.7564C3.3956 13.6751 3.34995 13.5649 3.34995 13.45H2.48328Z" fill="#CCCCCC"/>
</svg>""",
)
@dataclass
class MissingDocstringAgent(agent.ThirdPartyAgent):
    agent_type: ClassVar[str] = "missing_docstring_agent"
    params_cls: ClassVar[Any] = Params
    debug: bool = Config.debug

    @classmethod
    async def create(cls, params: Any, server: LspServer) -> Any:
        state = State(params=params, messages=[], response_lock=asyncio.Lock())
        obj: agent.ThirdPartyAgent = cls(state=state, agent_id=params.agent_id, server=server)
        return obj

    def process_response(
        self,
        document: IR.Code,
        language: IR.Language,
        functions_missing_docstrings: List[FunctionMissingDocstring],
        response: str,
    ) -> Tuple[List[IR.CodeEdit], List[IR.ValueDeclaration]]:
        if self.debug:
            logger.info(f"response: {response}")
        code_blocks = extract_blocks_from_response(response)
        logger.info(f"{code_blocks=}")
        if self.debug:
            logger.info(f"code_blocks: \n{code_blocks}\n")
        filter_function_ids = [
            function.function_declaration.get_qualified_id()
            for function in functions_missing_docstrings
        ]
        x = replace_functions_from_code_blocks(
            code_blocks=code_blocks,
            document=document,
            language=language,
            filter_function_ids=filter_function_ids,
            replace=Replace.DOC,
        )
        logger.info(x)
        return x

    async def code_edits_for_missing_files(
        self,
        document: IR.Code,
        language: IR.Language,
        functions_missing_docstrings: List[FunctionMissingDocstring],
    ) -> Tuple[List[IR.CodeEdit], List[IR.ValueDeclaration]]:
        prompt = MissingDocStringPrompt.create_prompt_for_file(
            language=language,
            functions_missing_docstrings=functions_missing_docstrings,
        )
        response_stream = TextStream()
        collected_messages: List[str] = []

        async def feed_task():
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            completion: List[Dict[str, Any]] = openai.ChatCompletion.create(  # type: ignore
                model=Config.model,
                messages=prompt,
                temperature=Config.temperature,
                stream=True,
            )
            for chunk in completion:
                await asyncio.sleep(0.0001)
                chunk_message_dict = chunk["choices"][0]  # type: ignore
                chunk_message: str = chunk_message_dict["delta"].get(
                    "content"
                )  # extract the message
                if chunk_message_dict["finish_reason"] is None and chunk_message:
                    collected_messages.append(chunk_message)  # save the message
                    response_stream.feed_data(chunk_message)
            response_stream.feed_eof()

        response_stream._feed_task = asyncio.create_task(  # type: ignore
            self.add_task(  # type: ignore
                f"Write doc strings for {'/'.join(function.function_declaration.name for function in functions_missing_docstrings)}",
                feed_task,
            ).run()
        )

        await self.send_chat_update(response_stream)
        response = "".join(collected_messages)
        return self.process_response(
            document=document,
            language=language,
            functions_missing_docstrings=functions_missing_docstrings,
            response=response,
        )

    def split_missing_docstrings_in_groups(
        self, functions_missing_docstrings: List[FunctionMissingDocstring]
    ) -> List[List[FunctionMissingDocstring]]:
        """Split the missing doc strings in groups of at most Config.max_size_group_missing_docstrings, and that don't contain functions with the same name."""
        groups: List[List[FunctionMissingDocstring]] = []
        group: List[FunctionMissingDocstring] = []
        for function in functions_missing_docstrings:
            group.append(function)
            split = len(group) == Config.max_size_group_missing_docstrings
            # also split if a function with the same name is in the current group (e.g. from another class)
            for function2 in group:
                if function.function_declaration.name == function2.function_declaration.name:
                    split = True
                    break
            if split:
                groups.append(group)
                group = []
        if len(group) > 0:
            groups.append(group)
        return groups

    async def process_file(self, file_process: FileProcess, project: IR.Project) -> None:
        file_missing_docstrings = file_process.file_missing_docstrings
        language = file_missing_docstrings.language
        document = file_missing_docstrings.ir_code
        groups_of_functions_missing_docstrings = self.split_missing_docstrings_in_groups(
            file_missing_docstrings.functions_missing_docstrings
        )

        for group in groups_of_functions_missing_docstrings:
            code_edits, updated_functions = await self.code_edits_for_missing_files(
                document=document,
                language=language,
                functions_missing_docstrings=group,
            )
            file_process.edits.extend(code_edits)
            file_process.updated_functions.extend(updated_functions)
        edit_import = update_typing_imports(
            code=document,
            language=language,
            updated_functions=file_process.updated_functions,
        )
        if edit_import is not None:
            file_process.edits.append(edit_import)

        old_num_missing = len(file_missing_docstrings.functions_missing_docstrings)
        logger.info(f"ABOUT TO APPLY EDITS: {file_process.edits}")
        new_document = document.apply_edits(file_process.edits)
        logger.info(f"{new_document=}")
        dummy_file = IR.File("dummy")
        parser.parse_code_block(dummy_file, new_document, language)
        new_num_missing = len(functions_missing_docstrings_in_file(dummy_file))
        await self.send_chat_update(
            f"Received docs for `{file_missing_docstrings.ir_name.path}` ({new_num_missing}/{old_num_missing} missing)"
        )
        if self.debug:
            logger.info(f"new_document:\n{new_document}\n")
        path = os.path.join(project.root_path, file_missing_docstrings.ir_name.path)
        file_change = file_diff.get_file_change(path=path, new_content=str(new_document))
        if self.debug:
            logger.info(f"file_change:\n{file_change}\n")
        file_process.file_change = file_change
        file_process.new_num_missing = new_num_missing

    async def apply_file_changes(
        self, file_changes: List[file_diff.FileChange]
    ) -> lsp.ApplyWorkspaceEditResponse:
        """
        Apply file changes to the workspace.
        :param updates: The updates to be applied.
        :return: The response from applying the workspace edit.
        """
        return await self.get_server().apply_workspace_edit(
            lsp.ApplyWorkspaceEditParams(
                file_diff.edits_from_file_changes(
                    file_changes,
                    user_confirmation=True,
                )
            )
        )

    def get_state(self) -> State:
        if not isinstance(self.state, State):
            raise Exception("Agent not initialized")
        return self.state

    def get_server(self) -> LspServer:
        if self.server is None:
            raise Exception("Server not initialized")
        return self.server

    async def run(self) -> Result:
        async def info_update(msg: str):
            logger.info(msg)
            await self.send_chat_update(msg)

        async def log_missing(file_missing_docstrings: FileMissingDocstrings):
            await info_update(f"File: {file_missing_docstrings.ir_name.path}")
            for function in file_missing_docstrings.functions_missing_docstrings:
                logger.info(f"Missing: {function.function_declaration.name}")

        async def get_user_response() -> str:
            result = await self.request_chat(
                agent.RequestChatRequest(messages=self.get_state().messages)
            )
            return result

        await self.send_progress()
        text_document = self.get_state().params.textDocument
        if text_document is not None:
            parsed = urlparse(text_document.uri)
            current_file_uri = url2pathname(
                unquote(parsed.path)
            )  # Work around bug: https://github.com/scikit-hep/uproot5/issues/325#issue-850683423
        else:
            raise Exception("Missing textDocument")

        await self.send_chat_update(
            "Reply with 'c' to start generating missing doc strings to the current file, or specify files and directories by typing @ and following autocomplete."
        )

        get_user_response_task = AgentTask("Get user response", get_user_response)
        self.set_tasks([get_user_response_task])
        user_response_coro = cast(
            Coroutine[None, None, Optional[str]], get_user_response_task.run()
        )
        user_response_task = asyncio.create_task(user_response_coro)
        await self.send_progress()
        user_response = await user_response_task
        if user_response is None:
            user_uris = []
        else:
            self.get_state().messages.append(openai_types.Message.user(user_response))
            user_uris = re.findall(r"\[uri\]\((\S+)\)", user_response)
        if user_uris == []:
            user_uris = [current_file_uri]
        user_references = [IR.Reference.from_uri(uri) for uri in user_uris]
        symbols_per_file: Dict[str, Set[IR.QualifiedId]] = {}
        for ref in user_references:
            if ref.qualified_id:
                if ref.file_path not in symbols_per_file:
                    symbols_per_file[ref.file_path] = set()
                symbols_per_file[ref.file_path].add(ref.qualified_id)
        user_paths = [ref.file_path for ref in user_references]
        project = parser.parse_files_in_paths(paths=user_paths)
        if self.debug:
            logger.info(f"\n=== Project Map ===\n{project.dump_map()}\n")

        files_missing_docstrings_ = files_missing_docstrings_in_project(project)
        files_missing_docstrings: List[FileMissingDocstrings] = []
        for file_missing_docstrings in files_missing_docstrings_:
            full_path = os.path.join(project.root_path, file_missing_docstrings.ir_name.path)
            if full_path not in symbols_per_file:  # no symbols in this file
                files_missing_docstrings.append(file_missing_docstrings)
            else:  # filter missing doc strings to only include symbols in symbols_per_file
                functions_missing_docstrings = [
                    function_missing_docstrings
                    for function_missing_docstrings in file_missing_docstrings.functions_missing_docstrings
                    if function_missing_docstrings.function_declaration.get_qualified_id()
                    in symbols_per_file[full_path]
                ]
                if functions_missing_docstrings != []:
                    file_missing_docstrings.functions_missing_docstrings = (
                        functions_missing_docstrings
                    )
                    files_missing_docstrings.append(file_missing_docstrings)

        file_processes: List[FileProcess] = []
        total_num_missing = 0
        await info_update("\n=== Missing Docs ===\n")
        files_missing_str = ""
        for file_missing_docstrings in files_missing_docstrings:
            await log_missing(file_missing_docstrings)
            files_missing_str += f"{file_missing_docstrings.ir_name.path}\n"
            total_num_missing += len(file_missing_docstrings.functions_missing_docstrings)
            file_processes.append(FileProcess(file_missing_docstrings=file_missing_docstrings))
        if total_num_missing == 0:
            await self.send_chat_update("No missing doc strings in the current file.")
            return Result()
        await self.send_chat_update(
            f"Missing {total_num_missing} doc strings in {files_missing_str}"
        )

        tasks: List[asyncio.Task[Any]] = [
            asyncio.create_task(self.process_file(file_process=file_processes[i], project=project))
            for i in range(len(files_missing_docstrings))
        ]
        await asyncio.gather(*tasks)

        file_changes: List[file_diff.FileChange] = []
        total_new_num_missing = 0
        for file_process in file_processes:
            if file_process.file_change is not None:
                file_changes.append(file_process.file_change)
            if file_process.new_num_missing is not None:
                total_new_num_missing += file_process.new_num_missing
            else:
                total_new_num_missing += len(
                    file_process.file_missing_docstrings.functions_missing_docstrings
                )
        await self.apply_file_changes(file_changes)
        await self.send_chat_update(
            f"Missing doc strings after responses: {total_new_num_missing}/{total_num_missing} ({total_new_num_missing/total_num_missing*100:.2f}%)"
        )
        return Result()
