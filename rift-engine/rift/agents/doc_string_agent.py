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
from rift.ir.missing_doc_strings import (
    FunctionMissingDocString,
    FileMissingDocStrings,
    functions_missing_doc_strings_in_file,
    files_missing_doc_strings_in_project,
)
from rift.ir.response import (
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
    max_size_group_missing_doc_strings = 10  # Max number of functions to process at once


class MissingDocStringPrompt:
    @staticmethod
    def mk_user_msg(
        functions_missing_doc_strings: List[FunctionMissingDocString], code: IR.Code
    ) -> str:
        missing_str = ""
        n = 0
        for function in functions_missing_doc_strings:
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
    def code_for_missing_doc_string_functions(
        functions_missing_doc_strings: List[FunctionMissingDocString],
    ) -> IR.Code:
        bytes = b""
        for function in functions_missing_doc_strings:
            bytes += function.function_declaration.get_substring()
            bytes += b"\n"
        return IR.Code(bytes)

    @staticmethod
    def create_prompt_for_file(
        language: IR.Language,
        functions_missing_doc_strings: List[FunctionMissingDocString],
    ) -> Prompt:
        example_py = '''
            ```python
                def foo(a: t1, b : t2) -> t3
                    """
                    ...
                    """
            ```
        '''
        example_js = """
            ```javascript
                /**
                    ...
                */
                function foo(a: t1, b : t2) : t3 {
            ```
        """
        example_ts = """
            ```typescript
                /**
                    ...
                */
                function foo(a: t1, b : t2): t3 {
            ```
        """
        example_ocaml = """
            ```ocaml
                (** ...
                *)
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

        code = MissingDocStringPrompt.code_for_missing_doc_string_functions(
            functions_missing_doc_strings
        )
        return [
            dict(role="system", content=system_msg),
            dict(
                role="user",
                content=MissingDocStringPrompt.mk_user_msg(
                    functions_missing_doc_strings=functions_missing_doc_strings,
                    code=code,
                ),
            ),
        ]


@dataclass
class FileProcess:
    file_missing_doc_strings: FileMissingDocStrings
    edits: List[IR.CodeEdit] = field(default_factory=list)
    updated_functions: List[IR.ValueDeclaration] = field(default_factory=list)
    file_change: Optional[file_diff.FileChange] = None
    new_num_missing: Optional[int] = None


@registry.agent(
    agent_description="Generate missing doc strings for functions",
    display_name="Generate Doc Strings",
    agent_icon="""\
<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
<g clip-path="url(#clip0_636_8979)">
<path d="M11.4446 5.05713H9.07153V12.8574C9.07153 13.3066 8.97144 13.6411 8.77124 13.8608C8.57104 14.0757 8.31226 14.1831 7.99487 14.1831C7.67261 14.1831 7.40894 14.0732 7.20386 13.8535C7.00366 13.6338 6.90356 13.3018 6.90356 12.8574V5.05713H4.53052C4.15942 5.05713 3.88354 4.97656 3.70288 4.81543C3.52222 4.64941 3.43188 4.43213 3.43188 4.16357C3.43188 3.88525 3.52466 3.66553 3.71021 3.50439C3.90063 3.34326 4.17407 3.2627 4.53052 3.2627H11.4446C11.8206 3.2627 12.0989 3.3457 12.2795 3.51172C12.4651 3.67773 12.5579 3.89502 12.5579 4.16357C12.5579 4.43213 12.4651 4.64941 12.2795 4.81543C12.094 4.97656 11.8157 5.05713 11.4446 5.05713Z" fill="#CCCCCC"/>
<rect x="13.8284" y="8.2998" width="1" height="4" rx="0.5" transform="rotate(45 13.8284 8.2998)" fill="#D9D9D9"/>
<rect x="11" y="6.8999" width="1" height="4" rx="0.5" transform="rotate(-45 11 6.8999)" fill="#D9D9D9"/>
<rect width="1" height="4" rx="0.5" transform="matrix(-0.707107 0.707107 0.707107 0.707107 2.30737 8.40674)" fill="#D9D9D9"/>
<rect width="1" height="4" rx="0.5" transform="matrix(-0.707107 -0.707107 -0.707107 0.707107 5.13574 7.00684)" fill="#D9D9D9"/>
</g>
<defs>
<clipPath id="clip0_636_8979">
<rect width="16" height="16" fill="white"/>
</clipPath>
</defs>
</svg>""",
)
@dataclass
class MissingDocStringAgent(agent.ThirdPartyAgent):
    agent_type: ClassVar[str] = "missing_doc_string_agent"
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
        functions_missing_doc_strings: List[FunctionMissingDocString],
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
            for function in functions_missing_doc_strings
        ]
        x = replace_functions_from_code_blocks(
            code_blocks=code_blocks,
            document=document,
            language=language,
            filter_function_ids=filter_function_ids,
            replace_body=True,
        )
        logger.info(x)
        return x

    async def code_edits_for_missing_files(
        self,
        document: IR.Code,
        language: IR.Language,
        functions_missing_doc_strings: List[FunctionMissingDocString],
    ) -> Tuple[List[IR.CodeEdit], List[IR.ValueDeclaration]]:
        prompt = MissingDocStringPrompt.create_prompt_for_file(
            language=language,
            functions_missing_doc_strings=functions_missing_doc_strings,
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
                f"Write doc strings for {'/'.join(function.function_declaration.name for function in functions_missing_doc_strings)}",
                feed_task,
            ).run()
        )

        await self.send_chat_update(response_stream)
        response = "".join(collected_messages)
        return self.process_response(
            document=document,
            language=language,
            functions_missing_doc_strings=functions_missing_doc_strings,
            response=response,
        )

    def split_missing_doc_strings_in_groups(
        self, functions_missing_doc_strings: List[FunctionMissingDocString]
    ) -> List[List[FunctionMissingDocString]]:
        """Split the missing doc strings in groups of at most Config.max_size_group_missing_doc_strings, and that don't contain functions with the same name."""
        groups: List[List[FunctionMissingDocString]] = []
        group: List[FunctionMissingDocString] = []
        for function in functions_missing_doc_strings:
            group.append(function)
            split = len(group) == Config.max_size_group_missing_doc_strings
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
        file_missing_doc_strings = file_process.file_missing_doc_strings
        language = file_missing_doc_strings.language
        document = file_missing_doc_strings.ir_code
        groups_of_functions_missing_doc_strings = self.split_missing_doc_strings_in_groups(
            file_missing_doc_strings.functions_missing_doc_strings
        )

        for group in groups_of_functions_missing_doc_strings:
            code_edits, updated_functions = await self.code_edits_for_missing_files(
                document=document,
                language=language,
                functions_missing_doc_strings=group,
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

        old_num_missing = len(file_missing_doc_strings.functions_missing_doc_strings)
        logger.info(f"ABOUT TO APPLY EDITS: {file_process.edits}")
        new_document = document.apply_edits(file_process.edits)
        logger.info(f"{new_document=}")
        dummy_file = IR.File("dummy")
        parser.parse_code_block(dummy_file, new_document, language)
        new_num_missing = len(functions_missing_doc_strings_in_file(dummy_file))
        await self.send_chat_update(
            f"Received docs for `{file_missing_doc_strings.ir_name.path}` ({new_num_missing}/{old_num_missing} missing)"
        )
        if self.debug:
            logger.info(f"new_document:\n{new_document}\n")
        path = os.path.join(project.root_path, file_missing_doc_strings.ir_name.path)
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

        async def log_missing(file_missing_doc_strings: FileMissingDocStrings):
            await info_update(f"File: {file_missing_doc_strings.ir_name.path}")
            for function in file_missing_doc_strings.functions_missing_doc_strings:
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

        files_missing_doc_strings_ = files_missing_doc_strings_in_project(project)
        files_missing_doc_strings: List[FileMissingDocStrings] = []
        for file_missing_doc_strings in files_missing_doc_strings_:
            full_path = os.path.join(project.root_path, file_missing_doc_strings.ir_name.path)
            if full_path not in symbols_per_file:  # no symbols in this file
                files_missing_doc_strings.append(file_missing_doc_strings)
            else:  # filter missing doc strings to only include symbols in symbols_per_file
                functions_missing_doc_strings = [
                    function_missing_doc_strings
                    for function_missing_doc_strings in file_missing_doc_strings.functions_missing_doc_strings
                    if function_missing_doc_strings.function_declaration.get_qualified_id()
                    in symbols_per_file[full_path]
                ]
                if functions_missing_doc_strings != []:
                    file_missing_doc_strings.functions_missing_doc_strings = (
                        functions_missing_doc_strings
                    )
                    files_missing_doc_strings.append(file_missing_doc_strings)

        file_processes: List[FileProcess] = []
        total_num_missing = 0
        await info_update("\n=== Missing Docs ===\n")
        files_missing_str = ""
        for file_missing_doc_strings in files_missing_doc_strings:
            await log_missing(file_missing_doc_strings)
            files_missing_str += f"{file_missing_doc_strings.ir_name.path}\n"
            total_num_missing += len(file_missing_doc_strings.functions_missing_doc_strings)
            file_processes.append(FileProcess(file_missing_doc_strings=file_missing_doc_strings))
        if total_num_missing == 0:
            await self.send_chat_update("No missing doc strings in the current file.")
            return Result()
        await self.send_chat_update(
            f"Missing {total_num_missing} doc strings in {files_missing_str}"
        )

        tasks: List[asyncio.Task[Any]] = [
            asyncio.create_task(self.process_file(file_process=file_processes[i], project=project))
            for i in range(len(files_missing_doc_strings))
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
                    file_process.file_missing_doc_strings.functions_missing_doc_strings
                )
        await self.apply_file_changes(file_changes)
        await self.send_chat_update(
            f"Missing doc strings after responses: {total_new_num_missing}/{total_num_missing} ({total_new_num_missing/total_num_missing*100:.2f}%)"
        )
        return Result()
