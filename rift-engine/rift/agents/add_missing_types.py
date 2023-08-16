import asyncio
import functools
import logging
import os
import re
from dataclasses import dataclass, field
from textwrap import dedent
from typing import ClassVar, Dict, List, Optional, cast
from urllib.parse import urlparse

import openai

import rift.agents.abstract as agent
import rift.agents.registry as registry
import rift.ir.IR as IR
import rift.ir.parser as parser
import rift.llm.openai_types as openai_types
import rift.lsp.types as lsp
import rift.util.file_diff as file_diff
from rift.agents.agenttask import AgentTask
from rift.ir.missing_types import (
    FileMissingTypes,
    MissingType,
    files_missing_types_in_project,
    functions_missing_types_in_file,
)
from rift.ir.response import extract_blocks_from_response, replace_functions_from_code_blocks
from rift.llm.create import ModelConfig
from rift.lsp import LspServer
from rift.util.TextStream import TextStream


@dataclass
class MissingTypesParams(agent.AgentParams):
    ...


@dataclass
class MissingTypesResult(agent.AgentRunResult):
    ...


@dataclass
class MissingTypesAgentState(agent.AgentState):
    params: MissingTypesParams
    messages: list[openai_types.Message]
    response_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Config:
    debug = False
    max_size_group_missing_types = 10  # maximum size for a group of missing types
    model = "gpt-3.5-turbo-0613"  # ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k"]
    temperature = 0


logger = logging.getLogger(__name__)

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
        return dedent(
            f"""
        Add missing types for the following functions:
        {missing_str}

        The code is:
        ```
        {code}
        ```
        """
        ).lstrip()

    @staticmethod
    def code_for_missing_types(missing_types: List[MissingType]) -> IR.Code:
        bytes = b""
        for mt in missing_types:
            bytes += mt.function_declaration.get_substring()
            bytes += b"\n"
        return IR.Code(bytes)

    @staticmethod
    def example_code_block() -> str:
        return dedent(
            """
            ```python
                def mul(a: t1, b : t2) -> t3
                    ...
            ```
        """
        ).lstrip()

    @staticmethod
    def create_prompt_for_file(language: IR.Language, missing_types: List[MissingType]) -> Prompt:
        code = MissingTypePrompt.code_for_missing_types(missing_types)
        example_py = """
            ```python
                def foo(a: t1, b : t2) -> t3
                    ...
            ```
        """
        example_ts = """
            ```typescript
                function foo(a: t1, b : t2): t3 {
                    ...
                }
            ```
        """
        if language in ["javascript", "typescript", "tsx"]:
            example = example_ts
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
        return [
            dict(role="system", content=system_msg),
            dict(
                role="user",
                content=MissingTypePrompt.mk_user_msg(
                    missing_types=missing_types, code=code),
            ),
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


@registry.agent(
    agent_description="Infer missing type signatures",
    display_name="Type Inference",
    agent_icon="""\
<svg width="34px" height="24px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- T remains unchanged -->
    <path transform="scale(0.8) translate(-3,3)" d="M5 7V6C5 4.89543 5.89543 4 7 4H12M19 7V6C19 4.89543 18.1046 4 17 4H12M12 4V20M12 20H9M12 20H15" stroke="#292929" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/>

    <!-- ? moved further to the right -->
    <path transform="translate(0,0)" d="M15.1777 8.79421C15.1777 5.06857 21.0323 5.0686 21.0323 8.79421C21.0323 11.4554 18.3711 10.9231 18.3711 14.1164" stroke="#292929" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path transform="translate(0,0)" d="M18.3711 18.385L18.3817 18.3732" stroke="#292929" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>""",
)
@dataclass
class MissingTypesAgent(agent.ThirdPartyAgent):
    agent_type: ClassVar[str] = "missing_types"
    params_cls: ClassVar[type[MissingTypesParams]] = MissingTypesParams

    debug = Config.debug

    @classmethod
    async def create(cls, params: MissingTypesParams, server):
        state = MissingTypesAgentState(
            params=params,
            messages=[],
        )
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )
        return obj

    def process_response(
        self,
        document: IR.Code,
        language: IR.Language,
        missing_types: List[MissingType],
        response: str,
    ) -> List[IR.CodeEdit]:
        if self.debug:
            logger.info(f"response:\n{response}\n")
        code_blocks = extract_blocks_from_response(response)
        if self.debug:
            logger.info(f"code_blocks:\n{code_blocks}\n")
        filter_function_ids = [
            mt.function_declaration.get_qualified_id() for mt in missing_types]
        edits = replace_functions_from_code_blocks(
            code_blocks=code_blocks,
            document=document,
            filter_function_ids=filter_function_ids,
            language=language,
            replace_body=False,
        )
        return edits

    async def code_edits_for_missing_files(
        self, document: IR.Code, language: IR.Language, missing_types: List[MissingType]
    ) -> List[IR.CodeEdit]:
        loop = asyncio.get_event_loop()
        prompt = MissingTypePrompt.create_prompt_for_file(
            language=language, missing_types=missing_types
        )
        # Partially apply parameters to ChatCompletion.create for later execution
        func = functools.partial(
            openai.ChatCompletion.create,
            model=Config.model,
            messages=prompt,
            temperature=Config.temperature,
            stream=True,
        )
        # Run OpenAI API call concurrently to avoid blocking event loop
        # completion =  await loop.run_in_executor(None, func)

        # completion = func()

        futs = []

        # def stream_handler(chunk):
        #     futs.append(asyncio.run_coroutine_threadsafe(self.send_chat_update(chunk), loop))
        response_stream = TextStream()
        collected_messages = []

        async def feed_task():
            completion = openai.ChatCompletion.create(
                model=Config.model, messages=prompt, temperature=Config.temperature, stream=True
            )
            for chunk in completion:
                await asyncio.sleep(0.0001)
                chunk_message_dict = chunk["choices"][0]  # type: ignore
                chunk_message = chunk_message_dict["delta"].get(
                    "content")  # extract the message
                if chunk_message_dict["finish_reason"] is None and chunk_message:
                    collected_messages.append(
                        chunk_message)  # save the message
                    response_stream.feed_data(chunk_message)
            response_stream.feed_eof()

        response_stream._feed_task = asyncio.create_task(
            self.add_task(
                f"Generate type annotations for {'/'.join(mt.function_declaration.name for mt in missing_types)}",
                feed_task,
            ).run()
        )

        await self.send_chat_update(response_stream)
        response = "".join(collected_messages)
        edits = self.process_response(
            document=document, language=language, missing_types=missing_types, response=response
        )
        return edits

    def split_missing_types_in_groups(
        self, missing_types: List[MissingType]
    ) -> List[List[MissingType]]:
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

    async def process_file(self, file_process: FileProcess, project: parser.Project) -> None:
        fmt = file_process.file_missing_types
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
        await self.send_chat_update(
            f"Received types for `{fmt.file.path}` ({new_num_missing}/{old_num_missing} missing)"
        )
        if self.debug:
            logger.info(f"new_document:\n{new_document}\n")
        path = os.path.join(project.root_path, fmt.file.path)
        file_change = file_diff.get_file_change(
            path=path, new_content=str(new_document))
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

    def get_state(self) -> MissingTypesAgentState:
        if not isinstance(self.state, MissingTypesAgentState):
            raise Exception("Agent not initialized")
        return self.state

    def get_server(self) -> LspServer:
        if self.server is None:
            raise Exception("Server not initialized")
        return self.server

    async def run(self) -> MissingTypesResult:
        async def info_update(msg):
            logger.info(msg)
            await self.send_chat_update(msg)

        async def log_missing(fmt: FileMissingTypes) -> None:
            await info_update(f"File: {fmt.file.path}")
            for mt in fmt.missing_types:
                await info_update(f"  {mt}")
            await info_update("")

        async def get_user_response() -> str:
            result = await self.request_chat(
                agent.RequestChatRequest(messages=self.get_state().messages)
            )
            return result

        config = cast(ModelConfig,
                      self.get_server().model_config  # type: ignore
                      )

        logger.info(f"config: {config}")

        if config.openaiKey is not None:
            openai.api_key = config.openaiKey.get_secret_value()
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")

        if openai.api_key is None:
            await self.send_chat_update(
                "OpenAI key missing: set the Openai Key in the Rift settings or as the `OPENAI_API_KEY` environment variable and run the agent again."
            )
            return MissingTypesResult()

        await self.send_progress()
        text_document = self.get_state().params.textDocument
        if text_document is not None:
            current_file_uri = text_document.uri
        else:
            raise Exception("Missing textDocument")

        await self.send_chat_update(
            "Reply with 'c' to start adding missing types to the current file, or specify files and directories by typing @ and following autocomplete."
        )

        get_user_response_task = AgentTask(
            "Get user response", get_user_response)
        self.set_tasks([get_user_response_task])
        user_response_task = asyncio.create_task(get_user_response_task.run())
        await self.send_progress()
        user_response = await user_response_task
        if user_response is None:
            user_paths = []
        else:
            user_paths = re.findall(r"\[uri\]\((\S+)\)", user_response)
        if user_paths == []:
            user_paths = [urlparse(current_file_uri).path]

        file_processes: List[FileProcess] = []
        tot_num_missing = 0
        project = parser.parse_files_in_paths(paths=user_paths)
        if self.debug:
            logger.info(f"\n=== Project Map ===\n{project.dump_map()}\n")
        files_missing_types = files_missing_types_in_project(project)
        await info_update("\n=== Missing Types ===\n")
        files_missing_str = ""
        for fmt in files_missing_types:
            files_missing_str += f"`{fmt.file.path}` "
            await log_missing(fmt)
            tot_num_missing += count_missing(fmt.missing_types)
            file_processes.append(FileProcess(file_missing_types=fmt))
        if tot_num_missing == 0:
            await self.send_chat_update("No missing types found in the current file.")
            return MissingTypesResult()
        await self.send_chat_update(f"Missing {tot_num_missing} types in {files_missing_str}")

        tasks: List[asyncio.Task] = [
            asyncio.create_task(self.process_file(
                file_process=file_processes[i], project=project))
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
        await self.apply_file_changes(file_changes)
        await self.send_chat_update(
            f"Missing types after responses: {tot_new_missing}/{tot_num_missing} ({tot_new_missing/tot_num_missing*100:.2f}%)"
        )
        return MissingTypesResult()
