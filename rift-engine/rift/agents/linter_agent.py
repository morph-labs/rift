import asyncio
import logging
import os
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, ClassVar, Coroutine, Dict, List, Optional, cast
from urllib.parse import urlparse

import nest_asyncio  # type: ignore
import openai

import rift.agents.abstract as agent
from rift.ir.metalanguage import MetaLanguage
import rift.agents.registry as registry
import rift.ir.IR as IR
import rift.ir.parser as parser
import rift.llm.openai_types as openai_types
import rift.lsp.types as lsp
import rift.util.file_diff as file_diff
from rift.agents.agenttask import AgentTask
from rift.ir.response import extract_blocks_from_response
from rift.lsp import LspServer
from rift.util.TextStream import TextStream

nest_asyncio.apply()  # type: ignore


@dataclass
class Result(agent.AgentRunResult):
    ...


@dataclass
class State(agent.AgentState):
    pass


class Config:
    model = "gpt-3.5-turbo-0613"  # ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k"]
    temperature = 0


logger = logging.getLogger(__name__)

Message = Dict[str, str]
Prompt = List[Message]


class GenerateCodePrompt:
    @staticmethod
    def create(user_prompt: str) -> Prompt:
        system_msg = dedent(
            """
            You are an agent that generates a *code block* to implement a linter function given a description from the user.
            
            You MUST format your response as a *code block*:
                ```python
                for x in __set__: ...
                ```
            
            The function must be written in the following subset of Python:
            - `for each __set__: __body__`
            - sets `__set__` can be one of ["$Class", "$File", "$Function", "$Method", "$Module", "$Namespace", "$TypeDefinition"]
            - __body__ performs a check on using the following functions:
                - `$check(x, __expression__)` where `x` is an element of a __set__
            - __expression__ can be one of the following categories:
                - `x.name`
                - boolean, and equality operations
                - string operations
                - `d.name` and `d.type` when `d` is a type definition
                - `t.kind in ["record", "array", ...]` when `t` is a type
                - `t.name in ["option", "list", "int", ...] when `t` is a type
                - `t.fields` when `d` is a record type definition
                - `f.name` and `f.type` and `f.optional` when `f` is a field

            Example:
            Given this user prompt:
            ```
            Check that all function names begin with letter 'f'
            ```
            You will generate the following *code block*:
            ```python
            for x in $Function: $check(x, x.name[0] == 'f')
            ```

            Example:
            Given this user prompt:
            ```
            Check that all type definitions that are records have a field named 'name'
            ```
            You will generate the following *code block*:
            ```python
            for x in $TypeDefinition: if x.type.kind == 'record': $check(x, 'name' in x.type.fields)
            ```
            """
        ).lstrip()
        return [
            dict(role="system", content=system_msg),
            dict(role="user", content=user_prompt),
        ]


@registry.agent(
    agent_description="Natural language linter",
    display_name="Linter",
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
class LinterAgent(agent.ThirdPartyAgent):
    agent_type: ClassVar[str] = "linter"

    @classmethod
    async def create(cls, params: agent.AgentParams, server: LspServer) -> agent.Agent:
        state = State(
            params=params,
            messages=[],
        )
        obj: agent.ThirdPartyAgent = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )
        return obj

    def get_project_root(self) -> str:
        text_document = self.get_state().params.textDocument
        if text_document is not None:
            path = urlparse(text_document.uri).path
            path = os.path.dirname(path)
            # walk up the path until ".git" is found
            while True:
                if os.path.exists(os.path.join(path, ".git")):
                    return path
                parent = os.path.dirname(path)
                if parent == path:
                    raise Exception("Could not find .git")
                path = parent
        else:
            raise Exception("Missing textDocument")

    def parse_current_repo(self) -> IR.Project:
        project_root = self.get_project_root()
        return parser.parse_files_in_paths([project_root])

    async def generate_code(self, user_prompt: str) -> Optional[IR.Code]:
        prompt = GenerateCodePrompt.create(user_prompt)
        response_stream = TextStream()
        collected_messages: List[str] = []

        async def feed_task():
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            completion: List[Dict[str, Any]] = openai.ChatCompletion.create(  # type: ignore
                model=Config.model, messages=prompt, temperature=Config.temperature, stream=True
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
                f"Generate code",
                feed_task,
            ).run()
        )

        await self.send_chat_update(response_stream)
        response = "".join(collected_messages)
        code_blocks = extract_blocks_from_response(response)
        if len(code_blocks) == 1:
            return code_blocks[0]

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

    async def create_get_user_response_task(self) -> Optional[str]:
        async def get_user_response() -> str:
            result = await self.request_chat(
                agent.RequestChatRequest(messages=self.get_state().messages)
            )
            return result

        get_user_response_task = AgentTask("Get user response", get_user_response)
        self.set_tasks([get_user_response_task])
        user_response_coro = cast(
            Coroutine[None, None, Optional[str]], get_user_response_task.run()
        )
        user_response_task = asyncio.create_task(user_response_coro)
        await self.send_progress()
        user_response = await user_response_task
        if user_response is not None:
            self.get_state().messages.append(openai_types.Message.user(user_response))
        return user_response

    async def run(self) -> Result:
        await self.send_progress()
        await self.send_chat_update("What yould you like to check in your project?")
        user_response = await self.create_get_user_response_task()
        if user_response is None:
            return Result()
        if user_response.strip() == "":  # TODO: this is for testing
            user_response = "Check that all class names have at least 5 letters"
            await self.send_chat_update(f"Using default prompt: `{user_response}`")
        code = await self.generate_code(user_response)
        if code is not None:
            await self.send_chat_update(f"Here is the code I generated for you:\n{code}\n")

            def report_check_failed(msg: str) -> None:
                # logging.info(msg)
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.send_chat_update(msg))

            ml = MetaLanguage(
                project=self.parse_current_repo(),
                raw_code=str(code),
                report_check_failed=report_check_failed,
            )
            ml.eval()
        else:
            await self.send_chat_update(f"I couldn't generate any code for you. Please try again.")
        return Result()
