import asyncio
import logging
from asyncio import Lock
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional

import rift.agents.registry as registry
import rift.llm.openai_types as openai
import rift.lsp.types as lsp
from rift.agents.abstract import AgentProgress  # AgentTask,
from rift.agents.abstract import Agent, AgentParams, AgentRunResult, AgentState, RequestChatRequest
from rift.agents.agenttask import AgentTask
from rift.llm.abstract import AbstractChatCompletionProvider
from rift.lsp import LspServer
from rift.util.context import resolve_inline_uris

logger = logging.getLogger(__name__)


@dataclass
class ChatRunResult(AgentRunResult):
    ...


@dataclass
class ChatProgress(
    AgentProgress
):  # reports what tasks are active and responsible for reporting new tasks
    response: Optional[str] = None
    done_streaming: bool = False


@dataclass
class RiftChatAgentState(AgentState):
    model: AbstractChatCompletionProvider
    document: lsp.TextDocumentItem


@registry.agent(
    agent_description="Ask questions about your code.",
    display_name="Chat",
)
@dataclass
class RiftChatAgent(Agent):
    state: Optional[RiftChatAgentState] = None
    agent_type: ClassVar[str] = "rift_chat"

    @classmethod
    async def create(cls, params: AgentParams, server: LspServer) -> "RiftChatAgent":
        # logger.info(f"RiftChatAgent.create {params=}")
        model = await server.ensure_chat_model()
        if params.textDocument is None:
            document = None
        else:
            document = server.documents[params.textDocument.uri]
        state = RiftChatAgentState(
            model=model,
            messages=[openai.Message.assistant("Hello! How can I help you today?")],
            document=document,
            params=params,
        )
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )
        return obj

    async def run(self) -> AgentRunResult:
        response_lock = Lock()

        async def get_user_input() -> str:
            # logger.info(f"getting user input for {self.state.messages=}")
            chat_result = await self.request_chat(RequestChatRequest(messages=self.state.messages))
            # logger.info(f"got input {chat_result=}")
            return chat_result

        async def generate_assistant_response(user_input: str):
            # logger.info(f"generating assistant response for {user_input=}")
            assistant_response = ""
            documents: List[lsp.Document] = resolve_inline_uris(user_input, self.server)
            logger.info(f"resolved document uris {documents=}")

            doc_text = self.state.document.text if self.state.document is not None else ""

            logger.info("running chat")
            with lsp.setdoc(self.state.document):
                cursor_offset_start = (
                    self.state.document.position_to_offset(self.state.params.selection.first)
                    if self.state.params.selection is not None
                    else None
                )
                cursor_offset_end = (
                    self.state.document.position_to_offset(self.state.params.selection.second)
                    if self.state.params.selection is not None
                    else None
                )
                stream = await self.state.model.run_chat(
                    doc_text,
                    self.state.messages,
                    user_response,
                    cursor_offset_start,
                    cursor_offset_end,
                    documents=documents,
                )
            async for delta in stream.text:
                assistant_response += delta
                # logger.info(f"{delta=}")
                async with response_lock:
                    await self.send_progress(ChatProgress(response=assistant_response))
            if stream.event:
                if stream.event.is_set():
                    raise Exception(f"[{self.agent_type}] generation failed")
            await self.send_progress(ChatProgress(response=assistant_response, done_streaming=True))
            logger.info(f"{self} finished streaming response.")
            return assistant_response

        async def generate_response_dummy():
            return True

        get_user_input_task = AgentTask("Get user input", get_user_input)
        old_generate_response_task = AgentTask(
            "Generate assistant response", generate_response_dummy
        )
        self.set_tasks([get_user_input_task, old_generate_response_task])
        await old_generate_response_task.run()
        while True:
            get_user_input_task = AgentTask("Get user input", get_user_input)

            sentinel_f = asyncio.get_running_loop().create_future()

            async def generate_response_task_args():
                return [await sentinel_f]

            generate_response_task = AgentTask(
                "Generate assistant response",
                generate_assistant_response,
                args=generate_response_task_args,
            )

            self.set_tasks([get_user_input_task, old_generate_response_task])

            await self.send_progress()
            user_response_task = asyncio.create_task(get_user_input_task.run())
            user_response_task.add_done_callback(lambda f: sentinel_f.set_result(f.result()))

            user_response = await user_response_task

            async with response_lock:
                self.state.messages.append(openai.Message.user(content=user_response))
            self.set_tasks([get_user_input_task, generate_response_task])
            await self.send_progress()
            assistant_response = await generate_response_task.run()
            await self.send_progress()
            async with response_lock:
                self.state.messages.append(openai.Message.assistant(content=assistant_response))

            old_generate_response_task = generate_response_task
            await self.send_progress()
