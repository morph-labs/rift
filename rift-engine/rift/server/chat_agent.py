import asyncio
from dataclasses import dataclass, field
import logging
from typing import ClassVar, Optional, List, Any
from typing import Literal
from rift.lsp import LspServer as BaseLspServer, rpc_method
from rift.rpc import RpcServerStatus
import rift.lsp.types as lsp
from rift.llm.abstract import (
    AbstractCodeCompletionProvider,
    AbstractChatCompletionProvider,
)
from rift.llm.create import ModelConfig
from rift.server.agent import *
from rift.server.selection import RangeSet
from rift.llm.openai_types import Message
from rift.util.ofdict import ofdict

logger = logging.getLogger(__name__)

@dataclass
class RunChatParams:
    message: str
    messages: List[Message]
    position: Optional[lsp.Position]
    textDocument: Optional[lsp.TextDocumentIdentifier]

ChatAgentLogs = AgentLogs        

class ChatAgent:
    count: ClassVar[int] = 0
    id: int
    cfg: RunChatParams
    running: bool
    server: "LspServer"
    change_futures: dict[str, asyncio.Future[None]]
    cursor: Optional[lsp.Position]
    """ The position of the cursor (where text will be inserted next). This position is changed if other edits occur above the cursor. """
    task: Optional[asyncio.Task]
    subtasks: set[asyncio.Task]

    @property
    def uri(self) -> Optional[str]:
        try:
            return self.cfg.textDocument.uri
        except AttributeError:
            return None

    def __str__(self):
        return f"<ChatAgent {self.id}>"

    def __init__(
        self,
        cfg: RunChatParams,
        model: AbstractChatCompletionProvider,
        server: "LspServer",
    ):
        ChatAgent.count += 1
        self.model = model
        self.id = CodeCompletionAgent.count
        self.cfg = cfg
        self.server = server
        self.running = False
        self.change_futures = {}
        self.cursor = cfg.position
        try:
            self.document = server.documents[self.cfg.textDocument.uri]
        except AttributeError:
            self.document = None
        self.task = None
        self.subtasks = set()

    def cancel(self, msg):
        logger.info(f"{self} cancel run: {msg}")
        if self.task is not None:
            self.task.cancel(msg)

    async def run(self):
        self.task = asyncio.create_task(self.worker())
        self.running = True
        try:
            return await self.task
        except asyncio.CancelledError as e:
            logger.info(f"{self} run task got cancelled")
            return f"I stopped! {e}"
        finally:
            self.running = False

    async def send_progress(
        self,
        response: str = "",
        logs: Optional[ChatAgentLogs] = None,
        done: bool = False,
    ):
        await self.server.send_chat_agent_progress(
            self.id,
            response=response,
            log=logs,
            done=done,
            # textDocument=to_text_document_id(self.document),
            # cursor=self.cursor,
            # status="running" if self.running else "done",
        )

    async def worker(self):
        response = ""
        from asyncio import Lock

        response_lock = Lock()
        assert self.running
        async with response_lock:
            await self.send_progress(response)
        if self.document:
            doc_text = self.document.text
        else:
            doc_text = None
        pos = self.cursor
        offset = None if pos is None else self.document.position_to_offset(pos)

        stream = await self.model.run_chat(doc_text, self.cfg.messages, self.cfg.message, offset)

        async for delta in stream.text:
            response += delta
            async with response_lock:
                await self.send_progress(response)
        logger.info(f"{self} finished streaming response.")

        self.running = False
        async with response_lock:
            await self.send_progress(response, done=True)
