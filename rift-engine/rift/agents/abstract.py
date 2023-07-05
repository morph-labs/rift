import asyncio
from dataclasses import dataclass, field
from abc import ABC
from typing import ClassVar, Dict, Literal
from rift.lsp import LspServer as BaseLspServer, rpc_method
from rift.llm.openai_types import Message as ChatMessage
from enum import Enum


class Status(Enum):
    running = "running"
    done = "done"
    error = "error"
    accepted = "accepted"
    rejected = "rejected"


@dataclass
class RequestInputRequest:
    msg: str


@dataclass
class RequestInputResponse:
    response: str


@dataclass
class RequestChatRequest:
    messages: List[ChatMessage]


@dataclass
class RequestChatResponse:
    message: ChatMessage  # TODO make this richer


AgentTaskId = str


@dataclass
class AgentTask:
    status: Literal["running", "done", "error"] = "running"
    description: str
    subtasks: List[AgentTaskId] = field(default_factory=list)
    parent: Optional[AgentTaskId] = None
    id: Optional[str] = None

    def __post_init__(self):
        self.id = str(uuid.uuidv4())[:8]


@dataclass
class AgentRunParams(ABC):
    ...


@dataclass
class AgentProgress(ABC):
    tasks: Optional[Dict[AgentTaskId, AgentTask]] = None


@dataclass
class AgentRunResult(ABC):
    ...


@dataclass
class AgentState(ABC):
    ...


@dataclass
class Agent:
    status: Literal["running", "done", "error"]
    state: AgentState
    tasks: Dict[str, AgentTask]
    server: BaseLspServer
    id: int

    @property
    def task(self):
        if self.active_task_id is None:
            return None
        else:
            return self.tasks.get(self.active_task_id)

    def __str__(self):
        return f"<{type(self).__name__}> {self.id}"

    async def run(self, params: AgentRunParams) -> AgentRunResult:
        ...

    def add_task(self, task: AgentTask):
        self.tasks[task.id] = task
        return task.id

    def cancel(self, msg):
        logger.info(f"{self} cancel run: {msg}")
        # if self.task is not None:
        #     self.task.cancel(msg)
        for _, task in tasks.items():
            if task is not None:
                task.cancel()

    async def request_input(self, req: RequestInputRequest) -> asyncio.Future[RequestInputResponse]:
        ...

    async def request_chat(
        self,
    ) -> ...:  # don't have a way to send a message back, unless if the request_chat_callback attaches another callback
        ...

    async def send_progress(self) -> ...:
        ...

    async def send_result(self) -> ...:
        ...
