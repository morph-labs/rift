import asyncio
import logging
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel

import rift.llm.openai_types as openai
import rift.lsp.types as lsp
from rift.agents.agenttask import AgentTask
from rift.llm.openai_types import Message as ChatMessage
from rift.lsp import LspServer as BaseLspServer

logger = logging.getLogger(__name__)


class Status(Enum):
    running = "running"
    done = "done"
    error = "error"
    accepted = "accepted"
    rejected = "rejected"


@dataclass
class RequestInputRequest:
    msg: str
    place_holder: str = ""


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
class AgentParams:
    agent_type: str
    agent_id: str
    textDocument: Optional[lsp.TextDocumentIdentifier]
    selection: Optional[lsp.Selection]
    position: Optional[lsp.Position]
    workspaceFolderPath: Optional[str]


@dataclass
class AgentProgress:
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    tasks: Optional[Dict[str, Any]] = None
    payload: Optional[Any] = None


@dataclass
class AgentRunResult(ABC):
    """
    Abstract base class for AgentRunResult
    """


@dataclass
class AgentState(ABC):
    """
    Abstract base class for AgentState. Always contains a copy of the params used to create the Agent.
    """

    params: AgentParams


@dataclass
class Agent:
    """
    Agent is the base class for all agents.

    `agent_type` is a string that is defined in the source code and represents the type of the agent.
    `agent_id` is a unique identifier for the agent that is generated by convention in the lsp's handler for 'morph/run'.
    `state` is a namespace that encapsulates all special state for the agent.
    `tasks` is a list of `AgentTask`s and is used to report the progress of the agent.
    `server` is a handle to the global language server.
    """

    agent_type: ClassVar[str]
    server: Optional[BaseLspServer] = None
    state: Optional[AgentState] = None
    agent_id: Optional[str] = None
    tasks: List[AgentTask] = field(default_factory=list)
    task: Optional[AgentTask] = None
    params_cls: Type[AgentParams] = AgentParams

    # def get_display(self):
    #     """Get agent display information"""
    #     return self.agent_type, self.description

    def __str__(self):
        """Get string representation of the agent"""
        return f"<{self.agent_type}> {self.agent_id}"

    @classmethod
    async def create(cls, params: AgentParams, server: BaseLspServer):
        """
        Factory function which is responsible for constructing the agent's state.
        """
        ...

    async def run(self) -> AgentRunResult:
        """
        Run the agent.
        """
        ...

    def set_tasks(self, tasks: List[AgentTask]):
        self.tasks = tasks

    def add_task(self, *args, **kwargs):
        """
        Register a subtask.
        """
        # Capture the current loop context
        try:
            loop = asyncio.get_running_loop()

            # On completion of the task we call the callable object done_cb with the provided arguments
            # done_cb tries to send_progress() using the captured loop context irrespective of where it's called from.
            def done_cb(*args):
                nonlocal self
                asyncio.run_coroutine_threadsafe(self.send_progress(), loop=loop)

            kwargs["done_callback"] = done_cb
            kwargs["start_callback"] = done_cb
        # Pass if loop context doesn't exist(not running any asynchronous code)
        except:
            pass

        # Create AgentTask using provided arguments
        task = AgentTask(*args, **kwargs)

        # Append the created task in the task list
        self.tasks.append(task)

        # Return the created task
        return task

    async def cancel(self, msg: Optional[str] = None, send_progress=True):
        """
        Cancel all tasks and update progress. Assumes that `Agent.main()` has been called and that the main task has been created.
        """
        if self.task.cancelled:
            return
        logger.info(f"{self.agent_type} {self.agent_id} cancel run {msg or ''}")
        self.task.cancel()
        for task in self.tasks:
            if task is not None:
                task.cancel()
        if send_progress:
            await self.send_progress()

    async def done(self):
        for task in self.tasks:
            if task is not None:
                task.cancel()
                task._done = True
        await self.send_progress()

    async def request_input(self, req: RequestInputRequest) -> str:
        """
        Prompt the user for more information.
        """
        try:
            # request user responses/data from server
            response = await self.server.request(
                f"morph/{self.agent_type}_{self.agent_id}_request_input", req
            )
            return response["response"]
        except Exception as e:  # return the response from the user
            return response["response"]
        # exception handling block

    async def send_update(self, msg: str):
        """
        Creates a notification toast in the Rift extension by default.
        """
        await self.server.notify(
            f"morph/{self.agent_type}_{self.agent_id}_send_update",
            {"msg": f"[{self.agent_type}] {msg}"},
        )
        await self.send_progress()

    async def request_chat(self, req: RequestChatRequest) -> str:
        """Send chat request"""
        try:
            response = await self.server.request(
                f"morph/{self.agent_type}_{self.agent_id}_request_chat", req
            )
            return response["message"].strip()
        except Exception as exception:
            logger.info(f"[request_chat] failed, caught {exception=}")
            raise exception

    async def send_chat_update(self, msg: str, prepend: bool = False):
        await self.send_progress(dict(response=msg))
        if not prepend:
            self.state.messages += [openai.Message.assistant(msg)]
        else:
            self.state.messages = [openai.Message.assistant(msg)] + self.state.messages
        await self.send_progress(dict(done_streaming=True, messages=self.state.messages))

    async def send_progress(self, payload: Optional[Any] = None, payload_only: bool = False):
        """
        Send an update about the progress of the agent's tasks to the server at `morph/{agent_type}_{agent_id}_send_progress`.
        It will try to package the description and status of the main and subtasks into the payload, unless the 'payload_only' parameter is set to True.

        Parameters:
        - payload (dict, optional): A dictionary containing arbitrary data about the agent's progress. Default is None.
        - payload_only (bool, optional): If set to True, the function will not include task updates and will send only the payload. Default is False.

        Note:
        This function assumes that `Agent.main()` has been run and the main task has been created.

        Returns:
        This function does not return a value.
        """
        # Check whether we're only sending payload or also tasks' data
        # logging.getLogger().info(f"sending progress with payload={payload}")
        if payload_only:
            # If only payload is to be sent, set tasks to None
            tasks = None
        else:
            # Try to wrap main and subtasks' data into tasks dictionary
            try:
                tasks = {
                    "task": {
                        "description": AGENT_REGISTRY.registry[self.agent_type].display_name,
                        "status": self.task.status,
                    },
                    "subtasks": (
                        [{"description": x.description, "status": x.status} for x in self.tasks]
                    ),
                }
            # If unable to create tasks dictionary due to an exception, log the exception and set tasks to None
            except Exception as e:
                logger.debug(f"Caught exception: {e}")
                tasks = None

        # Package all agent's progress into an AgentProgress object
        progress = AgentProgress(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            tasks=tasks,
            payload=payload,
        )

        # If the main task's status is 'error', log it as an info level message
        if self.task.status == "error":
            logger.info(f"[error]: {self.task._task.exception()}")

        # logger.info(f"{progress=}")
        await self.server.notify(f"morph/{self.agent_type}_{self.agent_id}_send_progress", progress)

    async def main(self):
        """
        The main method called by the LSP server to handle method `morph/run`.

        This method:
            - Creates a task to be run
           - Logs the status of the running task
            - Awaits the result of the running task
            - Sends progress of the task
            - Handles cancellation and exception situations

        Raises:
            asyncio.CancelledError: If the task being run was cancelled.
        """
        # Create a task to run with assigned description and run method
        self.task = AgentTask(description=self.agent_type, task=self.run)

        try:
            # Log the status of the running task
            logger.info(f"{self} running")

            # Await to get the result of the task
            result_t = asyncio.create_task(self.task.run())
            result_t.add_done_callback(
                lambda fut: asyncio.run_coroutine_threadsafe(
                    self.send_update("finished"), loop=asyncio.get_running_loop()
                )
            )
            await self.send_progress()
            result = await result_t
            # Send the progress of the task
            await self.send_progress()
            await self.done()
            await self.send_progress()
            return result
        except asyncio.CancelledError as e:
            # Log information if task is cancelled
            logger.info(f"{self} cancelled: {e}")

            # Call the cancel method if a CancelledError exception happens
            await self.cancel()


@dataclass
class ThirdPartyAgent(Agent):
    third_party_warning_message: str = (
        "This is a third-party agent. It does not use Rift's primitives for on-device LLMs."
    )

    async def main(self):
        # Create a task to run with assigned description and run method
        self.task = AgentTask(description=self.agent_type, task=self.run)

        try:
            # Log the status of the running task
            logger.info(f"{self} running")

            # Await to get the result of the task
            await self.send_progress()

            await self.send_chat_update(self.third_party_warning_message, prepend=True)

            result_t = asyncio.create_task(self.task.run())
            result_t.add_done_callback(
                lambda fut: asyncio.run_coroutine_threadsafe(
                    self.send_update("finished"), loop=asyncio.get_running_loop()
                )
            )
            await self.send_progress()

            result = await result_t
            # Send the progress of the task
            await self.send_progress()
            await self.done()
            await self.send_progress()
            return result
        except asyncio.CancelledError as e:
            # Log information if task is cancelled
            logger.info(f"{self} cancelled: {e}")

            # Call the cancel method if a CancelledError exception happens
            await self.cancel()


@dataclass
class AgentRegistryItem:
    """
    Stored in the registry by the @agent decorator, created upon Rift initialization.
    """

    agent: Type[Agent]
    agent_description: str
    display_name: Optional[str] = None
    agent_icon: Optional[str] = None

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.agent.agent_type


@dataclass
class AgentRegistryResult:
    """
    To be returned as part of a list of available agent workflows to the language server client.
    """

    agent_type: str
    agent_description: str
    display_name: Optional[str] = None
    agent_icon: Optional[str] = None  # svg icon information


@dataclass
class AgentRegistry:
    """
    AgentRegistry is an organizational class that is used to track all agents in one central location.
    """

    # Initial registry to store agents
    registry: Dict[str, AgentRegistryItem] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.get_agent(key)

    def register_agent(
        self,
        agent: Type[Agent],
        agent_description: str,
        display_name: Optional[str] = None,
        agent_icon: Optional[str] = None,
    ) -> None:
        if agent.agent_type in self.registry:
            raise ValueError(f"Agent '{agent.agent_type}' is already registered.")
        self.registry[agent.agent_type] = AgentRegistryItem(
            agent=agent,
            agent_description=agent_description,
            display_name=display_name,
            agent_icon=agent_icon,
        )

    def get_agent(self, agent_type: str) -> Type[Agent]:
        result: AgentRegistryItem | None = self.registry.get(agent_type)
        if result is not None:
            return result.agent
        else:
            raise ValueError(f"Agent not found: {agent_type}")

    def get_agent_icon(self, item: AgentRegistryItem) -> ...:
        return None  # TODO

    def list_agents(self) -> List[AgentRegistryResult]:
        return [
            AgentRegistryResult(
                agent_type=item.agent.agent_type,
                agent_description=item.agent_description,
                agent_icon=item.agent_icon,
                display_name=item.display_name,
            )
            for item in self.registry.values()
        ]


AGENT_REGISTRY = AgentRegistry()  # Creating an instance of AgentRegistry


def agent(
    agent_description: str, display_name: Optional[str] = None, agent_icon: Optional[str] = None
):
    def decorator(cls: Type[Agent]) -> Type[Agent]:
        AGENT_REGISTRY.register_agent(
            cls, agent_description, display_name, agent_icon
        )  # Registering the agent
        return cls

    return decorator
