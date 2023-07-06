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
# from rift.server.agent import *
from rift.server.selection import RangeSet
from rift.llm.openai_types import Message
from rift.util.ofdict import ofdict
from rift.server.chat_agent import RunChatParams, ChatAgentLogs, ChatAgent
from rift.server.agent import *
from rift.agents.code_completion import CodeCompletionAgent
from rift.agents.smol import SmolAgent, SmolAgentParams

logger = logging.getLogger(__name__)

@dataclass
class AgentProgress:
    id: int
    textDocument: lsp.TextDocumentIdentifier
    status: Literal["running", "done", "error"]
    log: Optional[AgentLogs] = field(default=None)
    ranges: Optional[RangeSet] = field(default=None)
    cursor: Optional[lsp.Position] = field(default=None)


class LspLogHandler(logging.Handler):
    def __init__(self, server: "LspServer"):
        super().__init__()
        self.server = server
        self.tasks: set[asyncio.Task] = set()

    def emit(self, record: logging.LogRecord) -> None:
        if self.server.status != RpcServerStatus.running:
            return
        t_map = {
            logging.DEBUG: 4,
            logging.INFO: 3,
            logging.WARNING: 2,
            logging.ERROR: 1,
        }
        level = t_map.get(record.levelno, 4)
        if level > 3:
            return
        t = asyncio.create_task(
            self.server.notify(
                "window/logMessage",
                {
                    "type": level,
                    "message": self.format(record),
                },
            )
        )
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

@dataclass
class ChatAgentProgress:
    id: int
    response: str = ""
    log: Optional[AgentLogs] = field(default=None)
    done: bool = False


@dataclass
class RunAgentResult:
    id: int


@dataclass
class RunAgentSyncResult:
    id: int
    text: str


class LspServer(BaseLspServer):
    active_agents: dict[int, CodeCompletionAgent]
    active_chat_agents: dict[int, asyncio.Task]
    model_config: ModelConfig
    completions_model: Optional[AbstractCodeCompletionProvider] = None
    chat_model: Optional[AbstractChatCompletionProvider] = None

    def __init__(self, transport):
        super().__init__(transport)
        self.model_config = ModelConfig.default()
        self.capabilities.textDocumentSync = lsp.TextDocumentSyncOptions(
            openClose=True,
            change=lsp.TextDocumentSyncKind.incremental,
        )
        self.active_agents = {}
        self.active_chat_agents = {}
        self._loading_task = None
        self._chat_loading_task = None
        self.logger = logging.getLogger(f"rift")
        self.logger.addHandler(LspLogHandler(self))

    @rpc_method("workspace/didChangeConfiguration")
    async def on_workspace_did_change_configuration(self, params: lsp.DidChangeConfigurationParams):
        logger.info("workspace/didChangeConfiguration")
        await self.get_config()

    async def get_config(self):
        """This should be called whenever the user changes the model config settings.

        It should also be called immediately after initialisation."""
        if self._loading_task is not None:
            idx = getattr(self, "_loading_idx", 0) + 1
            logger.debug(f"Queue of set_model_config tasks: {idx}")
            self._loading_idx = idx
            self._loading_task.cancel()
            # give user typing in config some time to settle down
            await asyncio.sleep(1)
            try:
                await self._loading_task
            except (asyncio.CancelledError, TypeError):
                pass
            if self._loading_idx != idx:
                logger.debug(f"loading task {idx} was cancelled, but a new one was started")
                return
            # only the most recent request will make it here.
        settings = await self.get_workspace_configuration(section="rift")
        if not isinstance(settings, list) or len(settings) != 1:
            raise RuntimeError(f"Invalid settings:\n{settings}\nExpected a list of dictionaries.")
        settings = settings[0]
        config = ModelConfig.parse_obj(settings)
        if self.chat_model and self.completions_model and self.model_config == config:
            logger.debug("config unchanged")
            return
        self.model_config = config
        logger.info(f"{self} recieved model config {config}")
        for k, h in self.active_agents.items():
            h.cancel("config changed")
        self.completions_model = config.create_completions()
        self.chat_model = config.create_chat()

        self._loading_task = asyncio.gather(
            self.completions_model.load(),
            self.chat_model.load(),
        )
        try:
            await self._loading_task
        except asyncio.CancelledError:
            logger.debug("loading cancelled")
        else:
            logger.info(f"{self} finished loading")
        finally:
            self._loading_task = None

    async def send_agent_progress(
        self,
        id: int,
        textDocument: lsp.TextDocumentIdentifier,
        log: Optional[AgentLogs] = None,
        cursor: Optional[lsp.Position] = None,
        ranges: Optional[RangeSet] = None,
        status: Literal["running", "done", "error"] = "running",
    ):
        progress = AgentProgress(
            id=id,
            textDocument=textDocument,
            log=log,
            cursor=cursor,
            status=status,
            ranges=ranges,
        )
        await self.notify("morph/progress", progress)

    async def send_chat_agent_progress(
        self,
        id: int,
        response: str,
        log: Optional[ChatAgentLogs] = None,
        done: bool = False,
        # textDocument: lsp.TextDocumentIdentifier,
        # log: Optional[AgentLogs] = None,
        # cursor: Optional[lsp.Position] = None,
        # status: Literal["running", "done", "error"] = "running",
    ):
        progress = ChatAgentProgress(
            # id=id, textDocument=textDocument, log=log, cursor=cursor, status=status
            id=id,
            response=response,
            log=log,
            done=done,
        )
        await self.notify("morph/chat_progress", progress)

    async def ensure_completions_model(self):
        try:
            if self.completions_model is None:
                await self.get_config()
            assert self.completions_model is not None
            return self.completions_model
        except:
            config = ModelConfig(chatModel="openai:gpt-3.5-turbo", completionsModel="openai:gpt-3.5-turbo")
            return config.create_completions()

    async def ensure_chat_model(self):
        try:
            if self.chat_model is None:
                await self.get_config()
            assert self.chat_model is not None
            return self.chat_model
        except:
            config = ModelConfig(chatModel="openai:gpt-3.5-turbo", completionsModel="openai:gpt-3.5-turbo")
            return config.create_chat()

    @rpc_method("morph/run")
    async def on_run(self, params: RunAgentParams):
        agent_type = params.agent_type
        if agent_type == "chat":
            # prepare params for ChatAgent construction
            model = await self.ensure_chat_model()
            agent_params = ofdict(RunChatParams, params.agent_params)
            agent = ChatAgent(agent_params, model=model, server=self)
        elif agent_type == "code_completion":
            model = await self.ensure_completions_model()
            agent_params = ofdict(CodeCompletionAgentParams, params.agent_params)
            agent = CodeCompletionAgent.create(agent_params, model=model, server=self)
        elif agent_type == "smol_dev":
            model = await self.ensure_chat_model()
            agent_params = ofdict(SmolAgentParams, params.agent_params)
            agent = SmolAgent.create(params=agent_params, model=model, server=self)
        else:
            raise Exception(f"unsupported agent type={agent_type}")
        t = asyncio.Task(agent.run())
        self.active_agents[agent.id] = agent
        return RunAgentResult(id=agent.id)

    @rpc_method("morph/run_agent")
    async def on_run_agent(self, params: CodeCompletionAgentParams):
        model = await self.ensure_completions_model()
        try:
            agent = CodeCompletionAgent(params, model=model, server=self)
        except LookupError:
            # [hack] wait a bit for textDocumentChanged notification to come in
            logger.debug("request too early: waiting for textDocumentChanged notification")
            await asyncio.sleep(3)
            agent = CodeCompletionAgent(params, model=model, server=self)
        logger.debug(f"starting agent {agent.id}")
        # agent holds a reference to worker task
        agent.run()
        self.active_agents[agent.id] = agent
        return RunAgentResult(id=agent.id)

    @rpc_method("morph/run_chat")
    async def on_run_chat(self, params: RunChatParams):
        chat = await self.ensure_chat_model()
        chat_agent = ChatAgent(params, model=chat, server=self)
        logger.debug(f"starting chat agent {chat_agent.id}")
        task = asyncio.create_task(chat_agent.run())
        self.active_chat_agents[chat_agent.id] = task

    @rpc_method("morph/cancel")
    async def on_cancel(self, params: AgentIdParams):
        agent = self.active_agents.get(params.id)
        if agent is not None:
            agent.cancel()

    @rpc_method("morph/accept")
    async def on_accept(self, params: AgentIdParams):
        agent = self.active_agents.get(params.id)
        if agent is not None:
            await agent.accept()
            self.active_agents.pop(params.id, None)

    @rpc_method("morph/reject")
    async def on_reject(self, params: AgentIdParams):
        agent = self.active_agents.get(params.id)
        if agent is not None:
            await agent.reject()
            self.active_agents.pop(params.id, None)
        else:
            logger.error(f"no agent with id {params.id}")

    @rpc_method("hello_world")
    def on_hello(self, params):
        logger.debug("hello world")
        return "hello world"
