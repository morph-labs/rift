import asyncio
import glob
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import rift.lsp.types as lsp
from rift.agents import AGENT_REGISTRY, Agent, AgentParams, AgentRegistryResult
from rift.llm.abstract import AbstractChatCompletionProvider, AbstractCodeCompletionProvider
from rift.llm.create import ModelConfig, parse_type_name_path
from rift.lsp import LspServer as BaseLspServer
from rift.lsp import rpc_method
from rift.rpc import RpcServerStatus
from rift.util.ofdict import ofdict
import pydantic

logger = logging.getLogger(__name__)


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
        if level == 1:
            t = asyncio.create_task(self.server.send_error(self.format(record)))
            self.tasks.add(t)
            t.add_done_callback(self.tasks.discard) # cute
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
class LoadFilesResult:
    documents: dict[lsp.DocumentUri, lsp.TextDocumentItem]


@dataclass
class LoadFilesParams:
    patterns: List[str]


@dataclass
class CreateAgentResult:
    id: str


@dataclass
class RunAgentSyncResult:
    id: int
    text: str


@dataclass
class AgentIdParams:
    id: str


class LspServer(BaseLspServer):
    active_agents: dict[str, Agent]
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
        self._loading_task = None
        self._chat_loading_task = None
        self.logger = logging.getLogger(f"rift")
        self.logger.addHandler(LspLogHandler(self))

    @rpc_method("workspace/didChangeConfiguration")
    async def on_workspace_did_change_configuration(self, params: lsp.DidChangeConfigurationParams):
        logger.info("workspace/didChangeConfiguration")
        await self.get_config()

    @rpc_method("morph/loadFiles")
    def load_documents(self, params: LoadFilesParams) -> LoadFilesResult:
        """
        Accepts a set of file paths, processes them into full, qualified paths.
        Opens each file and reads its text, stores the text into a list of TextDocumentItem.
        If a language can be determined from the file's extension using languages.json, sets languageId, else defaults to "*".
        Updates dictionary of documents tracked by LspServer.
        """
        # Try getting absolute path to current directory, default to current working directory on exception.
        try:
            current_dir = os.path.abspath(__file__)
        except:
            current_dir = os.getcwd()
        # Open the languages.json file to get details for mapping file extension to language id
        with open(os.path.join(current_dir, "languages.json"), "r") as f:
            language_map = json.loads(f)

        # Helper function to find the matching language id, given a file path and language map
        # The id of the language is returned, if found in the language map
        def find_matching_language(
            filepath: str, language_map: Dict[str, List[Dict[str, str]]]
        ) -> Optional[str]:
            # Getting the file extension
            extension = filepath.split(".")[-1]

            # loop over the details in the language map
            for details in language_map["languages"]:
                # Check if file extension matches any in the list, if yes, return that language id
                if extension in details.get("extensions", []):
                    return details["id"]

            # No language match found, return None
            return None

        # Helper function to expand all environment variables in file paths
        def preprocess_filepaths(filepaths: List[str]) -> List[str]:
            processed_filepaths = []
            for filepath in filepaths:
                # Expanding all env variables in the file path
                processed_filepaths.append(os.path.expandvars(filepath))
            return processed_filepaths

        # Helper function to join all the file paths provided
        def join_filepaths(filepaths: List[str]) -> List[str]:
            for filepath in filepaths:
                # Yielding all file paths matching the glob pattern
                yield from glob.glob(filepath, root="/" if filepath.startswith("/") else None)

        # Initialize dictionary to store resulting TextDocumentItems
        result_documents: Dict[str, lsp.TextDocumentItem] = dict()
        # Process and combine all file paths, and for each...
        for file_path in join_filepaths(preprocess_filepaths(params.patterns)):
            # Open the file for reading
            with open(file_path, "r") as f:
                # Reading the content of the file
                text = f.read()
            # Creating a TextDocumentItem instance for each file
            doc_item = lsp.TextDocumentItem(
                text=text,
                # Constructing the Uri for each file
                uri="file://" + os.path.join(os.getcwd(), str(file_path))
                if not file_path.startswith("/")
                else str(file_path),
                # Finding the language ID for each file, or using "*" if language ID cannot be determined
                languageId=find_matching_language(file_path, language_map) or "*",
                version=1,
            )
            # Adding the TextDocumentItem to the result documents dictionary
            result_documents[doc_item.uri] = doc_item

    @rpc_method("morph/applyWorkspaceEdit")
    async def on_workspace_did_change_configuration(self, params: lsp.ApplyWorkspaceEditParams):
        return await self.apply_workspace_edit(params)

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
        config: ModelConfig = ModelConfig.parse_obj(settings)
        if self.chat_model and self.completions_model and self.model_config == config:
            logger.debug("config unchanged")
            return
        self.model_config = config
        logger.info(f"{self} recieved model config {config}")
        for k, h in self.active_agents.items():
            asyncio.create_task(h.cancel("config changed"))
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

    def parse_current_chat_config(self) -> Tuple[str, str, str]:
        return parse_type_name_path(self.model_config.chatModel)

    def parse_current_completions_config(self) -> Tuple[str, str, str]:
        return parse_type_name_path(self.model_config.completionsModel)

    async def send_update(self, msg: str):
        await self.notify("morph/send_update", {"msg": msg})

    async def ensure_completions_model(self):
        try:
            if self.completions_model is None:
                await self.get_config()
            assert self.completions_model is not None
            return self.completions_model
        except:
            config = ModelConfig(
                chatModel="openai:gpt-3.5-turbo", completionsModel="openai:gpt-3.5-turbo"
            )
            return config.create_completions()

    async def ensure_chat_model(self):
        try:
            if self.chat_model is None:
                await self.get_config()
            assert self.chat_model is not None
            return self.chat_model
        except:
            config = ModelConfig(
                chatModel="openai:gpt-3.5-turbo", completionsModel="openai:gpt-3.5-turbo"
            )
            return config.create_chat()

    @rpc_method("morph/restart_agent")
    async def on_restart_agent(self, params: AgentIdParams) -> CreateAgentResult:
        logger.info(f"morph/restart_agent firing with {params=}")
        agent_id = params.id
        old_agent = self.active_agents[agent_id]
        old_params = old_agent.state.params
        logger.info(old_agent.state.params)
        agent_type = old_agent.agent_type
        agent_id = old_agent.agent_id
        return await self.on_create(
            AgentParams(
                agent_type=agent_type,
                agent_id=agent_id,
                textDocument=old_params.textDocument,
                selection=old_params.selection,
                workspaceFolderPath=old_params.workspaceFolderPath,
            )
        )

    @rpc_method("morph/create_agent")
    async def on_create(self, params_as_dict: Any):
        try:
            agent_type = params_as_dict["agent_type"]
            agent_id = str(uuid.uuid4())[:8]
            params_as_dict["agent_id"] = agent_id

            logger = logging.getLogger(__name__)
            agent_cls = AGENT_REGISTRY[agent_type]

            logger.info(f"[on_create] {agent_cls.params_cls=}")
            params_with_id = ofdict(agent_cls.params_cls, params_as_dict)
            agent = await agent_cls.create(params=params_with_id, server=self)

            self.active_agents[agent_id] = agent
            t = asyncio.create_task(agent.main())

            def main_callback(fut):
                if fut.exception():
                    logger.info(f"[on_run] caught exception={fut.exception()=}")

            t.add_done_callback(main_callback)
            return CreateAgentResult(id=agent_id)
        except pydantic.error_wrappers.ValidationError as e:
            await self.send_error(f"Error (possibly due to unset API key):\n{e}")

        except Exception as e:
            await self.send_error(str(e))

    async def send_error(self, msg: str):
        await self.notify("morph/error", {"msg": msg})

    @rpc_method("morph/cancel")
    async def on_cancel(self, params: AgentIdParams):
        agent: Agent = self.active_agents.get(params.id)
        if agent is not None:
            asyncio.create_task(agent.cancel())

    @rpc_method("morph/delete")
    async def on_delete(self, params: AgentIdParams):
        agent: Agent = self.active_agents.pop(params.id)
        await agent.cancel("cancel bc delete", False)
        del agent

    @rpc_method("morph/listAgents")
    def on_list_agents(self, _: Any) -> List[AgentRegistryResult]:
        return AGENT_REGISTRY.list_agents()

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
