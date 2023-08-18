import asyncio
import logging
import os
import re
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Type

logger = logging.getLogger(__name__)

import mentat.app
import rift.agents.abstract as agent
import rift.llm.openai_types as openai
import rift.lsp.types as lsp
import rift.util.file_diff as file_diff
from mentat.app import get_user_feedback_on_changes
from mentat.code_file_manager import CodeFileManager
from mentat.config_manager import ConfigManager
from mentat.conversation import Conversation
from mentat.llm_api import CostTracker
from mentat.user_input_manager import UserInputManager
from rift.util.TextStream import TextStream


@dataclass
class MentatAgentParams(agent.AgentParams):
    paths: List[str] = field(default_factory=list)


@dataclass
class MentatAgentState(agent.AgentState):
    params: MentatAgentParams
    messages: list[openai.Message]
    response_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _response_buffer: str = ""


@dataclass
class MentatRunResult(agent.AgentRunResult):
    ...


@agent.agent(agent_description="Request codebase-wide edits through chat", display_name="Mentat")
@dataclass
class Mentat(agent.ThirdPartyAgent):
    agent_type: ClassVar[str] = "mentat"
    run_params: Type[MentatAgentParams] = MentatAgentParams
    state: Optional[MentatAgentState] = None

    @classmethod
    async def create(cls, params: MentatAgentParams, server):
        logger.info(f"{params=}")
        state = MentatAgentState(
            params=params,
            messages=[],
        )
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )
        return obj

    async def apply_file_changes(self, updates) -> lsp.ApplyWorkspaceEditResponse:
        return await self.server.apply_workspace_edit(
            lsp.ApplyWorkspaceEditParams(
                file_diff.edits_from_file_changes(
                    updates,
                    user_confirmation=True,
                )
            )
        )

    async def _run_chat_thread(self, response_stream):

        before, after = response_stream.split_once("感")
        try:
            async with self.state.response_lock:
                async for delta in before:
                    self.state._response_buffer += delta
                    await self.send_progress({"response": self.state._response_buffer})
            await asyncio.sleep(0.1)
            await self._run_chat_thread(after)
        except Exception as e:
            logger.info(f"[_run_chat_thread] caught exception={e}, exiting")

    async def run(self) -> MentatRunResult:
        response_stream = TextStream()

        run_chat_thread_task = asyncio.create_task(self._run_chat_thread(response_stream))

        loop = asyncio.get_running_loop()

        def send_chat_update_wrapper(prompt: str = "感", *args, end="\n", **kwargs):
            async def _worker():
                response_stream.feed_data(prompt + end)

            asyncio.run_coroutine_threadsafe(_worker(), loop=loop)

        def request_chat_wrapper(prompt: Optional[str] = None, *args, **kwargs):
            async def request_chat():
                response_stream.feed_data("感")
                await asyncio.sleep(0.1)
                await self.state.response_lock.acquire()
                await self.send_progress(
                    dict(response=self.state._response_buffer, done_streaming=True)
                )
                self.state.messages.append(
                    openai.Message.assistant(content=self.state._response_buffer)
                )
                self.state._response_buffer = ""
                if prompt is not None:
                    self.state.messages.append(openai.Message.assistant(content=prompt))

                resp = await self.request_chat(
                    agent.RequestChatRequest(messages=self.state.messages)
                )

                def refactor_uri_match(resp):
                    pattern = f"\[uri\]\({self.state.params.workspaceFolderPath}/(\S+)\)"
                    replacement = r"`\1`"
                    resp = re.sub(pattern, replacement, resp)
                    return resp

                try:
                    resp = refactor_uri_match(resp)
                except:
                    pass
                self.state.messages.append(openai.Message.user(content=resp))
                self.state.response_lock.release()
                return resp

            t = asyncio.run_coroutine_threadsafe(request_chat(), loop)
            futures.wait([t])
            result = t.result()
            return result

        import inspect

        def collect_user_input(self) -> str:
            user_input = request_chat_wrapper().strip()
            if user_input.lower() == "q":
                raise mentat.user_input_manager.UserQuitInterrupt()
            return user_input

        def colored(*args, **kwargs):
            return args[0]

        def highlight(*args, **kwargs):
            return args[0]

        file_changes = []

        from collections import defaultdict

        from mentat.code_change import CodeChange, CodeChangeAction

        event = asyncio.Event()
        event2 = asyncio.Event()

        async def set_event():
            event.set()

        def write_changes_to_files(self, code_changes: list[CodeChange]) -> None:
            files_to_write = dict()
            file_changes_dict = defaultdict(list)
            for code_change in code_changes:
                rel_path = code_change.file
                if code_change.action == CodeChangeAction.CreateFile:
                    send_chat_update_wrapper(f"Creating new file {rel_path}")
                    files_to_write[rel_path] = code_change.code_lines
                elif code_change.action == CodeChangeAction.DeleteFile:
                    self._handle_delete(code_change)
                else:
                    changes = file_changes_dict[rel_path]
                    logging.getLogger().info(f"{changes=}")
                    changes.append(code_change)

            for file_path, changes in file_changes_dict.items():
                new_code_lines = self._get_new_code_lines(changes)
                if new_code_lines:
                    files_to_write[file_path] = new_code_lines

            for rel_path, code_lines in files_to_write.items():
                file_path = os.path.join(self.git_root, rel_path)
                if file_path not in self.file_paths:
                    logging.info(f"Adding new file {file_path} to context")
                    self.file_paths.append(file_path)
                file_changes.append(file_diff.get_file_change(file_path, "\n".join(code_lines)))
            asyncio.run_coroutine_threadsafe(set_event(), loop=loop)
            while True:
                if not event2.is_set():
                    time.sleep(0.25)
                    continue
                break

        for n, m in inspect.getmembers(mentat, inspect.ismodule):
            setattr(m, "cprint", send_chat_update_wrapper)
            setattr(m, "print", send_chat_update_wrapper)
            setattr(m, "colored", colored)
            setattr(m, "highlight", highlight)
            setattr(m, "change_delimiter", "```")

        mentat.user_input_manager.UserInputManager.collect_user_input = collect_user_input
        mentat.code_file_manager.CodeFileManager.write_changes_to_files = write_changes_to_files
        # mentat.parsing.change_delimiter = ("yeehaw" * 10)

        def extract_path(uri: str):
            if uri.startswith("file://"):
                return uri[7:]
            if uri.startswith("uri://"):
                return uri[6:]

        # TODO: revisit auto-context population at some point
        # paths = (
        #     [extract_path(x.textDocument.uri) for x in self.state.params.visibleEditorMetadata]
        #     if self.state.params.visibleEditorMetadata
        #     else []
        # )

        paths: List[str] = []

        self.state.messages.append(openai.Message.assistant(content="Which files should be visible to me for this conversation? (You can @-mention as many files as you want.)"))

        # Add a new task to request the user for the file names that should be visible
        get_repo_context_t = self.add_task(
            "get_repo_context",
            self.request_chat,
            [agent.RequestChatRequest(self.state.messages)]
        )
        
        # Wait for the user's response
        user_visible_files_response = await get_repo_context_t.run()
        self.state.messages.append(openai.Message.user(content=user_visible_files_response))
        await self.send_progress()
        
        # Return the response from the user
        from rift.util.context import resolve_inline_uris
        uris: List[str] = [extract_path(x.uri) for x in resolve_inline_uris(user_visible_files_response, server=self.server)]
        logger.info(f"{uris=}")

        paths += uris

        finished = False

        def done_cb(fut):
            nonlocal finished
            finished = True
            event.set()
        
        async def mentat_loop():
            nonlocal file_changes

            fut = loop.run_in_executor(None, mentat.app.run, mentat.app.expand_paths(paths))
            fut.add_done_callback(done_cb)
            while True:
                await event.wait()
                if finished:
                    break
                if len(file_changes) > 0:
                    await self.apply_file_changes(file_changes)
                    file_changes = []
                event2.set()
                event.clear()
            try:
                await fut
            except SystemExit as e:
                logger.info(f"[mentat] caught {e}, exiting")
            except Exception as e:
                logger.error(f"[mentat] caught {e}, exiting")
            finally:
                await self.send_progress()

        await self.add_task("Mentat main loop", mentat_loop).run()
