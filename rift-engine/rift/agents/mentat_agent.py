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
from mentat.app import get_user_feedback_on_changes
from mentat.code_file_manager import CodeFileManager
from mentat.config_manager import ConfigManager
from mentat.conversation import Conversation
from mentat.llm_api import CostTracker
from mentat.user_input_manager import UserInputManager

import rift.agents.abstract as agent
import rift.ir.IR as IR
import rift.llm.openai_types as openai
import rift.lsp.types as lsp
import rift.util.file_diff as file_diff
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


@agent.agent(
    agent_description="Request codebase-wide edits through chat",
    display_name="Mentat",
    agent_icon="""\
<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
<path fill-rule="evenodd" clip-rule="evenodd" d="M11.4468 9.20793C11.4245 9.23707 11.4251 9.23772 11.4536 9.21488C11.4835 9.19087 11.4929 9.17578 11.4782 9.17578C11.4745 9.17578 11.4603 9.19026 11.4468 9.20793ZM6.33459 10.3838C6.31229 10.4129 6.31292 10.4136 6.34137 10.3907C6.37125 10.3667 6.38074 10.3516 6.36598 10.3516C6.36227 10.3516 6.34813 10.3661 6.33459 10.3838Z" fill="#CCCCCC"/>
<path fill-rule="evenodd" clip-rule="evenodd" d="M4.18497 2.33661L4.18549 3.67323L4.32167 3.78346C4.55389 3.97144 5.14662 4.43307 5.25399 4.50957C5.27496 4.52451 5.43236 4.64889 5.60378 4.78597C5.93772 5.05304 6.12901 5.2027 6.77643 5.70349C7.00334 5.879 7.24673 6.07054 7.31732 6.12914C7.38788 6.18775 7.54932 6.31471 7.67607 6.41126C7.90371 6.58471 8.33586 6.92292 8.59709 7.13207C8.67108 7.19133 8.84059 7.32346 8.97378 7.42574C9.43705 7.78149 9.49275 7.82553 9.49336 7.8365C9.4937 7.84256 9.45158 7.88168 9.39979 7.92342C9.34801 7.96517 9.10845 8.15861 8.86748 8.35333C8.62651 8.54804 8.3561 8.7646 8.26657 8.83458C8.17704 8.90456 8.02676 9.02444 7.93261 9.10098C7.83844 9.17752 7.73027 9.26496 7.69225 9.29528C7.6542 9.32559 7.50587 9.44832 7.36263 9.56802C7.21938 9.68772 7.07327 9.8044 7.03796 9.82735C7.00264 9.85028 6.909 9.92184 6.82988 9.98636C6.75076 10.0509 6.64179 10.1383 6.58772 10.1806C6.53366 10.2229 6.47328 10.2729 6.45355 10.2918C6.40388 10.3392 5.92662 10.7267 5.76295 10.8525C5.57884 10.9939 5.47868 11.0741 5.12545 11.3629C4.95813 11.4997 4.67796 11.7261 4.50284 11.866L4.18445 12.1205V13.5602C4.18445 14.3521 4.18872 15 4.19392 15C4.22312 15 4.61443 14.6944 5.08669 14.3028C5.20309 14.2063 5.42564 14.0281 5.58123 13.9068C5.98968 13.5885 6.1503 13.4614 6.17801 13.4344C6.22933 13.3844 6.74193 12.9715 7.02156 12.755C7.17168 12.6387 8.09057 11.903 8.35744 11.6854C8.40815 11.6441 8.53327 11.5443 8.63548 11.4637C8.73768 11.3831 8.95744 11.2053 9.12381 11.0686C9.29018 10.932 9.45969 10.7959 9.5005 10.7662C9.57879 10.7092 9.92981 10.43 10.1949 10.2139C10.4306 10.0218 10.8509 9.68525 10.9798 9.58538C11.0423 9.53697 11.2809 9.34442 11.51 9.15748C11.739 8.97054 11.9801 8.77625 12.0457 8.72572L12.165 8.63386L12.1658 7.80413L12.1667 6.9744L12.1165 6.93142C12.0889 6.9078 11.9911 6.83471 11.8993 6.76903C11.6835 6.61485 11.4859 6.46759 11.2318 6.27179C10.9726 6.07206 10.6337 5.81613 10.5523 5.75862C10.4981 5.72039 10.2486 5.53391 9.75406 5.16213C9.685 5.11021 9.51549 4.9807 9.37737 4.87433C9.23925 4.76795 9.07681 4.6468 9.01638 4.60511C8.91539 4.53544 8.68531 4.36252 8.12103 3.93221C7.79248 3.68168 7.76296 3.65958 7.66459 3.59055C7.61418 3.55518 7.39368 3.38983 7.17455 3.2231C6.79155 2.93167 6.64824 2.82406 6.09243 2.41042C5.95302 2.30667 5.67553 2.09777 5.47577 1.94619C5.27602 1.79462 5.03494 1.61324 4.94001 1.54315C4.84509 1.47304 4.64126 1.32215 4.48709 1.20785C4.3329 1.09354 4.20174 1 4.19561 1C4.18947 1 4.18468 1.60147 4.18497 2.33661Z" fill="#CCCCCC"/>
<path fill-rule="evenodd" clip-rule="evenodd" d="M3.00058 5.4438V6.94911L3.08392 7.01992C3.12975 7.05885 3.1923 7.11669 3.22293 7.14844C3.29336 7.22145 3.74226 7.6059 3.80533 7.6472C3.83134 7.66423 3.86165 7.68654 3.87269 7.69677C3.92589 7.74608 4.08655 7.88351 4.1166 7.90541C4.13522 7.91899 4.14801 7.93652 4.14503 7.94438C4.13988 7.95794 3.49345 8.50416 3.15753 8.77881L3.00058 8.90713V10.3736C3.00058 11.1801 3.00752 11.8401 3.01599 11.8401C3.02447 11.8401 3.06684 11.8086 3.11016 11.7701C3.15348 11.7316 3.40686 11.5193 3.67324 11.2984C3.93961 11.0775 4.18984 10.8694 4.2293 10.836C4.26876 10.8026 4.50285 10.6088 4.74949 10.4054C4.99613 10.202 5.22381 10.0089 5.25544 9.97623C5.28708 9.9436 5.36376 9.87809 5.42584 9.83067C5.87572 9.48704 6.31921 9.14329 6.57661 8.93871C6.74296 8.8065 6.88485 8.69832 6.89195 8.69832C6.89904 8.69832 6.9978 8.62599 7.1114 8.53756C7.22498 8.44915 7.41975 8.29826 7.54418 8.20226C7.77 8.02807 7.80462 7.98261 7.73895 7.9466C7.7208 7.93666 7.67961 7.90332 7.64741 7.8725C7.61523 7.84171 7.53873 7.77722 7.47742 7.7292C7.3824 7.65477 6.88661 7.23778 6.77644 7.13965C6.7567 7.12207 6.61141 6.99892 6.45356 6.86596C6.29571 6.73301 5.95448 6.44395 5.69527 6.22361C5.43607 6.00327 5.15759 5.76697 5.07642 5.69852C4.99527 5.63006 4.78761 5.4549 4.61496 5.30927C4.44231 5.16363 4.2728 5.01818 4.23827 4.98602C4.12702 4.8824 3.7828 4.59086 3.71808 4.54543C3.68355 4.5212 3.64655 4.49283 3.63584 4.48239C3.62513 4.47196 3.58314 4.4358 3.54253 4.40203C3.50192 4.36828 3.45176 4.32562 3.43108 4.30725C3.4104 4.28887 3.3636 4.24865 3.32708 4.2179C3.29056 4.18712 3.21628 4.1197 3.16202 4.06807C2.98153 3.8963 3.00058 3.73392 3.00058 5.4438Z" fill="#CCCCCC"/> </svg>
""",
)
@dataclass
class Mentat(agent.ThirdPartyAgent):
    agent_type: ClassVar[str] = "mentat"
    run_params: Type[MentatAgentParams] = MentatAgentParams
    state: Optional[MentatAgentState] = None

    @classmethod
    async def create(cls, params: MentatAgentParams, server):
        """
        Create a new Mentat agent instance with the given parameters and server.
        :param params: The MentatAgentParams containing agent configuration.
        :param server: The server instance.
        :return: A new Mentat agent instance.
        """        
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
        """
        This is the main method of the Mentat agent. It starts the chat thread and handles the main loop of the agent.
        """
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

                dropped_symbols = False

                def refactor_uri_match(resp: str):
                    uri_pattern = r"\[uri\]\((\S+)\)"

                    def replacement(m: re.Match[str]):
                        parsed_uri = m.group(1)
                        if "#" in uri:
                            nonlocal dropped_symbols
                            dropped_symbols = True
                            uri, symbol = parsed_uri.split("#")[0], parsed_uri.split("#")[1]
                        else:
                            uri = parsed_uri

                        reference = IR.Reference.from_uri(uri)
                        file_path = reference.file_path
                        relative_path = os.path.relpath(
                            file_path, self.state.params.workspaceFolderPath
                        )
                        return f"`{relative_path}`" if not dropped_symbols else f"{symbol} @ `{relative_path}`"

                    resp = re.sub(uri_pattern, replacement, resp)
                    return resp

                try:
                    resp = refactor_uri_match(resp)

                    if dropped_symbols:
                        # await self.send_update(
                        #     "This agent does not support symbol references. A plain file reference will be used instead."
                        # )
                        pass
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

        dropped_symbols = False

        def extract_path(uri: str):
            if "#" in uri:
                nonlocal dropped_symbols
                dropped_symbols = True
                uri = uri.split("#")[0]
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

        self.state.messages.append(
            openai.Message.assistant(
                content="Which files should be visible to me for this conversation? (You can @-mention as many files as you want.)"
            )
        )

        # Add a new task to request the user for the file names that should be visible
        get_repo_context_t = self.add_task(
            "get_repo_context", self.request_chat, [agent.RequestChatRequest(self.state.messages)]
        )

        # Wait for the user's response
        user_visible_files_response = await get_repo_context_t.run()
        self.state.messages.append(openai.Message.user(content=user_visible_files_response))
        await self.send_progress()

        # Return the response from the user
        from rift.util.context import resolve_inline_uris

        uris: List[str] = [
            extract_path(x.uri)
            for x in resolve_inline_uris(user_visible_files_response, server=self.server)
        ]
        logger.info(f"{uris=}")

        if dropped_symbols:
            await self.send_update(
                "This agent does not support symbol references. A file reference will be used instead."
            )

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
