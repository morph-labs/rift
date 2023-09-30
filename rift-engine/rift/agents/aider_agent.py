import logging
import os
import re
from concurrent import futures

aider_available = False
try:
    import aider
    import aider.coders
    import aider.coders.base_coder
    import aider.io
    import aider.main
    from aider.coders.base_coder import ExhaustedContextWindow

    try:
        aider.__author__
        aider_available = True
    except AttributeError:
        raise Exception(
            'Wrong version of `aider` installed. Please try `pip install -e "rift-engine[aider]" --force-reinstall` from the Rift root directory.'
        )

except ImportError:
    print(
        "aider not available, is git in your path? Installation will continue without aider support."
    )


import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, ClassVar, List, Optional

from rich.text import Text

import rift.agents.abstract as agent
import rift.agents.registry as registry
import rift.llm.openai_types as openai
import rift.lsp.types as lsp
import rift.util.file_diff as file_diff
from rift.util.TextStream import TextStream

logger = logging.getLogger(__name__)


@dataclass
class AiderRunResult(agent.AgentRunResult):
    ...


@dataclass
class AiderAgentParams(agent.AgentParams):
    ...


@dataclass
class AiderAgentState(agent.AgentState):
    """
    A data class that holds the state of an Aider agent.
    It has the following attributes:
    - params (AiderAgentParams) : The parameters associated with the Aider agent.
    - messages (List[openai.Message]) : A list of messages communicated with the openai API during the agent's run.
    """

    params: AiderAgentParams
    messages: list[openai.Message]
    response_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@registry.agent(
    agent_description="Request codebase-wide edits through chat.",
    display_name="Aider",
    agent_icon="""\
<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M14.7369 5.47266H10.4606V1.26318H5.53949V5.47266H1.26318V10.3166H5.53949V14.5261H10.4606V10.3166H14.7369V5.47266Z" fill="#CCCCCC"/>
</svg>""",
)
@dataclass
class Aider(agent.ThirdPartyAgent):
    agent_type: ClassVar[str] = "aider"
    params_cls: ClassVar[Any] = AiderAgentParams

    @classmethod
    async def create(cls, params: AiderAgentParams, server: Any) -> agent.ThirdPartyAgent:
        """
        Class method to create an instance of the Aider class.
        :param params: Parameters for the Aider agent.
        :param server: The server where the Aider agent is running.
        :return: An instance of the Aider class.
        """
        if not aider_available:
            raise Exception("aider not available, is git in your path?")
        state = AiderAgentState(
            params=params,
            messages=[],
        )
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )
        return obj

    async def apply_file_changes(
        self, updates: List[file_diff.FileChange]
    ) -> lsp.ApplyWorkspaceEditResponse:
        """
        Apply file changes to the workspace.
        :param updates: The updates to be applied.
        :return: The response from applying the workspace edit.
        """
        return await self.server.apply_workspace_edit(
            lsp.ApplyWorkspaceEditParams(
                file_diff.edits_from_file_changes(
                    updates,
                    user_confirmation=True,
                )
            )
        )

    async def _run_chat_thread(self, response_stream: str) -> None:
        """
        Run the chat thread.
        :param response_stream: The stream of responses from the chat.
        """

        before, after = response_stream.split_once("感")
        try:
            async with self.state.response_lock:
                async for delta in before:
                    self._response_buffer += delta
                    await self.send_progress({"response": self._response_buffer})
            await asyncio.sleep(0.1)
            await self._run_chat_thread(after)
        except Exception as e:
            logger.info(f"[_run_chat_thread] caught exception={e}, exiting")

    async def run(self) -> AiderRunResult:
        """
        Run the Aider agent.
        :return: The result of running the Aider agent.
        """
        await self.send_progress()
        self._response_buffer = ""

        response_stream = TextStream()

        run_chat_thread_task = asyncio.create_task(self._run_chat_thread(response_stream))

        loop = asyncio.get_running_loop()

        def send_chat_update_wrapper(prompt: str = "感", end="", eof=False):
            async def _worker():
                response_stream.feed_data(prompt)

            asyncio.run_coroutine_threadsafe(_worker(), loop=loop)

        def request_chat_wrapper(prompt: Optional[str] = None):
            async def request_chat():
                # logger.info("acquiring response lock")
                response_stream.feed_data("感")
                await asyncio.sleep(0.1)
                await self.state.response_lock.acquire()
                # logger.info("acquired response lock")
                await self.send_progress(dict(response=self._response_buffer, done_streaming=True))
                # logger.info(f"{self.RESPONSE=}")
                self.state.messages.append(openai.Message.assistant(content=self._response_buffer))
                self._response_buffer = ""
                if prompt is not None:
                    self.state.messages.append(openai.Message.assistant(content=prompt))
                # logger.info(f"MESSAGE HISTORY BEFORE REQUESTING: {self.state.messages}")

                resp = await self.request_chat(
                    agent.RequestChatRequest(messages=self.state.messages)
                )

                def refactor_uri_match(resp) -> str:

                    dropped_symbols = False

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
                        if not resp.startswith("/add"):
                            return f"`{relative_path}`" if not dropped_symbols else f"{symbol} @ `{relative_path}`"
                        else:
                            return f"{relative_path}" if not dropped_symbols else f"{symbol} @ {relative_path}"
                    
                    # def process_path(path):
                    #     relative_path = os.path.relpath(path, self.state.params.workspaceFolderPath)
                    #     if not resp.startswith("/add"):  # /add does not like a quoted path
                    #         relative_path = f"`{relative_path}`"
                    #     return relative_path

                    resp = re.sub(f"\[uri\]\((\S+)\)", lambda m: replacement(m), resp)
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

        ##### PATCHES

        def confirm_ask(self, question, default="y"):
            # print(f"[confirm_ask] question={question}")
            self.num_user_asks += 1

            if self.yes is True:
                res = "yes"
            elif self.yes is False:
                res = "no"
            else:
                res = request_chat_wrapper(str(question) + " (y/n)")
                # res = "yes"

            hist = f"{question.strip()} {res.strip()}"

            # TODO: modify agent state chat history here
            self.append_chat_history(hist, linebreak=True, blockquote=True)
            if self.yes in (True, False):
                self.tool_output(hist)

            if not res or not res.strip():
                return
            return res.strip().lower().startswith("y")

        aider.io.InputOutput.confirm_ask = confirm_ask

        def get_input(self, root, rel_fnames, addable_rel_fnames, commands):
            try:
                rel_fnames = list(rel_fnames)
                show = None
                if len(rel_fnames) > 0:
                    show = "[aider] Current context:\n" + "\n".join(rel_fnames)

                inp = ""
                multiline_input = False

                while True:
                    if multiline_input:
                        show = ". "

                    session_kwargs = {
                        "message": show,
                        "reserve_space_for_menu": 4,
                        "input": self.input,
                        "output": self.output,
                    }
                    line = request_chat_wrapper(show)

                    if line and line[0] == "{" and not multiline_input:
                        multiline_input = True
                        inp += line[1:] + "\n"
                        continue
                    elif line and line[-1] == "}" and multiline_input:
                        inp += line[:-1] + "\n"
                        break
                    elif multiline_input:
                        inp += line + "\n"
                    else:
                        inp = line
                        break

                self.user_input(inp)
                return inp
            except Exception as e:
                print(f"EXCEPTION={e}")
                raise e

        aider.io.InputOutput.get_input = get_input

        def prompt_ask(self, question, default=None):
            self.num_user_asks += 1

            if self.yes is True:
                res = "yes"
            elif self.yes is False:
                res = "no"
            else:
                res = request_chat_wrapper(question)

            hist = f"{question.strip()} {res.strip()}"
            self.append_chat_history(hist, linebreak=True, blockquote=True)
            if self.yes in (True, False):
                self.tool_output(hist)

            return res

        aider.io.InputOutput.prompt_ask = prompt_ask

        def tool_error(self, message):
            self.num_error_outputs += 1

            if message.strip():
                hist = f"{message.strip()}"
                self.append_chat_history(hist, linebreak=True, blockquote=True)

            message = Text(message)
            style = dict(style=self.tool_error_color) if self.tool_error_color else dict()
            send_chat_update_wrapper(str(message))
            send_chat_update_wrapper("\n")

        aider.io.InputOutput.tool_error = tool_error

        def tool_output(self, *messages, log_only=False):
            hist = None
            if messages:
                hist = " ".join(messages)
                hist = f"{hist.strip()}"
                self.append_chat_history(hist, linebreak=True, blockquote=True)

            if not log_only:
                messages = list(map(Text, messages))
                style = dict(style=self.tool_output_color) if self.tool_output_color else dict()
                if hist:
                    # print(f"{hist=}")
                    send_chat_update_wrapper(hist + "\n")

        aider.io.InputOutput.tool_output = tool_output

        def show_send_output_stream(self, completion, silent=False):
            for chunk in completion:
                if chunk.choices[0].finish_reason == "length":
                    raise ExhaustedContextWindow()

                try:
                    func = chunk.choices[0].delta.function_call

                    for k, v in func.items():
                        if k in self.partial_response_function_call:
                            self.partial_response_function_call[k] += v
                        else:
                            self.partial_response_function_call[k] = v
                except AttributeError:
                    pass

                try:
                    text = chunk.choices[0].delta.content
                    if text:
                        self.partial_response_content += text
                        send_chat_update_wrapper(text)
                except AttributeError:
                    pass

                if silent:
                    continue

        aider.coders.base_coder.Coder.show_send_output_stream = show_send_output_stream

        file_changes: List[file_diff.FileChange] = []
        event = asyncio.Event()
        event2 = asyncio.Event()

        # This is called every time aider writes a file
        # Instead of writing, this stores the file change in a list
        def on_write(filename: str, new_content: str):
            if isinstance(filename, PurePath):
                filename = filename.__fspath__()
            file_change = file_diff.get_file_change(path=filename, new_content=new_content)
            file_changes.append(file_change)
            # print(f"{file_changes=}")

        # This is called when aider wants to commit after writing all the files
        # This is where the user should accept/reject the changes

        async def set_event():
            event.set()

        def on_commit():
            asyncio.run_coroutine_threadsafe(set_event(), loop=loop)
            while True:
                if not event2.is_set():
                    time.sleep(0.25)
                    continue
                break

        aider_finished = False

        def done_cb(fut):
            nonlocal aider_finished
            aider_finished = True
            event.set()

        with futures.ThreadPoolExecutor(1) as pool:
            aider_fut = loop.run_in_executor(
                pool,
                aider.main.main,
                [],
                on_write,
                on_commit,
                None,
                None,
                self.state.params.workspaceFolderPath,
            )
            aider_fut.add_done_callback(done_cb)
            logger.info("Running aider thread")

            while True:
                await event.wait()
                if aider_finished:
                    break
                if len(file_changes) > 0:
                    await self.apply_file_changes(file_changes)
                    file_changes = []
                event2.set()
                event.clear()
            try:
                await aider_fut
            except SystemExit as e:
                logger.info(f"[aider] caught {e}, exiting")
            except Exception as e:
                logger.error(f"[aider] caught {e}, exiting")
            finally:
                await self.send_progress()
