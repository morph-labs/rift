from concurrent import futures

import rift.agents.registry as registry

try:
    import gpt_engineer
    import gpt_engineer.chat_to_files
    import gpt_engineer.db
    from gpt_engineer.ai import AI, fallback_model
    from gpt_engineer.collect import collect_learnings
    from gpt_engineer.db import DB, DBs, archive
    from gpt_engineer.learning import collect_consent
    from gpt_engineer.steps import STEPS
    from gpt_engineer.steps import Config as StepsConfig

except ImportError:
    raise Exception(
        '`gpt_engineer` not found. Try `pip install -e "rift-engine[gpt-engineer]"` from the repository root directory.'
    )

try:
    gpt_engineer.__author__
except AttributeError:
    raise Exception(
        'Wrong version of `gpt-engineer` installed. Please try `pip install -e "rift-engine[gpt-engineer]" --force-reinstall` from the Rift root directory.'
    )


import asyncio
import functools
import logging
import os
import re
import time
from asyncio import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import typer

import rift.lsp.types as lsp
from rift.agents.abstract import AgentProgress  # AgentTask,
from rift.agents.abstract import (
    Agent,
    AgentParams,
    AgentRunResult,
    AgentState,
    RequestChatRequest,
    ThirdPartyAgent,
)
from rift.util import file_diff
from rift.util.context import contextual_prompt, resolve_inline_uris
from rift.util.TextStream import TextStream

STEPS_AGENT_TASKS_NAME_QUEUE = asyncio.Queue()
STEPS_AGENT_TASKS_EVENT_QUEUE = asyncio.Queue()

SEEN = set()

import json

import rift.llm.openai_types as openai

logger = logging.getLogger(__name__)


def __fix_windows_path(path: str) -> str:
    """
    Replace a windows path represented as "/c%3A"... with "c:"...

    :param path: Original path
    :return: Usable windows path, or original path if not a windows path
    """
    pattern = r"^/(.)%3A"

    match = re.match(pattern, path)

    if match:
        drive_letter = match.group(1)
        return path.replace(f"/{drive_letter}%3A", f"{drive_letter}:")
    else:
        return path


def _fix_windows_path(path: str) -> str:
    """
    Replace a windows path represented as "/c%3A"... with "c:"...

    :param path: Original path
    :return: Usable windows path, or original path if not a windows path
    """
    pattern = r"^/(.)%3A"

    match = re.match(pattern, path)

    if match:
        drive_letter = match.group(1)
        return path.replace(f"/{drive_letter}%3A", f"{drive_letter}:")
    else:
        return path


response_lock = asyncio.Lock()


# dataclass for representing the result of the code completion agent run
@dataclass
class EngineerRunResult(AgentRunResult):
    ...


@dataclass
class EngineerAgentParams(AgentParams):
    instructionPrompt: Optional[str] = None


@dataclass
class EngineerProgress(
    AgentProgress
):  # reports what tasks are active and responsible for reporting new tasks
    response: Optional[str] = None
    done_streaming: bool = False


@dataclass
class EngineerAgentState(AgentState):
    params: EngineerAgentParams
    messages: list[openai.Message]
    change_futures: Dict[str, Future] = field(default_factory=dict)
    _done: bool = False


# decorator for creating the code completion agent
@registry.agent(
    agent_description="Specify what you want it to build, the AI asks for clarification, and then builds it.",
    display_name="GPT Engineer",
    agent_icon="""\
<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M14.6245 4.13419C14.5656 3.89656 14.2682 3.81603 14.0951 3.98919L12.1382 5.94603L10.3519 5.6484L10.0543 3.86209L12.0111 1.90525C12.1853 1.73104 12.1014 1.4342 11.8622 1.37472C10.6153 1.06578 9.24244 1.39867 8.26797 2.37288C7.22481 3.41604 6.93771 4.92814 7.37192 6.24656L1.75641 11.8621C1.09878 12.5197 1.09878 13.586 1.75641 14.2436C2.41404 14.9013 3.48035 14.9013 4.13798 14.2436L9.74875 8.63287C11.0677 9.0726 12.5769 8.78234 13.6269 7.73234C14.6024 6.75682 14.9348 5.38182 14.6245 4.13419ZM2.94746 13.6842C2.59877 13.6842 2.31588 13.4013 2.31588 13.0526C2.31588 12.7036 2.59877 12.421 2.94746 12.421C3.29614 12.421 3.57903 12.7036 3.57903 13.0526C3.57903 13.4013 3.29614 13.6842 2.94746 13.6842Z" fill="#CCCCCC"/>
</svg>""",
)
@dataclass
class EngineerAgent(ThirdPartyAgent):
    state: EngineerAgentState
    agent_type: ClassVar[str] = "engineer"
    params_cls: ClassVar[Any] = EngineerAgentParams

    async def _main(
        self,
        prompt: Optional[str] = None,
        project_path: str = "",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        steps_config: Any = None,
        verbose: bool = typer.Option(False, "--verbose", "-v"),
        **kwargs,
    ) -> None:
        """
        Main function for the EngineerAgent. It initializes the AI model and starts the engineering process.

        :param prompt: The initial prompt for the AI.
        :param project_path: The path to the project directory.
        :param model: The AI model to use.
        :param temperature: The temperature for the AI model's output.
        :param steps_config: The configuration for the engineering steps.
        :param verbose: Whether to output verbose logs.
        :param kwargs: Additional parameters.
        """
        loop = asyncio.get_event_loop()

        request_chat_event = asyncio.Event()

        def send_chat_update_wrapper(prompt: str = "感", end="", sync=False):
            async def _worker():
                # logger.info(f"_worker {sync=}")
                if not sync:
                    self.response_stream.feed_data(prompt)
                # logger.info("fed data")
                if sync or prompt == "感":
                    self.response_stream.feed_data("感")
                    # logger.info("with sync")
                    async with response_lock:
                        request_chat_event.set()
                        # logger.info("acquired lock")
                        if self.RESPONSE:
                            self.state.messages.append(
                                openai.Message.assistant(content=self.RESPONSE)
                            )
                        if prompt and prompt != "感":
                            self.state.messages.append(openai.Message.assistant(prompt))
                        await self.send_progress(
                            dict(
                                done_streaming=True,
                                **{"response": None if not self.RESPONSE else self.RESPONSE},
                                messages=self.state.messages,
                            )
                        )
                        # logger.info("done streaming")
                        # logger.info(f"{self.state.messages=}")
                        self.RESPONSE = ""
                    await asyncio.sleep(0.1)

            fut = asyncio.run_coroutine_threadsafe(_worker(), loop)
            # futures.wait([fut])

        async def request_chat(prompt=""):
            async with response_lock:
                await request_chat_event.wait()
                if self.RESPONSE:
                    self.state.messages.append(openai.Message.assistant(content=self.RESPONSE))
                    await self.send_progress(dict(response=self.RESPONSE))
                if prompt:
                    self.state.messages.append(openai.Message.assistant(prompt))

                if self.RESPONSE:
                    await self.send_progress(
                        dict(done_streaming=True, messages=self.state.messages)
                    )

                    self.RESPONSE = ""
            request_chat_event.clear()
            return await self.request_chat(RequestChatRequest(messages=self.state.messages))

        def request_chat_wrapper(prompt="", loop=None):
            asyncio.set_event_loop(loop)
            fut = asyncio.run_coroutine_threadsafe(request_chat(prompt), loop)
            futures.wait([fut])
            return fut.result()

        _colored = lambda x, y: x
        gpt_engineer.ai.print = send_chat_update_wrapper
        gpt_engineer.steps.colored = _colored
        gpt_engineer.steps.print = functools.partial(send_chat_update_wrapper, sync=True)
        gpt_engineer.steps.input = functools.partial(request_chat_wrapper, loop=loop)
        gpt_engineer.learning.input = functools.partial(request_chat_wrapper, loop=loop)
        gpt_engineer.learning.colored = _colored
        gpt_engineer.learning.print = functools.partial(send_chat_update_wrapper, sync=True)
        # TODO: more coverage
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        model = fallback_model(model)
        ai = AI(
            model=model,
            temperature=temperature,
        )

        input_path = _fix_windows_path(project_path)

        gpteng_path = os.path.join(input_path, ".gpteng")

        if not os.path.exists(gpteng_path):
            os.makedirs(gpteng_path)

        if prompt:
            with open(os.path.join(input_path, "prompt"), "w") as f:
                f.write(prompt)

        memory_path = os.path.join(gpteng_path, "memory")
        workspace_path = os.path.join(input_path)  # pipe files directly into the workspace
        archive_path = os.path.join(gpteng_path, "archive")

        dbs = DBs(
            memory=DB(memory_path),
            logs=DB(os.path.join(memory_path, "logs")),
            input=DB(workspace_path),
            workspace=DB(workspace_path, in_memory_dict={}),  # in_memory_dict={}),
            preprompts=DB(Path(gpt_engineer.__file__).parent / "preprompts"),
            archive=DB(archive_path),
        )

        steps_config = StepsConfig.DEFAULT

        steps = STEPS[steps_config][:-1]  # TODO: restore after debugging

        step_events: Dict[int, asyncio.Event] = dict()
        for i, step in enumerate(steps):
            event = asyncio.Event()
            step_events[i] = event

            async def _step_task(event: asyncio.Event):
                await event.wait()

            _ = asyncio.create_task(
                self.add_task(description=step.__name__, task=_step_task, args=[event]).run()
            )

        counter = 0
        with futures.ThreadPoolExecutor(1) as pool:
            for i, step in enumerate(steps):
                await asyncio.sleep(0.1)
                messages = await loop.run_in_executor(pool, step, ai, dbs)
                await asyncio.sleep(0.1)
                dbs.logs[step.__name__] = json.dumps(messages)
                items = list(dbs.workspace.in_memory_dict.items())
                updates = [x for x in items if x[0] not in SEEN]
                if len(updates) > 0:
                    # for file_path, new_contents in updates:
                    await self.server.apply_workspace_edit(
                        lsp.ApplyWorkspaceEditParams(
                            file_diff.edits_from_file_changes(
                                [
                                    file_diff.get_file_change(file_path, new_contents)
                                    for file_path, new_contents in updates
                                ],
                                user_confirmation=True,
                            )
                        )
                    )
                    for x in items:
                        if x[0] in SEEN:
                            pass
                        else:
                            SEEN.add(x[0])

                step_events[i].set()
                await asyncio.sleep(0.5)
                counter += 1

    async def _run_chat_thread(self, response_stream: str) -> None:
        # logger.info("Started handler thread")
        before, after = response_stream.split_once("感")
        try:
            async with response_lock:
                async for delta in before:
                    self.RESPONSE += delta
                    await self.send_progress(dict(response=self.RESPONSE))
            await self._run_chat_thread(after)
        except Exception as e:
            logger.info(f"[_run_chat_thread] caught exception={e}, exiting")

    @classmethod
    async def create(cls, params: EngineerAgentParams, server: Any) -> ThirdPartyAgent:
        state = EngineerAgentState(
            params=params,
            messages=[openai.Message.assistant("What do you want to build?")],
        )
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )

        return obj

    async def run(self) -> AgentRunResult:  # main entry point
        self.RESPONSE = ""
        self.response_stream = TextStream()
        await self.send_progress()
        asyncio.create_task(self._run_chat_thread(self.response_stream))

        async def get_prompt():
            prompt = await self.request_chat(RequestChatRequest(messages=self.state.messages))
            self.state.messages.append(openai.Message.user(prompt))
            return prompt

        get_prompt_task = self.add_task("Get prompt for workspace", get_prompt)
        await self.send_progress()
        prompt = await get_prompt_task.run()

        documents = resolve_inline_uris(prompt, self.server)
        prompt = contextual_prompt(prompt, documents)

        await asyncio.create_task(
            self._main(prompt=prompt, project_path=self.state.params.workspaceFolderPath)
        )
