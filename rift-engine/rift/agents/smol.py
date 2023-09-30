from concurrent import futures

import rift.agents.registry as registry
from rift.util.TextStream import TextStream

try:
    import smol_dev
except ImportError:
    raise Exception(
        "`smol_dev` not found. Try `pip install -e 'rift-engine[smol_dev]' from the repository root directory.`"
    )

try:
    smol_dev.__author__
except AttributeError:
    raise Exception(
        'Wrong version of `smol_dev` installed. Please try `pip install -e "rift-engine[smol_dev]" --force-reinstall` from the Rift root directory.'
    )

import asyncio
import json
import logging
import os
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

import rift.llm.openai_types as openai
import rift.lsp.types as lsp
import rift.util.file_diff as file_diff
from rift.agents.abstract import AgentProgress  # AgentTask,
from rift.agents.abstract import (
    Agent,
    AgentParams,
    AgentRunResult,
    AgentState,
    RequestChatRequest,
    ThirdPartyAgent,
)
from rift.server.selection import RangeSet
from rift.util.context import contextual_prompt, extract_uris, resolve_inline_uris

logger = logging.getLogger(__name__)


@dataclass
class SmolRunResult(AgentRunResult):
    ...


# dataclass for representing the progress of the code completion agent
@dataclass
class SmolProgress(AgentProgress):
    response: Optional[str] = None
    thoughts: Optional[str] = None
    textDocument: Optional[lsp.TextDocumentIdentifier] = None
    cursor: Optional[lsp.Position] = None
    additive_ranges: Optional[RangeSet] = None
    negative_ranges: Optional[RangeSet] = None
    ready: bool = False


# dataclass for representing the parameters of the code completion agent
@dataclass
class SmolAgentParams(AgentParams):
    instructionPrompt: Optional[str] = None


# dataclass for representing the state of the code completion agent
@dataclass
class SmolAgentState(AgentState):
    params: SmolAgentParams
    _done: bool = False
    messages: List[openai.Message] = field(default_factory=list)
    response_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@registry.agent(
    agent_description="Quickly generate a workspace with smol_dev.",
    display_name="Smol Developer",
    agent_icon="""\
<svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M5.77183 20.9101L10.0069 23.8254L13.5818 21.1446L18.5284 23.8254L23.0823 21.1446L27.6163 24.1549L31.3134 20.9487C31.7148 23.956 31.026 27.4413 28.6551 29.5937C27.1353 30.9741 25.2454 31.9123 23.2929 32.4199C22.5558 32.6129 20.4524 33.0612 18.4459 33.0612C16.4365 33.0612 14.5267 32.6129 13.781 32.4199C11.8342 31.9093 9.94149 30.9741 8.42163 29.5937C7.22054 28.5012 6.38092 27.1177 5.88853 25.6215C5.37052 24.0421 5.33921 22.5191 5.77183 20.9101ZM5.77183 19.5801C5.62099 19.5801 5.47014 19.6068 5.32498 19.6662C5.13731 19.739 4.96882 19.8572 4.83328 20.0109C4.69775 20.1647 4.59901 20.3497 4.54513 20.5509C4.03851 22.433 4.08405 24.2321 4.68459 26.052C5.26806 27.8302 6.26992 29.4007 7.58485 30.5941C9.17302 32.0429 11.208 33.1176 13.4708 33.7084C14.4527 33.9637 16.3995 34.3882 18.443 34.3882C20.9192 34.3882 23.3328 33.7796 23.6003 33.7084C25.8602 33.1205 27.898 32.0429 29.489 30.5941C31.9453 28.3646 33.098 24.6893 32.5715 20.7646C32.5407 20.5299 32.45 20.3083 32.3088 20.1228C32.1675 19.9374 31.981 19.795 31.7688 19.7107C31.6208 19.6513 31.4643 19.6216 31.3078 19.6216C31.0146 19.6216 30.7271 19.7285 30.4937 19.9274L27.511 22.5162L23.7625 20.0284C23.5548 19.8888 23.3157 19.8205 23.0766 19.8205C22.8603 19.8176 22.6468 19.877 22.4504 19.9898L18.4971 22.3143L14.1652 19.966C13.9802 19.8651 13.7781 19.8176 13.5761 19.8176C13.3142 19.8176 13.0524 19.9007 12.8332 20.064L9.97564 22.2134L6.47484 19.8027C6.26422 19.6573 6.01945 19.5801 5.77183 19.5801Z" fill="#CCCCCC"/>
<path fill-rule="evenodd" clip-rule="evenodd" d="M34.8583 8.81283C34.0643 8.64919 33.3017 9.11097 32.6386 9.51255L32.6385 9.51258C32.5678 9.5554 32.4983 9.59753 32.4299 9.63814C31.2691 10.3328 29.9807 10.7781 28.6774 11.1136C28.6104 11.131 28.5351 11.1401 28.4567 11.1495L28.4567 11.1495L28.4102 11.1552C27.7036 9.28783 26.7685 7.52439 25.7829 6.30721C24.2183 4.37455 22.5855 2.99408 18.9221 3.00002C15.2616 2.99408 13.5665 4.65658 11.9218 6.51502C10.9035 7.66689 10.0129 9.34721 9.35974 11.1433C9.34097 11.1406 9.32114 11.1385 9.30103 11.1363C9.25433 11.1313 9.2062 11.1261 9.16677 11.1136C7.86645 10.7781 6.57802 10.3328 5.41427 9.63814C5.34589 9.59752 5.2763 9.55538 5.20558 9.51255C4.54245 9.11097 3.77989 8.64919 2.98583 8.81283C1.89927 9.04142 1.94677 10.8049 2.05067 11.5827C2.39208 14.1477 3.97145 16.8522 6.2663 18.2208C6.8838 18.5889 7.56067 18.8977 8.26724 19.0847C8.48395 21.2044 9.18161 23.1905 10.7521 24.6986C11.9782 25.8802 13.51 26.6788 15.0835 27.1152C15.6832 27.2785 17.2299 27.6644 18.8479 27.6644C20.4718 27.6644 22.1699 27.2785 22.7666 27.1152C24.343 26.6788 25.869 25.8802 27.098 24.6986C28.6685 23.1905 29.3661 21.2044 29.5858 19.0847C30.2894 18.8977 30.9663 18.5889 31.5838 18.2208C33.8786 16.8522 35.455 14.1477 35.7994 11.5827C35.9004 10.8019 35.9449 9.04142 34.8583 8.81283ZM18.9369 16.6022C18.7232 16.5903 18.5035 16.4983 18.3877 16.3588C18.2501 16.1887 18.1177 16.0116 17.9858 15.8351C17.876 15.6881 17.7664 15.5415 17.6544 15.3999C17.507 15.2143 17.3596 15.0293 17.2124 14.8446L17.2122 14.8443C17.0275 14.6125 16.8433 14.3813 16.6599 14.15C16.628 14.1099 16.5868 14.0694 16.5448 14.0281C16.4655 13.9501 16.3832 13.8692 16.3541 13.7819C16.3155 13.6661 16.3601 13.4999 16.3987 13.396C16.4628 13.1954 16.6421 12.9991 16.8093 12.8161C16.8821 12.7365 16.9526 12.6594 17.0102 12.5855L17.0967 12.4771L17.0967 12.477C17.5811 11.8696 18.0621 11.2663 18.5273 10.6499C18.6371 10.5014 18.7796 10.3471 19.0082 10.4094C19.2601 10.4798 19.4652 10.7774 19.6341 11.0225L19.6341 11.0225C19.6861 11.098 19.7346 11.1685 19.7801 11.2258C19.874 11.3453 19.9675 11.4648 20.0609 11.5842C20.2274 11.7969 20.3935 12.0092 20.5608 12.2203C20.6473 12.3292 20.7329 12.4386 20.8184 12.5478L20.8188 12.5483L20.819 12.5485C20.9595 12.728 21.0998 12.9072 21.2437 13.0842C21.3683 13.2446 21.4812 13.4286 21.4812 13.6305C21.4763 13.8033 21.3757 13.9242 21.2693 14.0521C21.2458 14.0802 21.2222 14.1087 21.1991 14.1382L21.1988 14.1386C20.8545 14.5778 20.5103 15.0171 20.163 15.4533C20.0859 15.5498 20.0064 15.6455 19.927 15.7413C19.8476 15.837 19.7682 15.9328 19.691 16.0292C19.6527 16.0771 19.616 16.1308 19.5785 16.1856C19.4997 16.3008 19.4176 16.4208 19.311 16.5013C19.2101 16.5785 19.0765 16.6082 18.9369 16.6022ZM13.4682 12.2913C14.0204 12.4338 14.6023 12.0093 14.7626 11.3354C14.9259 10.6614 14.6052 9.99645 14.0531 9.85395C13.4979 9.70551 12.919 10.136 12.7557 10.8069C12.5954 11.4808 12.9131 12.1458 13.4682 12.2913ZM24.4496 12.2913C23.8974 12.4338 23.3156 12.0093 23.1552 11.3354C22.992 10.6614 23.3096 9.99645 23.8677 9.85395C24.4199 9.70551 24.9988 10.136 25.1621 10.8069C25.3224 11.4808 25.0018 12.1458 24.4496 12.2913Z" fill="#CCCCCC"/>
</svg>""",
)
@dataclass
class SmolAgent(ThirdPartyAgent):
    state: SmolAgentState
    agent_type: ClassVar[str] = "smol_dev"
    params_cls: ClassVar[Any] = SmolAgentParams

    @classmethod
    async def create(cls, params: SmolAgentParams, server: Any) -> ThirdPartyAgent:
        state = SmolAgentState(
            params=params,
            _done=False,
            messages=[openai.Message.assistant("What do you want to build?")],
        )
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )
        return obj

    async def _run_chat_thread(self, response_stream: str) -> None:
        before, after = response_stream.split_once("感")
        try:
            async with self.state.response_lock:
                async for delta in before:
                    # logger.info(f"{delta=}")
                    self._response_buffer += delta
                    await self.send_progress({"response": self._response_buffer})
            await asyncio.sleep(0.1)
            await self._run_chat_thread(after)
        except Exception as e:
            logger.info(f"[_run_chat_thread] caught exception={e}, exiting")

    async def run(self) -> AgentRunResult:
        await self.send_progress()
        response_stream = TextStream()
        self._response_buffer = ""
        asyncio.create_task(self._run_chat_thread(response_stream))
        loop = asyncio.get_running_loop()

        def send_chat_update_wrapper(prompt: str = "感", end="", eof=False):
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

            async def _worker():
                # logger.info(f"[send_chat_update_wrapper] {prompt=}")
                response_stream.feed_data(prompt)

            return asyncio.run_coroutine_threadsafe(_worker(), loop=loop)

        prompt = await self.request_chat(RequestChatRequest(messages=self.state.messages))
        documents = resolve_inline_uris(prompt, self.server)
        prompt = contextual_prompt(prompt, documents)

        # logger.info(f"{prompt=}")

        self.state.messages.append(openai.Message.user(prompt))  # update messages history

        # # RESPONSE = ""

        async def flush_response_buffer():
            fut = send_chat_update_wrapper()
            waiter = loop.create_future()
            fut.add_done_callback(lambda fut: waiter.set_result(None))
            await waiter
            # logger.info(f"{self._response_buffer=}")
            async with self.state.response_lock:
                self.state.messages.append(openai.Message.assistant(self._response_buffer))
                await self.send_progress(
                    {
                        "response": self._response_buffer,
                        "done_streaming": True,
                        "messages": self.state.messages,
                    }
                )
                self._response_buffer = ""

        with futures.ThreadPoolExecutor(1) as executor:

            async def run_in_executor(fn, *args, **kwargs):
                from functools import partial

                return await loop.run_in_executor(executor, partial(fn, *args, **kwargs))

            async def get_plan():
                await run_in_executor(
                    smol_dev.prompts,
                    plan,
                    prompt,
                    stream_handler=send_chat_update_wrapper,
                    model="gpt-3.5-turbo",
                )

            plan_fut = loop.create_future()
            plan_task = self.add_task(
                "Generate plan",
                run_in_executor,
                [
                    smol_dev.prompts.plan,
                    prompt,
                    send_chat_update_wrapper,
                    "gpt-3.5-turbo",
                ],
            )

            async def get_file_paths_args():
                return [
                    smol_dev.prompts.specify_file_paths,
                    prompt,
                    await plan_fut,
                    "gpt-3.5-turbo",
                ]

            get_file_paths_task = self.add_task(
                "Generate filepaths",
                run_in_executor,
                args=get_file_paths_args,
            )

            await self.send_progress()

            plan = await plan_task.run()
            plan_fut.set_result(plan)
            file_paths = await get_file_paths_task.run()

            await self.send_progress()

            await flush_response_buffer()

            self.state.messages.append(
                openai.Message.assistant(
                    f"\nProposed filepaths:\n```\n{json.dumps(file_paths, indent=2)}\n```"
                    "\n\nWhere should the generated files be placed?\nTry `@`-mentioning a folder in the workspace, or say '.' to use the workspace root.\n"
                )
            )

            generated_code_location_response = await self.request_chat(
                RequestChatRequest(messages=self.state.messages)
            )

            self.state.messages.append(
                openai.Message.user(generated_code_location_response)
            )  # update messages history

            # Parse any URIs from the user's response
            matches = extract_uris(generated_code_location_response)
            if matches:
                parent_dir = matches[0]
                if not os.path.isdir(parent_dir):
                    if not os.path.exists(parent_dir):
                        os.mkdir(parent_dir)
            else:
                # If no URIs are found, default to the workspace folder
                parent_dir = self.state.params.workspaceFolderPath

            # file_changes = []

            send_chat_update_wrapper(f"Generating files in {parent_dir}.\n")
            await asyncio.sleep(0.1)
            await flush_response_buffer()

            @dataclass
            class PBarUpdater:
                pbars: Dict[int, Any] = field(default_factory=dict)
                dones: Dict[int, Any] = field(default_factory=dict)
                messages: Dict[int, Optional[str]] = field(default_factory=dict)
                lock: asyncio.Lock = asyncio.Lock()

                def update(self):
                    for position, pbar in self.pbars.items():
                        if self.dones[position]:
                            pbar.display(self.messages[position])
                        else:
                            pbar.update()

            PBarUpdater()

            logger.info(f"generate code target dir: {parent_dir}")

            async def generate_code_for_filepath(
                file_path: str, position: int
            ) -> file_diff.FileChange:
                code_future = asyncio.ensure_future(
                    smol_dev.generate_code(prompt, plan, file_path, model="gpt-3.5-turbo")
                )
                code = await code_future
                absolute_file_path = os.path.join(parent_dir, file_path)
                file_change = file_diff.get_file_change(path=absolute_file_path, new_content=code)
                return file_path, file_change

            fs = []
            loop = asyncio.get_running_loop()

            @dataclass
            class Counter:
                counter: int = 0

                def increment(self):
                    self.counter += 1

            done_counter = Counter()

            for i, fp in enumerate(file_paths):
                fut = asyncio.create_task(
                    self.add_task(
                        description=f"Generate {fp}",
                        task=generate_code_for_filepath,
                        kwargs=dict(file_path=fp, position=i),
                    ).run()
                )

                fut2 = loop.create_future()

                fut2.set_result(fp)

                def done_cb(fut):
                    fp, _ = fut.result()
                    done_counter.increment()

                    async def coro():
                        send_chat_update_wrapper(
                            f"Finished code generation for {fp}. ({done_counter.counter}/{len(file_paths)})\n"
                        )
                        # await flush_response_buffer()
                        await self.send_progress()

                    asyncio.run_coroutine_threadsafe(coro(), loop=loop)

                fut.add_done_callback(done_cb)
                fs.append(fut)
                send_chat_update_wrapper(f"Generating code for {fp}.\n")

            file_changes: List[file_diff.FileChange] = [
                file_change for _, file_change in await asyncio.gather(*fs)
            ]
            await asyncio.sleep(0.1)
            workspace_edit = file_diff.edits_from_file_changes(file_changes, user_confirmation=True)
            await self.server.apply_workspace_edit(
                lsp.ApplyWorkspaceEditParams(edit=workspace_edit, label="rift")
            )

    async def send_result(self, result: Any) -> None:
        ...  # unreachable
