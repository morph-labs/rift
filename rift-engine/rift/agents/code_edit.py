import asyncio
import logging
from asyncio import Future
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import rift.agents.registry as registry
import rift.llm.openai_types as openai
import rift.lsp.types as lsp
from rift.agents.abstract import AgentProgress  # AgentTask,
from rift.agents.abstract import Agent, AgentParams, AgentRunResult, AgentState, RequestChatRequest
from rift.llm.abstract import AbstractCodeEditProvider
from rift.server.selection import RangeSet
from rift.util.context import resolve_inline_uris
from rift.util.TextStream import TextStream

logger = logging.getLogger(__name__)


# dataclass for representing the result of the code completion agent run
@dataclass
class CodeEditRunResult(AgentRunResult):
    ...


# dataclass for representing the progress of the code completion agent
@dataclass
class CodeEditProgress(AgentProgress):
    response: Optional[str] = None
    thoughts: Optional[str] = None
    textDocument: Optional[lsp.TextDocumentIdentifier] = None
    cursor: Optional[lsp.Position] = None
    additive_ranges: Optional[RangeSet] = None
    negative_ranges: Optional[RangeSet] = None
    ready: bool = False


# dataclass for representing the parameters of the code completion agent
@dataclass
class CodeEditAgentParams(AgentParams):
    ...


# dataclass for representing the state of the code completion agent
@dataclass
class CodeEditAgentState(AgentState):
    model: AbstractCodeEditProvider
    document: lsp.TextDocumentItem
    active_range: lsp.Range
    cursor: lsp.Position
    params: CodeEditAgentParams
    selection: lsp.Selection
    messages: list[openai.Message]
    additive_ranges: RangeSet = field(default_factory=RangeSet)
    negative_ranges: RangeSet = field(default_factory=RangeSet)
    change_futures: Dict[str, Future] = field(default_factory=dict)
    _done: asyncio.Event = field(default_factory=asyncio.Event)


# decorator for creating the code completion agent
@registry.agent(
    agent_description="Generate code edit for currently selected region.",
    display_name="Code Edit",
)
@dataclass
class CodeEditAgent(Agent):
    state: CodeEditAgentState
    agent_type: ClassVar[str] = "code_edit"
    params_cls: ClassVar[Any] = CodeEditAgentParams

    @classmethod
    async def create(cls, params: CodeEditAgentParams, server: Any) -> Agent:
        logger.info(f"{params=}")
        model = await server.ensure_code_edit_model()  # TODO: not right, fix
        state = CodeEditAgentState(
            model=model,
            document=server.documents[params.textDocument.uri],
            active_range=lsp.Range(params.selection.start, params.selection.end),
            cursor=params.selection.second,  # begin at the start of the selection
            additive_ranges=RangeSet(),
            params=params,
            selection=params.selection,
            messages=[openai.Message.assistant("What do you want me to do?")],
        )
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )
        return obj

    async def run(self) -> AgentRunResult:  # main entry point
        try:
            self.DIFF = None
            self.RANGE = None

            async def get_user_response() -> str:
                response_fut = asyncio.create_task(
                    self.request_chat(RequestChatRequest(messages=self.state.messages))
                )

                async def waiter():
                    await self.state._done.wait()
                    return

                waiter_fut = asyncio.create_task(waiter())

                for fut in asyncio.as_completed([response_fut, waiter_fut]):
                    return await fut
                # return await self.request_chat(RequestChatRequest(messages=self.state.messages))

            await self.send_progress()
            self.RANGE = lsp.Range(self.state.selection.first, self.state.selection.second)
            # logger.info(f"{self.RANGE=}")
            with lsp.setdoc(self.state.document):
                urtext = self.state.document.text
                uroffset_start = self.state.document.position_to_offset(self.state.selection.first)
                uroffset_end = self.state.document.position_to_offset(self.state.selection.second)

            while True:
                try:
                    # get the next prompt
                    # logger.info("getting user response")
                    get_user_response_t = self.add_task("Get user response", get_user_response)
                    instructionPrompt: Optional[str] = await get_user_response_t.run()
                    if not instructionPrompt:
                        break
                    documents = resolve_inline_uris(instructionPrompt, self.server)
                    self.server.register_change_callback(self.on_change, self.state.document.uri)
                    from diff_match_patch import diff_match_patch

                    dmp = diff_match_patch()
                    edit_code_result = await self.state.model.edit_code(
                        urtext,
                        uroffset_start,
                        uroffset_end,
                        goal=instructionPrompt,
                        latest_region=None
                        if self.DIFF is None
                        else (self.accepted_diff_text(self.DIFF)),
                        documents=documents,
                    )

                    logger.info("started streaming result")
                    response_stream = TextStream()

                    async def generate_response():
                        response = ""
                        try:
                            async for delta in response_stream:
                                response += delta
                                await self.send_progress(CodeEditProgress(response=response))
                        except Exception as e:
                            logger.info(f"RESPONSE EXCEPTION: {e}")
                            raise e
                        finally:
                            await self.send_progress({"response": response, "done_streaming": True})
                        return response

                    generate_response_t = asyncio.create_task(generate_response())

                    async def gather_thoughts():
                        async for delta in edit_code_result.thoughts:
                            response_stream.feed_data(delta)

                    async def cleanup():
                        response_stream.feed_eof()

                    logger.info("created text stream")

                    all_deltas = []
                    offset_start = self.state.document.position_to_offset(
                        self.state.selection.first
                    )
                    offset_end = self.state.document.position_to_offset(self.state.selection.second)
                    self.selection_text = self.state.document.text[offset_start:offset_end]

                    logger.info("starting to iterate through text stream")
                    self.DIFF = None

                    diff_queue = asyncio.Queue()

                    async def send_diff(new_text: str):
                        max_fuel = 10
                        fuel = max_fuel
                        while True:
                            if self.state._done._value:
                                break
                            if fuel <= 0:
                                raise Exception(f"recursion_limit={max_fuel} reached")
                            try:
                                # diff = dmp.diff_lineMode(self.selection_text, new_text, None)
                                # # dmp.diff_cleanupSemantic(diff)
                                # dmp.diff_cleanupMerge

                                (x, y, linearray) = dmp.diff_linesToChars(
                                    self.selection_text, new_text
                                )

                                diff = dmp.diff_main(x, y, False)

                                # Convert the diff back to original text.
                                dmp.diff_charsToLines(diff, linearray)
                                # Eliminate freak matches (e.g. blank lines)
                                dmp.diff_cleanupSemantic(diff)

                                self.DIFF = diff  # store the latest diff
                                # logger.info(f"{diff=}")
                                diff_text = "".join([text for _, text in diff])
                                if diff_text == self.selection_text:
                                    break
                                # logger.info(f"{diff=}")

                                cf = asyncio.get_running_loop().create_future()
                                self.state.change_futures[diff_text] = cf

                                # logger.info("applying edit")
                                await self.server.apply_range_edit(
                                    self.state.document.uri, self.RANGE, diff_text
                                )
                                # logger.info("applied")

                                def add_pos_text(pos: lsp.Position, text: str):
                                    line_delta = text.count("\n")
                                    if line_delta == 0:
                                        offset = pos.character + len(text)
                                    else:
                                        offset = list(reversed(text)).index("\n")
                                    return lsp.Position(pos.line + line_delta, offset)

                                self.RANGE = lsp.Range(
                                    self.state.selection.first,
                                    add_pos_text(self.state.selection.first, diff_text),
                                )

                                try:
                                    await asyncio.wait_for(cf, timeout=2)
                                    break
                                except asyncio.TimeoutError:
                                    break
                                finally:
                                    del self.state.change_futures[diff_text]
                                    self.state.additive_ranges = RangeSet()
                                    self.state.negative_ranges = RangeSet()
                                    with lsp.setdoc(self.state.document):
                                        cursor = self.state.selection.first
                                        for op, text in diff:
                                            next_cursor = add_pos_text(cursor, text)
                                            if op == -1:  # delete
                                                self.state.negative_ranges.add(
                                                    lsp.Range(cursor, next_cursor)
                                                )
                                            elif op == 0:  # keep
                                                pass
                                            elif op == 1:  # add
                                                self.state.additive_ranges.add(
                                                    lsp.Range(cursor, next_cursor)
                                                )
                                            cursor = next_cursor

                                    progress = CodeEditProgress(
                                        response=None,
                                        textDocument=self.state.document,
                                        cursor=self.state.cursor,
                                        additive_ranges=list(self.state.additive_ranges),
                                        negative_ranges=list(self.state.negative_ranges),
                                    )
                                    await self.send_progress(progress)
                            except Exception as e:
                                logger.info(f"caught {e=} retrying")
                                fuel -= 1

                    async def generate_code():
                        # async def watch_event():
                        #     if edit_code_result.event:
                        #         while True:
                        #             # logger.info(f"{edit_code_result.event=}")
                        #             if edit_code_result.event.is_set():
                        #                 raise Exception(f"[{self.agent_type}] generation failed")
                        #             else:
                        #                 await asyncio.sleep(0.5)
                        #                 continue

                        # asyncio.create_task(watch_event())
                        nonlocal all_deltas
                        # async for substream in edit_code_result.code.asplit("\n"):
                        after = edit_code_result.code
                        line_flag = False

                        async def _watch_queue():
                            while True:
                                x = await diff_queue.get()
                                if x is None:
                                    return
                                else:
                                    await send_diff(x)

                        diff_queue_task = asyncio.create_task(_watch_queue())
                        while True:
                            if after.at_eof():
                                break
                            flag = False
                            before, after = after.split_once("\n")
                            # logger.info("yeehaw")
                            if line_flag:
                                all_deltas.append("\n")
                            async for delta in before:
                                if not flag:
                                    flag = True
                                all_deltas.append(delta)
                            # if not flag:
                            #     break
                            if not line_flag:
                                line_flag = True
                            # logger.info(f"{all_deltas=}")

                            await diff_queue.put("".join(all_deltas))
                        await diff_queue.put(None)
                        await diff_queue_task

                        # asyncio.create_task(send_diff("".join(all_deltas)))

                    await self.add_task("Generate code", generate_code).run()
                    await gather_thoughts()
                    t = asyncio.create_task(cleanup())
                    assistant_response = await generate_response_t
                    await t
                    self.state.messages += [
                        openai.Message.user(content=instructionPrompt),
                        openai.Message.assistant(content=assistant_response),
                    ]

                    await self.send_progress(
                        CodeEditProgress(
                            response=None,
                            textDocument=self.state.document,
                            cursor=self.state.cursor,
                            additive_ranges=list(self.state.additive_ranges),
                            negative_ranges=list(self.state.negative_ranges),
                            ready=True,
                        )
                    )
                    logger.info("sent ready=True")
                finally:
                    self.server.change_callbacks[self.state.document.uri].discard(self.on_change)
            return CodeEditRunResult()
        except asyncio.CancelledError as e:
            logger.info("cancelling")
            try:
                await self.reject()
            except:
                raise e

    async def on_change(
        self,
        *,
        before: lsp.TextDocumentItem,
        after: lsp.TextDocumentItem,
        changes: lsp.DidChangeTextDocumentParams,
    ) -> None:
        if self.task.status != "running":
            return
        """
        [todo]
        When a change happens:
        1. if the change is before our 'working area', then we stop the completion request and run again.
        2. if the change is in our 'working area', then the user is correcting something that
        3. if the change is after our 'working area', then just keep going.
        4. if _we_ caused the change, then just keep going.
        """
        assert changes.textDocument.uri == self.state.document.uri
        self.state.document = before
        for c in changes.contentChanges:
            # logger.info(f"contentChange: {c=}")
            # fut = self.state.change_futures.get(c.text)
            fut = None
            for span, vfut in self.state.change_futures.items():
                if c.text in span:
                    fut = vfut

            if fut is not None:
                # we caused this change
                try:
                    fut.set_result(None)
                except:
                    pass
            else:
                # someone else caused this change
                # [todo], in the below examples, we shouldn't cancel, but instead figure out what changed and restart the insertions with the new information.
                with lsp.setdoc(self.state.document):
                    self.state.additive_ranges.apply_edit(c)
                if c.range is None:
                    await self.cancel("the whole document got replaced")
                else:
                    if c.range.end <= self.state.cursor:
                        # some text was changed before our cursor
                        if c.range.end.line < self.state.cursor.line:
                            # the change is occurring on lines strictly above us
                            # so we can adjust the number of lines
                            lines_to_add = (
                                c.text.count("\n") + c.range.start.line - c.range.end.line
                            )
                            self.state.cursor += (lines_to_add, 0)
                        else:
                            # self.cancel("someone is editing on the same line as us")
                            pass  # temporarily disabled
                    elif self.state.cursor in c.range:
                        await self.cancel("someone is editing the same text as us")

        self.state.document = after

    async def send_result(self, result: Any) -> None:
        ...  # unreachable

    def accepted_diff_text(self, diff: List[Tuple[int, str]]) -> str:
        result = ""
        for op, text in diff:
            if op == -1:  # remove
                pass
            elif op == 0:
                result += text
            elif op == 1:
                result += text
        return result

    async def accept(self) -> None:
        logger.info(f"{self} user accepted result")

        await self.server.apply_range_edit(
            self.state.document.uri, self.RANGE, self.accepted_diff_text(self.DIFF)
        )
        # if self.task.status not in ["error", "done"]:
        #     logger.error(f"cannot accept status {self.task.status}")
        #     return
        # self.status = "done"
        await self.send_progress(
            payload="accepted",
            payload_only=True,
        )
        self.state._done.set()

    def rejected_diff_text(self, diff: List[Tuple[int, str]]) -> str:
        # result = ""
        # for op, text in diff:
        #     if op == -1:  # remove
        #         result += text
        #     elif op == 0:
        #         result += text
        #     elif op == 1:
        #         pass
        # return result
        return self.selection_text

    async def reject(self) -> None:
        logger.info(f"{self} user rejected result")

        await self.server.apply_range_edit(
            self.state.document.uri, self.RANGE, self.rejected_diff_text(self.DIFF)
        )
        await self.send_progress(
            payload="rejected",
            payload_only=True,
        )
        self.state._done.set()
