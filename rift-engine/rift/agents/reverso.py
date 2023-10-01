import asyncio
import logging
import random
from asyncio import Future
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional

import rift.lsp.types as lsp
from rift.agents.abstract import AgentProgress  # AgentTask,
from rift.agents.abstract import Agent, AgentParams, AgentRunResult, AgentState, agent
from rift.server.selection import RangeSet
from rift.util.TextStream import TextStream

logger = logging.getLogger(__name__)


# dataclass for representing the result of the code completion agent run
@dataclass
class ReversoRunResult(AgentRunResult):
    ...


# dataclass for representing the progress of the code completion agent
@dataclass
class ReversoProgress(AgentProgress):
    response: Optional[str] = None
    thoughts: Optional[str] = None
    textDocument: Optional[lsp.TextDocumentIdentifier] = None
    cursor: Optional[lsp.Position] = None
    additive_ranges: Optional[RangeSet] = None
    negative_ranges: Optional[RangeSet] = None


# dataclass for representing the state of the code completion agent
@dataclass
class ReversoAgentState(AgentState):
    document: lsp.TextDocumentItem
    active_range: lsp.Range
    cursor: lsp.Position
    selection: lsp.Selection
    additive_ranges: RangeSet = field(default_factory=RangeSet)
    negative_ranges: RangeSet = field(default_factory=RangeSet)
    change_futures: Dict[str, Future] = field(default_factory=dict)


# decorator for creating the code completion agent
@agent(
    agent_description="Reverse all lines in a region, line by line",
    display_name="Reverso",
)
@dataclass
class ReversoAgent(Agent):
    state: ReversoAgentState
    agent_type: ClassVar[str] = "reverso"

    @classmethod
    async def create(cls, params: Dict[Any, Any], server):
        state = ReversoAgentState(
            document=server.documents[params.textDocument.uri],
            active_range=lsp.Range(params.selection.start, params.selection.end),
            cursor=params.selection.second,  # begin at the start of the selection
            additive_ranges=RangeSet(),
            negative_ranges=RangeSet(),
            params=params,
            selection=params.selection,
        )
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )
        return obj

    async def run(self) -> AgentRunResult:  # main entry point
        self.server.register_change_callback(self.on_change, self.state.document.uri)
        from diff_match_patch import diff_match_patch

        logger.info("in run")
        dmp = diff_match_patch()
        logger.info("hawyee")
        EDIT = """\
def quicksort(nums: List[int]) -> List[int]:
    if len(nums) <= 1:
        return nums
    pivot = nums[len(nums) // 2]
    left = [x for x in nums if x < pivot]
    middle = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
        EDIT = "\n".join("".join(reversed(line)) for line in EDIT.split("\n"))
        logger.info("importe diff_match_patch")

        async def create_dummy_text_stream(msg: str):
            cursor = 0
            while True:
                chunk_size = random.choice(range(5)) + 1
                await asyncio.sleep(0.25)
                yield msg[cursor : cursor + chunk_size]
                cursor += chunk_size
                if cursor >= len(msg):
                    break

        logger.info("defined create_dummy_text_stream")

        logger.info("defined dummy task stream")

        # create stream of new text
        text_stream = TextStream.from_aiter(create_dummy_text_stream(EDIT))

        logger.info("created text stream")

        all_deltas = []
        RANGE = lsp.Range(self.state.selection.first, self.state.selection.second)
        logger.info(f"RANGE BEFORE ITERATION: {RANGE=}")
        # calculate the diff
        offset_start = self.state.document.position_to_offset(self.state.selection.first)
        offset_end = self.state.document.position_to_offset(self.state.selection.second)
        selection_text = self.state.document.text[offset_start:offset_end]
        async for delta in text_stream:
            logger.info(f"DELTA: {delta=}")
            fuel = 10
            while True:
                if fuel <= 0:
                    raise Exception(":(")
                try:
                    logger.info("in main try")
                    # assumption: RANGE is always the range of the last valid selection
                    all_deltas.append(delta)
                    new_text = "".join(all_deltas)

                    # logger.info(f"SELECTION TEXT: {selection_text=}")

                    # calculate diff

                    diff = dmp.diff_lineMode(selection_text, new_text, None)
                    dmp.diff_cleanupSemantic(diff)
                    logger.info(f"{diff=}")
                    diff_text = "".join([text for _, text in diff])

                    logger.info(f"got the diff_text: {diff_text}")

                    # set the stage to update the document and ranges
                    cf = asyncio.get_running_loop().create_future()
                    self.state.change_futures[diff_text] = cf

                    logger.info(f"VALS {RANGE=} {diff_text=}")
                    # refresh the displayed text
                    await self.server.apply_range_edit(self.state.document.uri, RANGE, diff_text)

                    # recalculate our ranges
                    with lsp.setdoc(self.state.document):
                        cursor = self.state.selection.first
                        for op, text in diff:
                            if op == -1:  # delete
                                self.state.negative_ranges.add(
                                    lsp.Range(cursor, cursor + len(text))
                                )
                            elif op == 0:  # keep
                                pass
                            elif op == 1:  # add
                                self.state.additive_ranges.add(
                                    lsp.Range(cursor, cursor + len(text))
                                )
                            cursor += len(text)

                    self.send_progress(
                        ReversoProgress(
                            response=None,
                            textDocument=self.state.document,
                            cursor=self.state.cursor,
                            additive_ranges=self.state.additive_ranges,
                            negative_ranges=self.state.negative_ranges,
                        )
                    )

                    # update doc
                    with lsp.setdoc(self.state.document):
                        RANGE = lsp.Range(
                            self.state.selection.first, self.state.selection.first + len(diff_text)
                        )

                    try:
                        await asyncio.wait_for(cf, timeout=2)
                        break
                    except asyncio.TimeoutError:
                        # [todo] this happens when an edit occured that clobbered this, try redoing.
                        logger.error(f"timeout waiting for change '{diff_text=}', retry the edit")
                    finally:
                        del self.state.change_futures[diff_text]
                except Exception as e:
                    logger.info(f"caught {e=} retrying")
                    fuel -= 1

            # correct the range

            # for op, text in diff:
            #     if op == -1: # negative diff
            #         self.state.negative_ranges.
            #     elif op == 0:
            #         ...
            #     elif op == 1:
            #         ...

        # logger.info("yeehaw")
        # offset_start = self.state.document.position_to_offset(self.state.selection.first)
        # offset_end = self.state.document.position_to_offset(self.state.selection.second)
        # selection_text = self.state.document.text[offset_start:offset_end]
        # logger.info("got selection")
        # logger.info("got new edit")

        # logger.info("imported diff match patch")
        # diff = dmp.diff_lineMode(selection_text, EDIT, None)
        # dmp.diff_cleanupSemantic(diff)

        # # for op, text in diff:
        # #     if op == -1: # remove
        # #         self.state.negative_ranges.add(Range)
        # #     elif op == 0: # leave
        # #         ...
        # #     elif op == 1: # add
        # #         ...

        # # logger.info(f"DIFF: {diff}")
        # # logger.info("got diffs")
        # final_edit = "".join(text for op, text in diff)
        # # logger.info(f"GOT IFNAL EDIT: {final_edit}")

        # await self.server.apply_range_edit(
        #     self.state.document.uri,
        #     lsp.Range(self.state.selection.first, self.state.selection.second),
        #     final_edit,
        # )
        # logger.info("done")
        """
really want something like
        async for delta in stream:
            all_deltas += delta
            all_text = ''.join(all_deltas)

            # calculate the diff
            # refresh the entire selection
            # request document from editor
            # find match for the entire selection after the anchor point
            # calculate the selection
            # recalculate the pos/neg ranges
            # update client
        """

        # async def reverseline(pos: lsp.Position, parity=True):
        #     offset_start = self.state.document.position_to_offset(lsp.Position(line=pos.line, character=0))
        #     offset_end = self.state.document.position_to_offset(lsp.Position(line=pos.line+1, character=0)) - 1
        #     line = self.state.document.text[offset_start:offset_end]
        #     # logger.info(f"LINE: {line}")
        #     await self.send_progress(
        #         ReversoProgress(
        #             response=None,
        #             textDocument=self.state.document,
        #             cursor=self.state.cursor,
        #             additive_ranges=self.state.additive_ranges,
        #             negative_ranges=self.state.negative_ranges,
        #         )
        #     )
        #     output_line = ''.join(reversed(line)) if parity else line

        #     await self.server.apply_range_edit(
        #         self.state.document.uri,
        #         range=lsp.Range.mk(pos.line, 0, pos.line+1, 0),
        #         text=output_line + "\n"
        #     )

        # # async def reverse_line():
        # #     """
        # #     reads the line, clears the line, then re-emits the line in reverse
        # #     """
        # #     self.state.document.text
        # self.state.additive_ranges.add(lsp.Range(self.state.selection.first, self.state.selection.second))
        # await self.send_progress(
        #     ReversoProgress(
        #         response=None,
        #         textDocument=self.state.document,
        #         cursor=self.state.cursor,
        #         additive_ranges=self.state.additive_ranges,
        #         negative_ranges=self.state.negative_ranges,
        #     )
        # )
        # async def reverse_region(parity: bool):
        #     for line_idx in range(self.state.selection.first.line, self.state.selection.second.line + 1):
        #         await reverseline(lsp.Position(line=line_idx, character=0), parity=parity)
        #         # await self.add_task(AgentTask(description="readline", task=reverseline, args=readline_args)).run()

        # async def spawn_task(parity: bool):
        #     def cb(fut: asyncio.Task, parity):
        #         if not fut.cancelled():
        #             asyncio.create_task(spawn_task(parity))
        #     await asyncio.sleep(1.0)
        #     async def parity_fn():
        #         return [parity]

        #     t = self.add_task(AgentTask(
        #         description="Reverse region",
        #         task=reverse_region,
        #         args=parity_fn,
        #         done_callback=functools.partial(cb, parity=not parity)
        #     ))
        #     await t.run()
        # await spawn_task(True)
        # await self.add_task(description="Reverse region", task=spawn_task, args=[True]).run()

        # await self.send_progress()
        # instructionPrompt = self.state.params.instructionPrompt or (
        #     await self.request_input(
        #         RequestInputRequest(
        #             msg="Describe what you want me to do",
        #             place_holder="Please implement the rest of this function",
        #         )
        #     )
        # )

        # self.server.register_change_callback(self.on_change, self.state.document.uri)
        # stream = await self.state.model.edit_code(
        #     self.state.document.text,
        #     self.state.document.position_to_offset(self.state.cursor),
        #     self.state.document.position_to_offset(self.state.cursor),
        #     goal=instructionPrompt,
        # )

        # # function to asynchronously generate the plan
        # async def generate_explanation():
        #     all_deltas = []

        #     if stream.thoughts is not None:
        #         async for delta in stream.thoughts:
        #             all_deltas.append(delta)
        #             await asyncio.sleep(0.01)

        #     await self.send_progress()
        #     return "".join(all_deltas)

        # # function to asynchronously generate the code
        # async def generate_code():
        #     try:
        #         all_deltas = []
        #         self.state.additive_ranges.add(lsp.Range(self.state.selection.first, self.state.selection.second))
        #         # ACTIVE_RANGE = self.state.additive_ranges.cover()
        #         async for delta in stream.code:
        #             all_deltas.append(delta)
        #             all_text = "".join(all_deltas)
        #             # self.state.additive_ranges.add(lsp.Range(self.state.cursor, self.state.cursor))
        #             RANGE = self.state.additive_ranges.cover()
        #             assert len(delta) > 0
        #             attempts = 10
        #             while True:
        #                 if attempts <= 0:
        #                     logger.error(f"too many edit attempts for '{delta}' dropped")
        #                     return
        #                 attempts -= 1
        #                 cf = asyncio.get_running_loop().create_future()
        #                 self.state.change_futures[delta] = cf
        #                 # logger.info(f"{ACTIVE_RANGE=}")
        #                 x = await self.server.apply_range_edit(
        #                     self.state.document.uri,
        #                     RANGE,
        #                     "".join(all_deltas),
        #                     self.state.document.version,
        #                 )
        #                 if x.applied == False:
        #                     logger.info(f"edit '{delta}' failed, retrying")
        #                     await asyncio.sleep(0.1)
        #                     continue
        #                 logger.info(f"applied edit for {delta=}")
        #                 break
        #                 try:
        #                     await asyncio.wait_for(cf, timeout=2)
        #                     break
        #                 except asyncio.TimeoutError:
        #                     # [todo] this happens when an edit occured that clobbered this, try redoing.
        #                     logger.error(f"timeout waiting for change '{delta}', retry the edit")
        #                 finally:
        #                     del self.state.change_futures[delta]

        #             with lsp.setdoc(self.state.document):
        #                 added_range = lsp.Range.of_pos(self.state.cursor, len(delta))
        #                 logger.info(f"{all_text=}")
        #                 logger.info(f"{self.state.selection.first=}")
        #                 # ACTIVE_RANGE = lsp.Range.of_pos(self.state.selection.first, len(all_text))
        #                 # logger.info(f"YEEHAW {[lsp.Range.of_pos(self.state.selection.first, k) for k in range(1, 10)]}")
        #                 # logger.info(f"{ACTIVE_RANGE=}")
        #                 self.state.additive_ranges.add(added_range)
        #                 self.state.cursor += len(delta)
        #                 # self.state.additive_ranges.add(added_range)
        #                 # send progress here because VSCode highlighting is triggered by the range
        #                 await self.send_progress(
        #                     ReversoProgress(
        #                         response=None,
        #                         textDocument=self.state.document,
        #                         cursor=self.state.cursor,
        #                         additive_ranges=self.state.additive_ranges,
        #                         negative_ranges=self.state.negative_ranges,
        #                     )
        #                 )
        #         all_text = "".join(all_deltas)
        #         logger.info(f"{self} finished streaming {len(all_text)} characters")
        #         await self.send_progress()
        #         return all_text

        #     except asyncio.CancelledError as e:
        #         logger.info(f"{self} cancelled: {e}")
        #         await self.cancel()
        #         return ReversoRunResult()

        #     except Exception as e:
        #         logger.exception("worker failed")
        #         # self.status = "error"
        #         return ReversoRunResult()

        #     finally:
        #         # self.server.change_callbacks[self.state.document.uri].discard(self.on_change)
        #         await self.send_progress(
        #             ReversoProgress(
        #                 response=None,
        #                 textDocument=self.state.document,
        #                 cursor=self.state.cursor,
        #                 additive_ranges=self.state.additive_ranges,
        #                 negative_ranges=self.state.negative_ranges,
        #             )
        #         )

        # await self.send_progress(
        #     ReversoProgress(
        #         response=None,
        #         textDocument=self.state.document,
        #         cursor=self.state.cursor,
        #         additive_ranges=self.state.additive_ranges,
        #         negative_ranges=self.state.negative_ranges,
        #     )
        # )

        # code_task = self.add_task(AgentTask("Generate code", generate_code))

        # await self.send_progress(
        #     ReversoProgress(
        #         response=None,
        #         textDocument=self.state.document,
        #         cursor=self.state.cursor,
        #         additive_ranges=self.state.additive_ranges,
        #         negative_ranges=self.state.negative_ranges,
        #     )
        # )

        # explanation_task = self.add_task(AgentTask("Explain code edit", generate_explanation))

        # await self.send_progress(
        #     ReversoProgress(
        #         response=None,
        #         textDocument=self.state.document,
        #         cursor=self.state.cursor,
        #         additive_ranges=self.state.additive_ranges,
        #         negative_ranges=self.state.negative_ranges,
        #     )
        # )

        # await code_task.run()
        # await self.send_progress()

        # explanation = await explanation_task.run()
        # await self.send_progress()

        # await self.send_update(explanation)

        return ReversoRunResult()

    async def on_change(
        self,
        *,
        before: lsp.TextDocumentItem,
        after: lsp.TextDocumentItem,
        changes: lsp.DidChangeTextDocumentParams,
    ):
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
                fut.set_result(None)
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

    async def send_result(self, result):
        ...  # unreachable

    async def accept(self):
        logger.info(f"{self} user accepted result")
        if self.task.status not in ["error", "done"]:
            logger.error(f"cannot accept status {self.task.status}")
            return
        # self.status = "done"
        await self.send_progress(
            payload="accepted",
            payload_only=True,
        )

    async def reject(self):
        # [todo] in this case we need to revert all of the changes that we made.
        logger.info(f"{self} user rejected result")
        # self.status = "done"
        with lsp.setdoc(self.state.document):
            if self.state.additive_ranges.is_empty:
                logger.error("no ranges to reject")
            else:
                edit = lsp.TextEdit(self.state.additive_ranges.cover(), "")
                params = lsp.ApplyWorkspaceEditParams(
                    edit=lsp.WorkspaceEdit(
                        documentChanges=[
                            lsp.TextDocumentEdit(
                                textDocument=self.state.document.id,
                                edits=[edit],
                            )
                        ]
                    )
                )
                x = await self.server.apply_workspace_edit(params)
                if not x.applied:
                    logger.error("failed to apply rejection edit")
            await self.send_progress(
                payload="rejected",
                payload_only=True,
            )
