from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from asyncio import Future
from rift.lsp import LspServer as BaseLspServer
from rift.agents.abstract import AgentState, AgentTask, AgentRunParams, AgentRunResult, Agent
from rift.llm.abstract import AbstractCodeCompletionProvider
from rift.lsp.document import TextDocumentItem
import rift.lsp.types as lsp
from rift.lsp.document import Position
from rift.server.selection import RangeSet
import asyncio


@dataclass
class CodeCompletionAgentState(AgentState):
    model: AbstractCodeCompletionProvider
    change_futures: Dict[str, Future] = field(default_factory=dict)
    document: lsp.TextDocumentItem
    cursor: Position
    ranges: RangeSet


@dataclass
class CodeCompletionAgent(Agent):
    id: int
    state: CodeCompletionAgentState
    tasks: Dict[str, AgentTask]
    server: BaseLspServer
    count: int = 0
    agent_type: str = "code_completion"

    @classmethod
    def create(cls, messages, server):
        CodeCompletionAgent.count += 1
        obj = CodeCompletionAgent(
            state=CodeCompletionAgentState(messages=messages),
            tasks=dict(),
            server=server,
            id=ChatAgent.count,
        )
        obj.generate_plan_task = obj.tasks[
            obj.add_task(AgentTask("running", "Plan code edits", [], None))
        ]
        obj.generate_code_tasks = obj.tasks[
            obj.add_task(AgentTask("running", "Generate code", [], None))
        ]
        obj._task = None
        return obj

    @property
    def task(self):
        # if self.active_task_id is None:
        #     return None
        # else:
        #     return self.tasks.get(self.active_task_id)
        return self._task

    def __str__(self):
        return f"<{type(self).__name__} {self.id}>"

    async def run(self, params: AgentRunParams) -> AgentRunResult:
        async def worker():
            try:
                self.server.register_change_callback(self.on_change, self.uri)
                model = self.state.model
                pos = self.state.cursor
                offset = self.state.document.position_to_offset(pos)
                doc_text = self.state.document.text

                stream: InsertCodeResult = await model.insert_code(
                    doc_text, offset, goal=params.task
                )
                logger.debug("starting streaming code")
                all_deltas = []
                async for delta in stream.code:
                    all_deltas.append(delta)
                    assert len(delta) > 0
                    attempts = 10
                    while True:
                        if attempts <= 0:
                            logger.error(f"too many edit attempts for '{delta}' dropped")
                            return
                        attempts -= 1
                        cf = asyncio.get_running_loop().create_future()
                        self.state.change_futures[delta] = cf
                        x = await self.server.apply_insert_text(
                            self.state.document.uri,
                            self.state.cursor,
                            delta,
                            self.state.document.version,
                        )
                        if x.applied == False:
                            logger.debug(f"edit '{delta}' failed, retrying")
                            await asyncio.sleep(0.1)
                            continue
                        try:
                            await asyncio.wait_for(cf, timeout=2)
                            break
                        except asyncio.TimeoutError:
                            # [todo] this happens when an edit occured that clobbers this, try redoing.
                            logger.error(f"timeout waiting for change '{delta}', retry the edit")
                        finally:
                            del self.state.change_futures[delta]
                    added_range = lsp.Range.of_pos(self.state.cursor, len(delta))
                    self.state.cursor += len(delta)
                    self.ranges.add(added_range)
                    await self.send_progress()
                all_text = "".join(all_deltas)
                logger.info(f"{self} finished streaming {len(all_text)} characters")
                if stream.thoughts is not None:
                    thoughts = await stream.thoughts.read()
                    return AgentRunResult(thoughts)
                else:
                    thoughts = "done!"
                await self.send_progress()
                return AgentRunResult(thoughts)

            except asyncio.CancelledError as e:
                logger.info(f"{self} cancelled: {e}")
                self.status = Status.error
                return AgentRunResult("Task cancelled")

            except Exception as e:
                logger.exception("worker failed")
                self.status = Status.error
                return AgentRunResult("Task failed")

            finally:
                self.task = None
                self.server.change_callbacks[self.state.document.uri].discard(self.on_change)
                await self.send_progress()

        t = asyncio.Task(worker())
        self._task = t
        await t

    async def on_change(
        self,
        *,
        before: lsp.TextDocumentItem,
        after: lsp.TextDocumentItem,
        changes: lsp.DidChangeTextDocumentParams,
    ):
        if self.status != "running":
            return
        """
        [todo]
        When a change happens:
        1. if the change is before our 'working area', then we stop the completion request and run again.
        2. if the change is in our 'working area', then the user is correcting something that
        3. if the change is after our 'working area', then just keep going.
        4. if _we_ caused the chagne, then just keep going.
        """
        assert changes.textDocument.uri == self.state.document.uri
        self.state.document = before
        for c in changes.contentChanges:
            fut = self.state.change_futures.get(c.text)
            if fut is not None:
                # we caused this change
                fut.set_result(None)
            else:
                # someone else caused this change
                # [todo], in the below examples, we shouldn't cancel, but instead figure out what changed and restart the insertions with the new information.
                with setdoc(self.state.document):
                    self.ranges.apply_edit(c)
                if c.range is None:
                    self.cancel("the whole document got replaced")
                else:
                    if c.range.end <= self.state.cursor:
                        # some text was changed before our cursor
                        if c.range.end.line < self.state.cursor.line:
                            # the change is occuring on lines strictly above us
                            # so we can adjust the number of lines
                            lines_to_add = (
                                c.text.count("\n") + c.range.start.line - c.range.end.line
                            )
                            self.state.cursor += (lines_to_add, 0)
                        else:
                            # self.cancel("someone is editing on the same line as us")
                            pass  # temporarily disable
                    elif self.state.cursor in c.range:
                        self.cancel("someone is editing the same text as us")

        self.state.document = after

    def add_task(self, task: AgentTask):
        self.tasks[task.id] = task
        return task.id

    def cancel(self, msg):
        logger.info(f"{self} cancel run: {msg}")
        self._task.cancel()
        # logger.info(f"{self} cancel run: {msg}")
        # for _, task in self.tasks.items():
        #     if task is not None:
        #         task.cancel()

    async def request_input(self, request_input_request):
        return await self.server.request(
            f"morph/{self.agent_type}_{self.id}_request_input", request_input_request
        )

    async def request_chat(self, request_chat_request):
        return await self.server.request(
            f"morph/{self.agent_type}_{self.id}_request_chat", request_chat_request
        )

    async def send_progress(self, progress):
        await self.notify("morph/{self.agent_type}_{self.id}_send_progress", progress)

    async def send_result(self, result):
        ...  # unreachable
