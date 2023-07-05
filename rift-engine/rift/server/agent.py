import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, ClassVar, Optional, List
from typing import Literal
from rift.lsp.document import setdoc
import rift.lsp.types as lsp
import importlib.util
from rift.llm.abstract import AbstractCodeCompletionProvider, InsertCodeResult
from rift.server.selection import RangeSet
from rift.agents.abstract import Status

logger = logging.getLogger(__name__)


@dataclass
class CodeCompletionAgentParams:
    task: str
    textDocument: lsp.TextDocumentIdentifier
    position: 


@dataclass
class RunAgentParams:
    agent_type: str
    agent_params: Any
    

@dataclass
class AgentIdParams:
    id: int


class CodeCompletionAgent:
    count: ClassVar[int] = 0
    id: int
    cfg: CodeCompletionAgentParams
    status: Status
    server: Any
    change_futures: dict[str, asyncio.Future[None]]
    cursor: lsp.Position
    ranges: RangeSet
    model: AbstractCodeCompletionProvider
    """ All of the ranges that we have inserted. """
    task: Optional[asyncio.Task]
    """ Worker task. (running self.worker()) """
    subtasks: set[asyncio.Task]
    document: lsp.TextDocumentItem
    """ The position of the cursor (where text will be inserted next).
    This position is changed if other edits occur above the cursor. """

    @property
    def uri(self):
        return self.cfg.textDocument.uri

    def __init__(self, cfg: CodeCompletionAgentParams, model: AbstractCodeCompletionProvider, server: Any):
        CodeCompletionAgent.count += 1
        self.id = CodeCompletionAgent.count
        self.cfg = cfg
        self.model = model
        self.server = server
        self.ranges = RangeSet()
        self.status = Status.running
        self.change_futures = {}
        self.cursor = cfg.position
        document = server.documents.get(self.cfg.textDocument.uri, None)
        if document is None:
            available_docs = "\n".join(server.documents.keys())
            # [note] this can happen when the model is still initializing and the model
            # queues up a request before the textDocument/didOpen notification is sent.
            raise LookupError(
                f"document {self.cfg.textDocument.uri} not found in\n{available_docs}"
            )
        self.document = document
        self.task = None
        self.subtasks = set()

    def cancel(self, msg="cancelled"):
        logger.info(f"{self} cancel run: {msg}")
        if self.task is not None:
            self.task.cancel(msg)

    async def accept(self):
        logger.info(f"{self} user accepted result")
        if self.status not in [Status.error, Status.done]:
            logger.error(f"cannot accept status {self.status}")
            return
        self.status = Status.accepted
        await self.send_progress()

    async def reject(self):
        # [todo] in this case we need to revert all of the changes that we made.
        logger.info(f"{self} user rejected result")
        self.status = Status.rejected
        with setdoc(self.document):
            if self.ranges.is_empty:
                logger.error("no ranges to reject")
            else:
                edit = lsp.TextEdit(self.ranges.cover(), "")
                params = lsp.ApplyWorkspaceEditParams(
                    edit=lsp.WorkspaceEdit(
                        documentChanges=[
                            lsp.TextDocumentEdit(
                                textDocument=self.document.id,
                                edits=[edit],
                            )
                        ]
                    )
                )
                x = await self.server.apply_workspace_edit(params)
                if not x.applied:
                    logger.error("failed to apply rejection edit")
            await self.send_progress()

    @property
    def running(self):
        return self.task is not None and not self.task.done()

    def run(self) -> asyncio.Task:
        if self.running:
            logger.error("already running")
            assert self.task is not None
            return self.task
        self.task = asyncio.create_task(self._worker())
        return self.task

    async def send_progress(self, message: Optional[str] = None):
        await self.server.send_agent_progress(
            self.id,
            textDocument=self.document.id,
            cursor=self.cursor,
            ranges=self.ranges,
            status=self.status.value,
        )

    def __str__(self):
        return f"<{type(self).__name__} {self.id}>"

    async def _worker(self):
        try:
            self.server.register_change_callback(self.on_change, self.uri)
            await self._worker_core()
        except asyncio.CancelledError as e:
            logger.info(f"{self} cancelled: {e}")
            self.status = Status.error
        except Exception as e:
            logger.exception("worker failed")
            self.status = Status.error
        else:
            self.status = Status.done
        finally:
            self.task = None
            self.server.change_callbacks[self.uri].discard(self.on_change)
            await self.send_progress()

    async def _worker_core(self):
        model = self.model
        pos = self.cursor
        offset = self.document.position_to_offset(pos)
        doc_text = self.document.text

        stream: InsertCodeResult = await model.insert_code(doc_text, offset, goal=self.cfg.task)
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
                self.change_futures[delta] = cf
                x = await self.server.apply_insert_text(
                    self.uri, self.cursor, delta, self.document.version
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
                    del self.change_futures[delta]
            with setdoc(self.document):
                added_range = lsp.Range.of_pos(self.cursor, len(delta))
                self.cursor += len(delta)
                self.ranges.add(added_range)
                await self.send_progress()
        all_text = "".join(all_deltas)
        logger.info(f"{self} finished streaming {len(all_text)} characters")
        if stream.thoughts is not None:
            thoughts = await stream.thoughts.read()
            return thoughts
        else:
            thoughts = "done!"
        await self.send_progress()
        return thoughts

    async def on_change(
        self,
        *,
        before: lsp.TextDocumentItem,
        after: lsp.TextDocumentItem,
        changes: lsp.DidChangeTextDocumentParams,
    ):
        if not self.running:
            return
        """
        [todo]
        When a change happens:
        1. if the change is before our 'working area', then we stop the completion request and run again.
        2. if the change is in our 'working area', then the user is correcting something that
        3. if the change is after our 'working area', then just keep going.
        4. if _we_ caused the chagne, then just keep going.
        """
        assert changes.textDocument.uri == self.uri
        self.document = before
        for c in changes.contentChanges:
            fut = self.change_futures.get(c.text)
            if fut is not None:
                # we caused this change
                fut.set_result(None)
            else:
                # someone else caused this change
                # [todo], in the below examples, we shouldn't cancel, but instead figure out what changed and restart the insertions with the new information.
                with setdoc(self.document):
                    self.ranges.apply_edit(c)
                if c.range is None:
                    self.cancel("the whole document got replaced")
                else:
                    if c.range.end <= self.cursor:
                        # some text was changed before our cursor
                        if c.range.end.line < self.cursor.line:
                            # the change is occuring on lines strictly above us
                            # so we can adjust the number of lines
                            lines_to_add = (
                                c.text.count("\n") + c.range.start.line - c.range.end.line
                            )
                            self.cursor += (lines_to_add, 0)
                        else:
                            # self.cancel("someone is editing on the same line as us")
                            pass  # temporarily disable
                    elif self.cursor in c.range:
                        self.cancel("someone is editing the same text as us")

        self.document = after


@dataclass
class AgentLogs:
    message: str
    severity: str
