from dataclasses import dataclass
from typing import Dict, Optional, Any
from asyncio import Future
from rift.lsp import LspServer as BaseLspServer

from rift.agents import AgentState, AgentTask, AgentRunParams, AgentRunResult
from rift.llm.abstract import AbstractCodeCompletionProvider

# from your_application.utils import logger, setdoc
# from your_application.status import Status
# from your_application.range_set import RangeSet
import asyncio


@dataclass
class CodeCompletionAgent:
    id: int
    state: AgentState
    tasks: Dict[str, AgentTask]
    server: BaseLspServer
    active_task_id: Optional[str] = None
    count: int = 0
    model: AbstractCodeCompletionProvider
    status: Status = Status.running
    change_futures: Dict[str, Future] = {}

    @classmethod
    def create(cls, messages, server):
        CodeCompletionAgent.count += 1
        obj = CodeCompletionAgent(
            state=CodeCompletionAgentState(messages=messages), tasks=dict(), server=server, id=ChatAgent.count
        )
        obj.active_task_id = obj.add_task(AgentTask("running", "Get user response", [], None))
        obj.wait_user_response_task = obj.tasks[active_task_id]
        obj.generate_response_task = obj.tasks[
            obj.add_task(AgentTask("done", "Generate response", [], None))
        ]
        return obj    

    @property
    def task(self):
        if self.active_task_id is None:
            return None
        else:
            return self.tasks.get(self.active_task_id)

    def __str__(self):
        return f"<{type(self).__name__} {self.id}>"

    async def run(self, params: AgentRunParams) -> AgentRunResult:
        try:
            self.server.register_change_callback(self.on_change, self.uri)
            model = self.model
            pos = self.cursor
            offset = self.document.position_to_offset(pos)
            doc_text = self.document.text

            stream: InsertCodeResult = await model.insert_code(doc_text, offset, goal=params.task)
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
                added_range = lsp.Range.of_pos(self.cursor, len(delta))
                self.cursor += len(delta)
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
            self.server.change_callbacks[self.uri].discard(self.on_change)
            await self.send_progress()

    def add_task(self, task: AgentTask):
        self.tasks[task.id] = task
        return task.id

    def cancel(self, msg):
        logger.info(f"{self} cancel run: {msg}")
        for _, task in self.tasks.items():
            if task is not None:
                task.cancel()

    async def request_input(self, req):
        # Logic to handle an input request
        pass

    async def request_chat(self):
        # Logic to handle a chat request
        pass

    async def send_progress(self):
        # Logic to send progress updates to the LSP server
        pass

    async def send_result(self):
        # Logic to send a result back to the LSP server
        pass
