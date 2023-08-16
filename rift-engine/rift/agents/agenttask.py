import asyncio
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """
    Represents a task associated with an agent

    Attributes:
        description (str): Description of the task
        task: thunk returning a Task
        started (bool): Indicates whether the main inner task has started execution
        _error (Optional[Exception]): Error encountered during task execution (if any)
        _cancelled (bool): Indicates whether the task has been cancelled
    """

    description: str
    task: Callable[Any, Awaitable[Any]]
    args: Union[Iterable, Callable[Any, Awaitable[Iterable[Any]]], None] = None
    kwargs: Union[Dict, Callable[Any, Awaitable[Dict[Any, Any]]], None] = None
    done_callback: Optional[Callable[Any, Any]] = None
    start_callback: Optional[Callable[Any, Any]] = None
    _task: Optional[asyncio.Task] = None
    _running: bool = False
    _error: Optional[Exception] = None
    _cancelled: bool = False
    _done: bool = False

    async def run(self):
        """
        Runs the task coroutine and handles exceptions
        """
        if self._running:
            raise Exception("Task is already running")

        self._running = True
        try:
            args = (
                list(await self.args())
                if callable(self.args)
                else list(self.args)
                if self.args
                else []
            )
            kwargs = (
                dict(await self.kwargs())
                if callable(self.kwargs)
                else dict(self.kwargs)
                if self.kwargs
                else {}
            )
            self._task: asyncio.Task = asyncio.create_task(self.task(*args, **kwargs))
            if self.done_callback is not None:
                self._task.add_done_callback(self.done_callback)
            if self.start_callback is not None:
                self.start_callback()
            return await self._task
        except asyncio.CancelledError as e:
            self._cancelled = True
            # raise e
        except Exception as e:
            self._error = e
            logger.error(f"[AgentTask] error: {e}")
            logger.info(traceback.format_exc())
        finally:
            self._running = False

    def cancel(self):
        """
        Cancels the task
        """
        if self.done:
            return
        if self._task:
            return self._task.cancel()
        self._cancelled = True

    @property
    def running(self) -> bool:
        return self._running

    @property
    def done(self) -> bool:
        """
        Returns whether the task is done
        """
        if self._done:
            return True
        if self._task:
            return self._task.done()
        else:
            return False

    @property
    def cancelled(self) -> bool:
        """
        Returns whether the task is cancelled
        """
        return self._cancelled or (self._task.cancelled() if self._task else False)

    @property
    def error(self) -> bool:
        """
        Returns whether an error occurred in the task
        """
        status = self._error is not None
        if status:
            logger.info(f"[AgentTask] exception={self._error}")
        return status

    @property
    def status(self):
        """
        Returns the status of the task

        Returns:
            str: The status of the task (scheduled, running, done, cancelled, error)
        """
        if self.error:
            return "error"
        elif self.cancelled:
            return "cancelled"
        elif self.done:
            return "done"
        elif self._running:
            return "running"
        else:
            return "scheduled"
