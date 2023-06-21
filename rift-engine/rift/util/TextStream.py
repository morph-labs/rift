import asyncio
from typing import Any, AsyncIterable, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class TextStream:
    _feed_task: Optional[asyncio.Task]
    _waiter: Optional[asyncio.Future[None]]
    _eof: bool
    _buffer: str  # [todo] use io.StringIO
    _loop: asyncio.AbstractEventLoop
    _on_cancel: Optional[Callable[[], None]]

    def __init__(self, loop=None, on_cancel=None):
        self._feed_task = None
        self._buffer = ""
        self._waiter = None
        self._eof = False
        self._loop = asyncio.get_event_loop() if loop is None else loop
        self._on_cancel = on_cancel

    def feed_eof(self):
        if self._eof:
            return
        self._eof = True
        self._wakeup_waiter()

    def at_eof(self):
        return self._eof and not self._buffer

    def feed_data(self, data: str):
        if self._eof:
            raise RuntimeError("feed_data() called after feed_eof()")
        if len(data) == 0:
            return
        self._buffer += data
        self._wakeup_waiter()

    def _wakeup_waiter(self):
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            if not waiter.cancelled():
                waiter.set_result(None)

    async def _wait_for_data(self, func_name="_wait_for_data()"):
        if self._waiter is not None:
            raise RuntimeError(
                f"{func_name}() called while another coroutine is already waiting for incoming data"
            )

        if self._eof:
            raise RuntimeError(f"{func_name} called after feed_eof()")
        if self._feed_task is not None:
            if self._feed_task.done():
                exn = self._feed_task.exception()
                if exn is not None:
                    raise exn
                if not self._eof:
                    raise RuntimeError("feeder is done but no eof")

        self._waiter = self._loop.create_future()
        try:
            await self._waiter
        finally:
            self._waiter = None

    async def read(self, n=-1):
        if n == 0:
            return ""
        if n < 0:
            while not self._eof:
                await self._wait_for_data("read()")
            text = self._buffer
            self._buffer = ""
            return text
        if not self._buffer and not self._eof:
            await self._wait_for_data(f"read({n})")
        return self.pop(n)

    def pop_all(self):
        text = self._buffer
        self._buffer = ""
        return text

    def pop(self, n: int):
        """Pop from the buffer and return them.

        If n is specified, pops at most n characters from the buffer.
        If n is negative, pops all but -n characters from the buffer.

        Note that this method does not wait for incoming data.
        """
        text = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return text

    async def readexactly(self, n: int):
        if n == 0:
            return ""
        if n < 0:
            raise ValueError("readexactly() called with negative size")
        while len(self._buffer) < n and not self._eof:
            await self._wait_for_data(f"readexactly()")
        if len(self._buffer) < n:
            assert self._eof
            incomplete: Any = self.pop_all()
            raise EOFError(
                f"expecting {n - len(incomplete)} more characters but got EOF"
            )
        return self.pop(n)

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Note this is different to StreamReader which yields lines.
        We just yield everything that is available in the buffer."""
        while len(self._buffer) == 0:
            if self._eof:
                raise StopAsyncIteration
            else:
                try:
                    await self._wait_for_data("__anext__")
                except asyncio.CancelledError:
                    if self._on_cancel is not None:
                        self._on_cancel()
                    raise
        return self.pop_all()
    @classmethod
    def from_aiter(cls, x: AsyncIterable[str], loop=None):
        self = cls(loop=loop)

        async def worker():
            async for line in x:
                self.feed_data(line)
            self.feed_eof()

        self._feed_task = self._loop.create_task(worker())
        return self

    @classmethod
    def from_bytestream(cls, reader: asyncio.StreamReader, encoding="utf-8"):
        self = cls()

        async def worker():
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.decode(encoding)
                self.feed_data(line)
            self.feed_eof()

        self._feed_task = self._loop.create_task(worker())
        return self

    async def readuntil(self, separator: str = "\n"):
        if not separator:
            raise ValueError("Separator can't be empty")
        while True:
            i = self._buffer.find(separator)
            if i >= 0:
                return self.pop(i + len(separator))
            if self._eof:
                incomplete: Any = self.pop_all()
                raise EOFError(f"got EOF before finding separator {separator}")
            await self._wait_for_data("readuntil()")

    def split_once(self, sep: str) -> tuple["TextStream", "TextStream"]:
        before = TextStream(self._loop)
        after = TextStream(self._loop)

        async def before_worker():
            while True:
                i = self._buffer.find(sep)
                if i >= 0:
                    before.feed_data(self.pop(i))
                    before.feed_eof()
                    return
                # separator not in buffer
                if self._eof:
                    before.feed_data(self.pop_all())
                    before.feed_eof()
                    return
                if len(self._buffer) > len(sep):
                    before.feed_data(self.pop(-len(sep)))
                await self._wait_for_data("split_once()")

        before_task = self._loop.create_task(before_worker())
        before._feed_task = before_task

        async def after_worker():
            await before_task
            if self.at_eof():
                logger.warning(f'got EOF before finding separator "{sep}"')
                after.feed_eof()
                return
            s = await self.readexactly(len(sep))
            if s != sep:
                raise RuntimeError(f'expected separator "{sep}" but got "{s}"')
            while True:
                after.feed_data(self.pop_all())
                if self._eof:
                    after.feed_eof()
                    return
                await self._wait_for_data("split_once()")

        after._feed_task = self._loop.create_task(after_worker())
        return before, after

    async def asplit(self, sep: str) -> AsyncIterable["TextStream"]:
        while True:
            ts = TextStream(self._loop)
            yield ts
            while True:
                i = self._buffer.find(sep)
                if i >= 0:
                    ts.feed_data(self.pop(i))
                    ts.feed_eof()
                    break
                # no separator in buffer
                if self._eof:
                    ts.feed_data(self.pop_all())
                    ts.feed_eof()
                    return
                if len(self._buffer) > len(sep):
                    ts.feed_data(self.pop(-len(sep)))
                await self._wait_for_data("asplit()")
