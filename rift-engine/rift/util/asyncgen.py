import asyncio
import itertools
import logging
import operator
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterable, Callable, Optional, TypeVar, overload

A = TypeVar("A")
B = TypeVar("B")


@overload
def accumulate(
    asg: AsyncIterable[A], func: Callable[[B, A], B], *, initial: B
) -> AsyncGenerator[B, None]:
    ...


@overload
def accumulate(asg: AsyncIterable[A], func: Callable[[B, A], B]) -> AsyncGenerator[B, None]:
    ...


@overload
def accumulate(
    asg: AsyncIterable[A],
) -> AsyncGenerator[A, None]:
    ...


async def accumulate(asg, func=operator.add, *, initial=None):
    xs = aiter(asg)  # TODO: this breaks outside of python3.10
    acc = initial or await anext(xs)  # TODO: this breaks outside of python 3.10
    async for x in xs:
        yield acc
        acc = func(acc, x)
    yield acc


async def takewhile(predicate: Callable[[A], bool], asg: AsyncIterable[A]):
    async for x in asg:
        if predicate(x):
            yield x
        else:
            break


async def map(
    fn: Callable[[A], B], asg: AsyncIterable[A], error_callback: Optional[Callable[Any, Any]] = None
) -> AsyncIterable[B]:
    async for x in asg:
        if not error_callback:
            yield fn(x)
        else:
            try:
                yield fn(x)
            except Exception as e:
                error_callback(e)


async def tolist(asg):
    xs = []
    async for x in asg:
        xs.append(x)
    return xs


async def buffer(asg: AsyncIterable[A], maxsize=0) -> AsyncIterable[A]:
    q = asyncio.Queue(maxsize=maxsize)

    async def worker():
        async for x in asg:
            await q.put(x)
        await q.put(StopAsyncIteration())

    t = asyncio.create_task(worker())
    while True:
        x = await q.get()
        if isinstance(x, StopAsyncIteration):
            break
        yield x
    await t
    # note: t will automatically get cancelled when its reference count drops to zero
