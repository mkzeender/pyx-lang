from asyncio import CancelledError, Task, get_running_loop
from collections.abc import Callable
from queue import Queue
from threading import Thread, Barrier
from time import sleep
from stopit import async_raise


class _Cancel(BaseException):
    pass


def _worker():
    while True:
        try:
            b2.wait()
            b1.wait()
            func, args, kwargs, fut = _to_do
            result = func(*args, **kwargs)
            fut.get_loop().call_soon_threadsafe(fut.set_result, result)
            while True:
                sleep(0.001)
        except _Cancel:
            pass


_worker_thread = Thread(target=_worker, daemon=True)
_worker_thread.start()

b1 = Barrier(2)
b2 = Barrier(2)
_to_do: tuple = ()


def to_worker[**P, R](
    func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> Task[R]:
    loop = get_running_loop()

    async def to_worker() -> R:
        global _to_do
        fut = loop.create_future()
        b2.wait()
        _to_do = (func, args, kwargs, fut)
        b1.wait()
        try:
            return await fut
        finally:
            async_raise(_worker_thread.ident, _Cancel)

    return loop.create_task(to_worker())
