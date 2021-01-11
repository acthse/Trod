from __future__ import annotations

import typing

from helo.util import adict
from helo.db.result import ExeResult


class Backend:

    async def connect(self) -> None:
        raise NotImplementedError()

    async def close(self) -> None:
        raise NotImplementedError()

    def connection(self) -> Connection:
        raise NotImplementedError()


class Connection:

    async def __aenter__(self) -> Connection:
        raise NotImplementedError()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        raise NotImplementedError()

    async def acquire(self) -> None:
        raise NotImplementedError()

    async def release(self) -> None:
        raise NotImplementedError()

    async def fetch(
        self,
        sql: str,
        params: typing.Optional[typing.Union[tuple, list]] = None,
        rows: typing.Optional[int] = None,
    ) -> typing.Union[None, adict, typing.List[adict]]:
        raise NotImplementedError()

    async def execute(
        self,
        sql: str,
        params: typing.Optional[typing.Union[tuple, list]] = None,
        many: bool = False,
    ) -> ExeResult:
        raise NotImplementedError()

    def transaction(self) -> Transaction:
        raise NotImplementedError()


class Transaction:

    async def __aenter__(self) -> Transaction:
        raise NotImplementedError()

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        raise NotImplementedError()

    def __call__(self, func: typing.Callable) -> typing.Callable:
        raise NotImplementedError()

    async def begin(self) -> None:
        raise NotImplementedError()

    async def commit(self) -> None:
        raise NotImplementedError()

    async def rollback(self) -> None:
        raise NotImplementedError()
