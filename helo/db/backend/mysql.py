from __future__ import annotations

import asyncio
import typing
from functools import wraps
from types import TracebackType

import aiomysql

from helo import err
from helo.util import adict
from helo.db import interface
from helo.db.url import URL
from helo.db.result import ExeResult


class Backend(interface.Backend):

    _CONN_KWARGS = (
        "user",
        "password",
        "host",
        "port",
        "db",
        "charset",
        "connect_timeout",
        "ssl",
    )
    _POOL_KWARGS = (
        'minsize',
        'maxsize',
        'pool_recycle',
        'loop',
    )

    def __init__(
        self,
        url: URL,
        **options: typing.Any,
    ) -> None:
        self._url = url
        self._options = options
        self._pool = None  # typing.Optional[aiomysql.Pool]

    def __repr__(self) -> str:
        return f"<MySQL Database({self._url.host}:{self._url.port}/{self._url.db})>"

    def __bool__(self) -> bool:
        return self._pool is not None

    @property
    def pool(self) -> typing.Optional[aiomysql.Pool]:
        return self._pool

    def _get_conn_kwargs(self) -> dict:
        kwargs = {}
        for arg in self._CONN_KWARGS:
            value = getattr(self._url, arg, None) or self._url.options.get(arg)
            if value is not None:
                kwargs[arg] = value
        kwargs["autocommit"] = True
        return kwargs

    def _get_pool_kwargs(self) -> dict:
        kwargs = {}
        for arg in self._POOL_KWARGS:
            value = self._options.get(arg) or self._url.options.get(arg)
            if value is None:
                continue
            if arg == "loop" and not isinstance(value, asyncio.AbstractEventLoop):
                raise TypeError(
                    "Invalid type for 'loop'. "
                    f"Expected 'asyncio.AbstractEventLoop', got {type(value)}"
                )
            kwargs[arg] = int(value)
        return kwargs

    async def connect(self) -> None:
        self._pool = await aiomysql.create_pool(
            **self._get_pool_kwargs(),
            **self._get_conn_kwargs()
        )

    async def close(self) -> None:
        if self._pool is None:
            raise err.UnconnectedError()

        self._pool.close()
        await self._pool.wait_closed()
        self._pool = None

    def connection(self) -> Connection:
        return Connection(self)


class Connection(interface.Connection):

    def __init__(self, database: Backend) -> None:
        self._database = database
        self._current = None  # type: typing.Optional[aiomysql.Connection]
        self._conn_counter = 0
        self._conn_lock = asyncio.Lock()
        self._tran_lock = asyncio.Lock()
        self._query_lock = asyncio.Lock()

    async def __aenter__(self) -> Connection:
        async with self._conn_lock:
            self._conn_counter += 1
            if self._conn_counter == 1:
                await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: typing.Type[BaseException] = None,
        exc_value: BaseException = None,
        traceback: TracebackType = None,
    ) -> None:
        async with self._conn_lock:
            self._conn_counter -= 1
            if self._conn_counter == 0:
                await self.release()

    async def acquire(self) -> None:
        if self._current is not None:
            raise AssertionError("Connection is already acquired")

        if not self._database.pool:
            raise AssertionError("Database backend is not running")

        self._current = await self._database.pool.acquire()

    async def release(self) -> None:
        if self._current is None:
            raise AssertionError("Connection is not acquired")

        if not self._database.pool:
            raise AssertionError("Database backend is not running")

        await self._database.pool.release(self._current)
        self._current = None

    async def fetch(
        self,
        sql: str,
        params: typing.Optional[typing.Tuple[typing.Any, ...]] = None,
        rows: typing.Optional[int] = None,
    ) -> typing.Union[None, adict, typing.List[adict]]:
        if self._current is None:
            raise AssertionError("Connection is not acquired")

        async with self._query_lock:
            async with self._current.cursor(_ADictCursor) as cur:
                await cur.execute(sql, params or ())
                if not rows:
                    result = await cur.fetchall()
                elif rows and rows == 1:
                    result = await cur.fetchone()
                else:
                    result = await cur.fetchmany(rows)
                return result

    async def execute(
        self,
        sql: str,
        params: typing.Optional[typing.Tuple[typing.Any, ...]] = None,
        many: bool = False,
    ) -> ExeResult:
        if self._current is None:
            raise AssertionError("Connection is not acquired")

        async with self._query_lock:
            async with self._current.cursor() as cur:
                if many is True:
                    await cur.executemany(sql, params or ())
                else:
                    await cur.execute(sql, params or ())

                affected, last_id = cur.rowcount, cur.lastrowid
                return ExeResult(affected, last_id)

    async def begin(self) -> None:
        async with self._tran_lock:
            await self.__aenter__()
            await self._current.begin()

    async def commit(self) -> None:
        async with self._tran_lock:
            await self._current.commit()
            await self.__aexit__()

    async def rollback(self) -> None:
        async with self._tran_lock:
            await self._current.rollback()
            await self.__aexit__()

    def transaction(self) -> Transaction:
        return Transaction(self)


class Transaction(interface.Transaction):

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    async def __aenter__(self) -> Transaction:
        await self.begin()
        return self

    async def __aexit__(
        self,
        exc_type: typing.Type[BaseException] = None,
        exc_value: BaseException = None,
        traceback: TracebackType = None,
    ) -> None:
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()

    def __call__(self, func: typing.Callable) -> typing.Callable:

        @wraps(func)
        async def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            async with self:
                return await func(*args, **kwargs)

        return wrapper

    async def begin(self) -> None:
        await self.connection.begin()

    async def commit(self) -> None:
        await self.connection.commit()

    async def rollback(self) -> None:
        await self.connection.rollback()


class _ADictCursor(aiomysql.DictCursor):
    dict_type = adict
