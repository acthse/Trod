import typing
from contextvars import ContextVar

from helo import err
from helo import _sql
from helo.db import url
from helo.db import result
from helo.db import logging
from helo.db import interface
from helo.util import import_object, adict

# @staticmethod
# def __ensure__(func):
#     @wraps(func)
#     def wrapper(connection):
#         if not connection._database.pool:
#             raise err.ProgrammingError("Database backend is not running")
#         return func(connection)
#     return wrapper

# from contextlib import asynccontextmanager

# @asynccontextmanager
# async def connect_manager(url_str: str, **kwargs: typing.Any):
#     try:
#         await connect(url_str, **kwargs)
#         yield
#     finally:
#         await disconnect()

logger = logging.create_logger()


class Database:

    _SUPPORTED_BACKENDS = {
        "mysql": "helo.db.backend.mysql.Backend",
    }

    def __init__(self, url_str: str, **options: typing.Any) -> None:
        if not url_str:
            raise ValueError("Database URL cannot be an empty")

        self.url = url.URL(url_str)
        self.options = options
        self.echo = options.pop("debug", False)

        backend_name = self._SUPPORTED_BACKENDS.get(self.url.scheme)
        if not backend_name:
            raise err.UnSupportedError(f"Helo not supported {self.url.scheme} now")
        if not self.url.db:
            raise ValueError("Must be specified the database name in url")

        self._backend = import_object(backend_name)(self.url, **options)
        self._connctx = ContextVar("connctx")  # type: ContextVar
        self._is_connected = False

    @property
    def is_connected(self):
        return self._is_connected

    async def connect(self) -> None:
        if self.is_connected:
            raise err.DuplicateConnect(
                f"Database already connected to {self._backend}"
            )

        await self._backend.connect()

        self._is_connected = True
        logger.info("Database is connected to %s", self._backend)

    async def close(self) -> None:
        if not self.is_connected:
            raise err.UnconnectedError()

        await self._backend.close()

        self._is_connected = False
        logger.info("Database connected to %s is closed", self._backend)

    async def execute(
        self, query: _sql.Query, **kwargs: typing.Any
    ) -> typing.Union[None, adict, typing.List[adict], result.ExeResult]:
        if not self.is_connected:
            raise err.UnconnectedError()

        if not isinstance(query, _sql.Query):
            raise TypeError(
                "Invalid type for 'query'"
                f"Expected 'Query', got {type(query)}"
            )

        if self.echo:
            logger.info(query)

        async with self.connection() as conn:
            if query.r:
                return await conn.fetch(
                    sql=query.sql,
                    params=query.params,
                    **kwargs
                )
            return await conn.execute(
                sql=query.sql,
                params=query.params,
                **kwargs
            )

    def connection(self) -> typing.Optional[interface.Connection]:
        if not self.is_connected:
            return None

        try:
            return self._connctx.get()
        except LookupError:
            current = self._backend.connection()
            self._connctx.set(current)
            return current

    def transaction(self) -> interface.Transaction:
        if not self.is_connected:
            raise err.UnconnectedError()

        return self.connection().transaction()
