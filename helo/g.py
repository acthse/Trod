"""
    helo.g
    ~~~~~~
"""

from types import ModuleType
from typing import Type, List, AsyncGenerator, Optional, Union, Any
from contextlib import asynccontextmanager

from helo import db
from helo import err
from helo import model
from helo import util
from helo import _sql


class Helo:

    def __init__(
        self,
        app: Optional[Any] = None,
        debug: bool = False,
    ) -> None:
        self.init_app(app)
        self.debug = debug
        self._database = None  # type: Optional[db.Database]
        self._model_cls = model.Model

    @property
    def Model(self) -> Type[model.Model]:
        return self._model_cls

    @property
    def is_connected(self) -> bool:
        return self._database is not None and self._database.is_connected

    def init_app(self, app) -> None:
        if not app:
            return None

        self.app = app
        self.app.db = self

        @self.app.before_request
        async def _first():
            if not self._database.is_connected:
                await self.connect()

        return None

    async def connect(self, url: str = "", **options: Any) -> None:
        if not self.is_connected:
            if not url and self.app is not None:
                url = self.app.config.get(db.ENV_KEY, "")
            self._database = db.Database(url, debug=self.debug, **options)

        self._model_cls.__db__ = self._database
        await self._database.connect()

    async def close(self) -> None:
        if self._database is None:
            raise err.UnconnectedError()

        await self._database.close()

    @asynccontextmanager
    async def c(self, url: str = "", **options) -> AsyncGenerator:
        """typing Any to ->
        >>> db = Helo()
        >>> async with db.c():
                await db.row()
                await User.get(1)
        """
        try:
            await self.connect(url, **options)
            yield
        finally:
            await self.close()

    def transaction(self) -> db.interface.Transaction:
        if self._database is None:
            raise err.UnconnectedError()

        return self._database.transaction()

    async def create_tables(
        self, models: List[Type[model.Model]], **options: Any
    ) -> bool:
        """Create table from Model list"""

        for m in models:
            if not issubclass(m, model.Model) or m is model.Model:
                raise TypeError(f"invalid model type: {type(m)}")

            await m.create(**options)
        return True

    async def create_all(self, module: ModuleType, **options: Any) -> bool:
        """Create all table from a model module"""

        if not isinstance(module, ModuleType):
            raise TypeError(f"{module!r} is not a module")

        return await self.create_tables(
            [
                m for _, m in vars(module).items()
                if issubclass(m, model.Model) and m is not model.Model
            ],
            **options
        )

    async def drop_tables(self, models: List[Type[model.Model]]) -> bool:
        """Drop table from Model list"""

        for m in models:
            if not issubclass(m, model.Model) or m is model.Model:
                raise TypeError(f"Invalid model type: {type(m)}")

            await m.drop()
        return True

    async def drop_all(self, module: ModuleType) -> bool:
        """Drop all table from a model module"""

        if not isinstance(module, ModuleType):
            raise TypeError(f"{module!r} is not a module")

        return await self.drop_tables([
            m for _, m in vars(module).items()
            if issubclass(m, model.Model) and m is not model.Model
        ])

    async def raw(
        self, query: Union[str, _sql.Query], **kwargs: Any
    ) -> Union[None, util.adict, List[util.adict], db.ExeResult]:
        """A coroutine that used to directly execute SQL query statements"""

        if not self.is_connected:
            raise err.UnconnectedError()

        if not isinstance(query, _sql.Query):
            query = _sql.Query(sql=query, params=kwargs.pop("params"))

        return await self._database.execute(query, **kwargs)

    async def execute(self):
        pass
