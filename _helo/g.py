"""
    helo.g
    ~~~~~~
"""

import warnings
from types import ModuleType
from typing import Any, Optional, Type, Union, List, Tuple

from . import db, model, util, _builder


@util.singleton
class G:
    """ The entry class of helo

    >>> import helo
    >>>
    >>> db = helo.G()

    :param app: Web application like Quart app
    :param debug: Record the executed SQL statement if true
    :param env_key: Environment variable key name of helo database url
    """

    def __init__(
        self,
        app: Optional[Any] = None,
        debug: bool = False,
        env_key: Optional[str] = None,
    ) -> None:
        self.init_app(app)
        self.debug = debug
        self.set_env_key(env_key)

    def __repr__(self):
        return f"<helo.G object, debug: {self.debug}>"

    __str__ = __repr__

    @property
    def isbound(self) -> bool:
        return db.isbound()

    @property
    def state(self) -> Optional[util.adict]:
        return db.state()

    def init_app(self, app) -> None:
        if not app:
            return None

        self.app = app
        self.app.db = self

        url = self.app.config.get(db.EnvKey.DFT, '')
        if not url:
            warnings.warn(f"The '{db.EnvKey.DFT}' not set for app, "
                          "getting from environment variable")

        @self.app.before_request
        async def _first():
            if not self.isbound:
                await self.bind(url)

        return None

    async def bind(self, url: Optional[str] = None, **kwargs: Any) -> None:
        """A coroutine that binding a database.

        :param url: Database url
        :param kwargs: see ``db.Pool``
        """

        url = url or db.EnvKey.get()
        return await db.binding(url, debug=self.debug, **kwargs)

    async def unbind(self) -> bool:
        """A coroutine that to unbind the database"""

        return await db.unbinding()

    def binder(self, url: Optional[str] = None, **kwargs: Any) -> db.Binder:
        """Handling of bound context"""

        return db.Binder(url, debug=self.debug, **kwargs)

    def set_env_key(self, key: Union[None, str]) -> None:
        """Set environment variable key name"""

        if key is None:
            return key
        return db.EnvKey.set(key)

    async def create_tables(
        self, models: List[Type[model.Model]], **options: Any
    ) -> bool:
        """Create table from Model list"""

        for m in models:
            await m.create(**options)
        return True

    async def create_all(self, module: ModuleType, **options: Any) -> bool:
        """Create all table from a model module"""

        if not isinstance(module, ModuleType):
            raise TypeError(f"{module!r} is not a module")

        return await self.create_tables(
            [m for _, m in vars(module).items()
             if isinstance(m, type) and issubclass(m, model.Model) and m is not model.Model
             ],
            **options
        )

    async def drop_tables(self, models: List[Type[model.Model]]) -> bool:
        """Drop table from Model list"""

        for m in models:
            await m.drop()
        return True

    async def drop_all(self, module: ModuleType) -> bool:
        """Drop all table from a model module"""

        if not isinstance(module, ModuleType):
            raise TypeError(f"{module!r} is not a module")

        return await self.drop_tables(
            [m for _, m in vars(module).items()
             if isinstance(m, type) and issubclass(m, model.Model) and m is not model.Model
             ]
        )

    async def raw(
        self, sql: Union[str, _builder.Query], **kwargs: Any
    ) -> Union[
        None, util.adict, Tuple[Any, ...], db.FetchResult, db.ExecResult
    ]:
        """A coroutine that used to directly execute SQL query statements"""

        query = sql
        if not isinstance(query, _builder.Query):
            query = _builder.Query(query, kwargs.pop('params', None))
        return await db.execute(query, **kwargs)
