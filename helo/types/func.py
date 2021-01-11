from typing import Callable

from helo import _sql
from helo.types import _abc


class Func(_sql.ClauseElement):

    __slots__ = ('_func', '_element')

    def __init__(
        self, func: str,
        element: _sql.ClauseElement
    ) -> None:
        self._func = func.upper()
        self._element = element

    def __getattr__(self, func: str) -> Callable:

        def decorator(*args, **kwargs):
            return Func(func, *args, **kwargs)

        return decorator

    def as_(self, alias: str) -> _abc.Alias:
        return _abc.Alias(self, alias)

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        ctx.literal(self._func)
        with ctx(parens=True):
            ctx.sql(self._element)
        return ctx


F = Func("", None)  # type: ignore
