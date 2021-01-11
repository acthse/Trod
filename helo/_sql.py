from __future__ import annotations

from typing import Any, Optional, Union, List, Tuple, Dict

from . import util


class Context:

    __slots__ = (
        '_elements', '_values', '_stack', '_state', 'aliases',
        'props', 'sources'
    )

    _SEMICOLON = ';'
    _MULTI_TYPES = (tuple, list)

    def __init__(self, **settings: Any) -> None:
        self._elements = []         # type: List[Any]
        self._values = []           # type: List[Any]
        self._stack = []            # type: List[Dict[str, Any]]
        self._state = settings      # type: Dict[str, Any]
        self.aliases = {}          # type: Dict[str, Any]
        self.props = util.adict()
        self.sources = util.adict()  # type: Dict[str, Any]

    def __sql__(self, ctx: Context) -> Context:
        ctx._elements.extend(self._elements)  # pylint: disable=protected-access
        ctx.values(self._values)
        return ctx

    def __enter__(self) -> Context:
        if self.parens:
            self.literal('(')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.parens:
            self.literal(')')
        self._state = self._stack.pop()

    def __call__(self, **overrides: Any) -> Context:
        self._stack.append(self._state)
        self._state = overrides
        return self

    @property
    def parens(self) -> Optional[bool]:
        return self._state.get('parens')

    def sql(self, obj: Any) -> Context:
        if isinstance(obj, (ClauseElement, Context)):
            return obj.__sql__(self)

        return self.values(obj)

    def literal(self, kwd: str) -> Context:
        self._elements.append(kwd)
        return self

    def values(self, value: Any) -> Context:
        converter = self._state.get('converter')
        if value is not None and converter:
            if isinstance(value, self._MULTI_TYPES):
                value = tuple(map(converter, value))
            else:
                value = converter(value)

        if self._state.get('params'):
            self.literal('%s')

        if isinstance(value, self._MULTI_TYPES):
            if not self._state.get('nesting'):
                self._values.extend(value)
                return self
        self._values.append(value)
        return self

    def alias(self, alias: str, realname: str) -> Context:
        if alias in self.aliases:
            raise ValueError(f"ambiguous alias: {alias}")

        self.aliases[alias] = realname
        return self

    def source(self, sources: List[str]) -> None:
        for idx, sou in enumerate(sources):
            alias = f"t{idx+1}"
            self.sources[alias] = sou
            self.sources[sou] = alias

    def query(self) -> Query:
        if self._elements[-1] != self._SEMICOLON:
            self.literal(self._SEMICOLON)
        return Query(
            sql=''.join(self._elements),
            params=tuple(self._values)
        )

    @classmethod
    def from_clause(cls, clause: ClauseElement) -> Context:
        return cls().sql(clause)


class Query:

    __slots__ = ('_sql', '_params', '_r')

    _QKS = ("SELECT", "SHOW")

    @util.argschecker(sql=str)
    def __init__(
        self,
        sql: str,
        params: Optional[list] = None,
        read: Optional[bool] = None,
    ) -> None:
        self._sql = sql
        self._params = params or []
        self._r = read

    def __repr__(self) -> str:
        return f"Query({self.sql} % {self.params})"

    def __str__(self) -> str:
        return f"{self.sql} % {self.params}"

    def __bool__(self) -> bool:
        return bool(self._sql)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Unsupported operation: '{self.__class__.__name__}' == '{type(other)}'"
            )
        return self.sql == other.sql and self.params == other.params

    @property
    def sql(self) -> str:
        return self._sql.strip()

    @property
    def params(self) -> Tuple[Any, ...]:
        if not isinstance(self._params, (tuple, list)):
            raise TypeError(
                "Invalid type for query params"
                f"Expected tuple, got {type(self._params)}"
            )
        return tuple(self._params)

    @property
    def r(self) -> bool:
        if self._r is not None:
            return self._r

        for k in self._QKS:
            if k in self.sql or k.lower() in self.sql:
                return True
        return False

    @r.setter
    def r(self, r: Optional[bool]) -> None:
        if r is not None and not isinstance(r, bool):
            raise TypeError(f'Invalid value {r!r} to set')
        self._r = r


def query(clause: ClauseElement) -> Query:
    return Context().sql(clause).query()


class ClauseElement:

    __slots__ = ()

    def __sql__(self, ctx: Context) -> Context:
        raise NotImplementedError


class ClauseElements(ClauseElement):

    __slots__ = ('elements', 'glue', 'parens')

    def __init__(
        self, elements: List[ClauseElement], glue: str = ' ', parens: bool = False
    ) -> None:
        self.elements = elements
        self.glue = glue
        self.parens = parens
        if parens and len(self.elements) == 1:
            if hasattr(self.elements[0], 'parens'):
                self.elements[0].parens = False  # type: ignore

    def __sql__(self, ctx: Context) -> Context:
        nlen = len(self.elements)
        if nlen == 0:
            return ctx.literal('()') if self.parens else ctx

        def paser(ctx, element):
            if isinstance(element, ClauseElement):
                ctx.sql(element)
            else:
                ctx.literal(element)  # type: ignore

        with ctx(parens=self.parens):
            for i in range(nlen - 1):
                paser(ctx, self.elements[i])
                ctx.literal(self.glue)

            paser(ctx, self.elements[-1])

        return ctx

    def add(
        self,
        elements: Union[ClauseElement, List[ClauseElement]]
    ) -> ClauseElements:
        if isinstance(elements, list):
            self.elements.extend(elements)
        else:
            self.elements.append(elements)
        return self


class EscapedElement(ClauseElement):

    __slots__ = ("element",)

    def __init__(self, element: str) -> None:
        self.element = element

    def __sql__(self, ctx: Context) -> Context:
        ctx.literal(f"`{self.element}`")
        return ctx


class SQL(ClauseElement):

    __slots__ = ('sql', 'params')

    def __init__(
        self,
        sql: str,
        params: Optional[Union[List[Any], Tuple[Any, ...]]] = None
    ) -> None:
        if params is not None and not isinstance(params, (tuple, list)):
            raise TypeError(
                "Invalid type for 'params'"
                f"Expected 'tuple' or 'list', got {type(params)}"
            )
        self.sql = sql
        self.params = params

    def __repr__(self) -> str:
        if self.params:
            return f"SQL({self.sql}) % {self.params}"
        return f"SQL({self.sql})"

    __str__ = __repr__

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Unsupported operation: '{self.__class__.__name__}' == '{type(other)}'"
            )
        return self.sql == other.sql and self.params == self.params

    def __sql__(self, ctx: Context) -> Context:
        ctx.literal(self.sql)
        if self.params is not None:
            ctx.values(self.params)
        return ctx


class Value(ClauseElement):

    __slots__ = ('_values',)

    def __init__(self, values: Any) -> None:
        self._values = values

    def __sql__(self, ctx: Context) -> Context:
        ctx.literal(
            '%s'
        ).values(
            self._values
        )
        return ctx


def CommaClauseElements(     # pylint: disable=invalid-name
    elements: List[ClauseElement]
) -> ClauseElements:
    return ClauseElements(elements, glue=', ')


def EnclosedClauseElements(  # pylint: disable=invalid-name
    elements: List[ClauseElement]
) -> ClauseElements:
    return ClauseElements(elements, glue=', ', parens=True)
