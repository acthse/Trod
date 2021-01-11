from __future__ import annotations

from typing import Any, Optional, Union, Callable, List, Tuple, Dict

from helo import _sql
from helo import err
from helo import util

SEQUENCE = (list, tuple, set, frozenset)
NULL = 'null'
OPERATOR = util.adict(
    AND='AND',
    OR='OR',
    ADD='+',
    SUB='-',
    MUL='*',
    DIV='/',
    BIN_AND='&',
    BIN_OR='|',
    XOR='#',
    MOD='%',
    EQ='=',
    LT='<',
    LTE='<=',
    GT='>',
    GTE='>=',
    NE='!=',
    IN='IN',
    NOT_IN='NOT IN',
    IS='IS',
    IS_NOT='IS NOT',
    LIKE='LIKE BINARY',
    ILIKE='LIKE',
    EXISTS='EXISTS',
    NEXISTS='NOT EXISTS',
    BETWEEN='BETWEEN',
    NBETWEEN='NOT BETWEEN',
    REGEXP='REGEXP BINARY',
    IREGEXP='REGEXP',
    BITWISE_NEGATION='~',
    CONCAT='||',
)
_ENCODINGS = (
    'ig5',
    'ec8',
    'p850',
    'p8',
    'oi8r',
    'atin1',
    'atin2',
    'ascii',
    'ujis',
    'sjis',
    'hebrew',
    'tis620',
    'euckr',
    'gb2312',
    'macce',
    'cp1251',
    'macroman',
    'cp1257',
    'binary',
    'armscii8',
    'cp1256',
    'cp866',
    'dec8',
    'greek',
    'hp8',
    'keybcs2',
    'koi8r',
    'koi8u',
    'latin2',
    'latin5',
    'latin7',
    'cp850',
    'cp852',
    'swe7',
    'big5',
    'gbk',
    'geostd8',
    'latin1',
    'cp932',
    'eucjpms',
    'cp1250',
    'utf16',
    'ucs2',
    'utf32',
    'utf8',
    'utf8mb4',
)
ENCODING = util.In(_ENCODINGS, 'Encoding')
ID = Union[int, str]
ON_CREATE = _sql.SQL('CURRENT_TIMESTAMP')
ON_UPDATE = _sql.SQL('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')


class Column(_sql.ClauseElement):

    __slots__ = ()

    def __and__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.AND, rhs)

    def __rand__(self, lhs: Any) -> Expression:
        return Expression(lhs, OPERATOR.AND, self)

    def __or__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.OR, rhs)

    def __ror__(self, lhs: Any) -> Expression:
        return Expression(lhs, OPERATOR.OR, self)

    def __add__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.ADD, rhs)

    def __radd__(self, lhs: Any) -> Expression:
        return Expression(lhs, OPERATOR.ADD, self)

    def __sub__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.SUB, rhs)

    def __rsub__(self, lhs: Any) -> Expression:
        return Expression(lhs, OPERATOR.SUB, self)

    def __mul__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.MUL, rhs)

    def __rmul__(self, lhs: Any) -> Expression:
        return Expression(lhs, OPERATOR.MUL, self)

    def __div__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.DIV, rhs)

    def __rdiv__(self, lhs: Any) -> Expression:
        return Expression(lhs, OPERATOR.DIV, self)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __xor__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.XOR, rhs)

    def __rxor__(self, lhs: Any) -> Expression:
        return Expression(lhs, OPERATOR.XOR, self)

    def __eq__(self, rhs: Any) -> Expression:  # type: ignore
        op = OPERATOR.IS if rhs is None else OPERATOR.EQ
        return Expression(self, op, rhs)

    def __ne__(self, rhs: Any) -> Expression:  # type: ignore
        op = OPERATOR.IS_NOT if rhs is None else OPERATOR.NE
        return Expression(self, op, rhs)

    def __lt__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.LT, rhs)

    def __le__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.LTE, rhs)

    def __gt__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.GT, rhs)

    def __ge__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.GTE, rhs)

    def __lshift__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.IN, rhs)

    def __rshift__(self, rhs: Any):
        return Expression(self, OPERATOR.IS, rhs)

    def __mod__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.LIKE, rhs)

    def __pow__(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.ILIKE, rhs)

    def __getitem__(self, item: slice) -> Expression:
        if isinstance(item, slice):
            if item.start is None or item.stop is None:
                raise ValueError(
                    "the BETWEEN range must have both a start and end-point."
                )
            return self.between(item.start, item.stop)
        return self == item

    def concat(self, rhs: Any) -> StrExpression:
        return StrExpression(self, OPERATOR.CONCAT, rhs)

    def binand(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.BIN_AND, rhs)

    def binor(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.BIN_OR, rhs)

    def in_(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.IN, rhs)

    def nin_(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.NOT_IN, rhs)

    def exists(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.EXISTS, rhs)

    def nexists(self, rhs: Any) -> Expression:
        return Expression(self, OPERATOR.NEXISTS, rhs)

    def isnull(self, is_null: bool = True) -> Expression:
        op = OPERATOR.IS if is_null else OPERATOR.IS_NOT
        return Expression(self, op, None)

    def regexp(self, rhs: Any, i: bool = True) -> Expression:
        if i:
            return Expression(self, OPERATOR.IREGEXP, rhs)
        return Expression(self, OPERATOR.REGEXP, rhs)

    def like(self, rhs: Any, i: bool = True) -> Expression:
        if i:
            return Expression(self, OPERATOR.ILIKE, rhs)
        return Expression(self, OPERATOR.LIKE, rhs)

    def contains(self, rhs: Any, i: bool = True) -> Expression:
        if i:
            return Expression(self, OPERATOR.ILIKE, f"%{rhs}%")
        return Expression(self, OPERATOR.LIKE, f"%{rhs}%")

    def startswith(self, rhs: Any, i: bool = True) -> Expression:
        if i:
            return Expression(self, OPERATOR.ILIKE, f"{rhs}%")
        return Expression(self, OPERATOR.LIKE, f"{rhs}%")

    def endswith(self, rhs: Any, i: bool = True) -> Expression:
        if i:
            return Expression(self, OPERATOR.ILIKE, f"%{rhs}")
        return Expression(self, OPERATOR.LIKE, f"%{rhs}")

    def between(self, low: Any, hig: Any) -> Expression:
        return Expression(
            self, OPERATOR.BETWEEN,
            _sql.ClauseElements(
                [_sql.Value(low), OPERATOR.AND, _sql.Value(hig)]
            )
        )

    def nbetween(self, low: Any, hig: Any) -> Expression:
        return Expression(
            self, OPERATOR.NBETWEEN,
            _sql.ClauseElements(
                [_sql.Value(low), OPERATOR.AND, _sql.Value(hig)]
            )
        )

    def asc(self) -> Ordering:
        return Ordering(self, "ASC")

    def desc(self) -> Ordering:
        return Ordering(self, "DESC")

    def as_(self, alias: str) -> _sql.ClauseElement:
        if alias:
            return Alias(self, alias)
        return self


class Ordering(Column):

    __slots__ = ("element", "key")

    def __init__(self, element: _sql.ClauseElement, key: str) -> None:
        self.element = element
        self.key = key

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        ctx.sql(
            self.element
        ).literal(
            f" {self.key} "
        )
        return ctx


class Alias(Column):

    __slots__ = ('element', 'alias')

    @util.argschecker(alias=str, nullable=False)
    def __init__(self, element: _sql.ClauseElement, alias: str) -> None:
        self.element = element
        self.alias = alias

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        ctx.sql(
            self.element
        ).literal(
            " AS "
        ).sql(
            _sql.EscapedElement(self.alias)
        )

        if isinstance(self.element, Field):
            table_alias = ctx.sources[self.element.__table__.name]
            ctx.alias(self.alias, f"{table_alias}.{self.element.name}")

        return ctx


class Expression(Column):

    __slots__ = ('lhs', 'op', 'rhs', 'parens')

    def __init__(
        self, lhs: Any, op: str, rhs: Any, parens: bool = True
    ) -> None:
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.parens = parens

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        overrides = {'parens': self.parens, 'params': True}

        if isinstance(self.lhs, Field):
            overrides['converter'] = self.lhs.db_value
        elif isinstance(self.rhs, Field):
            overrides['converter'] = self.rhs.db_value

        if self.op in (
            OPERATOR.IN,
            OPERATOR.NOT_IN,
            OPERATOR.EXISTS,
            OPERATOR.NEXISTS
        ):
            if not isinstance(self.rhs, (SEQUENCE, _sql.ClauseElement)):
                raise TypeError(
                    f"invalid values {self.rhs} for operator '{self.op}'")
            if isinstance(self.rhs, _sql.ClauseElement):
                self.rhs = _sql.EnclosedClauseElements([self.rhs])
            else:
                self.rhs = tuple(self.rhs)
            overrides['nesting'] = True

        with ctx(**overrides):
            ctx.sql(
                self.lhs
            ).literal(
                f" {self.op} "
            ).sql(self.rhs)

        return ctx


class StrExpression(Expression):

    __slots__ = ()

    def __add__(self, rhs: Any) -> StrExpression:
        return self.concat(rhs)

    def __radd__(self, lhs: Any) -> StrExpression:
        return StrExpression(lhs, OPERATOR.CONCAT, self)


class FieldDefine(_sql.ClauseElement):

    __slots__ = ("field", )
    __tpops__ = util.adict(
        sit=" {type} ",
        wlt=" {type}({length}) ",
        wdt=" {type}({length},{float_length}) ",
    )

    def __init__(self, field: Field) -> None:
        self.field = field

    def __str__(self) -> str:
        return _sql.query(self).sql

    def _dtyp(self) -> str:
        type_render = {'type': self.field.db_type}

        type_tpl = self.__tpops__.sit
        length = getattr(self.field, 'length', None)

        if length:
            type_render['length'] = length
            type_tpl = self.__tpops__.wlt
        if isinstance(length, tuple) and len(length) == 2:
            type_render['length'] = length[0]
            float_length = length[1]
            if float_length:
                type_render['float_length'] = float_length
                type_tpl = self.__tpops__.wdt

        return type_tpl.format(**type_render)

    def _ddft(self) -> str:

        def default_value(default):
            if isinstance(default, _sql.SQL):
                return f"DEFAULT {default.sql}"
            if callable(default):
                return None
            return "DEFAULT '{self.field.to_str(default)}'"

        if getattr(self.field, "auto", None):
            return "NOT NULL AUTO_INCREMENT"

        fd = getattr(self.field, 'default', NULL)
        if self.field.null:
            if fd is None:
                default = "DEFAULT NULL"
                if self.field.db_type == 'timestamp':
                    default = f"NULL {default}"
            elif fd == NULL:
                default = "NULL"
            else:
                default = default_value(fd)
                if default is None:
                    if self.field.db_type == 'timestamp':
                        default = "NULL DEFAULT NULL"
                    else:
                        default = "DEFAULT NULL"
        else:
            default = "NOT NULL"
            if fd not in (None, NULL):
                dv = default_value(fd)
                if dv:
                    default = f"{default} {dv}"

        return default if default is not None else None

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        ctx.sql(
            self.field
        ).literal(
            self._dtyp()
        )

        if getattr(self.field, "unsigned", None):
            ctx.literal("unsigned ")

        encoding = getattr(self.field, "enoding", None)
        if encoding:
            ctx.literal("CHARACTER SET {encoding} ")

        if getattr(self.field, "zerofill", None):
            ctx.literal("zerofill ")

        default = self._ddft()
        if default:
            ctx.literal(f"{default}")

        if self.field.comment:
            ctx.literal(f" COMMENT '{self.field.comment}'")

        return ctx


class IndexDefine(_sql.ClauseElement):

    __slots__ = ("index",)

    def __init__(self, index: Index) -> None:
        self.index = index

    def __str__(self) -> str:
        return _sql.query(self).sql

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        ctx.sql(
            _sql.ClauseElements([self.index.db_type, self.index])
        )
        ctx.literal(" ")
        ctx.sql(
            _sql.EnclosedClauseElements(self.index.fields)
        )
        if self.index.comment:
            ctx.literal(f"COMMENT '{self.index.comment}'")
        return ctx


class TableDefine(_sql.ClauseElement):

    __slots__ = ("table", )

    def __init__(self, table: Table) -> None:
        self.table = table

    def __str__(self) -> str:
        return _sql.query(self).sql

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        ctx.literal("CREATE ")
        if ctx.props.get("temporary") is True:
            ctx.literal("TEMPORARY")
        ctx.literal("TABLE ")
        if ctx.props.get("safe") is True:
            ctx.literal("IF NOT EXISTS")
        ctx.sql(self.table)

        columns = [f.__ddl__() for f in self.table.fields_dict.values()]
        columns.append(
            _sql.ClauseElements([
                _sql.SQL("PRIMARY KEY ("),
                self.table.primary.field,
                _sql.SQL(")")
            ], glue="")
        )
        if self.table.indexes:
            indexes = [i.__ddl__() for i in self.table.indexes]
            columns.extend(indexes)

        meta = []
        if ctx.props.get("scheme") == "mysql":
            meta.append(f"ENGINE={self.table.engine}")
        meta.append(
            f"AUTO_INCREMENT={self.table.auto_increment} "
            f"DEFAULT CHARSET={self.table.charset} "
            f"COMMENT='{self.table.comment}'"
        )

        ctx.sql(
            _sql.EnclosedClauseElements(columns)
        ).literal(
            " ".join(meta)
        )
        return ctx


class Field(Column):

    __slots__ = ('null', 'default', 'comment', 'name', '__table__')

    py_type = None  # type: Any
    db_type = None  # type: Any

    VALUEERR_MSG = "invalid value({}) for {} field"

    @util.argschecker(null=bool, comment=str)
    def __init__(
        self,
        null: bool,
        default: Any,
        comment: str,
        name: str = ''
    ) -> None:
        self.__table__ = None  # type: Any

        if self.py_type is None or self.db_type is None:
            raise NotImplementedError()

        if default:
            if isinstance(self.py_type, (list, tuple)):
                py_types = list(self.py_type)
                py_types.append(_sql.SQL)
            else:
                py_types = [self.py_type, _sql.SQL]
            if not (isinstance(default, tuple(py_types)) or callable(default)):
                raise err.FieldInitError(
                    f"invalid {self.__class__.__name__} default value ({default})"
                )

        self.null = null
        self.comment = comment
        self.default = default
        self.name = name

    def __repr__(self) -> str:
        return f"<types.{self.__class__.__name__} object '{self.name}'>"

    __str__ = __repr__

    def __hash__(self) -> int:
        if self.name:
            return hash(self.name)
        raise err.NoColumnNameError()

    def adapt(self, value: Any) -> Any:
        try:
            return self.py_type(value)  # pylint: disable=not-callable
        except ValueError:
            raise ValueError(
                f"illegal value {value!r} for "
                f"{self.__class__.__name__} field"
            )

    def to_str(self, value: Any) -> str:
        if value is None:
            raise ValueError("invalid None value")
        return str(self.db_value(value))

    def py_value(self, value: Any) -> Any:
        return value if value is None else self.adapt(value)

    def db_value(self, value: Any) -> Any:
        return value if value is None else self.adapt(value)

    def ddl(self) -> str:
        return str(self.__ddl__())

    def __ddl__(self) -> _sql.ClauseElement:
        return FieldDefine(self)

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        if ctx.props.get("select") is True:
            ctx.sql(
                _sql.EscapedElement(ctx.sources[self.__table__.name])
            ).literal(
                "."
            ).sql(
                _sql.EscapedElement(self.name)
            )
        else:
            ctx.sql(
                _sql.EscapedElement(self.name)
            )
        return ctx


class Index(_sql.ClauseElement):

    __slots__ = ('fields', 'comment', 'name')
    db_type = None  # type: _sql.SQL

    def __init__(
            self,
            name: str,
            fields: Union[
                str,
                List[str],
                Tuple[str, ...],
                Field,
                List[Field],
                Tuple[Field, ...]
            ],
            comment: Optional[str] = ''
    ) -> None:
        if self.db_type is None:
            raise NotImplementedError()

        self.name = name
        self.comment = comment

        if not isinstance(fields, SEQUENCE):
            fields = [fields]  # type: ignore

        self.fields = []       # type: List[_sql.ClauseElement]
        for f in fields:       # type: ignore
            if isinstance(f, str):
                self.fields.append(_sql.EscapedElement(f))
            elif isinstance(f, Field):
                self.fields.append(f)
            else:
                raise TypeError(f"invalid field type {type(f)}")

    def ddl(self) -> str:
        return str(self.__ddl__())

    def __ddl__(self) -> _sql.ClauseElement:
        return IndexDefine(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"<types.{self.__class__.__name__} object '{self.name}'>"

    __str__ = __repr__

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        ctx.sql(_sql.EscapedElement(self.name))
        return ctx


class Table(_sql.ClauseElement):

    __slots__ = (
        "name", "fields_dict", "primary", "indexes",
        "auto_increment", "engine", "charset", "comment",
        "_database",
    )

    PK = 'id'
    _META_DEFAULT = util.adict(
        auto_increment=1,
        engine="InnoDB",
        charset="utf8mb4",
        comment='',
    )

    def __init__(
        self,
        name: str,
        fields_dict: Dict[str, Field],
        primary: util.adict,
        indexes: Optional[Union[Tuple[Index, ...], List[Index]]] = None,
        engine: Optional[str] = None,
        charset: Optional[str] = None,
        comment: Optional[str] = None
    ) -> None:
        self.name = name
        self.fields_dict = fields_dict
        self.primary = primary
        self.indexes = indexes
        self.auto_increment = primary.begin or self._META_DEFAULT.auto_increment
        self.engine = engine or self._META_DEFAULT.engine
        self.charset = charset or self._META_DEFAULT.charset
        self.comment = comment or self._META_DEFAULT.comment

        for f in self.fields_dict:
            self.fields_dict[f].__table__ = self

        if not self.primary.field:
            raise err.NoPKError(
                f"primary key not found for table {self.name}"
            )

    def ddl(self) -> str:
        return str(self.__ddl__())

    def __ddl__(self) -> _sql.ClauseElement:
        return TableDefine(self)

    def __hash__(self) -> int:
        return hash(f"{self.name}")

    def __repr__(self) -> str:
        return f"<Table object '{self.name}'>"

    __str__ = __repr__

    def __sql__(self, ctx: _sql.Context) -> _sql.Context:
        if ctx.props.get("select") is True:
            ctx.sql(
                _sql.EscapedElement(self.name)
            ).literal(
                " AS "
            ).sql(
                _sql.EscapedElement(ctx.sources[self.name])
            )
        else:
            ctx.sql(_sql.EscapedElement(self.name))
        return ctx
