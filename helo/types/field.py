from __future__ import annotations

import datetime
import decimal
import uuid
import warnings
from typing import Any, Optional, Union, Callable, List, Tuple, Dict

from helo import _sql
from helo import _helper
from helo import err
from helo.types import _abc


class Tinyint(_abc.Field):

    __slots__ = ('length', 'unsigned', 'zerofill')

    py_type = int
    db_type = 'tinyint'
    default_length = 4

    def __init__(
        self,
        length: Optional[int] = None,
        unsigned: bool = False,
        zerofill: bool = False,
        null: bool = True,
        default: Optional[Union[int, _sql.SQL, Callable]] = None,
        comment: str = '',
        name: Optional[str] = None
    ) -> None:
        self.length = length or self.default_length
        self.unsigned = unsigned
        self.zerofill = zerofill
        super().__init__(
            null=null, default=default, comment=comment, name=name
        )


class Smallint(Tinyint):

    __slots__ = ()

    db_type = 'smallint'
    default_length = 6


class Int(Tinyint):

    __slots__ = ('primary_key', 'auto')

    db_type = 'int'
    default_length = 11

    def __init__(
            self,
            length: Optional[int] = None,
            unsigned: bool = False,
            zerofill: bool = False,
            primary_key: bool = False,
            auto: bool = False,
            null: bool = True,
            default: Optional[Union[int, _sql.SQL, Callable]] = None,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        self.length = length or self.default_length
        self.primary_key = primary_key
        self.auto = auto
        if self.primary_key is True:
            null = False
            if default is not None:
                raise err.FieldInitError("primary key field not allow set default")
        elif self.auto:
            raise err.FieldInitError(
                "the 'AUTO_INCREMENT' cannot be set for non-primary key fields",
            )
        super().__init__(
            length=length,
            unsigned=unsigned,
            zerofill=zerofill,
            null=null,
            default=default,
            comment=comment,
            name=name
        )


class Bigint(Int):

    __slots__ = ()

    db_type = 'bigint'
    default_length = 20


class Auto(Int):

    __slots__ = ()

    default_length = 11

    def __init__(
            self,
            length: Optional[int] = None,
            unsigned: bool = False,
            zerofill: bool = False,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        super().__init__(
            length=length or self.default_length,
            unsigned=unsigned,
            zerofill=zerofill,
            primary_key=True,
            auto=True,
            null=False,
            default=None,
            comment=comment,
            name=name
        )


class BigAuto(Auto):

    __slots__ = ()

    db_type = 'bigint'
    default_length = 20


class Bool(_abc.Field):

    __slots__ = ()

    py_type = bool
    db_type = 'bool'

    def __init__(
            self,
            null: bool = True,
            default: Optional[Union[bool, _sql.SQL, Callable]] = None,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        super().__init__(
            null=null,
            default=default,
            comment=comment,
            name=name
        )

    def to_str(self, value: Any) -> str:
        if self.py_value(value):
            return "1"
        return "0"


class Float(_abc.Field):

    __slots__ = ('length', 'unsigned')

    py_type = float  # type: Any
    db_type = 'float'

    def __init__(
            self,
            length: Optional[Union[int, Tuple[int, int]]] = None,
            unsigned: bool = False,
            null: bool = True,
            default: Optional[Union[float, int, _sql.SQL, Callable]] = None,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        if not length or isinstance(length, int):
            self.length = length
        else:
            if isinstance(length, _abc.SEQUENCE) and len(length) == 2:
                self.length = tuple(length)  # type: ignore
            else:
                raise err.FieldInitError(f"invalid `Float` length type({length})")
        self.unsigned = unsigned
        super().__init__(
            null=null,
            default=default,
            comment=comment,
            name=name
        )


class Double(Float):

    __slots__ = ()

    db_type = 'double'


class Decimal(_abc.Field):

    __slots__ = ('length', 'unsigned', 'auto_round', 'rounding')

    py_type = decimal.Decimal
    db_type = 'decimal'
    default_md = (10, 5)

    def __init__(
            self,
            length: Optional[Tuple[int, int]] = None,
            unsigned: bool = False,
            null: bool = True,
            auto_round: bool = False,
            rounding: Optional[str] = None,
            default: Optional[Union[str, float, decimal.Decimal, _sql.SQL, Callable]] = None,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        if length:
            if not isinstance(length, tuple) or len(length) != 2:
                raise err.FieldInitError(
                    "the `Decimal` length "
                    "must be tuple or list")
        self.length = tuple(length or self.default_md)
        self.unsigned = unsigned
        self.auto_round = auto_round
        self.rounding = rounding or decimal.DefaultContext.rounding
        super().__init__(
            null=null,
            default=default,
            comment=comment,
            name=name
        )

    def db_value(self, value: Any) -> Optional[decimal.Decimal]:
        if not value:
            return value if value is None else self.py_type(0)
        if self.auto_round:
            exp = self.py_type(10) ** (-self.length[1])  # type: ignore
            rounding = self.rounding
            return self.py_type(str(value)).quantize(exp, rounding=rounding)
        return self.py_type(str(value))

    def py_value(self, value: Any) -> Optional[decimal.Decimal]:
        if value is not None:
            if isinstance(value, self.py_type):
                return value
            return self.py_type(str(value))
        return None


class Text(_abc.Field):

    __slots__ = ('encoding',)

    py_type = str
    db_type = 'text'

    def __init__(  # pylint: disable=super-init-not-called
            self,
            encoding: Optional[str] = None,
            null: bool = True,
            comment: str = '',
            name: str = ''
    ) -> None:
        if encoding and encoding not in _abc.ENCODING:
            raise err.FieldInitError(f"unsupported encoding '{encoding}'")
        self.encoding = encoding
        self.null = null
        self.comment = comment
        self.name = name
        self.table = None

    def __add__(self, other: Any) -> _abc.StrExpression:
        return _abc.StrExpression(self, _abc.OPERATOR.CONCAT, other)

    def __radd__(self, other: Any) -> _abc.StrExpression:
        return _abc.StrExpression(other, _abc.OPERATOR.CONCAT, self)


class Char(_abc.Field):

    __slots__ = ('length', 'encoding',)

    py_type = str
    db_type = 'char'
    default_length = 254

    def __init__(
            self,
            length: Optional[int] = None,
            encoding: Optional[str] = None,
            null: bool = True,
            default: Optional[Union[str, _sql.SQL, Callable]] = None,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        self.length = length or self.default_length
        if encoding and encoding not in _abc.ENCODING:
            raise ValueError(f"unsupported encoding '{encoding}'")
        self.encoding = encoding
        super().__init__(
            null=null, default=default, comment=comment, name=name
        )

    def __add__(self, other: Any) -> _abc.StrExpression:
        return _abc.StrExpression(self, _abc.OPERATOR.CONCAT, other)

    def __radd__(self, other: Any) -> _abc.StrExpression:
        return _abc.StrExpression(other, _abc.OPERATOR.CONCAT, self)


class VarChar(Char):

    __slots__ = ()

    db_type = 'varchar'


class UUID(_abc.Field):

    __slots__ = ("primary_key",)

    py_type = uuid.UUID
    db_type = "varchar(40)"

    def __init__(
            self,
            primary_key: bool = False,
            default: Optional[Union[str, uuid.UUID, _sql.SQL, Callable]] = None,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        self.primary_key = primary_key
        if self.primary_key is True and default is not None:
            raise err.FieldInitError("primary key field not allow set default")
        super().__init__(
            null=False, default=default, comment=comment, name=name
        )

    def db_value(self, value: Any) -> str:
        if isinstance(value, str) and len(value) == 32:
            return value
        if isinstance(value, bytes) and len(value) == 16:
            value = self.py_type(bytes=value)

        if isinstance(value, self.py_type):
            return value.hex
        try:
            return self.py_type(value).hex
        except Exception:  # pylint: disable=broad-except
            return value

    def py_value(self, value: Any) -> Optional[uuid.UUID]:
        if isinstance(value, self.py_type):
            return value
        return self.py_type(value) if value is not None else None


class IP(Bigint):
    """IPV4"""

    __slots__ = ()

    py_type = str  # type: ignore
    MIN, MAX = 0, 4294967295

    def db_value(self, value: Optional[str]) -> Optional[int]:
        if value is not None:
            if isinstance(value, int) and (IP.MIN <= value <= IP.MAX):
                return value
            if not _helper.is_ipv4(value):
                raise ValueError(IP.VALUEERR_MSG.format(value, "IP"))
            return _helper.iptoint(str(value))
        return value

    def py_value(self, value: Union[str, int, None]) -> Optional[str]:
        if value is not None:
            if isinstance(value, int):
                return _helper.iptostr(value)
            if not isinstance(value, str):
                raise TypeError(f"invalid type({value!r}) for IP Field")
            if not _helper.is_ipv4(value):
                raise ValueError(IP.VALUEERR_MSG.format(value, "IP"))
        return value


class Email(VarChar):

    __slots__ = ()

    default_length = 100

    def adapt(self, value: Any) -> Optional[str]:
        if value is not None:
            if not isinstance(value, self.py_type):
                value = self.py_type(value)
            if not value:
                return value
            if not _helper.is_email(value):
                raise ValueError(Email.VALUEERR_MSG.format(value, "Email"))
        return value


class URL(VarChar):

    __slots__ = ()

    def adapt(self, value: Any) -> Optional[str]:
        if value is not None:
            if not isinstance(value, self.py_type):
                value = self.py_type(value)
            if not value:
                return value
            if not _helper.is_url(value):
                raise ValueError(URL.VALUEERR_MSG.format(value, "URL"))
        return value


class Date(_abc.Field):

    __slots__ = ('formats',)

    py_type = (datetime.datetime, datetime.date)  # type: Any
    db_type = 'date'

    FORMATS = (
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
    )

    def __init__(
            self,
            formats: Optional[Union[List[str], Tuple[str, ...]]] = None,
            null: bool = True,
            default: Optional[Union[datetime.datetime, datetime.date, str, _sql.SQL, Callable]] = None,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        if formats is not None:
            if isinstance(formats, str):
                formats = [formats]
        self.formats = formats or self.__class__.FORMATS
        super().__init__(
            null=null, default=default, comment=comment, name=name
        )

    def __call__(self, *_args: Any, **_kwargs: Any) -> datetime.date:
        return datetime.datetime.now().date()

    def adapt(self, value: Any) -> Optional[datetime.date]:
        if value and isinstance(value, str):
            value = _helper.format_datetime(value, self.formats, lambda x: x.date())
        elif value and isinstance(value, datetime.datetime):
            value = value.date()
        return value

    def to_str(self, value: Any) -> str:
        return _helper.dt_strftime(self.db_value(value), self.formats)


class Time(Date):

    __slots__ = ()

    py_type = (datetime.datetime, datetime.time)
    db_type = 'time'

    FORMATS = (  # type: ignore
        '%H:%M:%S.%f',
        '%H:%M:%S',
        '%H:%M',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
    )

    def __call__(self, *_args: Any, **_kwargs: Any) -> datetime.time:  # type: ignore
        return datetime.datetime.now().time()

    def adapt(self, value: Any) -> Optional[datetime.time]:  # type:ignore
        if value:
            if isinstance(value, str):
                value = _helper.format_datetime(value, self.formats, lambda x: x.time())  # type: ignore
            elif isinstance(value, datetime.datetime):
                value = value.time()
        if value is not None and isinstance(value, datetime.timedelta):
            return (datetime.datetime.min + value).time()
        return value


class DateTime(Date):

    __slots__ = ()

    py_type = datetime.datetime
    db_type = 'datetime'

    FORMATS = (
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    )

    def __call__(self, *args: Any, **kwargs: Any) -> datetime.datetime:
        return datetime.datetime.now()

    def adapt(self, value: Any) -> Optional[datetime.datetime]:  # type: ignore
        if value and isinstance(value, str):
            return _helper.format_datetime(value, self.formats)
        return value


class Timestamp(_abc.Field):

    __slots__ = ('utc',)

    py_type = datetime.datetime
    db_type = 'timestamp'

    FORMATS = (
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
    )

    def __init__(
            self,
            utc: bool = False,
            null: bool = True,
            default: Optional[Union[datetime.datetime, str, _sql.SQL, Callable]] = None,
            comment: str = '',
            name: Optional[str] = None
    ) -> None:
        self.utc = utc
        super().__init__(
            null=null, default=default, comment=comment, name=name
        )

    def db_value(
        self, value: Union[datetime.datetime, int, str]
    ) -> Optional[datetime.datetime]:
        if value is None:
            return value
        if not isinstance(value, datetime.datetime):
            if isinstance(value, datetime.date):
                value = datetime.datetime(value.year, value.month, value.day)
            elif isinstance(value, str):
                value = _helper.simple_datetime(value)
            else:
                value = int(round(value))
                if self.utc:
                    value = datetime.datetime.utcfromtimestamp(value)
                else:
                    value = datetime.datetime.fromtimestamp(value)
        return value  # type: ignore

    def py_value(self, value: Any) -> Optional[datetime.datetime]:
        if value is not None:
            if isinstance(value, (int, float)):
                if self.utc:
                    value = datetime.datetime.utcfromtimestamp(value)
                else:
                    value = datetime.datetime.fromtimestamp(value)
            else:
                value = _helper.simple_datetime(value)
        return value

    def to_str(self, value: Any) -> str:
        return _helper.dt_strftime(self.db_value(value), self.FORMATS)
