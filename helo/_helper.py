import re
import struct
import socket
from datetime import datetime
from typing import Any, Optional, List, Union, Callable, Type


_EMAIL_REGEX = re.compile(
    r'^ ('
    r'( ( [%(atext)s]+ (\.[%(atext)s]+)* ) | ("( [%(qtext)s\s] | \\[%(vchar)s\s] )*") )'
    r'@((?!-)[A-Z0-9-]{1,63}(?<!-)\.)+[A-Z]{2,63})$'
    % {
        'atext': '-A-Z0-9!#$%&\'*+/=?^_`{|}~',
        'qtext': '\x21\x23-\x5B\\\x5D-\x7E',
        'vchar': '\x21-\x7E'
    },
    re.I + re.X
)

_URL_REGEXP = re.compile(
    r'^(?:http|ftp)s?://'
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
    r'localhost|'
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
    r'(?::\d+)?'
    r'(?:/?|[/?]\S+)$',
    re.I
)

_TABLENAME_REGEX = re.compile(r'([a-z]|\d)([A-Z])')

_IPV4_REGEXP = re.compile(
    r'^( ((%(oct)s\.){3} %(oct)s) )'
    % {
        'oct': '( 25[0-5] | 2[0-4][0-9] | [0-1]?[0-9]{1,2} )'
    },
    re.I + re.X
)


def with_metaclass(meta: Type, *bases: Type) -> Type:

    class MetaClass(type):

        def __new__(cls, name, _this_bases, attrs):
            return meta(name, bases, attrs)

    return type.__new__(MetaClass, 'temporary_class', (), {})


def format_datetime(
        value: str,
        formats: Union[List[str], tuple],
        extractor: Optional[Callable] = None
) -> Any:
    extractor = extractor or (lambda x: x)
    for fmt in formats:
        try:
            return extractor(datetime.strptime(value, fmt))
        except ValueError:
            pass
    return value


def simple_datetime(value: str) -> Union[datetime, str]:
    try:
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        return value


def dt_strftime(value: Any, formats: Union[List[str], tuple]) -> str:
    if hasattr(value, 'strftime'):
        for fmt in formats:
            try:
                return value.strftime(fmt)
            except (TypeError, ValueError):
                pass
    return value


def iptoint(value: str) -> int:
    try:
        return struct.unpack('!I', socket.inet_aton(value))[0]
    except OSError:
        raise ValueError(f"illegal IP address({value}) for IP Field")


def iptostr(value: int) -> str:
    try:
        return socket.inet_ntoa(struct.pack('!I', value))
    except struct.error:
        raise ValueError("IP value must be range [0, 4294967295]"
                         f"got {value}")


def is_email(value: Any) -> bool:
    if not value:
        return False
    return _EMAIL_REGEX.match(value) is not None


def is_url(value: Any) -> bool:
    if not value:
        return False
    return _URL_REGEXP.match(value) is not None


def is_ipv4(value: Any) -> bool:
    if not value:
        return False
    return _IPV4_REGEXP.match(value) is not None


def snake_name(name: str) -> str:
    return re.sub(_TABLENAME_REGEX, r'\1_\2', name).lower()
