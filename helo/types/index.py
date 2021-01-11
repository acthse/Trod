from helo import _sql
from helo.types import _abc


class Key(_abc.Index):

    __slots__ = ()
    db_type = _sql.SQL("KEY")


class UKey(_abc.Index):

    __slots__ = ()
    db_type = _sql.SQL("UNIQUE KEY")
