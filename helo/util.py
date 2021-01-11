"""
    helo.util
    ~~~~~~~~~
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import wraps, reduce
from inspect import iscoroutinefunction, isclass, signature
from typing import Any, Dict, Callable, Type


__all__ = (
    'adict',
    'adictformatter',
    'asyncinit',
    'singleton',
    'singleton_asyncinit',
    'argschecker',
    'and_',
    'or_',
    'In',
    'FreeObject',
)


class adict(dict):  # pylint: disable=invalid-name
    """ Is a class that makes a dictionary behave like an object,
        with attribute-style access.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        keys = kwargs.pop("__keys__", None)
        values = kwargs.pop("__values__", None)

        super().__init__(*args, **kwargs)

        if keys and values:
            for key, value in zip(keys, values):
                self[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __iadd__(self, other: Dict[str, Any]) -> adict:
        self.update(other)
        return self

    def __add__(self, other: Dict[str, Any]) -> adict:
        td = self.copy()
        td.update(other)
        return td

    def copy(self) -> adict:
        return adict(**super().copy())


def adictformatter(func: Callable):
    """ A function decorator that convert the returned dict object to adict
        If it is a list, recursively convert its elements
    """

    if iscoroutinefunction(func):
        @wraps(func)
        async def convert(*args, **kwargs):
            result = await func(*args, **kwargs)
            return formatadict(result)
    else:
        @wraps(func)
        def convert(*args, **kwargs):
            result = func(*args, **kwargs)
            return formatadict(result)

    return convert


def import_object(name: str) -> Any:
    """Imports an object by name.

    ``import_object('x')`` is equivalent to ``import x``.
    ``import_object('x.y.z')`` is equivalent to ``from x.y import z``.
    """
    if name.count(".") == 0:
        return __import__(name)

    parts = name.split(".")
    obj = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
    try:
        return getattr(obj, parts[-1])
    except AttributeError:
        raise ImportError("No module named %s" % parts[-1])


def asyncinit(obj: Type):
    """A class decorator that add async `__init__` functionality."""

    if not isclass(obj):
        raise ValueError("decorated object must be a class")

    async def nnew(cls, *_args, **_kwargs):
        return object.__new__(cls)

    def force_async(f):
        if iscoroutinefunction(f):
            return f

        async def wrapped(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapped

    if obj.__new__ is object.__new__:
        cls_new = nnew
    else:
        cls_new = force_async(obj.__new__)

    @wraps(obj.__new__)
    async def new(cls, *args, **kwargs):
        self = await cls_new(cls, *args, **kwargs)

        cls_init = force_async(self.__init__)
        await cls_init(*args, **kwargs)

        return self

    obj.__new__ = new

    return obj


def singleton(cls: Type):
    """A singleton decorator of class"""

    instances = {}

    @wraps(cls)
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance


def singleton_asyncinit(cls: Type):
    """A singleton decorator of asyncinit class"""

    instances = {}

    @wraps(cls)
    async def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = await asyncinit(cls)(*args, **kwargs)
        return instances[cls]

    return getinstance


def argschecker(*cargs: Any, **ckwargs: Any):

    def decorator(func):
        # If in optimized mode, disable type checking
        if not __debug__:
            return func

        nullable = ckwargs.pop('nullable', True)

        # Map function argument names to supplied types
        sig = signature(func)
        bound_args = sig.bind_partial(*cargs, **ckwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_args:
                    if not isinstance(value, bound_args[name]):
                        raise TypeError(
                            f"argument {name} must be {bound_args[name]}, "
                            f"now got {type(value)}"
                        )
                    if not nullable and not value:
                        raise ValueError(f"arguments {name} cannot be empty")
            return func(*args, **kwargs)
        return wrapper

    return decorator


def and_(*exprs: Any) -> Any:
    return reduce(lambda a, b: a & b, exprs)


def or_(*exprs: Any) -> Any:
    return reduce(lambda a, b: a | b, exprs)


class In:

    def __init__(self, data, source):
        self._set = set(data)
        self._for = source

    def __str__(self):
        return f'<{self._for} object>'

    __repr__ = __str__

    def __len__(self):
        return len(self._set)

    def __contains__(self, name: str):
        return name.lower() in self._set

    def __getattr__(self, name: str):
        key = name.lower()
        if key in self._set:
            return key
        raise AttributeError(
            f"'{self._for}' object has no attribute '{name}'"
        )


class FreeObject:

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __bool__(self):
        return bool(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, name):
        return name in self.__dict__

    def __getitem__(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __delitem__(self, name):
        delattr(self, name)

    def __iter__(self):
        return iter(self.__dict__)

    def __iadd__(self, other):
        self.__dict__.update(other)
        return self

    def __add__(self, other):
        return self.as_new(**other.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return f'FreeObject({self.__dict__!r})'

    def as_new(self, **values):
        c = self.__class__.__new__(self.__class__)
        c.__dict__ = self.__dict__.copy()
        if values:
            c.__dict__.update(values)
        return c


def formatadict(original: Any):

    def do_format(ori_dict):
        td = adict()
        for key, value in ori_dict.items():
            td[key] = do_format(value) if isinstance(value, dict) else value
        return td

    if original is None:
        return original

    if isinstance(original, dict):
        return do_format(original)

    if isinstance(original, Iterable):
        fmted = []
        for item in original:
            if not isinstance(item, (list, tuple, dict)):
                raise TypeError(
                    f"invalid data type '{original}' to convert `adict`"
                )
            fmted.append(formatadict(item))
        return fmted
    raise TypeError(
        f"non-iterable object can not '{original}' to convert `adict`"
    )
