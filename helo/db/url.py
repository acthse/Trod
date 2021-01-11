import typing
from urllib import parse


class URL:

    __slots__ = ("_url", "_components", "_options")

    _BOOL_VALUES = ("True", "False", "true", "false")
    _BOOL_MAP = {"true": True, "false": False}

    def __init__(self, url: str) -> None:
        if not isinstance(url, str):
            raise TypeError(
                f"Invalid type for {self.__class__.__name__}. "
                f"Expected 'str', got {type(url)}"
            )
        self._url = url
        self._components = parse.urlsplit(url)  # type: parse.SplitResult

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    @property
    def scheme(self) -> str:
        return self._components.scheme

    @property
    def user(self) -> typing.Optional[str]:
        if self._components.username is None:
            return None
        return parse.unquote(self._components.username)

    @property
    def password(self) -> typing.Optional[str]:
        if self._components.password is None:
            return None
        return parse.unquote(self._components.password)

    @property
    def host(self) -> typing.Optional[str]:
        return self._components.hostname

    @property
    def port(self) -> typing.Optional[int]:
        return self._components.port

    @property
    def db(self) -> str:
        path = self._components.path
        if path.startswith("/"):
            path = path[1:]
        return parse.unquote(path)

    @property
    def options(self) -> typing.Dict[str, typing.Any]:
        if not hasattr(self, "_options"):
            options = dict(parse.parse_qsl(self._components.query))
            for arg, value in options.items():
                if value.isdigit():
                    options[arg] = int(value)  # type: ignore
                if value in self._BOOL_VALUES:
                    options[arg] = self._BOOL_MAP[value.lower()]  # type: ignore
            self._options = options
        return self._options
