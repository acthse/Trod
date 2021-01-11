"""
    helo.model
    ~~~~~~~~~~

    Implements the model.
"""
from __future__ import annotations

import warnings
import re
from copy import deepcopy
from typing import Any, Dict, Optional, List, Union, Tuple, Type

from . import db, util, err, types, _builder, _helper

__all__ = (
    'Model',
    "JOINTYPE",
    "ROWTYPE",
)


ROWTYPE = util.adict(
    MODEL=1,
    ADICT=2,
)
JOINTYPE = util.adict(
    INNER='INNER',
    LEFT='LEFT',
    RIGHT='RIGHT',
)
_TABLENAME_REGEX = re.compile(r'([a-z]|\d)([A-Z])')
_BUILTIN_MODEL_NAMES = ("ModelBase", "Model", "HeloModel")


class ModelType(type):

    def __new__(cls, name: str, bases: Tuple[type, ...], attrs: dict) -> ModelType:

        def __prepare__():
            model_fields, model_attrs = {}, {}
            for attr in attrs.copy():
                field = attrs[attr]
                if isinstance(field, types.FieldBase):
                    field.name = field.name or attr
                    model_fields[attr] = field
                    model_attrs[field.name] = attr
                    attrs.pop(attr)

            baseclass = bases[0] if bases else None
            if baseclass:
                base_table = deepcopy(baseclass.__table__)
                if base_table:
                    base_table.fields_dict.update(model_fields)
                    model_fields = base_table.fields_dict
                    base_names = deepcopy(baseclass.__attrs__)
                    base_names.update(model_attrs)
                    model_attrs = base_names

            metaclass = attrs.get('Meta')
            if not metaclass:
                baseclass = bases[0] if bases else metaclass
                metaclass = getattr(baseclass, 'Meta', None)

            indexes = getattr(metaclass, 'indexes', [])
            if indexes and not isinstance(indexes, (tuple, list)):
                raise TypeError("the Table.indexes type must be `tuple` or `list`")
            for index in indexes:
                if not isinstance(index, types.IndexBase):
                    raise TypeError(f"invalid index type {index!r}")

            primary = util.adict(auto=False, field=None, attr=None, begin=None)
            for attr_name, field in model_fields.items():
                if getattr(field, 'primary_key', None):
                    if primary.field is not None:
                        raise err.DuplicatePKError(
                            "duplicate primary key found for field "
                            f"{field.name}"
                        )
                    primary.field = field
                    primary.attr = attr_name
                    if getattr(field, "auto", False):
                        primary.auto = True
                        primary.begin = int(field.auto)
                        if field.name != types.Table.AIPK:
                            warnings.warn(
                                "The field name of AUTO_INCREMENT "
                                "primary key is suggested to use "
                                f"`id` instead of {field.name}",
                                err.ProgrammingWarning)

            attrs["__attrs__"] = model_attrs
            attrs["__table__"] = types.Table(
                database=getattr(metaclass, "db", None),
                name=getattr(metaclass, 'name',
                             re.sub(_TABLENAME_REGEX, r'\1_\2', name).lower()),
                fields_dict=model_fields,
                primary=primary,
                indexes=indexes,
                engine=getattr(metaclass, "engine", None),
                charset=getattr(metaclass, "charset", None),
                comment=getattr(metaclass, "comment", None),
            )

            return attrs

        attrs['__table__'] = None
        if name not in _BUILTIN_MODEL_NAMES:
            attrs = __prepare__()

        return type.__new__(cls, name, bases, attrs)  # type: ignore

    def __getattr__(cls, name: str) -> Any:
        if cls.__table__ is not None:
            if name in cls.__table__.fields_dict:
                return cls.__table__.fields_dict[name]

        raise AttributeError(
            f"'{cls.__name__}' class does not have attribute '{name}'"
        )

    # def __setattr__(cls, *_args: Any) -> None:
    #     raise err.NotAllowedError(
    #         f"Model '{cls.__name__}' class not allow set attribute")

    # def __delattr__(cls, name: str) -> None:
    #     raise err.NotAllowedError(
    #         f"Model '{cls.__name__}' class not allow delete attribute"
    #     )

    def __repr__(cls) -> str:
        if cls.__name__ in _BUILTIN_MODEL_NAMES:
            return f"helo.{cls.__name__}"
        return f"Model<{cls.__name__}>"

    def __str__(cls) -> str:
        return cls.__name__

    def __hash__(cls) -> int:
        if cls.__table__:
            return hash(cls.__table__)
        return 0

    def __aiter__(cls) -> Select:
        return ApiProxy.select(cls)  # type: ignore

    def __getitem__(cls, _id: types.ID) -> Model:
        raise NotImplementedError

    def __contains__(cls, _id: types.ID) -> bool:
        raise NotImplementedError


def get_table(m: Union[Type[Model], Model]) -> types.Table:
    try:
        return m.__table__
    except AttributeError:
        raise err.ProgrammingError("Must be ModelType")


def get_attrs(m: Union[Type[Model], Model]) -> Dict[str, Any]:
    try:
        return m.__attrs__
    except AttributeError:
        raise err.ProgrammingError("Must be ModelType")


class ModelBase:

    def __init__(self, **kwargs: Any) -> None:
        for attr in kwargs:
            setattr(self, attr, kwargs[attr])

    def __repr__(self) -> str:
        id_ = getattr(self, self.__table__.primary.attr, None)
        return f"<{self.__class__.__name__} object {id_}>"

    __str__ = __repr__

    def __hash__(self) -> int:
        return hash(self.__table__)

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__

    def __setattr__(self, name: str, value: Any) -> None:
        self.__setmodel__(name, value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__[name]
        except KeyError:
            if name in self.__table__.fields_dict:
                return None
            raise AttributeError(
                f"'{self.__class__}' object has no attribute '{name}'"
            )

    def __bool__(self) -> bool:
        return bool(self.__dict__)

    def __setmodel__(
            self, name: str, value: Any, __load__: bool = False
    ) -> None:
        f = self.__table__.fields_dict.get(name)
        if not f:
            raise err.NotAllowedError(
                f"{self.__class__.__name__} object not allowed "
                f"set attribute '{name}'"
            )

        if not __load__:
            if getattr(f, 'primary_key', None) and getattr(f, 'auto', None):
                raise err.NotAllowedError(
                    f"auto field '{f.name}' not allowed to set"
                )

        value = f.py_value(value)
        self.__dict__[name] = value

    @property
    def __self__(self) -> Dict[str, Any]:
        return deepcopy(self.__dict__)


class Model(_helper.with_metaclass(ModelType, ModelBase)):  # type: ignore
    """From Model defining your model is easy

    >>> import helo
    >>>
    >>> class User(helo.Model):
    ...     id = helo.Auto()
    ...     nickname = helo.VarChar(length=45)
    ...     password = helo.VarChar(length=100)
    """

    @classmethod
    async def create(cls, **options: Any) -> db.ExecResult:
        """Create a table in the database from the model"""

        return await ApiProxy.create_table(cls, **options)

    @classmethod
    async def drop(cls, **options: Any) -> db.ExecResult:
        """Drop a table in the database from the model"""

        return await ApiProxy.drop_table(cls, **options)

    @classmethod
    def show(cls) -> Show:
        """Show information about table"""

        return ApiProxy.show(cls)

    #
    # Simple API for short
    #
    @classmethod
    async def get(
        cls,
        by: Union[types.ID, types.Expression]
    ) -> Union[None, Model]:
        """Getting a row by the primary key
        or simple query expression

        >>> user = await User.get(1)
        >>> user
        <User objetc> at 1
        >>> user.nickname
        'at7h'
        """

        if not by:
            return None
        return await ApiProxy.get(cls, by)

    @classmethod
    async def mget(
        cls,
        by: Union[List[types.ID], types.Expression],
        columns: Optional[List[types.Column]] = None,
    ) -> db.FetchResult:
        """Getting rows by the primary key list
        or simple query expression

        >>> await User.mget([1, 2, 3])
        [<User object at 1>, <User object at 2>, <User object at 3>]
        """

        if not by:
            raise ValueError("no condition to mget")
        return await ApiProxy.get_many(cls, by, columns=columns)

    @classmethod
    async def add(
        cls,
        __row: Optional[Dict[str, Any]] = None,
        **values: Any
    ) -> types.ID:
        """Adding a row, simple and shortcut of ``insert``

        # Using keyword arguments:
        >>> await User.add(nickname='at7h', password='7777')
        1

        # Using values dict:
        >>> await User.add({'nickname': 'at7h', 'password': '777'})
        1
        """

        row = __row or values
        if not row:
            raise ValueError("no data to add")
        return await ApiProxy.add(cls, row)

    @classmethod
    async def madd(
        cls,
        rows: Union[List[Dict[str, Any]], List[Model]]
    ) -> int:
        """Adding multiple, simple and shortcut of ``minsert``

        # Using values dict list:
        >>> users = [
        ...    {'nickname': 'at7h', 'password': '777'}
        ...    {'nickname': 'mebo', 'password': '666'}]
        >>> await User.madd(users)
        2

        # Adding User object list:
        >>> users = [User(**u) for u in users]
        >>> await User.madd(users)
        2
        """

        if not rows:
            raise ValueError("no data to madd")
        return await ApiProxy.add_many(cls, rows)

    @classmethod
    async def set(cls, _id: types.ID, **values: Any) -> int:
        """Setting the value of a row with the primary key

        >>> user = await User.get(1)
        >>> user.password
        777
        >>> await User.set(1, password='888')
        1
        >>> user = await User.get(1)
        >>> user.password
        888
        """

        if not values:
            raise ValueError('no _id or values to set')
        return await ApiProxy.set(cls, _id, values)

    # API that translates directly from SQL statements(DQL, DML).
    # You have to explicitly execute them via methods like `do()`.
    @classmethod
    def select(cls, *columns: types.Column) -> Select:
        """Select Query, see ``Select``"""

        return ApiProxy.select(cls, *columns)

    @classmethod
    def insert(
        cls, __row: Optional[Dict[str, Any]] = None, **values: Any
    ) -> Insert:
        """Inserting a row

        # Using keyword arguments:
        >>> await User.insert(nickname='at7h', password='777').do()
        ExecResult(affected: 1, last_id: 1)

        # Using values dict list:
        >>> await User.insert({
        ...     'nickname': 'at7h',
        ...     'password': '777',
        ... }).do()
        ExecResult(affected: 1, last_id: 1)
        """

        row = __row or values
        if not row:
            raise ValueError("no data to insert")
        return ApiProxy.insert(cls, row)

    @classmethod
    def minsert(
        cls,
        rows: List[Union[Dict[str, Any], Tuple[Any, ...]]],
        columns: Optional[List[types.FieldBase]] = None
    ) -> Insert:
        """Inserting multiple

        # Using values dict list:
        >>> users = [
        ...    {'nickname': 'Bob', 'password': '666'},
        ...    {'nickname': 'Her', 'password: '777'},
        ...    {'nickname': 'Nug', 'password': '888'}]

        >>> result = await User.insert(users).do()

        # We can also specify row tuples
        # columns the tuple values correspond to:
        >>> users = [
        ...    ('Bob', '666'),
        ...    ('Her', '777'),
        ...    ('Nug', '888')]
        >>> result = await User.insert(
        ...    users, columns=[User.nickname, User.password]
        ... ).do()
        """

        if not rows:
            raise ValueError("no data to minsert {}")
        return ApiProxy.insert_many(cls, rows, columns=columns)

    @classmethod
    def insert_from(
        cls, from_: Select, columns: List[types.Column]
    ) -> Insert:
        """Inserting from select clause

        >>> select = Employee.Select(
        ...     Employee.id, Employee.name
        ... ).where(Employee.id < 10)
        >>>
        >>> User.insert_from(select, [User.id, User.name]).do()
        """

        if not columns:
            raise ValueError("insert_from must specify columns")
        return ApiProxy.insert(cls, list(columns), from_select=from_)

    @classmethod
    def update(cls, **values: Any) -> Update:
        """Updating record

        >>> await User.update(
        ...    password='888').where(User.id == 1
        ... ).do()
        ExecResult(affected: 1, last_id: 0)
        """
        if not values:
            raise ValueError("no data to update")
        return ApiProxy.update(cls, values)

    @classmethod
    def delete(cls) -> Delete:
        """Deleting record

        >>> await User.delete().where(User.id == 1).do()
        ExecResult(affected: 1, last_id: 0)
        """
        return ApiProxy.delete(cls)

    @classmethod
    def replace(
        cls, __row: Optional[Dict[str, Any]] = None, **values: Any
    ) -> Replace:
        """MySQL REPLACE, similar to ``insert``"""

        row = __row or values
        if not row:
            raise ValueError("no data to replace")
        return ApiProxy.replace(cls, row)

    @classmethod
    def mreplace(
        cls,
        rows: List[Union[Dict[str, Any], Tuple[Any, ...]]],
        columns: Optional[List[types.FieldBase]] = None
    ) -> Replace:
        """MySQL REPLACE, similar to ``minsert``"""

        if not rows:
            raise ValueError("no data to mreplace")
        return ApiProxy.replace_many(cls, rows, columns=columns)

    # instance

    async def save(self) -> types.ID:
        """Write objects in memory to database

        >>> user = User(nickname='at7h',password='777')
        >>> await user.save()
        1
        """
        return await ApiProxy.save(self)

    async def remove(self) -> int:
        """Removing a row

        >>> user = await User.get(1)
        >>> await user.remove()
        1
        >>> await User.get(1)
        None
        """
        return await ApiProxy.remove(self)


class ApiProxy:
    """Implementation of the Model API"""

    @classmethod
    async def create_table(
        cls, m: Type[Model], **options: Any
    ) -> db.ExecResult:
        """Do create table"""

        if m.__name__ in _BUILTIN_MODEL_NAMES:
            raise err.NotAllowedError(f"{m.__name__} is built-in model name")

        return await Create(get_table(m), **options).do()

    @classmethod
    async def drop_table(
        cls, m: Type[Model], **options: Any
    ) -> db.ExecResult:
        """Do drop table"""

        if m.__name__ in _BUILTIN_MODEL_NAMES:
            raise err.NotAllowedError(f"{m.__name__} is built-in model name")

        return await Drop(get_table(m), **options).do()

    @classmethod
    def show(cls, m: Type[Model]) -> Show:
        return Show(get_table(m))

    @classmethod
    async def get(
        cls,
        m: Type[Model],
        by: Union[types.ID, types.Expression],
    ) -> Union[None, Model]:

        where = by
        if not isinstance(where, types.Expression):
            where = get_table(m).primary.field == where
        return (await Select([_builder.SQL("*")], [m]).where(where)  # type: ignore
                .get())

    @classmethod
    @util.argschecker(by=(types.SEQUENCE, types.Expression))
    async def get_many(
        cls,
        m: Type[Model],
        by: Union[List[types.ID], types.Expression],
        columns: Optional[List[types.Column]] = None,
    ) -> db.FetchResult:

        where = by
        if isinstance(where, types.SEQUENCE):
            where = get_table(m).primary.field.in_(by)
        return await (
            Select(columns or [_builder.SQL("*")], [m]).where(where).all()  # type: ignore
        )

    @classmethod
    @util.argschecker(row=dict, nullable=False)
    async def add(
        cls,
        m: Type[Model],
        row: Dict[str, Any]
    ) -> types.ID:

        addrow = cls._gen_insert_row(m, row)
        return (
            await Insert(get_table(m), ValuesMatch(addrow)).do()
        ).last_id

    @classmethod
    @util.argschecker(rows=list, nullable=False)
    async def add_many(
        cls,
        m: Type[Model],
        rows: Union[List[Dict[str, Any]], List[Model]]
    ) -> int:

        addrows = []
        for row in rows:
            if isinstance(row, m):
                addrows.append(cls._gen_insert_row(m, row.__self__))
            elif isinstance(row, dict):
                addrows.append(cls._gen_insert_row(m, row))
            else:
                raise ValueError(f"invalid data {row!r} to add")

        return (
            await Insert(get_table(m), ValuesMatch(addrows), many=True).do()
        ).affected

    @classmethod
    @util.argschecker(values=dict, nullable=False)
    async def set(
        cls,
        m: Type[Model],
        _id: types.ID,
        values: Any
    ) -> int:

        table = get_table(m)
        values = cls._normalize_update_values(m, values)
        return (await Update(
            table, AssignmentList(values)
        ).where(
            table.primary.field == _id
        ).do()
        ).affected

    @classmethod
    def select(
        cls, m: Type[Model], *columns: types.Column
    ) -> Select:

        return Select(list(columns) or [_builder.SQL("*")], [m])  # type: ignore

    @classmethod
    def insert(
        cls,
        m: Type[Model],
        row: Union[Dict[str, Any], List[types.Column]],
        from_select: Optional[Select] = None
    ) -> Insert:

        if isinstance(row, dict):
            toinsert = cls._gen_insert_row(m, row.copy())
            return Insert(get_table(m), ValuesMatch(toinsert))
        if from_select is None:
            raise ValueError('`from_select` cannot be None')
        return Insert(get_table(m), row).from_(from_select)

    @classmethod
    @util.argschecker(rows=types.SEQUENCE)
    def insert_many(
        cls,
        m: Type[Model],
        rows: List[Union[Dict[str, Any], Tuple[Any, ...]]],
        columns: Optional[List[types.FieldBase]] = None
    ) -> Insert:

        normalize_rows = cls._normalize_insert_rows(m, rows, columns)
        return Insert(
            get_table(m), ValuesMatch(normalize_rows), many=True
        )

    @classmethod
    def update(cls, m: Type[Model], values: Dict[str, Any]) -> Update:

        values = cls._normalize_update_values(m, values)
        return Update(get_table(m), AssignmentList(values))

    @classmethod
    def delete(cls, m: Type[Model]) -> Delete:

        return Delete(get_table(m))

    @classmethod
    def replace(cls, m: Type[Model], row: Dict[str, Any]) -> Replace:

        toreplace = cls._gen_insert_row(m, row, for_replace=True)
        return Replace(get_table(m), ValuesMatch(toreplace))

    @classmethod
    def replace_many(
        cls,
        m: Type[Model],
        rows: List[Union[Dict[str, Any], Tuple[Any, ...]]],
        columns: Optional[List[types.FieldBase]] = None
    ) -> Replace:

        normalize_rows = cls._normalize_insert_rows(m, rows, columns, for_replace=True)
        return Replace(get_table(m), ValuesMatch(normalize_rows), many=True)

    @classmethod
    async def save(cls, mo: Model) -> types.ID:
        """ Save model object to db """

        has_id = False
        pk_attr = get_table(mo).primary.attr
        if pk_attr in mo.__self__:
            has_id = True

        row = cls._gen_insert_row(mo, mo.__self__, for_replace=has_id)
        result = await Replace(get_table(mo), ValuesMatch(row)).do()
        mo.__setmodel__(
            name=pk_attr,
            value=result.last_id,
            __load__=True
        )
        return result.last_id

    @classmethod
    async def remove(cls, mo: Model) -> int:

        table = get_table(mo)
        primary_value = getattr(mo, table.primary.attr, None)
        if not primary_value:
            raise RuntimeError("remove object has no primary key value")

        ret = await Delete(
            table
        ).where(
            table.primary.field == primary_value
        ).do()
        return ret.affected

    @classmethod
    @util.argschecker(row_data=dict, nullable=False)
    def _gen_insert_row(
        cls,
        m: Type[Model],
        row_data: Dict[str, Any],
        for_replace: bool = False
    ) -> Dict[str, Any]:

        toinserts = {}
        for name, field in get_table(m).fields_dict.items():
            # Primary key fields should not be included when not for_replace
            if name == get_table(m).primary.attr and not for_replace:
                continue

            value = row_data.pop(name, None)
            # if value is None, to get default
            if value is None:
                if hasattr(field, 'default'):
                    default = field.default() if callable(field.default) else field.default
                    if isinstance(default, _builder.SQL):
                        continue
                    value = default
            if value is None and not field.null:
                if not for_replace:
                    raise ValueError(
                        f"invalid data(None) for not null attribute {name}"
                    )
            try:
                toinserts[field.name] = field.db_value(value)
            except (ValueError, TypeError):
                raise ValueError(f'invalid data({value}) for {name}')

        for attr in row_data:
            if not for_replace and attr == get_table(m).primary.attr:
                raise err.NotAllowedError(
                    f"auto field {attr!r} not allowed to set"
                )
            raise ValueError(f"'{m!r}' has no attribute {attr}")

        return toinserts

    @classmethod
    def _normalize_insert_rows(
        cls,
        m: Type[Model],
        rows: List[Union[Dict[str, Any], Tuple[Any, ...]]],
        columns: Optional[List[types.FieldBase]] = None,
        for_replace: bool = False,
    ) -> List[Dict[str, Any]]:

        cleaned_rows = []  # type: List[Dict[str, Any]]

        if columns:
            if not isinstance(columns, list):
                raise ValueError("specify columns must be list")
            mattrs = get_attrs(m)
            for c in columns:
                if not isinstance(c, types.FieldBase):
                    raise TypeError(f"invalid type of columns element {c}")

                if c.name not in mattrs:
                    raise ValueError(f"{m!r} has no attribute {c.name}")
                c = mattrs[c.name]

            for row in rows:
                if not isinstance(row, types.SEQUENCE):
                    raise ValueError(f"invalid data {row!r} for specify columns")
                row = dict(zip(columns, row))  # type: ignore
                if len(row) != len(columns):
                    raise ValueError("no enough data for columns")

                cleaned_rows.append(cls._gen_insert_row(m, row, for_replace))
        else:
            cleaned_rows = [cls._gen_insert_row(m, r, for_replace) for r in rows]
        return cleaned_rows

    @classmethod
    def _normalize_update_values(
        cls, m: Type[Model], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        table = get_table(m)
        normalized_values = {}  # type: Dict[str, Any]
        for attr in values:
            f = table.fields_dict.get(attr)
            if f is None:
                raise ValueError(f"'{m!r}' has no attribute {attr}")
            v = values[attr]
            if not isinstance(v, _builder.Node):
                v = f.db_value(v)
            normalized_values[f.name] = v
        return normalized_values


class ValuesMatch(_builder.Node):

    __slots__ = ("_columns", "_params", "_values")

    def __init__(
        self, rows: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> None:

        if isinstance(rows, dict):
            columns = list(rows.keys())
            self._values = tuple(rows.values())
        elif isinstance(rows, list):
            columns = list(rows[0].keys())
            self._values = tuple([tuple(r.values()) for r in rows])
        else:
            raise ValueError("invalid data unpack to values")

        self._columns, self._params = [], []  # type: List[_builder.Node], List[_builder.Node]
        for col in columns:
            self._columns.append(_builder.SQL(col.join("``")))
            self._params.append(_builder.SQL("%s"))

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        ctx.literal(' ').sql(_builder.EnclosedNodeList(self._columns))
        ctx.literal(
            " VALUES "
        ).sql(
            _builder.EnclosedNodeList(self._params)
        ).values(self._values)
        return ctx


class Join(_builder.Node):

    __slots__ = ('lt', 'rt', 'join_type', '_on')

    def __init__(
        self,
        lt: types.Table,
        rt: types.Table,
        join_type: str = JOINTYPE.INNER,
        on: Optional[types.Expression] = None
    ):
        self.lt = lt
        self.rt = rt
        self.join_type = join_type
        self._on = on

    def on(self, expr: types.Expression):
        self._on = expr
        return self

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        with ctx(params=True):
            ctx.sql(
                self.lt
            ).literal(
                f' {self.join_type} JOIN '
            ).sql(
                self.rt
            )
            if self._on is not None:
                ctx.literal(' ON ').sql(self._on)
        return ctx


class AssignmentList(_builder.Node):

    __slots__ = ('_data_dict',)

    _VSM = "`{col}` = {val}"

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data_dict = data

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        values, params = [], []
        for col, value in self._data_dict.items():
            if isinstance(value, types.FieldBase):
                values.append(_builder.SQL(
                    self._VSM.format(
                        col=col,
                        val="{}.{}".format(
                            value.table.table_name,
                            value.column)
                    )
                ))
            elif isinstance(value, types.Expression):
                query = _builder.parse(value)
                values.append(_builder.SQL(
                    self._VSM.format(
                        col=col,
                        val=query.sql[0:-1]
                    )
                ))
                params.append(query.params)
            else:
                values.append(_builder.SQL(
                    self._VSM.format(col=col, val='%s')
                ))
                params.append(value)

        ctx.sql(
            _builder.CommaNodeList(values)  # type: ignore
        )
        if params:
            ctx.values(params)

        return ctx


class BaseQuery(_builder.Node):

    __slots__ = ('_props', '_aliases')
    __fread__ = True

    def __init__(self) -> None:
        self._props = util.adict()
        self._aliases = {}  # type: Dict[str,Any]

    def __repr__(self) -> str:
        return repr(self.query)

    def __str__(self) -> str:
        return str(self.query)

    def __query__(self) -> _builder.Query:
        ctx = _builder.Context.from_node(self)
        self._aliases = ctx.aliases
        return ctx.query_of()

    @property
    def query(self) -> _builder.Query:
        return self.__query__()

    async def __do__(self, **props) -> Any:
        query = self.query
        query.r = self.__fread__
        if props:
            self._props.update(props)
        return await db.execute(query, **self._props)

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        raise NotImplementedError


class WriteQuery(BaseQuery):

    __slots__ = ()
    __fread__ = False

    async def do(self) -> db.ExecResult:
        return await self.__do__()

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        raise NotImplementedError


class Select(BaseQuery):

    __slots__ = (
        '_models', '_columns', '_froms', '_where',
        '_group_by', '_having', '_order_by', '_limit',
        '_offset', '_rowtype', '_gotlist', '_gotidx',
    )
    _SINGLE = 1
    _BATCH = 200

    def __init__(
        self,
        columns: Union[List[types.Column], List[_builder.SQL]],
        models: List[Type[Model]]
    ) -> None:
        super().__init__()
        self._columns = columns
        self._models = models
        self._froms = [get_table(model) for model in models]  # type:List[_builder.Node]
        self._where = None
        self._group_by = None
        self._having = None
        self._order_by = None
        self._limit = None     # type: Optional[int]
        self._offset = None    # type: Optional[int]
        self._gotlist = []     # type: List[Model]
        self._gotidx = 0
        self._rowtype = ROWTYPE.MODEL

    def join(
        self,
        target: Type[Model],
        join_type: str = JOINTYPE.INNER,
        on: Optional[types.Expression] = None
    ) -> Select:
        lt = self._froms.pop()
        rt = get_table(target)
        self._froms.append(Join(lt, rt, join_type, on))  # type:ignore
        return self

    def where(self, *filters: _builder.Node) -> Select:
        self._where = util.and_(*filters) or None
        return self

    def group_by(self, *columns: types.Column) -> Select:
        if not columns:
            raise ValueError("group by clause cannot be empty")
        for f in columns:
            if not isinstance(f, types.Column):
                raise TypeError(
                    f"invalid value '{f}' for group_by field"
                )

        self._group_by = columns  # type: ignore
        return self

    def having(self, *filters: _builder.Node) -> Select:
        self._having = util.and_(*filters) or None
        return self

    def order_by(self, *columns: types.Column):
        if not columns:
            raise ValueError("order by clause cannot be empty")
        for f in columns:
            if not isinstance(f, types.Column):
                raise TypeError(
                    f"invalid value '{f}' for order_by field")

        self._order_by = columns  # type: ignore
        return self

    def limit(self, limit: int = 1000) -> Select:
        self._limit = limit
        return self

    def offset(self, offset: Optional[int] = 0) -> Select:
        if self._limit is None:
            raise err.ProgrammingError("offset clause has no limit")
        self._offset = offset
        return self

    #
    # Single
    #
    async def get(
        self, wrap: bool = True
    ) -> Union[None, util.adict, Model]:
        """If "wrap" is False, the returned row type is not
        wrapped as the ``Model`` object, and the original
        ``helo.util.adict`` is used
        """
        return await self.__do__(rows=self._SINGLE, wrap=wrap)

    async def first(
        self, wrap: bool = True
    ) -> Union[None, util.adict, Model]:
        """If "wrap" is False, the returned row type is not
        wrapped as the ``Model`` object, and the original
        ``helo.util.adict`` is used
        """
        self.limit(self._SINGLE)
        return await self.__do__(rows=self._SINGLE, wrap=wrap)

    #
    # Many
    #
    async def rows(
        self,
        rows: int,
        start: int = 0,
        wrap: bool = True
    ) -> db.FetchResult:
        """If "wrap" is False, the returned row type is not
        wrapped as the ``Model`` object, and the original
        ``helo.util.adict`` is used
        """
        self.limit(rows).offset(start)
        if rows <= 0:
            raise ValueError(f"invalid select rows: {rows}")
        return await self.__do__(wrap=wrap)

    async def paginate(
        self,
        page: int,
        size: int = 20,
        wrap: bool = True
    ) -> db.FetchResult:
        """If "wrap" is False, the returned row type is not
        wrapped as the ``Model`` object, and the original
        ``helo.util.adict`` is used
        """
        if page < 0 or size <= 0:
            raise ValueError("invalid page or size")
        if page > 0:
            page -= 1
        self._limit = size
        self._offset = page * size
        return await self.__do__(wrap=wrap)

    async def all(self, wrap: bool = True) -> db.FetchResult:
        """If "wrap" is False, the returned row type is not
        wrapped as the ``Model`` object, and the original
        ``helo.util.adict`` is used
        """
        return await self.__do__(wrap=wrap)

    #
    # Scalar
    #
    async def scalar(
        self, as_tuple=False
    ) -> Union[None, int, Tuple[Any, ...]]:
        self._props.adicts = False
        row = await self.first()
        return row[0] if row and not as_tuple else row  # type: ignore

    async def count(self) -> int:
        self._columns = [types.F.COUNT(_builder.SQL('1'))]  # type: ignore
        return await self.scalar()  # type: ignore

    async def exist(self) -> bool:
        return bool(await self.limit(self._SINGLE).scalar())

    async def __do__(self, **props) -> Any:
        wrap = props.pop('wrap', False) is True
        if wrap is True or len(self._models) != self._SINGLE:
            self._rowtype = ROWTYPE.ADICT
        return Loader(
            await super().__do__(**props),
            self._models[0], self._aliases, wrap=wrap
        ).do()

    async def __getrow__(self) -> Optional[Model]:
        async def sets():
            self._gotlist = await (
                self.limit(self._BATCH).offset(self._gotidx)
                .all())

        if not self._gotlist:
            await sets()
        elif self._gotlist and self._gotidx >= self._BATCH:
            await sets()
            self._gotidx = 0
        try:
            return self._gotlist[self._gotidx]
        except IndexError:
            return None

    def __aiter__(self) -> Select:
        return self

    async def __anext__(self) -> Optional[Model]:
        row = await self.__getrow__()
        if row is None:
            raise StopAsyncIteration
        self._gotidx += self._SINGLE
        return row

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        ctx.props.select = True
        ctx.literal(
            "SELECT "
        ).sql(
            _builder.CommaNodeList(self._columns)  # type: ignore
        ).literal(
            " FROM "
        ).sql(_builder.CommaNodeList(self._froms))

        if self._where:
            ctx.literal(" WHERE ").sql(self._where)

        if self._group_by:
            ctx.literal(
                " GROUP BY "
            ).sql(_builder.CommaNodeList(self._group_by))

        if self._having:
            ctx.literal(" HAVING ").sql(self._having)

        if self._order_by:
            ctx.literal(
                " ORDER BY "
            ).sql(_builder.CommaNodeList(self._order_by))

        if self._limit is not None:
            ctx.literal(f" LIMIT {self._limit}")

        if self._offset is not None:
            ctx.literal(f" OFFSET {self._offset}")
        return ctx


class Insert(WriteQuery):

    __slots__ = ('_table', '_values', '_from')

    def __init__(
        self,
        table: types.Table,
        values: Union[ValuesMatch, List[types.Column]],
        many: bool = False
    ) -> None:
        super().__init__()
        self._table = table
        self._values = values
        self._from = None  # type: Optional[Select]
        if many:
            self._props.many = True

    def from_(self, select: Select) -> Insert:
        if not isinstance(select, Select):
            raise TypeError(
                'from select clause must be `Select` object')
        self._from = select
        return self

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        ctx.literal(
            "INSERT INTO "
        ).sql(self._table)

        if isinstance(self._values, ValuesMatch):
            ctx.sql(self._values)
        elif isinstance(self._values, list):
            for i, f in enumerate(self._values):
                if isinstance(f, str):
                    self._values[i] = _builder.SQL(f.join('``'))
            ctx.literal(' ').sql(_builder.EnclosedNodeList(self._values))  # type: ignore
        if self._from:
            ctx.literal(' ').sql(self._from)

        return ctx


class Replace(WriteQuery):

    __slots__ = ('_table', '_values', '_from')

    def __init__(
        self,
        table: types.Table,
        values: Union[ValuesMatch],
        many: bool = False
    ) -> None:
        super().__init__()
        self._table = table
        self._values = values
        if many:
            self._props.many = True

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        ctx.literal(
            "REPLACE INTO "
        ).sql(self._table)

        ctx.sql(self._values)
        return ctx


class Update(WriteQuery):

    __slots__ = ('_table', '_values', '_from', '_where')

    def __init__(
        self, table: types.Table, values: AssignmentList
    ) -> None:
        super().__init__()
        self._table = table
        self._values = values
        self._from = None  # type: Optional[types.Table]
        self._where = None

    def from_(self, source: Type[Model]) -> Update:
        self._from = get_table(source)
        return self

    def where(self, *filters: types.Column) -> Update:
        self._where = util.and_(*filters) or None
        return self

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        ctx.literal(
            "UPDATE "
        ).sql(
            self._table
        ).literal(
            " SET "
        )

        if self._from is not None:
            ctx.props.update_from = True
        ctx.sql(self._values)
        if self._from is not None:
            ctx.literal(" FROM ").sql(self._from)

        if self._where is not None:
            ctx.literal(" WHERE ").sql(self._where)
        return ctx


class Delete(WriteQuery):

    __slots__ = ('_table', '_where', '_limit', '_force')

    def __init__(self, table: types.Table, force: bool = False) -> None:
        self._table = table
        self._where = None
        self._limit = None  # type: Optional[int]
        self._force = force
        super().__init__()

    def where(self, *filters: types.Column) -> Delete:
        self._where = util.and_(*filters) or None
        return self

    def limit(self, row_count: int) -> Delete:
        self._limit = row_count
        return self

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        ctx.literal("DELETE FROM ").sql(self._table)
        if self._where:
            ctx.literal(
                " WHERE "
            ).sql(self._where)
        elif not self._force:
            raise err.DangerousOperation(
                "delete is too dangerous as no where clause"
            )
        if self._limit is not None:
            ctx.literal(f" LIMIT {self._limit}")

        return ctx


class Show(BaseQuery):

    __slots__ = ("_table", "_key")

    _options = {
        "create": "SHOW CREATE TABLE ",
        "columns": "SHOW FULL COLUMNS FROM ",
        "indexes": "SHOW INDEX FROM ",
    }

    def __init__(self, table: types.Table) -> None:
        super().__init__()
        self._table = table
        self._key = None  # type: Optional[str]

    def __repr__(self) -> str:
        return f"<Show object for {self._table!r}>"

    __str__ = __repr__

    async def create_syntax(self) -> Optional[util.adict]:
        self._key = "create"
        return (await self.__do__(rows=1)).get("Create Table")

    async def columns(self) -> db.FetchResult:
        self._key = "columns"
        return await self.__do__()

    async def indexes(self) -> db.FetchResult:
        self._key = "indexes"
        return await self.__do__()

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        if self._key is not None:
            ctx.literal(
                self._options[self._key]
            ).sql(self._table)
        return ctx


class Create(WriteQuery):

    __slots__ = ('_table', '_options')

    def __init__(self, table: types.Table, **options: Any) -> None:
        self._table = table
        self._options = options
        super().__init__()

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        ctx.literal('CREATE ')
        if self._options.get('temporary'):
            ctx.literal('TEMPORARY ')
        ctx.literal('TABLE ')
        if self._options.get('safe', True):
            ctx.literal('IF NOT EXISTS ')
        ctx.sql(self._table)

        defs = [f.__def__() for f in self._table.fields_dict.values()]  # type: List[_builder.Node]
        defs.append(_builder.SQL(f"PRIMARY KEY ({self._table.primary.field.column})"))
        if self._table.indexes:
            defs.extend([i.__def__() for i in self._table.indexes])

        ctx.sql(
            _builder.EnclosedNodeList(defs)
        ).literal(
            f"ENGINE={self._table.engine} "
            f"AUTO_INCREMENT={self._table.auto_increment} "
            f"DEFAULT CHARSET={self._table.charset} "
            f"COMMENT='{self._table.comment}'"
        )
        return ctx


class Drop(Create):

    __slots__ = ()

    def __sql__(self, ctx: _builder.Context) -> _builder.Context:
        ctx.literal('DROP TABLE ').sql(self._table)
        return ctx


class Loader:

    __slots__ = ('_data', '_modelclass', '_wrap',
                 '_mattrs', '_mfields', '_aliases')

    def __init__(
        self,
        data: Union[None, util.adict, db.FetchResult],
        model: Type[Model],
        aliases: Dict[str, Any],
        wrap: bool = True
    ) -> None:
        self._data = data
        self._modelclass = model
        self._aliases = aliases
        self._wrap = wrap

        self._mattrs = get_attrs(self._modelclass)
        self._mfields = get_table(self._modelclass).fields_dict

    def do(self) -> Any:
        if not self._data:
            return self._data

        if isinstance(self._data, db.FetchResult):
            if self._wrap is True:
                for i in range(self._data.count):
                    mobj = self._convert_to_model(self._data[i])
                    self._data[i] = mobj or self._data[i]
            else:
                for i in range(self._data.count):
                    self._data[i] = self._convert_type(self._data[i])
        elif isinstance(self._data, dict):
            if self._wrap is True:
                self._data = self._convert_to_model(self._data) or self._data
            else:
                self._data = self._convert_type(self._data)
        return self._data

    def _convert_type(
        self, row: util.adict
    ) -> util.adict:
        if isinstance(row, dict):
            for name in row.copy():
                if name not in self._mattrs.values():
                    aname = self._aliases.get(name, name)
                    rname = self._mattrs.get(aname, aname)
                    row[rname] = row.pop(name)
                    name = rname

                f = self._mfields.get(name)
                if f and not isinstance(row[name], f.py_type):
                    row[name] = f.py_value(row[name])
        else:
            pass
        return row

    def _convert_to_model(self, row: util.adict) -> Optional[Model]:
        model = self._modelclass()
        for name, value in row.items():
            name = self._aliases.get(name, name)
            name = self._mattrs.get(name)
            if not name:
                return None
            try:
                model.__setmodel__(name, value, __load__=True)
            except Exception:  # pylint: disable=broad-except
                return None
        return model
