"""
Author: E.W.Ayers <contact@edayers.com>
This file is adapted from  https://github.com/EdAyers/sss
"""
import inspect
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from functools import singledispatch
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Type, TypeVar, Union, get_args, get_origin

from pydantic import ValidationError
from rift.util.misc import map_ctx

from .dispatch import classdispatch
from .type_util import as_list, as_newtype, as_optional, as_set, is_optional

try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard


T = TypeVar("T")
logger = logging.getLogger(__name__)

JsonLike = Optional[Union[dict, list, str, bool, int, float]]
JsonKey = Optional[Union[str, int, float, bool]]


def is_json_key(key: Any) -> TypeGuard[JsonKey]:
    return isinstance(key, (str, int, float, bool, type(None)))


def ofdict_dataclass(data_class_type: Type[T], json_like_object: JsonLike) -> T:
    """
    Converts JSON-like object passed as a parameter to the specified data class type.

    :param data_class_type: The data class type to which the JSON-like object will be converted.
    :param json_like_object: The JSON-like object to be converted to the specified data class type.

    :return: The data class type instance built from the JSON-like object
    """
    # Validation to ensure passed data_class_type is a dataclass
    assert is_dataclass(data_class_type)

    # Dictionary that will be constructed from json_like_object
    parsed_dict = {}

    # Iterating over all the fields of the data class
    for field in fields(data_class_type):
        # Check if the json_like_object is dictionary
        if not isinstance(json_like_object, dict):
            raise OfDictError(
                f"Error while decoding dataclass {data_class_type}, expected a dict but got {json_like_object} : {type(json_like_object)}"
            )

        # Fetch the field name
        key = field.name

        # Check if the key is not in the json_like_object
        if key not in json_like_object:
            # Check if key's type is optional
            if field.type is not None and is_optional(field.type):
                value = None
            else:
                raise OfDictError(
                    f"Missing {field.name} on input dict. Decoding {json_like_object} to type {data_class_type}."
                )
        else:
            value = json_like_object[key]

        # Process the value if the type is defined for field
        if field.type is not None:
            with dpath(key):
                parsed_dict[key] = ofdict(field.type, value)
        else:
            parsed_dict[key] = value

    # Return instance of the data class
    return data_class_type(**parsed_dict)  # type: ignore


ofdict_context: ContextVar[list[str]] = ContextVar("ofdict_context", default=[])


def atstr():
    p = ofdict_context.get()
    if len(p) == 0:
        return ""
    else:
        s = ".".join(p)
        return f" at {s}"


@contextmanager
def dpath(coord: str):
    with map_ctx(ofdict_context, lambda x: x + [coord]) as p:
        yield p


class OfDictError(TypeError):
    """Error caused by a failure to convert a dictionary-like object to a type using ofdict."""

    def __init__(self, msg: str):
        super().__init__(msg + atstr())


@classdispatch
def ofdict(A: Type[T], a: JsonLike) -> T:
    """Converts an ``a`` to an instance of ``A``, calling recursively if necessary.

    We assume that ``a`` is a nested type made of dicts, lists and scalars.

    The main usecase is to be able to treat dataclasses as a schema for json.
    Ideally, ``ofdict`` should be defined such that ``ofdict(type(x), json.loads(MyJsonEncoder().dumps(x)))`` is deep-equal to ``x`` for all ``x``.

    Similar to ` cattrs.structure <https://cattrs.readthedocs.io/en/latest/structuring.html#what-you-can-structure-and-how/>`_.

    [todo] I am hoping to retire this in favour of Pydantic.
    """
    # try:
    #     return A(**a)
    # except ValidationError as e:
    #     print(e.errors())
    #     raise e

    if isinstance(A, str):
        raise TypeError(
            f"please make sure your class {A} is referred using types and not string-escaped types"
        )
        # [todo] also https://docs.python.org/3/library/typing.html#typing.ForwardRef
    S = as_newtype(A)
    if S is not None:
        return A(ofdict(S, a))
    if inspect.isclass(A) and issubclass(A, OfDictUnion):
        class_key = getattr(A, "_class_key", "__class__")
        ct = getattr(A, "_class_table", None)
        if ct is None:
            raise OfDictError(f"failed to find class table for {A}")
        assert isinstance(a, dict)
        C: Optional[type] = a.get(class_key, None)
        if C is None:
            raise OfDictError(
                f"ofdict for a subclass of 'OfDictUnion' must include a '{class_key}' key"
            )
        if not issubclass(C, A):
            raise OfDictError(f"Expected {C} to be a subclass of {A}")
        A = C  # type: ignore

    if A is Any:
        return a  # type: ignore
    if A is type(None) and a is None:
        return a  # type: ignore
    if get_origin(A) is Literal:
        values = get_args(A)
        if a in values:
            return a  # type: ignore
        else:
            raise OfDictError(f"In Literal expected one of {values}, but got {a}")
    if is_optional(A):
        X = as_optional(A)
        if a is None:
            return None  # type: ignore
        else:
            return ofdict(X, a)  # type: ignore

    if get_origin(A) is Union:
        es = []
        for X in get_args(A):
            try:
                return ofdict(X, a)
            except OfDictError as e:
                # [note] python 3.11 has exception groups instead
                es.append(e)
        # [todo] raise everything?
        raise es[-1]
    od = getattr(A, "__ofdict__", None)
    if od is not None:
        return od(a)
    adapt = getattr(A, "__adapt__", None)
    if adapt is not None:
        result = adapt(a)
        if result is not None:
            return result
    if is_dataclass(A):
        return ofdict_dataclass(A, a)  # type: ignore
    if A in [float, str, int, bytes]:  # [todo] etc
        if isinstance(a, A):
            return a  # type: ignore
        else:
            raise OfDictError(f"Expected a {A.__name__} but was {type(a).__name__}: {a}")
    try:
        if isinstance(a, A):
            return a  # type: ignore
    except TypeError:
        pass
    raise NotImplementedError(f"No implementation of ofdict for {A.__name__}.")


@ofdict.register(list)
def _list_ofdict(A, a):
    if not isinstance(a, list):
        raise OfDictError(f"Expected a list but got a {type(a)}")
    X = as_list(A)
    if X is not None:
        return [ofdict(X, y) for y in a]
    else:
        return a


@ofdict.register(set)
def _set_ofdict(A, a):
    if not isinstance(a, list):
        raise OfDictError(f"Expected a list but got a {type(a)}")
    X = as_set(A)
    if X is not None:
        return set(ofdict(X, y) for y in a)
    else:
        return set(a)


@ofdict.register(dict)
def _dict_ofdict(A, a):
    if not isinstance(a, dict):
        raise OfDictError(f"Expected a {A} but got {type(a)}")
    o = get_origin(A)
    if o is None:
        return a
    K, V = get_args(A)
    d = {}
    for k, v in a.items():
        with dpath(str(k)):
            vk = ofdict(K, k)
            vv = ofdict(V, v)
            d[vk] = vv
    return o(d)


@ofdict.register(Enum)
def _ofdict_enum(A, a):
    return A(a)


@ofdict.register(datetime)
def _ofdict_datetime(_, t):
    return datetime.fromisoformat(t)


@ofdict.register(Path)
def _ofdict_path(_, t):
    return Path(t)


class TypedJsonDecoder(json.JSONDecoder):
    """Given a python type T, this will decode a json object to an instance of `T`, or fail otherwise.

    It makes use of the `ofdict` function defined above to convert plain json dictionaries to native python types.
    """

    def __init__(self, T: Type):
        self.T = T
        super().__init__()

    def decode(self, j):
        jj = super().decode(j)
        return ofdict(self.T, jj)


@classdispatch
def validate(t: Type, item) -> bool:
    """Validates that the given item is of the given type."""
    # [todo] type assertion `bool ↝ item is t`
    if t == Any:
        return True
    o = as_optional(t)
    if o is not None:
        if t is None:
            return True
        else:
            return validate(o, item)
    X = as_list(t)
    if X is not None:
        assert isinstance(item, list)

        return all([validate(X, x) for x in item])

    if isinstance(item, t):
        if is_dataclass(item):
            return all([validate(field.type, getattr(item, field.name)) for field in fields(item)])
        return True
    raise NotImplementedError(f"Don't know how to validate {t}")


def todict_dataclass(x: Any):
    assert is_dataclass(x)
    r = {}
    for field in fields(x):
        k = field.name
        v = getattr(x, k)
        if is_optional(field.type) and v is None:
            continue
        # [todo] shouldn't this not be recursive?
        r[k] = todict(v)
    return r


@singledispatch
def todict(x: Any) -> JsonLike:
    """Converts the given object to a JSON-compatible object.

    This should not recurse on the arguments. Just return one of dict, list, tuple, str, int, float, bool, or None.
    `MyJsonEncoder` will run todict recursively on any child objects.
    """
    if isinstance(x, OfDictUnion):
        j = _todict_core(x)
        cls = type(x)
        root = cls._root_class
        class_key = root._class_key
        assert issubclass(cls, root)
        assert cls in cls._class_table
        assert isinstance(j, dict)
        assert class_key not in j
        j[class_key] = cls.__name__
        return j
    else:
        return _todict_core(x)


def _todict_core(x: Any):
    if isinstance(x, (str, int, float, bool)):
        return x
    if x is None:
        return x
    # Something like PEP-246
    conform = getattr(x, "__todict__", None)
    if conform is not None:
        return conform()
    conform = getattr(x, "__conform__", None)
    if conform is not None:
        result = conform(JsonLike)
        # note in PEP-246 failing to conform will return None,
        # but None is a valid JsonLike so instead we use NotImplemented.
        if result is not NotImplemented:
            return result
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, (list, dict)):
        return x
    if is_dataclass(x):
        return todict_dataclass(x)
    raise NotImplementedError(f"Don't know how to convert {type(x)}")


@todict.register(Enum)
def _todict_enum(x):
    return x.value


@todict.register(datetime)
def _todict_datetime(x):
    return x.isoformat()


@todict.register(Path)
def _todict_path(x):
    return str(x)


@todict.register(set)
def _todict_set(x):
    return list(x)


class MyJsonEncoder(json.JSONEncoder):
    """Converts Python objects to Json.

    We have additional support for dataclasses and enums that are not present in the standard encoder.
    [todo] rename to TypedJsonEncoder?
    """

    def encode(self, obj):
        if isinstance(obj, dict):
            # json encoder doesn't recursively encode keys.
            obj = {todict_key(k): v for k, v in obj.items()}
            assert all(is_json_key(k) for k in obj.keys())
        return super().encode(obj)

    # [todo] needs to handle `None` by not setting json field.
    def default(self, o):
        j = todict(o)
        if j is NotImplemented:
            j = json.JSONEncoder.default(self, o)

        return j


@singledispatch
def todict_key(x: Any) -> JsonKey:
    """Converts the given object to a JSON-compatible dictionary key (ie a string or number)."""
    if is_json_key(x):
        return x
    tdk = getattr(x, "__todict_key__", None)
    if tdk is not None:
        r = tdk()
        if not is_json_key(r):
            raise TypeError(f"{type(x)}.__todict_key__ returned {type(r)} instead of str or int.")
        return r
    j = todict(x)
    if j is NotImplemented:
        raise NotImplementedError(f"Don't know how to convert {type(x)} to a dictionary key.")
    elif not is_json_key(j):
        raise TypeError(
            f"Type {type(x)} is being used as a dictionary key but {type(j)} is not a string or integer. "
        )
    else:
        return j


def todict_rec(x: Any) -> JsonLike:
    """Recursive version of `todict`."""
    if x is NotImplemented:
        raise TypeError(f"Oops")
    assert x is not NotImplemented
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    j = todict(x)
    if j is NotImplemented:
        if isinstance(x, (dict, list, tuple, str, int, float, bool)):
            j = x
        else:
            raise NotImplementedError(f"Don't know how to todict {type(x)}")
    if isinstance(j, dict):
        return {todict_key(k): todict_rec(v) for k, v in j.items()}
    elif isinstance(j, (list, tuple)):
        return [todict_rec(c) for c in j]
    elif isinstance(x, (str, int, float, bool)):
        return x
    else:
        raise TypeError(f"Don't know how to todict {type(x)}")


class OfDictUnion:
    _class_key: ClassVar[str] = "__class__"
    _class_table: ClassVar[dict[str, type]]
    _root_class: ClassVar[type["OfDictUnion"]]
    """ Use this when you have a class hierarchy of types that you want
    to be able to serialize using todict.

    It works by also storing a  `"__class__": cls.__name__` entry on a class table.
    When the dict is deserialised, it will look up this class name to find the subclass.


    Example usage:
    ```
    class Base(OfDictUnion):
      pass

    @dataclass
    class X(Base):
      x : str

    @dataclass
    class Y(Base):
      y : int

    assert todict(X(x="hello")) == {"__class__": "X", "x": "hello"}
    assert todict(Y(y=42)) == {"__class__": "Y", "y": 42}
    assert ofdict(Base, {"__class__": "X", "x": "hello"}) == X(x="hello")
    ```
    """

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "_class_table"):
            # first time
            cls._class_table = dict()
            cls._root_class = cls
        name = cls.__name__
        assert name not in cls._class_table
        cls._class_table[name] = cls
        super().__init_subclass__(**kwargs)


try:
    from pydantic import BaseModel

    @todict.register(BaseModel)
    def _todict_model(x: BaseModel):
        return x.dict()

    @ofdict.register(BaseModel)
    def _ofdict_model(ModelCls: type[BaseModel], item):
        try:
            return ModelCls.parse_obj(item)
        except ValidationError as e:
            raise OfDictError(f"Model {ModelCls.__name__} is invalid: {e}") from e

    # [todo] pydantic validation types like EmailStr, SecretStr etc
except ImportError:
    pass
