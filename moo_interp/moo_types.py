from typing import Union

from moo_interp.list import MOOList
from moo_interp.map import MOOMap

from .errors import MOOError
from .string import MOOString
from .waif import Waif

MOONumber = Union[int, float]
Addable = Union[MOONumber, MOOString, MOOList]
Subtractable = Union[MOONumber, MOOList]
Container = Union[MOOList, MOOMap, MOOString]
Comparable = Union[MOONumber, MOOString]
MapKey = Union[MOONumber, MOOString]
# Complete type for any MOO value (includes plain Python str/list/dict for compatibility)
MOOAny = Union[MOONumber, MOOString, Container, MOOError, bool, None, str, list, dict, Waif]


def to_moo(py_obj: Union[str, int, float, bool, list, dict]) -> MOOAny:
    py_type = type(py_obj)
    if py_type is str:
        return MOOString(py_obj)
    elif py_type is int:
        return py_obj
    elif py_type is float:
        return py_obj
    elif py_type is bool:
        return py_obj
    elif py_type is list:
        return MOOList(*[to_moo(item) for item in py_obj])
    elif py_type is dict:
        return MOOMap(**{key: to_moo(value) for key, value in py_obj.items()})
    else:
        raise TypeError(f"Cannot convert {py_type.__name__} to MOO type.")


def is_truthy(value: MOOAny) -> bool:
    if isinstance(value, MOOString):
        return bool(value)
    elif isinstance(value, str):
        # Handle plain Python strings (should be MOOString but handle anyway)
        return bool(value)
    elif isinstance(value, MOOList):
        return bool(value)
    elif isinstance(value, list):
        # Handle plain Python lists
        return bool(value)
    elif isinstance(value, MOOMap):
        return bool(value)
    elif isinstance(value, dict):
        # Handle plain Python dicts
        return bool(value)
    elif isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return bool(value)
    elif isinstance(value, float):
        return bool(value)
    else:
        raise TypeError(f"Cannot check truthiness of {type(value).__name__}.")
