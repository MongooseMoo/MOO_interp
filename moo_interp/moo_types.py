from typing import Union

from moo_interp.list import MOOList
from moo_interp.map import MOOMap

from .string import MOOString

MOONumber = Union[int, float]
Addable = Union[MOONumber, MOOString, MOOList]
Subtractable = Union[MOONumber, MOOList]
Container = Union[MOOList, MOOMap, MOOString]
Comparable = Union[MOONumber, MOOString]
MapKey = Union[MOONumber, MOOString]
MOOAny = Union[MOONumber, MOOString, Container, bool]


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
