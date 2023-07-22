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
