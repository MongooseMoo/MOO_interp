from collections.abc import MutableSequence
from typing import Iterable

from attr import define, field


@define(repr=False)
class MOOList(MutableSequence):
    _list = field(factory=list)

    def __init__(self, *args):
        self._list = list(args)

    def __getitem__(self, index):
        return self._list[index - 1]

    def __setitem__(self, index, value):
        self._list[index - 1] = value

    def __delitem__(self, index):
        del self._list[index - 1]

    def insert(self, index, value):
        self._list.insert(index - 1, value)

    def append(self, value):
        self._list.append(value)

    def __len__(self):
        return len(self._list)

    def __repr__(self):
        return f"MOOList({self._list})"

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        return MOOList(*self._list, *other._list)
