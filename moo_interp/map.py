from collections.abc import MutableMapping
from attr import define, field


@define(repr=False)
class MOOMap(MutableMapping):
    _map = field(factory=dict)

    def __getitem__(self, key):
        return self._map[key]

    def __setitem__(self, key, value):
        self._map[key] = value

    def __delitem__(self, key):
        del self._map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __repr__(self):
        return f"MOOMap({self._map})"
