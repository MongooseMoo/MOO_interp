from collections.abc import MutableMapping
import sys

from attr import define, field


@define(repr=False)
class MOOMap(MutableMapping):
    _map = field(factory=dict)
    _refcount = field(default=1, init=False)

    def __getitem__(self, key):
        return self._map[key]

    def __setitem__(self, key, value):
        self._map[key] = value

    def __delitem__(self, key):
        del self._map[key]

    def __iter__(self):
        # MOO maps are ordered by sorted keys (like C++ std::map / red-black tree)
        return iter(sorted(self._map.keys()))

    def __len__(self):
        return len(self._map)

    def __repr__(self):
        return f"MOOMap({self._map})"

    def refcount(self):
        """Return the reference count of this MOOMap."""
        return sys.getrefcount(self) - 1  # -1 to account for the call itself

    def shallow_copy(self):
        """Create a shallow copy for copy-on-write.

        Creates a new MOOMap with a new _map container, but values
        are not recursively copied (they remain as references).
        """
        new_map = MOOMap.__new__(MOOMap)
        new_map._map = self._map.copy()  # Shallow copy of the dict
        new_map._refcount = 1
        return new_map
