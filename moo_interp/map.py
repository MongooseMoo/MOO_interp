from collections.abc import MutableMapping
from functools import cmp_to_key
import sys

from attr import define, field


def moo_compare(a, b):
    """MOO value comparison function.

    Comparison rules:
    1. If types are different, compare type IDs (int < obj < str < err < float < ...)
    2. If types are the same, compare values:
       - int, obj, err: numeric comparison
       - str: lexicographic (case-insensitive for maps)
       - float: numeric comparison
    """
    # Get type priorities for MOO (matching toaststunt's TYPE_* enum order)
    def type_priority(val):
        """Return type priority for sorting (int=0, obj=1, str=2, err=3, float=4, ...)."""
        # Check ObjNum before int (ObjNum inherits from int)
        if type(val).__name__ == 'ObjNum':
            return 1  # TYPE_OBJ
        if isinstance(val, bool):
            return 6  # TYPE_BOOL (after float)
        if isinstance(val, int):
            # Check if it's a MOOError (IntEnum)
            if type(val).__name__ == 'MOOError':
                return 3  # TYPE_ERR
            return 0  # TYPE_INT
        if isinstance(val, str) or type(val).__name__ == 'MOOString':
            return 2  # TYPE_STR
        if isinstance(val, float):
            return 4  # TYPE_FLOAT
        # Lists and maps
        if hasattr(val, '_list'):
            return 5  # TYPE_LIST
        if hasattr(val, '_map'):
            return 6  # TYPE_MAP
        # Fallback: compare by type name
        return 10

    type_a = type_priority(a)
    type_b = type_priority(b)

    # Different types: compare type priorities
    if type_a != type_b:
        return type_a - type_b

    # Same type: compare values
    # For strings: case-insensitive comparison (for map key ordering)
    if isinstance(a, str) or type(a).__name__ == 'MOOString':
        a_lower = str(a).lower()
        b_lower = str(b).lower()
        if a_lower < b_lower:
            return -1
        elif a_lower > b_lower:
            return 1
        return 0

    # For numbers (int, float, obj, err): direct comparison
    if a < b:
        return -1
    elif a > b:
        return 1
    return 0


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
        # Use MOO's comparison function for sorting
        return iter(sorted(self._map.keys(), key=cmp_to_key(moo_compare)))

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
