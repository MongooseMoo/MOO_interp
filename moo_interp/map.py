from collections.abc import MutableMapping
from functools import cmp_to_key
import sys

from attr import define, field


def moo_compare(a, b):
    """MOO value comparison function.

    Comparison rules:
    1. If types are different, compare type IDs (int < obj < float < str < err < ...)
    2. If types are the same, compare values:
       - int, obj, err: numeric comparison
       - str: lexicographic (case-insensitive for maps)
       - float: numeric comparison
    """
    # Get type priorities for MOO map key sorting
    # Order: INT < OBJ < FLOAT < STR < ERR < LIST < MAP < BOOL
    def type_priority(val):
        """Return type priority for sorting."""
        # Check ObjNum before int (ObjNum inherits from int)
        if type(val).__name__ == 'ObjNum':
            return 1  # TYPE_OBJ
        if isinstance(val, bool):
            return 7  # TYPE_BOOL
        if isinstance(val, int):
            # Check if it's a MOOError (IntEnum)
            if type(val).__name__ == 'MOOError':
                return 4  # TYPE_ERR
            return 0  # TYPE_INT
        if isinstance(val, float):
            return 2  # TYPE_FLOAT (before STR)
        if isinstance(val, str) or type(val).__name__ == 'MOOString':
            return 3  # TYPE_STR
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


def _is_string_like(val):
    """Check if val is a string or MOOString."""
    return isinstance(val, str) or type(val).__name__ == 'MOOString'


@define(repr=False)
class MOOMap(MutableMapping):
    _map: dict = field(factory=dict, alias="_map")
    _refcount: int = field(default=1, init=False)

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

    def __eq__(self, other):
        """Compare maps with case-insensitive string comparison."""
        if not isinstance(other, MOOMap):
            return NotImplemented
        if len(self._map) != len(other._map):
            return False
        # Compare each key-value pair case-insensitively for strings
        for key in self._map:
            # Find matching key in other (case-insensitive for strings)
            other_key = None
            if _is_string_like(key):
                key_lower = str(key).lower()
                for k in other._map:
                    if _is_string_like(k) and str(k).lower() == key_lower:
                        other_key = k
                        break
            else:
                if key in other._map:
                    other_key = key

            if other_key is None:
                return False

            # Compare values case-insensitively for strings
            val1 = self._map[key]
            val2 = other._map[other_key]
            if _is_string_like(val1) and _is_string_like(val2):
                if str(val1).lower() != str(val2).lower():
                    return False
            elif val1 != val2:
                return False
        return True

    def __repr__(self):
        return f"MOOMap({self._map})"

    def equal_case_sensitive(self, other) -> bool:
        """Compare maps with case-sensitive string comparison.

        Used by the equal() builtin. Unlike __eq__ which is case-insensitive
        for the == operator, this performs strict case-sensitive comparison.
        """
        if not isinstance(other, MOOMap):
            return False
        if len(self._map) != len(other._map):
            return False
        # Compare each key-value pair case-sensitively
        for key in self._map:
            if key not in other._map:
                return False
            val1 = self._map[key]
            val2 = other._map[key]
            # Recurse for nested maps
            if isinstance(val1, MOOMap) and isinstance(val2, MOOMap):
                if not val1.equal_case_sensitive(val2):
                    return False
            elif val1 != val2:
                return False
        return True

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
