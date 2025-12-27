"""WAIF (Weakly Attached Instance Format) support for MOO.

WAIFs are lightweight objects that:
- Are NOT stored in the object database (not "valid")
- Inherit properties and verbs from a "class" object
- Have their own values for `:prop` prefixed properties
- Are reference-counted and auto-recycled when dereferenced
- Cannot reference themselves (E_RECMOVE protection)
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Waif:
    """A WAIF (Weakly Attached Instance Format) object.

    WAIFs are lightweight objects that inherit from a "class" object
    but are not stored in the object database.

    Attributes:
        class_obj: Object ID of the class object that defines this WAIF's
                   properties and verbs. The class must be a valid object.
        owner: Player ID of the creator/owner.
        propvals: Dictionary mapping `:property_name` to values for this
                  instance. Only `:` prefixed properties can be stored here.
        _valid: Whether the class is still valid. Set to False when the
                class object is recycled.
    """
    class_obj: int
    owner: int
    propvals: dict[str, Any] = field(default_factory=dict)
    _valid: bool = field(default=True)

    def __repr__(self) -> str:
        if self._valid:
            return f"[[Waif class #{self.class_obj}]]"
        else:
            return "[[invalid waif]]"

    def __str__(self) -> str:
        return self.__repr__()

    def invalidate(self) -> None:
        """Mark this WAIF as invalid (class was recycled)."""
        self._valid = False
        self.propvals.clear()

    def is_valid(self) -> bool:
        """Check if this WAIF is still valid."""
        return self._valid

    def get_class(self) -> int:
        """Get the class object ID."""
        return self.class_obj

    def get_owner(self) -> int:
        """Get the owner player ID."""
        return self.owner


def refers_to_waif(value: Any, target_waif: "Waif") -> bool:
    """Check if a value contains a reference to the target waif.

    Used to prevent self-reference in waif properties.

    Args:
        value: The value to check
        target_waif: The waif we're checking for references to

    Returns:
        True if value contains a reference to target_waif
    """
    if value is target_waif:
        return True

    # Check lists (MOOList or Python list)
    # Exclude strings - MOOString is UserString (not str) but is iterable
    from .string import MOOString
    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, MOOString)):
        try:
            for item in value:
                if refers_to_waif(item, target_waif):
                    return True
        except TypeError:
            pass

    # Check maps (MOOMap or Python dict)
    if hasattr(value, 'items'):
        try:
            for k, v in value.items():
                if refers_to_waif(k, target_waif) or refers_to_waif(v, target_waif):
                    return True
        except (TypeError, AttributeError):
            pass

    # Check nested waifs (but not self)
    if isinstance(value, Waif) and value is not target_waif:
        for prop_value in value.propvals.values():
            if refers_to_waif(prop_value, target_waif):
                return True

    return False
