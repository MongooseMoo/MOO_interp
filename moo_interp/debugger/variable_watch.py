"""Variable watch and inspection tracking plugin for MOO debugger."""

from typing import Any, Dict, List, Set
from .base import DebugPlugin


class VariableWatchPlugin(DebugPlugin):
    """Plugin that tracks variable values across execution.

    Records when watched variables change, including:
    - Step number when change occurred
    - New value
    - History across verb calls (by variable name)
    """

    def __init__(self, watch_vars: List[str] = None):
        """Initialize variable watch plugin.

        Args:
            watch_vars: List of variable names to watch (can be empty initially)
        """
        super().__init__()
        self.watch_vars: Set[str] = set(watch_vars or [])
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.current_values: Dict[str, Any] = {}

        # Initialize history for all watch vars
        for var in self.watch_vars:
            self.history[var] = []

    def add_watch(self, var_name: str) -> None:
        """Add a variable to the watch list.

        Args:
            var_name: Name of variable to watch
        """
        if var_name not in self.watch_vars:
            self.watch_vars.add(var_name)
            self.history[var_name] = []

    def remove_watch(self, var_name: str) -> None:
        """Remove a variable from the watch list.

        Args:
            var_name: Name of variable to stop watching
        """
        self.watch_vars.discard(var_name)
        # Keep history even after removing watch

    def on_step_after(self, frame, vm_state: Dict[str, Any]) -> None:
        """Check watched variables after each step.

        Args:
            frame: Current stack frame (or None if returned)
            vm_state: VM state after step
        """
        if not self.enabled or not self.watch_vars:
            return

        # Get current variables from state
        current_vars = vm_state.get('vars', {})
        step_count = vm_state.get('step_count', 0)

        # Check each watched variable
        for var_name in self.watch_vars:
            # Get current value (None if not in scope)
            current_value = current_vars.get(var_name)

            # Check if this is first observation or value changed
            if var_name not in self.current_values:
                # First observation
                self.current_values[var_name] = current_value
                self.history[var_name].append({
                    "step": step_count,
                    "value": self._make_serializable(current_value),
                })
            elif not self._values_equal(self.current_values[var_name], current_value):
                # Value changed
                self.current_values[var_name] = current_value
                self.history[var_name].append({
                    "step": step_count,
                    "value": self._make_serializable(current_value),
                })

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Check if two values are equal for tracking purposes.

        Args:
            val1: First value
            val2: Second value

        Returns:
            True if values are considered equal
        """
        # Handle None
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        # Handle MOO types with special comparison
        type1 = type(val1).__name__
        type2 = type(val2).__name__

        if type1 != type2:
            return False

        # For MOOList, compare contents
        if hasattr(val1, '_list') and hasattr(val2, '_list'):
            return val1._list == val2._list

        # For MOOString, compare string values
        if type1 == 'MOOString':
            return str(val1) == str(val2)

        # For ObjNum, compare numeric values
        if type1 == 'ObjNum':
            return str(val1) == str(val2)

        # Default comparison
        return val1 == val2

    def _make_serializable(self, obj: Any) -> Any:
        """Convert MOO types to JSON-serializable form.

        Args:
            obj: Object to convert

        Returns:
            Serializable version of object
        """
        if obj is None:
            return None
        if isinstance(obj, (int, float, str, bool)):
            return obj
        if hasattr(obj, '_list'):  # MOOList
            return [self._make_serializable(x) for x in obj._list]
        if hasattr(obj, '_map'):  # MOOMap
            return {str(k): self._make_serializable(v) for k, v in obj._map.items()}
        if type(obj).__name__ == 'MOOString':
            return str(obj)
        if type(obj).__name__ == 'ObjNum':
            return int(str(obj).lstrip('#'))
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(x) for x in obj]
        return str(obj)

    def get_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the complete variable history.

        Returns:
            Dict mapping variable names to their change history
        """
        return self.history

    def get_current_values(self) -> Dict[str, Any]:
        """Get current values of all watched variables.

        Returns:
            Dict mapping variable names to current values
        """
        return {
            var: self._make_serializable(val)
            for var, val in self.current_values.items()
        }

    def reset(self) -> None:
        """Reset tracking data."""
        self.history = {var: [] for var in self.watch_vars}
        self.current_values.clear()
