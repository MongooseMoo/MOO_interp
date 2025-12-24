"""Base plugin class for MOO debugger."""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


class DebugPlugin(ABC):
    """Base class for debugger plugins.

    Plugins can hook into various points in VM execution to observe
    or modify debugger behavior.
    """

    def __init__(self):
        """Initialize the plugin."""
        self.enabled = True

    def on_step_before(self, frame, vm_state: Dict[str, Any]) -> None:
        """Called before each VM step.

        Args:
            frame: Current stack frame
            vm_state: Current VM state snapshot
        """
        pass

    def on_step_after(self, frame, vm_state: Dict[str, Any]) -> None:
        """Called after each VM step.

        Args:
            frame: Current stack frame (or None if returned)
            vm_state: VM state after step
        """
        pass

    def on_verb_call(self, verb_name: str, this: int, args: list, depth: int, step: int) -> None:
        """Called when a verb is about to be called.

        Args:
            verb_name: Name of the verb being called
            this: Object the verb is on
            args: Arguments to the verb
            depth: Current call stack depth
            step: Step number
        """
        pass

    def on_verb_return(self, verb_name: Optional[str], return_value: Any, depth: int, step: int) -> None:
        """Called when a verb returns.

        Args:
            verb_name: Name of the verb returning (if known)
            return_value: Value being returned
            depth: Call stack depth after return
            step: Step number
        """
        pass

    def on_builtin_call(self, func_name: str, func_id: int, depth: int, step: int) -> None:
        """Called when a builtin function is about to be called.

        Args:
            func_name: Name of the builtin
            func_id: ID of the builtin
            depth: Current call stack depth
            step: Step number
        """
        pass

    def should_break(self, frame, vm_state: Dict[str, Any]) -> bool:
        """Check if execution should break at this point.

        Args:
            frame: Current stack frame
            vm_state: Current VM state

        Returns:
            True if debugger should stop execution
        """
        return False

    def get_data(self) -> Any:
        """Get plugin-specific data for reporting.

        Returns:
            Plugin data (typically a dict or list)
        """
        return None

    def reset(self) -> None:
        """Reset plugin state (called when starting new debug session)."""
        pass
