"""Builtin function call tracing plugin for MOO debugger."""

from typing import Any, Dict, List, Optional
from .base import DebugPlugin


class BuiltinTracePlugin(DebugPlugin):
    """Plugin that traces builtin function calls and return values.

    Captures:
    - Builtin function calls with arguments
    - Return values from builtins
    - Step numbers for correlation
    """

    def __init__(self):
        """Initialize builtin trace plugin."""
        super().__init__()
        self.trace: List[Dict[str, Any]] = []
        self.pending_builtin: Optional[Dict[str, Any]] = None

    def on_step_before(self, frame, vm_state: Dict[str, Any]) -> None:
        """Capture builtin arguments before execution (if this is a builtin call).

        Args:
            frame: Current stack frame
            vm_state: Current VM state snapshot
        """
        if not self.enabled:
            return

        # Check if this instruction is a builtin call
        opcode = vm_state.get("opcode")
        if opcode == "OP_BI_FUNC_CALL":
            # The top of stack should be the MOOList of args
            stack = vm_state.get("stack", [])
            if stack:
                args_list = stack[-1]
                # Convert MOOList to regular Python list
                if hasattr(args_list, "_list"):
                    # Store args for when on_builtin_call fires
                    self.pending_args = list(args_list._list)
                else:
                    self.pending_args = args_list
            else:
                self.pending_args = []

    def on_builtin_call(self, func_name: str, func_id: int, depth: int, step: int) -> None:
        """Record a builtin call (before execution).

        Args:
            func_name: Name of the builtin
            func_id: ID of the builtin
            depth: Current call stack depth
            step: Step number
        """
        if not self.enabled:
            return

        # Create the pending builtin entry with the args we captured
        self.pending_builtin = {
            "type": "builtin",
            "name": func_name,
            "func_id": func_id,
            "depth": depth,
            "step": step,
            "args": getattr(self, "pending_args", None),
            "return_value": None,  # Will be filled after execution
        }

    def on_step_after(self, frame, vm_state: Dict[str, Any]) -> None:
        """Capture builtin return value after execution.

        Args:
            frame: Current stack frame (or None if returned)
            vm_state: VM state after step
        """
        if not self.enabled:
            return

        # If we have a pending builtin call, complete it
        if self.pending_builtin:
            # The return value should be on top of the stack after execution
            stack = vm_state.get("stack", [])
            if stack:
                self.pending_builtin["return_value"] = stack[-1]

            self.trace.append(self.pending_builtin)
            self.pending_builtin = None
            self.pending_args = None

    def get_data(self) -> List[Dict[str, Any]]:
        """Get the complete builtin trace.

        Returns:
            List of builtin call events with args and return values
        """
        return self.trace

    def reset(self) -> None:
        """Reset trace data."""
        self.trace.clear()
        self.pending_builtin = None
