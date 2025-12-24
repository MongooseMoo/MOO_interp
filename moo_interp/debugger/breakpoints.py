"""Breakpoint plugin for MOO debugger."""

from typing import Any, Dict, Set, Tuple
from .base import DebugPlugin
from ..opcodes import Opcode


class BreakpointPlugin(DebugPlugin):
    """Plugin that handles breakpoint logic."""

    def __init__(self):
        """Initialize breakpoint plugin."""
        super().__init__()
        self.breakpoints: Set[Tuple[str, Any]] = set()

    def set_breakpoint(self, bp_type: str, value: Any) -> None:
        """Set a breakpoint.

        Args:
            bp_type: Type of breakpoint ('ip', 'opcode', 'verb', 'builtin')
            value: Value to break on (depends on bp_type)
        """
        self.breakpoints.add((bp_type, value))

    def remove_breakpoint(self, bp_type: str, value: Any) -> None:
        """Remove a breakpoint.

        Args:
            bp_type: Type of breakpoint
            value: Value that was being watched
        """
        self.breakpoints.discard((bp_type, value))

    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self.breakpoints.clear()

    def should_break(self, frame, vm_state: Dict[str, Any]) -> bool:
        """Check if any breakpoint condition is met.

        Args:
            frame: Current stack frame
            vm_state: Current VM state

        Returns:
            True if a breakpoint was hit
        """
        if frame is None or not self.enabled:
            return False

        # Can't check breakpoints if past end of instructions
        if frame.ip >= len(frame.stack):
            return False

        instr = frame.current_instruction

        for bp_type, value in self.breakpoints:
            if bp_type == 'ip':
                if frame.ip == value:
                    return True
            elif bp_type == 'opcode':
                if instr.opcode == value:
                    return True
            elif bp_type == 'verb':
                if frame.verb_name == value or frame.verb == value:
                    return True
            elif bp_type == 'builtin':
                if instr.opcode == Opcode.OP_BI_FUNC_CALL:
                    # Need to get function name from vm_state since we need bi_funcs
                    if vm_state.get('builtin_name') == value or vm_state.get('builtin_id') == value:
                        return True

        return False

    def get_data(self) -> list:
        """Get current breakpoints.

        Returns:
            List of (type, value) breakpoint tuples
        """
        return sorted(list(self.breakpoints))

    def reset(self) -> None:
        """Reset breakpoints."""
        self.breakpoints.clear()
