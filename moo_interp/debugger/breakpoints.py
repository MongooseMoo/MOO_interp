"""Breakpoint plugin for MOO debugger."""

from typing import Any, Dict, List, Optional, Set, Tuple
from .base import DebugPlugin
from .conditions import parse_condition, ConditionNode
from ..opcodes import Opcode


class BreakpointPlugin(DebugPlugin):
    """Plugin that handles breakpoint logic."""

    def __init__(self):
        """Initialize breakpoint plugin."""
        super().__init__()
        self.breakpoints: Set[Tuple[str, Any]] = set()
        self.conditional_breakpoints: List[Tuple[str, ConditionNode]] = []
        self.last_matched_condition: Optional[Dict[str, Any]] = None
        self._verb_call_match = False
        self._verb_return_match = False

    def set_breakpoint(self, bp_type: str, value: Any) -> None:
        """Set a breakpoint.

        Args:
            bp_type: Type of breakpoint ('ip', 'opcode', 'verb', 'builtin', 'conditional')
            value: Value to break on (depends on bp_type)
        """
        if bp_type == 'conditional':
            # Parse and store conditional breakpoint
            condition = parse_condition(value)
            self.conditional_breakpoints.append((value, condition))
        else:
            self.breakpoints.add((bp_type, value))

    def remove_breakpoint(self, bp_type: str, value: Any) -> None:
        """Remove a breakpoint.

        Args:
            bp_type: Type of breakpoint
            value: Value that was being watched
        """
        if bp_type == 'conditional':
            # Remove conditional breakpoint
            self.conditional_breakpoints = [
                (expr, cond) for expr, cond in self.conditional_breakpoints
                if expr != value
            ]
        else:
            self.breakpoints.discard((bp_type, value))

    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self.breakpoints.clear()
        self.conditional_breakpoints.clear()
        self.last_matched_condition = None

    def should_break(self, frame, vm_state: Dict[str, Any]) -> bool:
        """Check if any breakpoint condition is met.

        Args:
            frame: Current stack frame
            vm_state: Current VM state

        Returns:
            True if a breakpoint was hit
        """

        # Check if we matched on verb call/return hook
        if self._verb_call_match or self._verb_return_match:
            self._verb_call_match = False
            self._verb_return_match = False
            return True

        if frame is None or not self.enabled:
            return False

        # Can't check breakpoints if past end of instructions
        if frame.ip >= len(frame.stack):
            return False

        instr = frame.current_instruction

        # Check regular breakpoints
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

        # Check conditional breakpoints
        for expression, condition in self.conditional_breakpoints:
            context = self._build_context(frame, vm_state)
            try:
                if condition.evaluate(context):
                    # Store matched values for reporting
                    self.last_matched_condition = {
                        'expression': expression,
                        'matched': {k: v for k, v in context.items() if v is not None}
                    }
                    return True
            except Exception:
                # Condition evaluation failed, skip this breakpoint
                pass

        return False

    def _build_context(self, frame, vm_state: Dict[str, Any]) -> Dict[str, Any]:
        """Build evaluation context from frame and VM state.

        Args:
            frame: Current stack frame
            vm_state: Current VM state

        Returns:
            Dictionary with available variables for condition evaluation
        """
        context = {}

        # Add verb information
        context['verb'] = frame.verb_name if frame.verb_name else str(frame.verb)

        # Add return value from vm_state if available
        if 'result' in vm_state:
            context['return_value'] = vm_state['result']

        # Add arguments from frame rt_env if available
        # In MOO, args is typically at index 4 in rt_env (after player, this, caller, verb)
        if hasattr(frame, 'rt_env') and len(frame.rt_env) > 4:
            args_obj = frame.rt_env[4]
            # Convert MOOList to Python list for indexing
            if hasattr(args_obj, '_list'):
                context['args'] = list(args_obj._list)
            elif isinstance(args_obj, list):
                context['args'] = args_obj
            else:
                context['args'] = []

        # Add stack depth
        context['stack_depth'] = vm_state.get('stack_depth', 0)

        # Add current object
        if hasattr(frame, 'this'):
            # Extract numeric value from ObjNum if needed
            this = frame.this
            if hasattr(this, '__str__'):
                this_str = str(this)
                if this_str.startswith('#'):
                    context['this'] = int(this_str[1:])
                else:
                    context['this'] = this
            else:
                context['this'] = this

        return context

    def get_data(self) -> dict:
        """Get current breakpoints.

        Returns:
            Dict with 'regular' and 'conditional' breakpoint lists
        """
        return {
            'regular': sorted(list(self.breakpoints)),
            'conditional': [expr for expr, _ in self.conditional_breakpoints],
            'last_matched': self.last_matched_condition
        }


    def on_verb_call(self, verb_name: str, this, args: list, call_depth: int, step_count: int):
        """Hook called before a verb is executed."""
        if not self.enabled:
            return
        for expression, condition in self.conditional_breakpoints:
            context = {
                'verb': verb_name,
                'this': int(str(this).lstrip('#')) if hasattr(this, '__str__') else this,
                'args': args,
                'call_depth': call_depth,
                'stack_depth': call_depth,
            }
            try:
                if condition.evaluate(context):
                    self.last_matched_condition = {
                        'expression': expression,
                        'matched': {k: v for k, v in context.items() if v is not None}
                    }
                    self._verb_call_match = True
            except Exception:
                pass

    def on_verb_return(self, verb_name: Optional[str], return_value, call_depth: int, step_count: int):
        """Hook called after a verb returns."""
        if not self.enabled:
            return
        for expression, condition in self.conditional_breakpoints:
            context = {
                'verb': verb_name if verb_name else 'unknown',
                'return_value': return_value,
                'call_depth': call_depth,
                'stack_depth': call_depth,
            }
            try:
                if condition.evaluate(context):
                    self.last_matched_condition = {
                        'expression': expression,
                        'matched': {k: v for k, v in context.items() if v is not None}
                    }
                    self._verb_return_match = True
            except Exception:
                pass

    def reset(self) -> None:
        self.last_matched_condition = None
        self._verb_call_match = False
        self._verb_return_match = False
        """Reset breakpoints."""
        self.breakpoints.clear()
        self.conditional_breakpoints.clear()
        self.last_matched_condition = None
