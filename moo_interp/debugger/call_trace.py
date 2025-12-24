"""Call tree tracing plugin for MOO debugger."""

from typing import Any, Dict, List, Optional
from .base import DebugPlugin


class CallTracePlugin(DebugPlugin):
    """Plugin that traces verb calls and returns.

    Captures a complete call tree showing:
    - Verb calls with arguments
    - Verb returns with values
    - Call depth
    - Step numbers
    """

    def __init__(self):
        """Initialize call trace plugin."""
        super().__init__()
        self.trace: List[Dict[str, Any]] = []
        self.call_stack_tracker: List[Dict[str, Any]] = []  # Track pending calls

    def on_verb_call(self, verb_name: str, this: int, args: list, depth: int, step: int) -> None:
        """Record a verb call.

        Args:
            verb_name: Name of the verb being called
            this: Object the verb is on
            args: Arguments to the verb
            depth: Current call stack depth
            step: Step number
        """
        if not self.enabled:
            return

        call_entry = {
            "type": "call",
            "verb": verb_name,
            "object": this,
            "args": args,
            "depth": depth,
            "step": step,
        }
        self.trace.append(call_entry)

        # Track this call so we can pair it with its return
        self.call_stack_tracker.append({
            "verb": verb_name,
            "object": this,
            "depth": depth,
        })

    def on_verb_return(self, verb_name: Optional[str], return_value: Any, depth: int, step: int) -> None:
        """Record a verb return.

        Args:
            verb_name: Name of the verb returning (if known)
            return_value: Value being returned
            depth: Call stack depth after return
            step: Step number
        """
        if not self.enabled:
            return

        # Pop from our call tracker to get the verb that's returning
        returning_verb = None
        returning_obj = None
        if self.call_stack_tracker:
            call_info = self.call_stack_tracker.pop()
            returning_verb = call_info["verb"]
            returning_obj = call_info["object"]

        return_entry = {
            "type": "return",
            "verb": verb_name or returning_verb,
            "object": returning_obj,
            "return_value": return_value,
            "depth": depth,
            "step": step,
        }
        self.trace.append(return_entry)

    def get_data(self) -> List[Dict[str, Any]]:
        """Get the complete call trace.

        Returns:
            List of call/return events
        """
        return self.trace

    def reset(self) -> None:
        """Reset trace data."""
        self.trace.clear()
        self.call_stack_tracker.clear()
