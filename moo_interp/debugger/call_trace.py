"""Call tree tracing plugin for MOO debugger."""

import fnmatch
from typing import Any, Dict, List, Optional
from .base import DebugPlugin


class CallTracePlugin(DebugPlugin):
    """Plugin that traces verb calls and returns.

    Captures a complete call tree showing:
    - Verb calls with arguments
    - Verb returns with values
    - Call depth
    - Step numbers

    Supports filtering verbs by glob patterns:
    - "redlist*" - all verbs starting with redlist
    - "*login*" - all verbs containing login
    - "$login:*" - all verbs on the $login object
    - "$login:do_login" - specific verb on specific object
    """

    def __init__(self, filter_pattern: Optional[str] = None):
        """Initialize call trace plugin.

        Args:
            filter_pattern: Optional glob pattern to filter verbs.
                           Format: "verb_name" or "object_name:verb_name"
        """
        super().__init__()
        self.trace: List[Dict[str, Any]] = []
        self.call_stack_tracker: List[Dict[str, Any]] = []  # Track pending calls
        self.filter_pattern = filter_pattern

    def _matches_filter(self, verb_name: str, obj: int, obj_name: Optional[str] = None) -> bool:
        """Check if a verb matches the filter pattern.

        Args:
            verb_name: Name of the verb
            obj: Object number
            obj_name: Optional object name (like "$login")

        Returns:
            True if the verb matches the filter (or no filter is set)
        """
        if not self.filter_pattern:
            return True

        # Handle object:verb patterns like "$login:*" or "#0:do_command"
        if ":" in self.filter_pattern:
            filter_obj, filter_verb = self.filter_pattern.split(":", 1)

            # Match object part
            obj_matches = False
            if filter_obj.startswith("$"):
                # Match by object name
                obj_matches = obj_name and fnmatch.fnmatch(obj_name, filter_obj)
            elif filter_obj.startswith("#"):
                # Match by object number
                obj_matches = f"#{obj}" == filter_obj or fnmatch.fnmatch(f"#{obj}", filter_obj)
            else:
                # Try both
                obj_matches = (
                    f"#{obj}" == filter_obj or
                    (obj_name and fnmatch.fnmatch(obj_name, filter_obj))
                )

            if not obj_matches:
                return False

            # Match verb part
            return fnmatch.fnmatch(verb_name, filter_verb)
        else:
            # Just a verb pattern like "redlist*"
            return fnmatch.fnmatch(verb_name, self.filter_pattern)

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

        # Check filter
        if not self._matches_filter(verb_name, this):
            # Still track in call stack for return pairing, but mark as filtered
            self.call_stack_tracker.append({
                "verb": verb_name,
                "object": this,
                "depth": depth,
                "filtered": True,
            })
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
            "filtered": False,
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
        was_filtered = False
        if self.call_stack_tracker:
            call_info = self.call_stack_tracker.pop()
            returning_verb = call_info["verb"]
            returning_obj = call_info["object"]
            was_filtered = call_info.get("filtered", False)

        # Skip recording if this was a filtered call
        if was_filtered:
            return

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
