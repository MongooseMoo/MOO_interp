"""MOO VM Debugger with plugin architecture.

Provides step-by-step execution control and state inspection for the MOO VM
with a modular plugin system for extensibility.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from ..vm import VM, StackFrame, VMOutcome
from ..opcodes import Opcode, Extended_Opcode
from ..string import MOOString
from ..list import MOOList

from .base import DebugPlugin
from .breakpoints import BreakpointPlugin
from .call_trace import CallTracePlugin
from .builtin_trace import BuiltinTracePlugin
from .variable_watch import VariableWatchPlugin


class MooDebugger:
    """Interactive debugger for MOO VM execution with plugin support.

    Provides stepping, breakpoints, and state inspection capabilities.
    Can be extended with plugins for tracing, profiling, etc.
    """

    def __init__(self, vm: VM, plugins: Optional[List[DebugPlugin]] = None):
        """Initialize the debugger.

        Args:
            vm: The VM instance to debug
            plugins: Optional list of plugins to use (defaults to breakpoints only)
        """
        self.vm = vm
        self.history: List[Dict[str, Any]] = []
        self.paused = False
        self.step_count = 0

        # Initialize plugins
        if plugins is None:
            # Default: just breakpoints
            self.plugins = [BreakpointPlugin()]
        else:
            # Always include BreakpointPlugin if not already in list
            has_breakpoints = any(isinstance(p, BreakpointPlugin) for p in plugins)
            if has_breakpoints:
                self.plugins = plugins
            else:
                self.plugins = [BreakpointPlugin()] + plugins

        # Keep reference to breakpoint plugin for convenience
        self.breakpoint_plugin = None
        for plugin in self.plugins:
            if isinstance(plugin, BreakpointPlugin):
                self.breakpoint_plugin = plugin
                break

    def add_plugin(self, plugin: DebugPlugin):
        """Add a plugin to the debugger.

        Args:
            plugin: Plugin to add
        """
        self.plugins.append(plugin)
        if isinstance(plugin, BreakpointPlugin) and self.breakpoint_plugin is None:
            self.breakpoint_plugin = plugin

    def step(self) -> Optional[Dict[str, Any]]:
        """Execute one instruction and return state snapshot.

        Returns:
            State snapshot dict, or None if execution is complete
        """
        if not self.vm.call_stack or self.vm.state is not None:
            return None

        # Capture state before execution
        snapshot_before = self._capture_state()
        frame_before = self.vm.current_frame if self.vm.call_stack else None
        depth_before = len(self.vm.call_stack)

        # Call plugin hooks before step
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_step_before(frame_before, snapshot_before)

        # Check if this is a verb call or builtin call
        if frame_before and frame_before.ip < len(frame_before.stack):
            instr = frame_before.current_instruction

            # Detect verb calls
            if instr.opcode == Opcode.OP_CALL_VERB:
                # Get verb info from stack (they're about to be popped by exec_call_verb)
                if len(self.vm.stack) >= 3:
                    args = self.vm.stack[-1] if isinstance(self.vm.stack[-1], MOOList) else MOOList([])
                    verb_name = str(self.vm.stack[-2]) if len(self.vm.stack) >= 2 else "unknown"
                    this = self.vm.stack[-3] if len(self.vm.stack) >= 3 else -1

                    for plugin in self.plugins:
                        if plugin.enabled:
                            plugin.on_verb_call(verb_name, this, list(args._list), depth_before, self.step_count)

            # Detect builtin calls
            elif instr.opcode == Opcode.OP_BI_FUNC_CALL:
                func_id = instr.operand
                func_name = self.vm.bi_funcs.get_function_name_by_id(func_id) if self.vm.bi_funcs else f"builtin_{func_id}"

                for plugin in self.plugins:
                    if plugin.enabled:
                        plugin.on_builtin_call(func_name, func_id, depth_before, self.step_count)

        # Execute one step
        self.vm.step()
        self.step_count += 1

        # Capture state after execution
        snapshot_after = self._capture_state()
        snapshot_after['step_count'] = self.step_count
        frame_after = self.vm.current_frame if self.vm.call_stack else None
        depth_after = len(self.vm.call_stack)

        # Detect returns (call depth decreased)
        if depth_after < depth_before:
            # A return happened - the return value is on top of the value stack
            # (pushed there by the returning verb). Only use vm.result when the
            # entire VM has finished (empty call stack).
            if depth_after == 0 and not self.vm.call_stack:
                # Final return - VM is done, result is in vm.result
                return_value = self.vm.result
            else:
                # Mid-execution return - return value was pushed onto the stack
                return_value = self.vm.stack[-1] if self.vm.stack else None

            for plugin in self.plugins:
                if plugin.enabled:
                    # We don't have verb_name from the returned frame, so pass None
                    plugin.on_verb_return(None, return_value, depth_after, self.step_count)

        # Call plugin hooks after step
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_step_after(frame_after, snapshot_after)

        # Add to history
        self.history.append(snapshot_after)

        return snapshot_after

    def continue_until_breakpoint(self) -> Optional[Dict[str, Any]]:
        """Run until a breakpoint is hit or execution completes.

        Returns:
            State snapshot when stopped, or None if execution complete
        """
        while True:
            # Check if done
            if not self.vm.call_stack or self.vm.state is not None:
                return None

            # Check for breakpoints before stepping
            if self._check_breakpoint():
                return self._capture_state()

            # Step
            snapshot = self.step()
            if snapshot is None:
                return None

    def set_breakpoint(self, bp_type: str, value: Any):
        """Set a breakpoint (convenience method).

        Args:
            bp_type: Type of breakpoint ('ip', 'opcode', 'verb', 'builtin')
            value: Value to break on (depends on bp_type)
        """
        if self.breakpoint_plugin:
            self.breakpoint_plugin.set_breakpoint(bp_type, value)

    def remove_breakpoint(self, bp_type: str, value: Any):
        """Remove a breakpoint (convenience method).

        Args:
            bp_type: Type of breakpoint
            value: Value that was being watched
        """
        if self.breakpoint_plugin:
            self.breakpoint_plugin.remove_breakpoint(bp_type, value)

    def clear_breakpoints(self):
        """Clear all breakpoints (convenience method)."""
        if self.breakpoint_plugin:
            self.breakpoint_plugin.clear_breakpoints()

    def _check_breakpoint(self) -> bool:
        """Check if any breakpoint condition is met.

        Returns:
            True if a breakpoint was hit
        """
        if not self.vm.call_stack:
            return False

        frame = self.vm.current_frame
        state = self._capture_state()

        # Check all plugins for should_break
        for plugin in self.plugins:
            if plugin.enabled and plugin.should_break(frame, state):
                return True

        return False

    def get_plugin_data(self, plugin_type: type) -> Any:
        """Get data from a specific plugin type.

        Args:
            plugin_type: Type of plugin to get data from

        Returns:
            Plugin data, or None if plugin not found
        """
        for plugin in self.plugins:
            if isinstance(plugin, plugin_type):
                return plugin.get_data()
        return None

    def inspect_stack(self) -> List[Any]:
        """Return current operand stack contents.

        Returns:
            List of values on the stack (bottom to top)
        """
        return list(self.vm.stack)

    def inspect_vars(self) -> Dict[str, Any]:
        """Return current frame's variable bindings.

        Returns:
            Dict mapping variable names to values
        """
        if not self.vm.call_stack:
            return {}

        frame = self.vm.current_frame
        result = {}

        for i, var_name in enumerate(frame.prog.var_names):
            if i < len(frame.rt_env):
                result[str(var_name)] = frame.rt_env[i]

        return result

    def inspect_call_stack(self) -> List[Dict[str, Any]]:
        """Return call stack with frame information.

        Returns:
            List of dicts describing each stack frame
        """
        result = []
        for i, frame in enumerate(self.vm.call_stack):
            frame_info = {
                'depth': i,
                'verb': frame.verb,
                'verb_name': frame.verb_name,
                'this': frame.this,
                'player': frame.player,
                'ip': frame.ip,
                'instructions_remaining': len(frame.stack) - frame.ip,
            }
            result.append(frame_info)
        return result

    def get_current_instruction(self) -> Optional[Dict[str, Any]]:
        """Return details about the current instruction.

        Returns:
            Dict with instruction details, or None if no current instruction
        """
        if not self.vm.call_stack:
            return None

        frame = self.vm.current_frame

        if frame.ip >= len(frame.stack):
            return None

        instr = frame.current_instruction

        result = {
            'ip': frame.ip,
            'opcode': instr.opcode.name if hasattr(instr.opcode, 'name') else str(instr.opcode),
            'operand': instr.operand,
        }

        # Add extra details for specific opcodes
        if instr.opcode == Opcode.OP_BI_FUNC_CALL:
            func_id = instr.operand
            func_name = self.vm.bi_funcs.get_function_name_by_id(func_id)
            result['builtin_name'] = func_name
            result['builtin_id'] = func_id
        elif instr.opcode == Opcode.OP_CALL_VERB:
            result['note'] = 'verb call'
        elif instr.opcode == Opcode.OP_EXTENDED:
            if isinstance(instr.operand, Extended_Opcode):
                result['extended_opcode'] = instr.operand.name

        return result

    def get_source_context(self, lines_before: int = 2, lines_after: int = 2) -> Optional[str]:
        """Return MOO source around current position if available.

        Args:
            lines_before: Number of lines to show before current
            lines_after: Number of lines to show after current

        Returns:
            String with source context, or None if not available
        """
        # This would require source line mapping from the compiler
        # For now, return a placeholder
        return None

    def _capture_state(self) -> Dict[str, Any]:
        """Capture current VM state as a snapshot.

        Returns:
            Dict with complete state information
        """
        if not self.vm.call_stack:
            return {
                'state': 'done',
                'result': self.vm.result,
                'outcome': self.vm.state.name if self.vm.state else None,
                'opcode': None,
                'ip': None,
            }

        frame = self.vm.current_frame

        # Get current instruction info
        instr_info = self.get_current_instruction()

        snapshot = {
            'ip': frame.ip,
            'opcode': instr_info['opcode'] if instr_info else None,
            'operand': instr_info.get('operand') if instr_info else None,
            'stack': list(self.vm.stack),
            'stack_depth': len(self.vm.stack),
            'call_depth': len(self.vm.call_stack),
            'current_verb': frame.verb,
            'current_verb_name': frame.verb_name,
            'current_this': frame.this,
            'vars': self.inspect_vars(),
            'state': 'running',
        }

        # Add builtin info if applicable
        if instr_info and 'builtin_name' in instr_info:
            snapshot['builtin_name'] = instr_info['builtin_name']
            snapshot['builtin_id'] = instr_info['builtin_id']

        return snapshot

    def format_state(self, state: Dict[str, Any]) -> str:
        """Format a state snapshot as a readable string.

        Args:
            state: State snapshot from step() or continue_until_breakpoint()

        Returns:
            Formatted string representation
        """
        if state.get('state') == 'done':
            return f"Execution complete. Result: {state.get('result')}"

        lines = []
        lines.append(f"IP: {state['ip']} | {state['opcode']}")

        if state.get('operand') is not None:
            lines.append(f"  Operand: {state['operand']}")

        if state.get('builtin_name'):
            lines.append(f"  Builtin: {state['builtin_name']} (id={state['builtin_id']})")

        lines.append(f"  Stack depth: {state['stack_depth']}, Call depth: {state['call_depth']}")
        lines.append(f"  Verb: {state['current_verb_name'] or state['current_verb']} on #{state['current_this']}")

        if state['stack']:
            stack_str = ", ".join(repr(v) for v in state['stack'][-5:])
            if state['stack_depth'] > 5:
                stack_str = f"... {stack_str}"
            lines.append(f"  Stack: [{stack_str}]")

        return "\n".join(lines)

    def print_state(self, state: Optional[Dict[str, Any]] = None):
        """Print current or given state in readable format.

        Args:
            state: Optional state to print (defaults to current state)
        """
        if state is None:
            state = self._capture_state()
        print(self.format_state(state))

    def print_vars(self):
        """Print current variables."""
        vars_dict = self.inspect_vars()
        if not vars_dict:
            print("No variables in current frame")
            return

        print("Variables:")
        for name, value in vars_dict.items():
            print(f"  {name} = {repr(value)}")

    def print_call_stack(self):
        """Print the call stack."""
        call_stack = self.inspect_call_stack()
        if not call_stack:
            print("Call stack is empty")
            return

        print("Call stack:")
        for frame_info in call_stack:
            depth = frame_info['depth']
            verb = frame_info['verb_name'] or frame_info['verb']
            this = frame_info['this']
            ip = frame_info['ip']
            remaining = frame_info['instructions_remaining']
            print(f"  [{depth}] {verb} on #{this} (ip={ip}, {remaining} instructions left)")

    def print_breakpoints(self):
        """Print all active breakpoints."""
        if self.breakpoint_plugin:
            bps = self.breakpoint_plugin.get_data()
            if not bps:
                print("No breakpoints set")
                return

            print("Breakpoints:")
            for bp_type, value in bps:
                print(f"  {bp_type}: {value}")
        else:
            print("No breakpoint plugin available")

    def run_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute VM in oneshot mode with query parameters.

        Args:
            query: Query configuration dict with keys:
                - breakpoints: List of (type, value) tuples for breakpoints
                - max_steps: Maximum steps to execute (default: 10000)
                - capture_calls: Whether to log verb calls/returns (default: False)
                - stop_on_error: Whether to stop on VM errors (default: True)

        Returns:
            Result dict with keys:
                - stopped_reason: 'breakpoint' | 'complete' | 'error' | 'max_steps'
                - steps_executed: Number of steps run
                - final_state: Final state snapshot dict
                - call_trace: List of call/return events (if capture_calls=True)
                - error: Error message or None
        """
        # Extract query parameters
        breakpoints = query.get('breakpoints', [])
        max_steps = query.get('max_steps', 10000)
        capture_calls = query.get('capture_calls', False)
        stop_on_error = query.get('stop_on_error', True)

        # Set up breakpoints
        if self.breakpoint_plugin:
            self.breakpoint_plugin.clear_breakpoints()
            for bp_type, value in breakpoints:
                self.breakpoint_plugin.set_breakpoint(bp_type, value)

        # Add call trace plugin if requested
        call_trace_plugin = None
        if capture_calls:
            call_trace_plugin = CallTracePlugin()
            self.add_plugin(call_trace_plugin)

        # Initialize tracking
        steps_executed = 0
        stopped_reason = None
        error_msg = None

        # Run until a stop condition
        while steps_executed < max_steps:
            # Check if execution is complete
            if not self.vm.call_stack or self.vm.state is not None:
                if self.vm.state == VMOutcome.OUTCOME_ABORTED:
                    stopped_reason = 'error'
                    error_msg = str(self.vm.result) if self.vm.result else "VM error"
                else:
                    stopped_reason = 'complete'
                break

            # Check for breakpoints before stepping
            if self._check_breakpoint():
                stopped_reason = 'breakpoint'
                break

            # Step (this will trigger plugin hooks)
            self.step()
            steps_executed += 1

        # Check if we hit max_steps
        if stopped_reason is None and steps_executed >= max_steps:
            stopped_reason = 'max_steps'

        # Capture final state
        final_state = self._capture_state()

        # Get call trace if it was captured
        call_trace = None
        if call_trace_plugin:
            call_trace = call_trace_plugin.get_data()

        return {
            'stopped_reason': stopped_reason,
            'steps_executed': steps_executed,
            'final_state': final_state,
            'call_trace': call_trace,
            'error': error_msg,
        }

    def get_matched_condition(self) -> Optional[Dict[str, Any]]:
        """Get information about the last matched conditional breakpoint.

        Returns:
            Dict with 'expression' and 'matched' keys, or None if no conditional breakpoint matched
        """
        if self.breakpoint_plugin and hasattr(self.breakpoint_plugin, 'last_matched_condition'):
            return self.breakpoint_plugin.last_matched_condition
        return None


# Re-export for backwards compatibility
__all__ = ['MooDebugger', 'DebugPlugin', 'BreakpointPlugin', 'CallTracePlugin']
