"""MOO VM Debugger

Provides step-by-step execution control and state inspection for the MOO VM.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from .vm import VM, StackFrame, VMOutcome
from .opcodes import Opcode, Extended_Opcode
from .string import MOOString


class MooDebugger:
    """Interactive debugger for MOO VM execution.

    Provides stepping, breakpoints, and state inspection capabilities.
    """

    def __init__(self, vm: VM):
        """Initialize the debugger.

        Args:
            vm: The VM instance to debug
        """
        self.vm = vm
        self.breakpoints: Set[Tuple[str, Any]] = set()
        self.history: List[Dict[str, Any]] = []
        self.paused = False
        self.step_count = 0

    def step(self) -> Optional[Dict[str, Any]]:
        """Execute one instruction and return state snapshot.

        Returns:
            State snapshot dict, or None if execution is complete
        """
        if not self.vm.call_stack or self.vm.state is not None:
            return None

        # Capture state before execution
        snapshot_before = self._capture_state()

        # Execute one step
        self.vm.step()
        self.step_count += 1

        # Capture state after execution
        snapshot_after = self._capture_state()
        snapshot_after['step_count'] = self.step_count

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
        """Set a breakpoint.

        Args:
            bp_type: Type of breakpoint ('ip', 'opcode', 'verb', 'builtin')
            value: Value to break on (depends on bp_type)
        """
        self.breakpoints.add((bp_type, value))

    def remove_breakpoint(self, bp_type: str, value: Any):
        """Remove a breakpoint.

        Args:
            bp_type: Type of breakpoint
            value: Value that was being watched
        """
        self.breakpoints.discard((bp_type, value))

    def clear_breakpoints(self):
        """Clear all breakpoints."""
        self.breakpoints.clear()

    def _check_breakpoint(self) -> bool:
        """Check if any breakpoint condition is met.

        Returns:
            True if a breakpoint was hit
        """
        if not self.vm.call_stack:
            return False

        frame = self.vm.current_frame

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
                    func_id = instr.operand
                    func_name = self.vm.bi_funcs.get_function_name_by_id(func_id)
                    if func_name == value or func_id == value:
                        return True

        return False

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
        if not self.breakpoints:
            print("No breakpoints set")
            return

        print("Breakpoints:")
        for bp_type, value in sorted(self.breakpoints):
            print(f"  {bp_type}: {value}")
