"""Test nested indexed assignment: x[1][2] = value"""
import pytest
from moo_interp.moo_ast import parse, compile, disassemble
from moo_interp.vm import VM, VMOutcome
from moo_interp.list import MOOList
from moo_interp.string import MOOString


def run_code(code: str):
    """Compile and run MOO code, return the VM after execution."""
    frame = compile(code)
    vm = VM()
    vm.call_stack = [frame]
    for _ in vm.run():
        pass
    return vm


def get_var(vm, name: str):
    """Get a variable's value from the VM's current frame."""
    frame = vm.call_stack[0] if vm.call_stack else None
    if frame is None:
        # Frame was popped, check the last frame's state
        return None
    try:
        idx = frame.prog.var_names.index(MOOString(name))
        return frame.rt_env[idx]
    except (ValueError, IndexError):
        return None


class TestSimpleIndexedAssignment:
    """Test simple x[i] = value"""

    def test_list_single_index_assignment(self):
        """x = {1, 2, 3}; x[2] = 99; return x;"""
        code = 'x = {1, 2, 3}; x[2] = 99; return x;'
        vm = run_code(code)
        assert vm.state == VMOutcome.OUTCOME_DONE
        result = vm.result
        assert isinstance(result, MOOList)
        assert list(result._list) == [1, 99, 3]

    def test_list_assignment_returns_value(self):
        """The assignment expression should return the assigned value."""
        code = 'x = {1, 2, 3}; return (x[2] = 99);'
        vm = run_code(code)
        assert vm.state == VMOutcome.OUTCOME_DONE
        assert vm.result == 99


class TestNestedIndexedAssignment:
    """Test nested x[i][j] = value"""

    def test_nested_list_assignment(self):
        """x = {{1, 2}, {3, 4}}; x[1][2] = 99; return x;"""
        code = 'x = {{1, 2}, {3, 4}}; x[1][2] = 99; return x;'
        vm = run_code(code)
        assert vm.state == VMOutcome.OUTCOME_DONE
        result = vm.result
        assert isinstance(result, MOOList)
        # x[1] should be {1, 99}, x[2] should be {3, 4}
        inner1 = result._list[0]
        inner2 = result._list[1]
        assert list(inner1._list) == [1, 99]
        assert list(inner2._list) == [3, 4]

    def test_nested_assignment_returns_value(self):
        """The nested assignment expression should return the assigned value."""
        code = 'x = {{1, 2}, {3, 4}}; return (x[1][2] = 99);'
        vm = run_code(code)
        assert vm.state == VMOutcome.OUTCOME_DONE
        assert vm.result == 99

    def test_triple_nested_assignment(self):
        """x = {{{1}}}; x[1][1][1] = 99; return x;"""
        code = 'x = {{{1}}}; x[1][1][1] = 99; return x;'
        vm = run_code(code)
        assert vm.state == VMOutcome.OUTCOME_DONE
        result = vm.result
        # Should be {{{99}}}
        assert result._list[0]._list[0]._list[0] == 99


class TestBytecodeGeneration:
    """Test that the compiler generates correct bytecode."""

    def test_simple_assignment_bytecode(self):
        """Check bytecode for x[1] = 2"""
        code = 'x = {0}; x[1] = 2;'
        frame = compile(code)
        # Just check it compiles without error
        assert frame is not None

    def test_nested_assignment_uses_push_ref(self):
        """Check that nested assignment uses OP_PUSH_REF, not OP_REF."""
        from moo_interp.opcodes import Opcode
        code = 'x = {{0}}; x[1][1] = 2;'
        frame = compile(code)

        # Find all opcodes in the bytecode
        opcodes = [instr.opcode for instr in frame.stack]

        # Should have OP_PUSH_REF for nested indexing
        assert Opcode.OP_PUSH_REF in opcodes, "Nested assignment should use OP_PUSH_REF"

        # Should have multiple OP_INDEXSET for the chain
        indexset_count = sum(1 for op in opcodes if op == Opcode.OP_INDEXSET)
        assert indexset_count >= 2, f"Expected at least 2 OP_INDEXSET, got {indexset_count}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
