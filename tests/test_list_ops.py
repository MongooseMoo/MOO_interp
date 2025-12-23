"""Tests for list operation opcodes."""

from contextlib import contextmanager

import pytest
from moo_interp.list import MOOList
from moo_interp.string import MOOString
from moo_interp.opcodes import Extended_Opcode, Opcode
from moo_interp.vm import VM, Instruction, Program, StackFrame, VMError


@contextmanager
def create_vm():
    vm = VM()
    try:
        yield vm
    finally:
        pass


def test_eop_length_list():
    """EOP_LENGTH returns list length."""
    with create_vm() as vm:
        test_list = MOOList(1, 2, 3, 4, 5)
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_LENGTH.value),
        ]))
        vm.step()  # Push list
        vm.step()  # Execute EOP_LENGTH
        assert vm.stack[-1] == 5


def test_eop_length_string():
    """EOP_LENGTH returns string length."""
    with create_vm() as vm:
        test_string = MOOString("hello")
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_string),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_LENGTH.value),
        ]))
        vm.step()  # Push string
        vm.step()  # Execute EOP_LENGTH
        assert vm.stack[-1] == 5


def test_eop_first_list():
    """EOP_FIRST returns first element of list."""
    with create_vm() as vm:
        test_list = MOOList(10, 20, 30)
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_FIRST.value),
        ]))
        vm.step()  # Push list
        vm.step()  # Execute EOP_FIRST
        assert vm.stack[-1] == 10


def test_eop_first_empty_list():
    """EOP_FIRST on empty list raises error."""
    with create_vm() as vm:
        test_list = MOOList()
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_FIRST.value),
        ]))
        vm.step()  # Push list
        with pytest.raises(VMError):
            vm.step()  # Execute EOP_FIRST


def test_eop_last_list():
    """EOP_LAST returns last element of list."""
    with create_vm() as vm:
        test_list = MOOList(10, 20, 30)
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_LAST.value),
        ]))
        vm.step()  # Push list
        vm.step()  # Execute EOP_LAST
        assert vm.stack[-1] == 30


def test_eop_last_empty_list():
    """EOP_LAST on empty list raises error."""
    with create_vm() as vm:
        test_list = MOOList()
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_LAST.value),
        ]))
        vm.step()  # Push list
        with pytest.raises(VMError):
            vm.step()  # Execute EOP_LAST


def test_eop_rangeset_basic():
    """EOP_RANGESET replaces range of elements in list."""
    with create_vm() as vm:
        test_list = MOOList(1, 2, 3, 4, 5)
        replacement = MOOList(10, 20, 30)
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            Instruction(opcode=Opcode.OP_IMM, operand=2),  # start index
            Instruction(opcode=Opcode.OP_IMM, operand=4),  # end index
            Instruction(opcode=Opcode.OP_IMM, operand=replacement),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_RANGESET.value),
        ]))
        vm.step()  # Push list
        vm.step()  # Push start index
        vm.step()  # Push end index
        vm.step()  # Push replacement
        vm.step()  # Execute EOP_RANGESET

        # Should replace indices 2-4 with new values: [1, 10, 20, 30, 5]
        result = vm.stack[-1]
        assert isinstance(result, MOOList)
        assert result._list == [1, 10, 20, 30, 5]


def test_eop_complement():
    """EOP_COMPLEMENT returns bitwise complement."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_COMPLEMENT.value),
        ]))
        vm.step()  # Push value
        vm.step()  # Execute EOP_COMPLEMENT
        assert vm.stack[-1] == ~5


def test_eop_scatter_simple():
    """{a, b, c} = {1, 2, 3} assigns correctly."""
    with create_vm() as vm:
        test_list = MOOList(1, 2, 3)
        # Set up variables to scatter into
        prog = Program(var_names=[MOOString("a"), MOOString("b"), MOOString("c")])

        # Create scatter instruction with pattern
        scatter_instr = Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_SCATTER.value)
        scatter_instr.scatter_pattern = [("a", False), ("b", False), ("c", False)]

        vm.call_stack.append(StackFrame(0, prog, ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            scatter_instr,
        ]))

        # Initialize runtime environment
        vm.call_stack[-1].rt_env = [0, 0, 0]

        vm.step()  # Push list
        vm.step()  # Execute EOP_SCATTER

        # After scatter, variables should have values
        assert vm.call_stack[-1].rt_env[0] == 1
        assert vm.call_stack[-1].rt_env[1] == 2
        assert vm.call_stack[-1].rt_env[2] == 3


def test_eop_scatter_with_rest():
    """{a, @rest} = {1, 2, 3, 4} captures remainder."""
    with create_vm() as vm:
        test_list = MOOList(1, 2, 3, 4)
        # Set up variables: a and rest
        prog = Program(var_names=[MOOString("a"), MOOString("rest")])

        # Create scatter instruction with rest pattern
        scatter_instr = Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_SCATTER.value)
        scatter_instr.scatter_pattern = [("a", False), ("rest", True)]  # True = is_rest

        vm.call_stack.append(StackFrame(0, prog, ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            scatter_instr,
        ]))

        # Initialize runtime environment
        vm.call_stack[-1].rt_env = [0, MOOList()]

        vm.step()  # Push list
        vm.step()  # Execute EOP_SCATTER

        # After scatter with @rest pattern
        # a should get first element, rest should get remainder
        assert vm.call_stack[-1].rt_env[0] == 1
        assert vm.call_stack[-1].rt_env[1]._list == [2, 3, 4]
