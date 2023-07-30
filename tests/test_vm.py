import operator
from contextlib import contextmanager

import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, sampled_from
from moo_interp.list import MOOList

from moo_interp.opcodes import Extended_Opcode, Opcode
from moo_interp.vm import VM, Instruction, Program, StackFrame, VMError


@contextmanager
def create_vm():
    vm = VM()
    try:
        yield vm
    finally:
        # Clean up after each test case if needed
        pass


opcode_mapping = {
    Opcode.OP_MULT: operator.mul,
    # Opcode.OP_DIV: operator.floordiv,
    # Opcode.OP_MOD: operator.mod,
    Opcode.OP_ADD: operator.add,
    Opcode.OP_MINUS: operator.sub,
    Opcode.OP_EQ: operator.eq,
    Opcode.OP_NE: operator.ne,
    Opcode.OP_LT: operator.lt,
    Opcode.OP_LE: operator.le,
    Opcode.OP_GT: operator.gt,
    Opcode.OP_GE: operator.ge,
    Extended_Opcode.EOP_EXP:     operator.pow,
}


@given(
    values=lists(integers(min_value=-1e5, max_value=1e5),
                 min_size=2, max_size=2),
    opcode=sampled_from(list(opcode_mapping.keys())),
)
def test_vm_math_operations(values, opcode):
    with create_vm() as vm:
        # Prepare instructions
        instruction_list = []
        for value in values:
            instruction_list.append(Instruction(
                opcode=Opcode.OP_PUSH, operand=value))
        if isinstance(opcode, Extended_Opcode):
            # For extended opcodes, use Opcode.OP_EXTENDED and opcode value as operand

            instruction_list.append(Instruction(
                opcode=Opcode.OP_EXTENDED, operand=opcode.value))
        else:
            instruction_list.append(Instruction(opcode=opcode))
        prog = Program()
        frame1 = StackFrame(0, prog=prog, ip=0, stack=instruction_list)
        vm.call_stack.append(frame1)

        # Perform the operation using operator module
        try:
            expected_result = opcode_mapping[opcode](*values)
        except ZeroDivisionError:
            expected_result = None

        # Ensure the values are pushed to the stack
        for _ in range(len(values)):
            vm.step()

        # Now execute the opcode operation
        vm.step()

        # Handle potential VMError in case of an operation error (like ZeroDivisionError)
        try:
            assert vm.stack[-1] == expected_result
        except VMError:
            pass


# MOOList tests

def test_vm_creating_empty_list():
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(Program(), 0, ip=0, stack=[
            Instruction(opcode=Opcode.OP_MAKE_EMPTY_LIST),
        ]))
        vm.step()
        assert vm.stack[-1] == MOOList()
