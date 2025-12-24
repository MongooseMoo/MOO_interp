"""Comprehensive tests for bitwise extended opcodes.

Tests all bitwise operations: BITOR, BITAND, BITXOR, BITSHL, BITSHR, COMPLEMENT
with various edge cases including negative numbers and zero.

Note: These opcodes are tested via direct VM execution since bitwise operators
are not part of MOO language syntax (they would be implemented as built-in functions).
"""

from contextlib import contextmanager

import pytest
from moo_interp.opcodes import Extended_Opcode, Opcode
from moo_interp.vm import VM, Instruction, Program, StackFrame


@contextmanager
def create_vm():
    vm = VM()
    try:
        yield vm
    finally:
        pass


def run_bitwise_op(op1, op2, extended_opcode):
    """Helper to run a bitwise operation through the VM."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=op1),
            Instruction(opcode=Opcode.OP_IMM, operand=op2),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=extended_opcode.value),
        ]))
        vm.step()  # Push op1
        vm.step()  # Push op2
        vm.step()  # Execute operation
        return vm.stack[-1]


def run_unary_op(operand, extended_opcode):
    """Helper to run a unary operation through the VM."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=operand),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=extended_opcode.value),
        ]))
        vm.step()  # Push operand
        vm.step()  # Execute operation
        return vm.stack[-1]


# ===== EOP_BITOR tests =====

def test_eop_bitor_basic():
    """EOP_BITOR performs bitwise OR."""
    result = run_bitwise_op(0b1010, 0b1100, Extended_Opcode.EOP_BITOR)
    assert result == 0b1110  # 14


def test_eop_bitor_with_zero():
    """EOP_BITOR with zero is identity."""
    assert run_bitwise_op(42, 0, Extended_Opcode.EOP_BITOR) == 42


def test_eop_bitor_with_self():
    """EOP_BITOR with self returns self."""
    assert run_bitwise_op(42, 42, Extended_Opcode.EOP_BITOR) == 42


def test_eop_bitor_all_bits():
    """EOP_BITOR with complementary bits."""
    assert run_bitwise_op(0b1010, 0b0101, Extended_Opcode.EOP_BITOR) == 0b1111


def test_eop_bitor_negative():
    """EOP_BITOR with negative numbers."""
    # -1 has all bits set
    assert run_bitwise_op(-1, 5, Extended_Opcode.EOP_BITOR) == -1


# ===== EOP_BITAND tests =====

def test_eop_bitand_basic():
    """EOP_BITAND performs bitwise AND."""
    result = run_bitwise_op(0b1010, 0b1100, Extended_Opcode.EOP_BITAND)
    assert result == 0b1000  # 8


def test_eop_bitand_with_zero():
    """EOP_BITAND with zero returns zero."""
    assert run_bitwise_op(42, 0, Extended_Opcode.EOP_BITAND) == 0


def test_eop_bitand_with_self():
    """EOP_BITAND with self returns self."""
    assert run_bitwise_op(42, 42, Extended_Opcode.EOP_BITAND) == 42


def test_eop_bitand_masking():
    """EOP_BITAND can be used for bit masking."""
    assert run_bitwise_op(0xFF, 0x0F, Extended_Opcode.EOP_BITAND) == 0x0F


def test_eop_bitand_negative():
    """EOP_BITAND with negative numbers."""
    # -1 has all bits set, so result is the other operand
    assert run_bitwise_op(-1, 5, Extended_Opcode.EOP_BITAND) == 5


# ===== EOP_BITXOR tests =====

def test_eop_bitxor_basic():
    """EOP_BITXOR performs bitwise XOR."""
    result = run_bitwise_op(0b1010, 0b1100, Extended_Opcode.EOP_BITXOR)
    assert result == 0b0110  # 6


def test_eop_bitxor_with_zero():
    """EOP_BITXOR with zero is identity."""
    assert run_bitwise_op(42, 0, Extended_Opcode.EOP_BITXOR) == 42


def test_eop_bitxor_with_self():
    """EOP_BITXOR with self returns zero."""
    assert run_bitwise_op(42, 42, Extended_Opcode.EOP_BITXOR) == 0


def test_eop_bitxor_toggle():
    """EOP_BITXOR can toggle bits."""
    assert run_bitwise_op(0b1010, 0b0101, Extended_Opcode.EOP_BITXOR) == 0b1111


def test_eop_bitxor_negative():
    """EOP_BITXOR with negative numbers."""
    assert run_bitwise_op(-1, 5, Extended_Opcode.EOP_BITXOR) == -6


# ===== EOP_BITSHL tests =====

def test_eop_bitshl_basic():
    """EOP_BITSHL performs left shift."""
    result = run_bitwise_op(5, 2, Extended_Opcode.EOP_BITSHL)
    assert result == 20  # 5 << 2 = 20


def test_eop_bitshl_by_zero():
    """EOP_BITSHL by zero is identity."""
    assert run_bitwise_op(42, 0, Extended_Opcode.EOP_BITSHL) == 42


def test_eop_bitshl_by_one():
    """EOP_BITSHL by 1 is multiply by 2."""
    assert run_bitwise_op(10, 1, Extended_Opcode.EOP_BITSHL) == 20


def test_eop_bitshl_large_shift():
    """EOP_BITSHL with large shift value."""
    assert run_bitwise_op(1, 10, Extended_Opcode.EOP_BITSHL) == 1024


def test_eop_bitshl_negative():
    """EOP_BITSHL with negative number."""
    assert run_bitwise_op(-4, 2, Extended_Opcode.EOP_BITSHL) == -16


# ===== EOP_BITSHR tests =====

def test_eop_bitshr_basic():
    """EOP_BITSHR performs right shift."""
    result = run_bitwise_op(20, 2, Extended_Opcode.EOP_BITSHR)
    assert result == 5  # 20 >> 2 = 5


def test_eop_bitshr_by_zero():
    """EOP_BITSHR by zero is identity."""
    assert run_bitwise_op(42, 0, Extended_Opcode.EOP_BITSHR) == 42


def test_eop_bitshr_by_one():
    """EOP_BITSHR by 1 is divide by 2 (floor)."""
    assert run_bitwise_op(10, 1, Extended_Opcode.EOP_BITSHR) == 5
    assert run_bitwise_op(11, 1, Extended_Opcode.EOP_BITSHR) == 5  # Rounds down


def test_eop_bitshr_to_zero():
    """EOP_BITSHR can reduce to zero."""
    assert run_bitwise_op(5, 10, Extended_Opcode.EOP_BITSHR) == 0


def test_eop_bitshr_negative():
    """EOP_BITSHR with negative number (arithmetic shift)."""
    # Python uses arithmetic right shift for negative numbers
    assert run_bitwise_op(-16, 2, Extended_Opcode.EOP_BITSHR) == -4


# ===== EOP_COMPLEMENT tests =====

def test_eop_complement_basic():
    """EOP_COMPLEMENT performs bitwise NOT."""
    result = run_unary_op(5, Extended_Opcode.EOP_COMPLEMENT)
    assert result == ~5  # -6 in two's complement


def test_eop_complement_zero():
    """EOP_COMPLEMENT of zero."""
    assert run_unary_op(0, Extended_Opcode.EOP_COMPLEMENT) == -1


def test_eop_complement_negative_one():
    """EOP_COMPLEMENT of -1."""
    assert run_unary_op(-1, Extended_Opcode.EOP_COMPLEMENT) == 0


def test_eop_complement_double():
    """EOP_COMPLEMENT applied twice is identity."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_COMPLEMENT.value),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_COMPLEMENT.value),
        ]))
        vm.step()  # Push 5
        vm.step()  # ~5 = -6
        vm.step()  # ~(-6) = 5
        assert vm.stack[-1] == 5


def test_eop_complement_pattern():
    """EOP_COMPLEMENT inverts all bits."""
    # In two's complement, ~n = -(n+1)
    assert run_unary_op(10, Extended_Opcode.EOP_COMPLEMENT) == -11
    assert run_unary_op(-10, Extended_Opcode.EOP_COMPLEMENT) == 9


# ===== Complex bitwise expressions =====

def test_bitwise_combination_or_and():
    """Multiple bitwise operations: (5 | 3) & 7."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_IMM, operand=3),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_BITOR.value),
            Instruction(opcode=Opcode.OP_IMM, operand=7),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_BITAND.value),
        ]))
        vm.step()  # Push 5
        vm.step()  # Push 3
        vm.step()  # 5 | 3 = 7
        vm.step()  # Push 7
        vm.step()  # 7 & 7 = 7
        assert vm.stack[-1] == 7


def test_bitwise_combination_xor_or():
    """Multiple bitwise operations: (10 ^ 6) | 1."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=10),
            Instruction(opcode=Opcode.OP_IMM, operand=6),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_BITXOR.value),
            Instruction(opcode=Opcode.OP_IMM, operand=1),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_BITOR.value),
        ]))
        vm.step()  # Push 10
        vm.step()  # Push 6
        vm.step()  # 10 ^ 6 = 12
        vm.step()  # Push 1
        vm.step()  # 12 | 1 = 13
        assert vm.stack[-1] == 13


def test_bitwise_masking_pattern():
    """Common bitwise masking patterns."""
    # Extract low nibble: 0xAB & 0x0F = 0x0B
    assert run_bitwise_op(0xAB, 0x0F, Extended_Opcode.EOP_BITAND) == 0x0B

    # Extract high nibble (shifted): (0xAB & 0xF0) >> 4 = 0x0A
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=0xAB),
            Instruction(opcode=Opcode.OP_IMM, operand=0xF0),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_BITAND.value),
            Instruction(opcode=Opcode.OP_IMM, operand=4),
            Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_BITSHR.value),
        ]))
        vm.step()  # Push 0xAB
        vm.step()  # Push 0xF0
        vm.step()  # 0xAB & 0xF0 = 0xA0
        vm.step()  # Push 4
        vm.step()  # 0xA0 >> 4 = 0x0A
        assert vm.stack[-1] == 0x0A


def test_power_of_two_check():
    """Check if number is power of 2 using bitwise: n & (n-1) == 0."""
    # 8 is power of 2: 8 & 7 = 0
    assert run_bitwise_op(8, 7, Extended_Opcode.EOP_BITAND) == 0

    # 6 is not power of 2: 6 & 5 = 4
    assert run_bitwise_op(6, 5, Extended_Opcode.EOP_BITAND) == 4
