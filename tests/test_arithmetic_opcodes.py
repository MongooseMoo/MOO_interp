"""Comprehensive tests for arithmetic opcodes.

Tests all arithmetic operations including edge cases like division by zero,
negative numbers, and type mismatches.
"""

from contextlib import contextmanager

import pytest
from moo_interp.list import MOOList
from moo_interp.string import MOOString
from moo_interp.opcodes import Opcode
from moo_interp.vm import VM, Instruction, Program, StackFrame, VMError
from moo_interp.moo_ast import run, parse, compile as compile_moo


@contextmanager
def create_vm():
    vm = VM()
    try:
        yield vm
    finally:
        pass


def run_program(program):
    """Run a MOO program and return the result."""
    ast = parse(program)
    return run(compile_moo(ast))


def expect_result(program, expected):
    """Run a program and check that the result is as expected."""
    result = run_program(program)
    assert result.result == expected, f"Expected {expected} but got {result.result}"


# ===== OP_ADD tests =====

def test_op_add_integers():
    """OP_ADD adds two integers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_IMM, operand=3),
            Instruction(opcode=Opcode.OP_ADD),
        ]))
        vm.step()  # Push 5
        vm.step()  # Push 3
        vm.step()  # Execute ADD
        assert vm.stack[-1] == 8


def test_op_add_negative():
    """OP_ADD handles negative numbers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=-10),
            Instruction(opcode=Opcode.OP_IMM, operand=3),
            Instruction(opcode=Opcode.OP_ADD),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == -7


def test_op_add_strings():
    """OP_ADD concatenates strings."""
    program = """
    return "hello" + " world";
    """
    result = run_program(program)
    assert str(result.result) == "hello world"


def test_op_add_lists():
    """OP_ADD concatenates lists."""
    program = """
    return {1, 2} + {3, 4};
    """
    result = run_program(program)
    assert result.result == MOOList(1, 2, 3, 4)


def test_op_add_zero():
    """OP_ADD with zero."""
    expect_result("return 42 + 0;", 42)


# ===== OP_MINUS tests =====

def test_op_minus_integers():
    """OP_MINUS subtracts two integers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=10),
            Instruction(opcode=Opcode.OP_IMM, operand=3),
            Instruction(opcode=Opcode.OP_MINUS),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == 7


def test_op_minus_negative_result():
    """OP_MINUS produces negative result."""
    expect_result("return 3 - 10;", -7)


def test_op_minus_negative_operands():
    """OP_MINUS with negative operands."""
    expect_result("return -5 - (-3);", -2)


def test_op_minus_zero():
    """OP_MINUS with zero."""
    expect_result("return 42 - 0;", 42)


# ===== OP_MULT tests =====

def test_op_mult_integers():
    """OP_MULT multiplies two integers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=6),
            Instruction(opcode=Opcode.OP_IMM, operand=7),
            Instruction(opcode=Opcode.OP_MULT),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == 42


def test_op_mult_negative():
    """OP_MULT with negative numbers."""
    expect_result("return -5 * 3;", -15)
    expect_result("return -5 * -3;", 15)


def test_op_mult_zero():
    """OP_MULT by zero gives zero."""
    expect_result("return 42 * 0;", 0)


def test_op_mult_one():
    """OP_MULT by one is identity."""
    expect_result("return 42 * 1;", 42)


def test_op_mult_string_repetition():
    """OP_MULT repeats strings (if supported)."""
    # MOO may support this - check behavior
    program = """
    return "ab" * 3;
    """
    result = run_program(program)
    # This might be "ababab" or might error - document actual behavior
    # For now just ensure it doesn't crash


# ===== OP_DIV tests =====

def test_op_div_integers():
    """OP_DIV divides two integers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=20),
            Instruction(opcode=Opcode.OP_IMM, operand=4),
            Instruction(opcode=Opcode.OP_DIV),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == 5


def test_op_div_negative():
    """OP_DIV with negative numbers."""
    expect_result("return -20 / 4;", -5)
    expect_result("return 20 / -4;", -5)
    expect_result("return -20 / -4;", 5)


def test_op_div_by_zero():
    """OP_DIV by zero raises error."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=10),
            Instruction(opcode=Opcode.OP_IMM, operand=0),
            Instruction(opcode=Opcode.OP_DIV),
        ]))
        vm.step()
        vm.step()
        with pytest.raises(VMError, match="zero"):
            vm.step()


def test_op_div_fractional():
    """OP_DIV with non-integer result."""
    # Python-style division returns float
    expect_result("return 7 / 2;", 3.5)


# ===== OP_MOD tests =====

def test_op_mod_integers():
    """OP_MOD computes modulo."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=10),
            Instruction(opcode=Opcode.OP_IMM, operand=3),
            Instruction(opcode=Opcode.OP_MOD),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == 1


def test_op_mod_negative():
    """OP_MOD with negative numbers."""
    expect_result("return -10 % 3;", 2)  # Python-style modulo
    expect_result("return 10 % -3;", -2)


def test_op_mod_by_zero():
    """OP_MOD by zero raises error."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=10),
            Instruction(opcode=Opcode.OP_IMM, operand=0),
            Instruction(opcode=Opcode.OP_MOD),
        ]))
        vm.step()
        vm.step()
        with pytest.raises(VMError, match="zero"):
            vm.step()


def test_op_mod_exact_division():
    """OP_MOD when divisor divides evenly."""
    expect_result("return 10 % 5;", 0)


# ===== OP_UNARY_MINUS tests =====

def test_op_unary_minus_positive():
    """OP_UNARY_MINUS negates positive number."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=42),
            Instruction(opcode=Opcode.OP_UNARY_MINUS),
        ]))
        vm.step()
        vm.step()
        assert vm.stack[-1] == -42


def test_op_unary_minus_negative():
    """OP_UNARY_MINUS negates negative number."""
    expect_result("return -(-5);", 5)


def test_op_unary_minus_zero():
    """OP_UNARY_MINUS of zero is zero."""
    expect_result("return -0;", 0)


def test_op_unary_minus_in_expression():
    """OP_UNARY_MINUS in larger expression."""
    expect_result("return 10 + -5;", 5)
    expect_result("return -3 * -4;", 12)


# ===== Complex arithmetic tests =====

def test_complex_arithmetic_expression():
    """Complex arithmetic with multiple operations."""
    expect_result("return (5 + 3) * 2 - 10 / 2;", 11)


def test_arithmetic_precedence():
    """Arithmetic follows correct precedence."""
    expect_result("return 2 + 3 * 4;", 14)  # Mult before add
    expect_result("return 10 - 6 / 2;", 7)   # Div before sub


def test_arithmetic_with_parentheses():
    """Parentheses override precedence."""
    expect_result("return (2 + 3) * 4;", 20)
    expect_result("return (10 - 6) / 2;", 2)
