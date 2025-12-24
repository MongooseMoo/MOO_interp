"""Comprehensive tests for comparison and logical opcodes.

Tests all comparison operations (EQ, NE, LT, LE, GT, GE, IN) and
logical operations (AND, OR, NOT) with various data types and edge cases.
"""

from contextlib import contextmanager

import pytest
from moo_interp.list import MOOList
from moo_interp.string import MOOString
from moo_interp.opcodes import Opcode
from moo_interp.vm import VM, Instruction, Program, StackFrame
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


# ===== OP_EQ tests =====

def test_op_eq_integers_equal():
    """OP_EQ returns true for equal integers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_EQ),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == True


def test_op_eq_integers_not_equal():
    """OP_EQ returns false for different integers."""
    expect_result("return 5 == 3;", False)


def test_op_eq_strings():
    """OP_EQ compares strings."""
    expect_result('return "hello" == "hello";', True)
    expect_result('return "hello" == "world";', False)


def test_op_eq_lists():
    """OP_EQ compares lists."""
    expect_result("return {1, 2, 3} == {1, 2, 3};", True)
    expect_result("return {1, 2} == {1, 2, 3};", False)


def test_op_eq_empty_collections():
    """OP_EQ handles empty collections."""
    expect_result("return {} == {};", True)
    expect_result('return "" == "";', True)


def test_op_eq_mixed_types():
    """OP_EQ compares different types."""
    expect_result("return 0 == 0;", True)  # Same type
    # Type mismatches should return False
    expect_result('return 0 == "";', False)


# ===== OP_NE tests =====

def test_op_ne_integers():
    """OP_NE returns true for different integers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_IMM, operand=3),
            Instruction(opcode=Opcode.OP_NE),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == True


def test_op_ne_equal_values():
    """OP_NE returns false for equal values."""
    expect_result("return 5 != 5;", False)
    expect_result('return "hello" != "hello";', False)


def test_op_ne_lists():
    """OP_NE compares lists."""
    expect_result("return {1, 2} != {1, 3};", True)
    expect_result("return {1, 2} != {1, 2};", False)


# ===== OP_LT tests =====

def test_op_lt_integers():
    """OP_LT compares integers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=3),
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_LT),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == True


def test_op_lt_false():
    """OP_LT returns false when not less than."""
    expect_result("return 5 < 3;", False)
    expect_result("return 5 < 5;", False)


def test_op_lt_negative():
    """OP_LT handles negative numbers."""
    expect_result("return -5 < 3;", True)
    expect_result("return -10 < -5;", True)


def test_op_lt_strings():
    """OP_LT compares strings lexicographically."""
    expect_result('return "abc" < "def";', True)
    expect_result('return "xyz" < "abc";', False)


# ===== OP_LE tests =====

def test_op_le_less_than():
    """OP_LE returns true when less than."""
    expect_result("return 3 <= 5;", True)


def test_op_le_equal():
    """OP_LE returns true when equal."""
    expect_result("return 5 <= 5;", True)


def test_op_le_greater_than():
    """OP_LE returns false when greater than."""
    expect_result("return 7 <= 5;", False)


def test_op_le_strings():
    """OP_LE compares strings."""
    expect_result('return "abc" <= "abc";', True)
    expect_result('return "abc" <= "def";', True)
    expect_result('return "def" <= "abc";', False)


# ===== OP_GT tests =====

def test_op_gt_integers():
    """OP_GT compares integers."""
    with create_vm() as vm:
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=5),
            Instruction(opcode=Opcode.OP_IMM, operand=3),
            Instruction(opcode=Opcode.OP_GT),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == True


def test_op_gt_false():
    """OP_GT returns false when not greater than."""
    expect_result("return 3 > 5;", False)
    expect_result("return 5 > 5;", False)


def test_op_gt_negative():
    """OP_GT handles negative numbers."""
    expect_result("return 3 > -5;", True)
    expect_result("return -5 > -10;", True)


# ===== OP_GE tests =====

def test_op_ge_greater_than():
    """OP_GE returns true when greater than."""
    expect_result("return 5 >= 3;", True)


def test_op_ge_equal():
    """OP_GE returns true when equal."""
    expect_result("return 5 >= 5;", True)


def test_op_ge_less_than():
    """OP_GE returns false when less than."""
    expect_result("return 3 >= 5;", False)


# ===== OP_IN tests =====

def test_op_in_list_found():
    """OP_IN returns 1-based index when item in list."""
    with create_vm() as vm:
        test_list = MOOList(10, 20, 30)
        vm.call_stack.append(StackFrame(0, Program(), ip=0, stack=[
            Instruction(opcode=Opcode.OP_IMM, operand=20),
            Instruction(opcode=Opcode.OP_IMM, operand=test_list),
            Instruction(opcode=Opcode.OP_IN),
        ]))
        vm.step()
        vm.step()
        vm.step()
        assert vm.stack[-1] == 2  # 1-based index


def test_op_in_list_not_found():
    """OP_IN returns 0 when item not in list."""
    program = """
    return 99 in {1, 2, 3};
    """
    result = run_program(program)
    assert result.result == 0


def test_op_in_list_first():
    """OP_IN finds first element."""
    program = """
    return 1 in {1, 2, 3};
    """
    result = run_program(program)
    assert result.result == 1


def test_op_in_list_last():
    """OP_IN finds last element."""
    program = """
    return 3 in {1, 2, 3};
    """
    result = run_program(program)
    assert result.result == 3


def test_op_in_string_found():
    """OP_IN returns 1-based position when substring in string."""
    program = """
    return "world" in "hello world";
    """
    result = run_program(program)
    assert result.result == 7  # "world" starts at position 7 (1-based)


def test_op_in_string_not_found():
    """OP_IN returns 0 when substring not in string."""
    program = """
    return "xyz" in "hello world";
    """
    result = run_program(program)
    assert result.result == 0


def test_op_in_empty_list():
    """OP_IN returns 0 for empty list."""
    program = """
    return 1 in {};
    """
    result = run_program(program)
    assert result.result == 0


def test_op_in_empty_string():
    """OP_IN handles empty string."""
    program = """
    return "x" in "";
    """
    result = run_program(program)
    assert result.result == 0


# ===== OP_AND tests =====

def test_op_and_both_true():
    """OP_AND returns second operand when both operands are truthy (Python-style)."""
    expect_result("return 1 && 1;", 1)
    expect_result("return 5 && 3;", 3)


def test_op_and_first_false():
    """OP_AND returns false when first operand is false."""
    expect_result("return 0 && 1;", False)


def test_op_and_second_false():
    """OP_AND returns false when second operand is false."""
    expect_result("return 1 && 0;", False)


def test_op_and_both_false():
    """OP_AND returns false when both operands are false."""
    expect_result("return 0 && 0;", False)


def test_op_and_with_expressions():
    """OP_AND works with comparison expressions."""
    expect_result("return (5 > 3) && (2 < 4);", True)
    expect_result("return (5 < 3) && (2 < 4);", False)


# ===== OP_OR tests =====

def test_op_or_both_true():
    """OP_OR returns true when both operands are true."""
    expect_result("return 1 || 1;", True)


def test_op_or_first_true():
    """OP_OR returns true when first operand is true."""
    expect_result("return 1 || 0;", True)


def test_op_or_second_true():
    """OP_OR returns true when second operand is true."""
    expect_result("return 0 || 1;", True)


def test_op_or_both_false():
    """OP_OR returns false when both operands are false."""
    expect_result("return 0 || 0;", False)


def test_op_or_with_expressions():
    """OP_OR works with comparison expressions."""
    expect_result("return (5 > 3) || (2 > 4);", True)
    expect_result("return (5 < 3) || (2 > 4);", False)


# ===== OP_NOT tests =====

def test_op_not_true():
    """OP_NOT negates true to false."""
    expect_result("return !1;", False)


def test_op_not_false():
    """OP_NOT negates false to true."""
    expect_result("return !0;", True)


def test_op_not_truthy():
    """OP_NOT negates truthy values."""
    expect_result("return !5;", False)


def test_op_not_double():
    """OP_NOT applied twice is identity."""
    expect_result("return !!1;", True)
    expect_result("return !!0;", False)


def test_op_not_with_comparison():
    """OP_NOT works with comparison results."""
    expect_result("return !(5 > 3);", False)
    expect_result("return !(5 < 3);", True)


# ===== Complex logical expressions =====

def test_complex_logical_and_or():
    """Complex expression with AND and OR."""
    expect_result("return (1 && 1) || 0;", True)
    expect_result("return 0 && (1 || 1);", False)


def test_logical_precedence():
    """Logical operators have correct precedence."""
    # AND has higher precedence than OR
    expect_result("return 0 || 1 && 0;", False)  # 0 || (1 && 0) = 0 || 0 = 0
    expect_result("return 1 && 0 || 1;", True)   # (1 && 0) || 1 = 0 || 1 = 1


def test_comparison_with_logical():
    """Comparison and logical operators together."""
    expect_result("return 5 > 3 && 2 < 4;", True)
    expect_result("return 5 < 3 || 2 < 4;", True)
    expect_result("return 5 > 3 && !(2 > 4);", True)


def test_chained_comparisons():
    """Chained comparison expressions."""
    expect_result("return 1 < 2 && 2 < 3;", True)
    expect_result("return 5 == 5 && 5 <= 10;", True)
