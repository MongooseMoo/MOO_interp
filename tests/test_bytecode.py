"""Tests for bytecode generation.

These tests verify that the compiler generates correct bytecode for various
MOO language constructs.
"""
from moo_interp.moo_ast import parse, compile
from typing import Any, List, Tuple, Union

from moo_interp.opcodes import Opcode


def get_bytecode(code: str) -> List[Tuple[Union[Opcode, int], Any]]:
    """Compile code and return list of (opcode, operand) tuples."""
    actual = compile(parse(code))
    return [(i.opcode, i.operand) for i in actual.stack]


def test_variable_assignment():
    """Test bytecode generation for variable assignments."""
    code = "a = 1; b = 2; c = 3;"
    bytecode = get_bytecode(code)

    # Small integers -10 to 192 use optimized opcodes (OPTIM_NUM_START + value - OPTIM_NUM_LOW)
    # So 1 = opcode 64, 2 = opcode 65, etc. (using optim_num_to_opcode)
    # Each assignment: push value, put var, pop (OP_PUT peeks, doesn't pop)
    assert bytecode[0] == (64, None)  # push 1
    assert bytecode[1] == (Opcode.OP_PUT, 'a')
    assert bytecode[2] == (Opcode.OP_POP, 1)  # clean up stack
    assert bytecode[3] == (65, None)  # push 2
    assert bytecode[4] == (Opcode.OP_PUT, 'b')
    assert bytecode[5] == (Opcode.OP_POP, 1)  # clean up stack
    assert bytecode[6] == (66, None)  # push 3
    assert bytecode[7] == (Opcode.OP_PUT, 'c')
    assert bytecode[8] == (Opcode.OP_POP, 1)  # clean up stack
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_arithmetic_expr():
    """Test bytecode generation for arithmetic expressions."""
    # Note: Parser evaluates left-to-right, use parentheses for precedence
    code = "a = 1 + (3 * 2);"
    bytecode = get_bytecode(code)

    # 1 + (3 * 2) = 1 + 6 = 7
    # Bytecode: push 1, push 3, push 2, mult, add, put a, pop
    assert bytecode[0] == (64, None)  # push 1
    assert bytecode[1] == (66, None)  # push 3
    assert bytecode[2] == (65, None)  # push 2
    assert bytecode[3] == (Opcode.OP_MULT, None)
    assert bytecode[4] == (Opcode.OP_ADD, None)
    assert bytecode[5] == (Opcode.OP_PUT, 'a')
    assert bytecode[6] == (Opcode.OP_POP, 1)  # clean up stack
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_if_condition():
    """Test bytecode generation for IF conditions."""
    # MOO uses lowercase keywords
    code = "if (a > b) c = 1; endif"
    bytecode = get_bytecode(code)

    # Should have: push a, push b, gt, if (jump offset), push 1, put c, pop, done
    assert bytecode[0] == (Opcode.OP_PUSH, 'a')
    assert bytecode[1] == (Opcode.OP_PUSH, 'b')
    assert bytecode[2] == (Opcode.OP_GT, None)
    assert bytecode[3][0] == Opcode.OP_IF  # jump offset varies
    assert bytecode[4] == (64, None)  # push 1
    assert bytecode[5] == (Opcode.OP_PUT, 'c')
    assert bytecode[6] == (Opcode.OP_POP, 1)  # clean up stack
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_for_loop():
    """Test bytecode generation for FOR range loop."""
    # MOO uses lowercase keywords and [start..end] syntax
    code = "for i in [1..10] a = a + i; endfor"
    bytecode = get_bytecode(code)

    # Should have: push 1, push 10, for_range, push a, push i, add, put a, pop, jump back, done
    assert bytecode[0] == (64, None)  # push 1
    assert bytecode[1] == (73, None)  # push 10
    assert bytecode[2][0] == Opcode.OP_FOR_RANGE
    assert bytecode[3] == (Opcode.OP_PUSH, 'a')
    assert bytecode[4] == (Opcode.OP_PUSH, 'i')
    assert bytecode[5] == (Opcode.OP_ADD, None)
    assert bytecode[6] == (Opcode.OP_PUT, 'a')
    assert bytecode[7] == (Opcode.OP_POP, 1)  # clean up stack
    assert bytecode[8][0] == Opcode.OP_JUMP  # backward jump
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_while_loop():
    """Test bytecode generation for WHILE loop."""
    # MOO uses lowercase keywords
    code = "while (a < b) a = a + 1; endwhile"
    bytecode = get_bytecode(code)

    # Should have: push a, push b, lt, while (jump offset), push a, push 1, add, put a, pop, jump back, done
    assert bytecode[0] == (Opcode.OP_PUSH, 'a')
    assert bytecode[1] == (Opcode.OP_PUSH, 'b')
    assert bytecode[2] == (Opcode.OP_LT, None)
    assert bytecode[3][0] == Opcode.OP_WHILE  # jump offset varies
    assert bytecode[4] == (Opcode.OP_PUSH, 'a')
    assert bytecode[5] == (64, None)  # push 1
    assert bytecode[6] == (Opcode.OP_ADD, None)
    assert bytecode[7] == (Opcode.OP_PUT, 'a')
    assert bytecode[8] == (Opcode.OP_POP, 1)  # clean up stack
    assert bytecode[9][0] == Opcode.OP_JUMP  # backward jump
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_small_integers_use_optimized_opcodes():
    """Small integers -10 to 192 use optimized opcodes."""
    code = "return 0;"
    bytecode = get_bytecode(code)
    assert bytecode[0] == (63, None)  # 0 = opcode 63

    code = "return 42;"
    bytecode = get_bytecode(code)
    assert bytecode[0] == (105, None)  # 42 = opcode 105

    code = "return 192;"  # 192 is OPTIM_NUM_HI (max optimized value)
    bytecode = get_bytecode(code)
    assert bytecode[0] == (255, None)  # 192 uses opcode 255 (the highest opcode)
