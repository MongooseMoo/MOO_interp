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

    # Small integers 0-255 use optimized opcodes (113 + value)
    # So 1 = opcode 114, 2 = opcode 115, 3 = opcode 116
    assert bytecode[0] == (114, None)  # push 1
    assert bytecode[1] == (Opcode.OP_PUT, 'a')
    assert bytecode[2] == (115, None)  # push 2
    assert bytecode[3] == (Opcode.OP_PUT, 'b')
    assert bytecode[4] == (116, None)  # push 3
    assert bytecode[5] == (Opcode.OP_PUT, 'c')
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_arithmetic_expr():
    """Test bytecode generation for arithmetic expressions."""
    # Note: Parser evaluates left-to-right, use parentheses for precedence
    code = "a = 1 + (3 * 2);"
    bytecode = get_bytecode(code)

    # 1 + (3 * 2) = 1 + 6 = 7
    # Bytecode: push 1, push 3, push 2, mult, add, put a
    assert bytecode[0] == (114, None)  # push 1
    assert bytecode[1] == (116, None)  # push 3
    assert bytecode[2] == (115, None)  # push 2
    assert bytecode[3] == (Opcode.OP_MULT, None)
    assert bytecode[4] == (Opcode.OP_ADD, None)
    assert bytecode[5] == (Opcode.OP_PUT, 'a')
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_if_condition():
    """Test bytecode generation for IF conditions."""
    # MOO uses lowercase keywords
    code = "if (a > b) c = 1; endif"
    bytecode = get_bytecode(code)

    # Should have: push a, push b, gt, if (jump offset), push 1, put c, done
    assert bytecode[0] == (Opcode.OP_PUSH, 'a')
    assert bytecode[1] == (Opcode.OP_PUSH, 'b')
    assert bytecode[2] == (Opcode.OP_GT, None)
    assert bytecode[3][0] == Opcode.OP_IF  # jump offset varies
    assert bytecode[4] == (114, None)  # push 1
    assert bytecode[5] == (Opcode.OP_PUT, 'c')
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_for_loop():
    """Test bytecode generation for FOR range loop."""
    # MOO uses lowercase keywords and [start..end] syntax
    code = "for i in [1..10] a = a + i; endfor"
    bytecode = get_bytecode(code)

    # Should have: push 1, push 10, for_range, push a, push i, add, put a, jump back, done
    assert bytecode[0] == (114, None)  # push 1
    assert bytecode[1] == (123, None)  # push 10 (113 + 10)
    assert bytecode[2][0] == Opcode.OP_FOR_RANGE
    assert bytecode[3] == (Opcode.OP_PUSH, 'a')
    assert bytecode[4] == (Opcode.OP_PUSH, 'i')
    assert bytecode[5] == (Opcode.OP_ADD, None)
    assert bytecode[6] == (Opcode.OP_PUT, 'a')
    assert bytecode[7][0] == Opcode.OP_JUMP  # backward jump
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_while_loop():
    """Test bytecode generation for WHILE loop."""
    # MOO uses lowercase keywords
    code = "while (a < b) a = a + 1; endwhile"
    bytecode = get_bytecode(code)

    # Should have: push a, push b, lt, while (jump offset), push a, push 1, add, put a, jump back, done
    assert bytecode[0] == (Opcode.OP_PUSH, 'a')
    assert bytecode[1] == (Opcode.OP_PUSH, 'b')
    assert bytecode[2] == (Opcode.OP_LT, None)
    assert bytecode[3][0] == Opcode.OP_WHILE  # jump offset varies
    assert bytecode[4] == (Opcode.OP_PUSH, 'a')
    assert bytecode[5] == (114, None)  # push 1
    assert bytecode[6] == (Opcode.OP_ADD, None)
    assert bytecode[7] == (Opcode.OP_PUT, 'a')
    assert bytecode[8][0] == Opcode.OP_JUMP  # backward jump
    assert bytecode[-1] == (Opcode.OP_DONE, None)


def test_small_integers_use_optimized_opcodes():
    """Small integers 0-255 use optimized opcodes (113 + value)."""
    code = "return 0;"
    bytecode = get_bytecode(code)
    assert bytecode[0] == (113, None)  # 0 = opcode 113

    code = "return 42;"
    bytecode = get_bytecode(code)
    assert bytecode[0] == (155, None)  # 42 = opcode 113 + 42 = 155

    code = "return 255;"
    bytecode = get_bytecode(code)
    assert bytecode[0] == (368, None)  # 255 = opcode 113 + 255 = 368
