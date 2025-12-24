"""Tests for MOO $ (dollar/last_index) expressions.

The $ symbol refers to the length of the current list being indexed.
Example: x[1..$] means x[1..length(x)]
"""
import pytest
from moo_interp.moo_ast import parse, compile, run


def test_dollar_in_range():
    """Dollar in range expression."""
    program = """
    x = {1, 2, 3, 4, 5};
    y = x[2..$];
    return y;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert list(vm.result) == [2, 3, 4, 5], "Should get elements 2 to end"


def test_dollar_in_function_call():
    """Dollar inside function call - the failing case from server_started."""
    program = """
    x = {1, 2, 3, 4, 5};
    y = x[1..min($, 3)];
    return y;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    # min(5, 3) = 3, so x[1..3] = {1, 2, 3}
    assert list(vm.result) == [1, 2, 3], "Should get elements 1 to min($, 3)"


def test_dollar_alone():
    """Dollar as array index."""
    program = """
    x = {1, 2, 3, 4, 5};
    y = x[$];
    return y;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 5, "Should get last element"
