"""Tests for MOO splicer (@) expressions.

The syntax @expr splices a list into another list.
Example: {1, @{2, 3}, 4} => {1, 2, 3, 4}
"""
import pytest
from moo_interp.moo_ast import parse, compile, run


def test_splicer_basic():
    """Splicer splices list into list."""
    program = """
    x = {2, 3};
    y = {1, @x, 4};
    return y;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert list(vm.result) == [1, 2, 3, 4], "Should splice list into list"


def test_splicer_empty():
    """Splicer with empty list."""
    program = """
    x = {};
    y = {1, @x, 2};
    return y;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert list(vm.result) == [1, 2], "Should splice empty list"


def test_splicer_inline():
    """Splicer with inline list expression."""
    program = """
    y = {1, @{2, 3}, 4};
    return y;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert list(vm.result) == [1, 2, 3, 4], "Should splice inline list"


def test_splicer_with_range():
    """Splicer with range expression - from server_started verb."""
    program = """
    x = {1, 2, 3, 4, 5};
    y = {{0}, @x[2..4]};
    return y;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    # Expected: {{0}, 2, 3, 4} - nested list then spliced elements
    assert len(vm.result) == 4, "Should have 4 elements"
