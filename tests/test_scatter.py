"""Tests for MOO scatter (destructuring) assignments.

Scatter syntax:
{a, b, c} = list;           # Required variables
{a, ?b, c} = list;          # Optional variable (no default)
{a, ?b = 0, c} = list;      # Optional with default
{a, @rest} = list;          # Rest variable (captures remaining)
"""
import pytest
from moo_interp.moo_ast import parse, compile, run


def test_scatter_basic():
    """Basic scatter - required variables only."""
    program = """
    x = {1, 2, 3};
    {a, b, c} = x;
    return {a, b, c};
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert list(vm.result) == [1, 2, 3], "Should unpack all elements"


def test_scatter_optional():
    """Scatter with optional variable."""
    program = """
    x = {1, 2};
    {a, ?b, ?c} = x;
    return {a, b, c};
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    # Optional vars get 0 by default when not provided
    assert list(vm.result) == [1, 2, 0], "Optional should default to 0"


def test_scatter_optional_with_default():
    """Scatter with optional variable and explicit default."""
    program = """
    x = {1, 2};
    {a, ?b, ?c = 99} = x;
    return {a, b, c};
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert list(vm.result) == [1, 2, 99], "Should use explicit default"


def test_scatter_rest():
    """Scatter with rest variable."""
    program = """
    x = {1, 2, 3, 4, 5};
    {a, @rest} = x;
    return {a, rest};
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    # a=1, rest={2, 3, 4, 5}
    # MOOList is 1-indexed
    assert vm.result[1] == 1
    assert list(vm.result[2]) == [2, 3, 4, 5]


def test_scatter_from_args():
    """Scatter from args - the common use case."""
    program = """
    args = {"foo", "bar", 42};
    {target, thelist, ?indx = 1} = args;
    return {target, thelist, indx};
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert list(vm.result) == ["foo", "bar", 42]
