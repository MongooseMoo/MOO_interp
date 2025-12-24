"""Tests for MOO inline try-catch expressions.

The syntax `expr ! ANY => default' evaluates expr and returns default if any error.
Also known as "scatter expression" or "catch expression".
"""
import pytest
from moo_interp.moo_ast import parse, compile, run


def test_try_expr_no_error():
    """Inline try expression returns value when no error."""
    program = """
    x = `1 + 1 ! ANY => 99';
    return x;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 2, "Should return expression value when no error"


def test_try_expr_catches_error():
    """Inline try expression returns default when error occurs."""
    program = """
    x = `1 / 0 ! ANY => 42';
    return x;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 42, "Should return default value when error occurs"


def test_try_expr_in_condition():
    """Inline try expression works in if condition."""
    # This is the real use case from incoming_connection:
    # if (index(`connection_name(what) ! ANY => ""', " to "))
    program = """
    x = `1 / 0 ! ANY => ""';
    if (x == "")
        return 1;
    endif
    return 0;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 1, "Should use default empty string and match condition"


def test_try_expr_specific_error():
    """Inline try expression with specific error code."""
    program = """
    x = `1 / 0 ! E_DIV => 99';
    return x;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 99, "Should catch E_DIV and return default"


def test_try_expr_wrong_error():
    """Inline try expression doesn't catch different error type."""
    program = """
    x = `1 / 0 ! E_TYPE => 99';
    return x;
    """
    ast = parse(program)
    frame = compile(ast)

    # E_TYPE won't catch E_DIV, so error should propagate
    with pytest.raises(Exception):
        run(frame)
