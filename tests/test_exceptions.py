"""Tests for MOO try/except/finally exception handling opcodes."""
import pytest
from moo_interp.moo_ast import parse, compile, run
from moo_interp.errors import MOOError
from moo_interp.vm import VMError

# Note: Requires exception handling opcodes in VM


def test_try_except_catches_error():
    """try/except catches raised error."""
    # Division by zero should raise E_DIV, but try/except catches it
    program = """
    try
        x = 1 / 0;
        return 99;
    except (E_DIV)
        return 0;
    endtry
    """

    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 0, "except block should catch E_DIV and return 0"


def test_try_except_no_error():
    """try block executes normally when no error is raised."""
    program = """
    try
        x = 1 + 1;
        return x;
    except (E_DIV)
        return 0;
    endtry
    """

    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 2, "try block should execute normally and return 2"


@pytest.mark.skip(reason="Indexing not implemented yet")
def test_try_except_specific_error():
    """except (E_DIV) only catches division errors."""
    # Type error should NOT be caught by E_DIV handler
    program = """
    try
        x = 1[2];
        return 99;
    except (E_DIV)
        return 0;
    endtry
    """

    # This should raise an uncaught error since we're catching E_DIV but raising E_TYPE
    ast = parse(program)
    frame = compile(ast)

    with pytest.raises((VMError, Exception)):
        run(frame)


def test_try_except_any():
    """except (ANY) catches all errors."""
    program = """
    try
        x = 1 / 0;
        return 99;
    except (ANY)
        return 42;
    endtry
    """

    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 42, "except (ANY) should catch all errors"


def test_try_except_with_error_binding():
    """except e (E_DIV) binds the error to variable e."""
    program = """
    try
        x = 1 / 0;
        return 99;
    except e (E_DIV)
        return e;
    endtry
    """

    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    # Should return the error as a 4-element list: {error_code, message, value, traceback}
    # Per MOO spec, except e binds e to {E_DIV, "message", value, traceback}
    # MOOList is 1-indexed (MOO convention)
    from moo_interp.errors import MOOError
    from moo_interp.list import MOOList
    assert isinstance(vm.result, MOOList), "except should bind error list to variable e"
    assert len(vm.result) == 4, "error list should have 4 elements"
    assert vm.result[1] == MOOError.E_DIV, "first element should be E_DIV"


def test_try_finally_runs_always():
    """finally block runs even after error."""
    program = """
    x = 0;
    try
        x = 1;
        y = 2 / 0;
    finally
        x = x + 10;
    endtry
    return x;
    """

    # Even though division raises error, finally should run and set x = 11
    # Then the error should propagate
    ast = parse(program)
    frame = compile(ast)

    # The error will still be raised, but finally should have run
    with pytest.raises((VMError, Exception)):
        run(frame)

    # TODO: check that x was set to 11 before error propagated


def test_try_finally_no_error():
    """finally block runs when no error occurs."""
    program = """
    x = 0;
    try
        x = 1;
    finally
        x = x + 10;
    endtry
    return x;
    """

    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 11, "finally block should run and modify x"


def test_try_except_finally():
    """try/except inside try/finally works correctly."""
    # MOO doesn't support try/except/finally in same statement,
    # but you can nest try/except inside try/finally
    program = """
    x = 0;
    try
        try
            y = 1 / 0;
        except (E_DIV)
            x = 5;
        endtry
    finally
        x = x + 10;
    endtry
    return x;
    """
    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    # except sets x = 5, finally adds 10, so x = 15
    assert vm.result == 15, "except and finally should both run"


def test_nested_try():
    """Nested try/except works correctly."""
    program = """
    try
        try
            x = 1 / 0;
            return 1;
        except (E_TYPE)
            return 2;
        endtry
    except (E_DIV)
        return 3;
    endtry
    """

    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    # Inner except doesn't catch E_DIV, outer except does
    assert vm.result == 3, "outer except should catch error from inner try"


def test_try_except_multiple_errors():
    """except (E_DIV, E_TYPE) catches multiple error types."""
    program = """
    try
        x = 1 / 0;
        return 99;
    except (E_DIV, E_TYPE)
        return 7;
    endtry
    """

    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 7, "except should catch E_DIV from list"


def test_try_except_specific_error():
    """except (E_DIV) only catches division errors."""
    # Type error should NOT be caught by E_DIV handler
    program = """
    try
        x = 1[2];
        return 99;
    except (E_DIV)
        return 0;
    endtry
    """

    # This should raise an uncaught error since we're catching E_DIV but raising E_TYPE
    ast = parse(program)
    frame = compile(ast)

    with pytest.raises((VMError, Exception)):
        run(frame)


def test_try_except_multiple_errors_second():
    """except (E_DIV, E_TYPE) catches E_TYPE too."""
    program = """
    try
        x = 1[2];
        return 99;
    except (E_DIV, E_TYPE)
        return 8;
    endtry
    """

    ast = parse(program)
    frame = compile(ast)
    vm = run(frame)

    assert vm.result == 8, "except should catch E_TYPE from list"
