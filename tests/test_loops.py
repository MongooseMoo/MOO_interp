"""Tests for loop opcodes (for list, for range, while, continue, break)

TDD approach: These tests are written first and should fail until loop opcodes are implemented.
"""

from moo_interp.moo_ast import run, parse, compile


def run_program(program):
    """Run a program and return the result."""
    ast = parse(program)
    return run(compile(ast))


def expect_result(program, expected):
    """Run a program and check that the result is as expected."""
    result = run_program(program)
    assert result.result == expected, f"Expected {expected} but got {result.result}"


# For list tests


def test_for_list_simple():
    """for x in ({1,2,3}) accumulates values."""
    program = """
    res = 0;
    for x in ({1, 2, 3})
        res = res + x;
    endfor
    return res;
    """
    expect_result(program, 6)


def test_for_list_empty():
    """for loop over empty list doesn't execute body."""
    program = """
    res = 0;
    for x in ({})
        res = res + 1;
    endfor
    return res;
    """
    expect_result(program, 0)


def test_for_list_nested_access():
    """for loop can access loop variable."""
    program = """
    res = {};
    for x in ({1, 2, 3})
        res = {@res, x};
    endfor
    return res;
    """
    result = run_program(program)
    from moo_interp.list import MOOList
    assert result.result == MOOList(1, 2, 3)


# For range tests


def test_for_range_simple():
    """for i in [1..5] iterates 1 through 5."""
    program = """
    res = 0;
    for i in [1..5]
        res = res + i;
    endfor
    return res;
    """
    expect_result(program, 15)  # 1+2+3+4+5 = 15


def test_for_range_negative():
    """for i in [5..1] is empty range (MOO doesn't support backwards ranges)."""
    program = """
    res = 0;
    for i in [5..1]
        res = res + i;
    endfor
    return res;
    """
    expect_result(program, 0)  # Empty range, body never executes


def test_for_range_single():
    """for i in [3..3] iterates once."""
    program = """
    res = 0;
    for i in [3..3]
        res = res + 1;
    endfor
    return res;
    """
    expect_result(program, 1)


# Nested loops


def test_nested_loops():
    """Nested loops work correctly."""
    program = """
    res = 0;
    for i in [1..3]
        for j in [1..2]
            res = res + 1;
        endfor
    endfor
    return res;
    """
    expect_result(program, 6)  # 3 * 2 = 6


def test_nested_loops_with_list():
    """Nested loops with list and range."""
    program = """
    res = 0;
    for x in ({10, 20})
        for i in [1..2]
            res = res + x + i;
        endfor
    endfor
    return res;
    """
    expect_result(program, 66)  # (10+1)+(10+2)+(20+1)+(20+2) = 11+12+21+22 = 66


# Continue tests


def test_continue_in_for_list():
    """continue skips to next iteration in for-list loop."""
    program = """
    res = 0;
    for x in ({1, 2, 3, 4, 5})
        if (x == 3)
            continue;
        endif
        res = res + x;
    endfor
    return res;
    """
    expect_result(program, 12)  # 1+2+4+5 = 12 (skips 3)


def test_continue_in_for_range():
    """continue skips to next iteration in for-range loop."""
    program = """
    res = 0;
    for i in [1..5]
        if (i == 3)
            continue;
        endif
        res = res + i;
    endfor
    return res;
    """
    expect_result(program, 12)  # 1+2+4+5 = 12 (skips 3)


def test_continue_in_while():
    """continue skips to next iteration in while loop."""
    program = """
    res = 0;
    i = 0;
    while (i < 5)
        i = i + 1;
        if (i == 3)
            continue;
        endif
        res = res + i;
    endwhile
    return res;
    """
    expect_result(program, 12)  # 1+2+4+5 = 12 (skips 3)


# Break/exit tests


def test_break_in_for_list():
    """break exits the loop in for-list loop."""
    program = """
    res = 0;
    for x in ({1, 2, 3, 4, 5})
        if (x == 3)
            break;
        endif
        res = res + x;
    endfor
    return res;
    """
    expect_result(program, 3)  # 1+2 = 3 (exits before 3)


def test_break_in_for_range():
    """break exits the loop in for-range loop."""
    program = """
    res = 0;
    for i in [1..10]
        if (i == 4)
            break;
        endif
        res = res + i;
    endfor
    return res;
    """
    expect_result(program, 6)  # 1+2+3 = 6 (exits before 4)


def test_break_in_while():
    """break exits the loop in while loop."""
    program = """
    res = 0;
    i = 0;
    while (i < 10)
        i = i + 1;
        if (i == 4)
            break;
        endif
        res = res + i;
    endwhile
    return res;
    """
    expect_result(program, 6)  # 1+2+3 = 6 (exits when i becomes 4)


def test_break_nested_inner():
    """break in nested loop only exits inner loop."""
    program = """
    res = 0;
    for i in [1..3]
        for j in [1..3]
            if (j == 2)
                break;
            endif
            res = res + 1;
        endfor
    endfor
    return res;
    """
    expect_result(program, 3)  # Each outer iteration: 1 inner iteration = 3 total


# While loop tests (some already exist in test_compiler.py)


def test_while_basic():
    """Basic while loop."""
    program = """
    i = 0;
    while (i < 5)
        i = i + 1;
    endwhile
    return i;
    """
    expect_result(program, 5)


def test_while_with_accumulation():
    """While loop with accumulation."""
    program = """
    i = 0;
    res = 0;
    while (i < 5)
        i = i + 1;
        res = res + i;
    endwhile
    return res;
    """
    expect_result(program, 15)  # 1+2+3+4+5 = 15


def test_while_false_never_executes():
    """While loop with false condition never executes."""
    program = """
    res = 0;
    while (0)
        res = res + 1;
    endwhile
    return res;
    """
    expect_result(program, 0)
