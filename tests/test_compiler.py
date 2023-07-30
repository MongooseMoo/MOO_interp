from moo_interp.moo_ast import run, parse, compile, disassemble


# first, let's set up some simple programs and their expected results

# a program that adds two numbers
add_program = """
    return 1 + 2;
"""

# a program that adds two numbers, but with a bug
add_program_bug = """
    return 1 + 2    # missing semicolon
"""

# A simple test of unary operators

unary_program = """
    return -1;
"""

# conditional program
conditional_program = """
    if (1)
        return 1;
    else
        return 0;
    endif
"""

# while loop program

while_program = """

    i = 0;
    while (i < 10)
        i = i + 1;
    endwhile
    return i;
"""
# for loop program

for_program = """
res = 0;
    for i in ({1, 2, 3})
        res = res + i;
    endfor
    return res;
"""


# some helper functions


def run_program(program):
    """Run a program and return the result."""
    ast = parse(program)
    return run(compile(ast))


def expect_result(program, expected):
    """Run a program and check that the result is as expected."""
    result = run_program(program)
    assert result.result == expected, "Expected %s but got %s" % (expected, result)


def test_add():
    expect_result(add_program, 3)

def test_unary():
    expect_result(unary_program, -1)

def test_conditional():
    expect_result(conditional_program, 1)


def test_while():
    expect_result(while_program, 10)




def test_for():
    expect_result(for_program, 6)
