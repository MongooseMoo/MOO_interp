"""
Test string escape sequence handling in the parser.

The parser must properly unescape escape sequences like \" \n \t etc.
"""
import pytest
from moo_interp.moo_ast import parse, StringLiteral


def test_escaped_quote():
    """Parser should unescape \" to \" in string literals"""
    ast = parse('return "say \\"hello\\"";')
    # Navigate: VerbCode -> _SingleStatement -> ReturnStatement -> StringLiteral
    single_stmt = ast.children[0]
    return_stmt = single_stmt.statement
    string_literal = return_stmt.value

    assert isinstance(string_literal, StringLiteral)
    # The actual string value should have unescaped quotes
    assert string_literal.value == 'say "hello"', \
        f"Expected 'say \"hello\"' but got {repr(string_literal.value)}"


def test_escaped_newline():
    """Parser should unescape \\n to actual newline"""
    ast = parse('return "line1\\nline2";')
    string_literal = ast.children[0].statement.value

    assert isinstance(string_literal, StringLiteral)
    assert string_literal.value == 'line1\nline2', \
        f"Expected newline but got {repr(string_literal.value)}"


def test_escaped_tab():
    """Parser should unescape \\t to actual tab"""
    ast = parse('return "col1\\tcol2";')
    string_literal = ast.children[0].statement.value

    assert isinstance(string_literal, StringLiteral)
    assert string_literal.value == 'col1\tcol2', \
        f"Expected tab but got {repr(string_literal.value)}"


def test_escaped_backslash():
    """Parser should unescape \\\\ to single backslash"""
    ast = parse('return "path\\\\file";')
    string_literal = ast.children[0].statement.value

    assert isinstance(string_literal, StringLiteral)
    assert string_literal.value == 'path\\file', \
        f"Expected single backslash but got {repr(string_literal.value)}"


def test_multiple_escapes():
    """Parser should handle multiple escape sequences in one string"""
    ast = parse('return "Quote: \\"Hi!\\" Tab:\\tNewline:\\n";')
    string_literal = ast.children[0].statement.value

    assert isinstance(string_literal, StringLiteral)
    expected = 'Quote: "Hi!" Tab:\tNewline:\n'
    assert string_literal.value == expected, \
        f"Expected {repr(expected)} but got {repr(string_literal.value)}"
