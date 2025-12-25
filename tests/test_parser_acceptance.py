"""
Parser Acceptance Tests

Tests that valid MOO code parses without errors. These tests define the
language constructs we need to support, based on the ToastStunt parser.y.

Tests are organized by language feature. Many will initially fail - that's
expected. We fix them incrementally.
"""
import pytest
from moo_interp.moo_ast import parse


# =============================================================================
# LITERALS
# =============================================================================

LITERAL_TESTS = [
    # Numbers
    ("return 0;", "zero"),
    ("return 42;", "positive integer"),
    ("return -1;", "negative integer"),

    # Floats
    ("return 3.14;", "float"),
    ("return -2.5;", "negative float"),
    ("return 1.0e10;", "scientific notation"),
    ("return 1e-5;", "scientific negative exponent"),

    # Strings
    ('return "hello";', "simple string"),
    ('return "";', "empty string"),
    ('return "hello\\nworld";', "string with escape"),
    ('return "say \\"hello\\"";', "string with escaped quotes"),

    # Booleans
    ("return true;", "true literal"),
    ("return false;", "false literal"),

    # Object references
    ("return #0;", "object zero"),
    ("return #123;", "positive object"),
    ("return #-1;", "negative object (invalid)"),

    # Lists
    ("return {};", "empty list"),
    ("return {1, 2, 3};", "simple list"),
    ("return {1, {2, 3}, 4};", "nested list"),

    # Maps
    ("return [];", "empty map"),
    ('return ["a" -> 1, "b" -> 2];', "simple map"),
    ('return ["nested" -> ["inner" -> 1]];', "nested map"),
]

@pytest.mark.parametrize("code,desc", LITERAL_TESTS)
def test_literal(code, desc):
    """Test literal parsing: {desc}"""
    ast = parse(code)
    assert ast is not None


# =============================================================================
# OPERATORS
# =============================================================================

ARITHMETIC_TESTS = [
    ("return 1 + 2;", "addition"),
    ("return 3 - 1;", "subtraction"),
    ("return 2 * 3;", "multiplication"),
    ("return 6 / 2;", "division"),
    ("return 7 % 3;", "modulo"),
    ("return 2 ^ 3;", "exponentiation"),
]

@pytest.mark.parametrize("code,desc", ARITHMETIC_TESTS)
def test_arithmetic(code, desc):
    ast = parse(code)
    assert ast is not None


COMPARISON_TESTS = [
    ("return 1 == 1;", "equality"),
    ("return 1 != 2;", "inequality"),
    ("return 1 < 2;", "less than"),
    ("return 2 > 1;", "greater than"),
    ("return 1 <= 1;", "less or equal"),
    ("return 1 >= 1;", "greater or equal"),
    ("return 1 in {1, 2};", "in operator"),
]

@pytest.mark.parametrize("code,desc", COMPARISON_TESTS)
def test_comparison(code, desc):
    ast = parse(code)
    assert ast is not None


LOGICAL_TESTS = [
    ("return 1 && 1;", "logical and"),
    ("return 0 || 1;", "logical or"),
    ("return !0;", "logical not"),
]

@pytest.mark.parametrize("code,desc", LOGICAL_TESTS)
def test_logical(code, desc):
    ast = parse(code)
    assert ast is not None


BITWISE_TESTS = [
    ("return 5 &. 3;", "bitwise and"),
    ("return 5 |. 3;", "bitwise or"),
    ("return 5 ^. 3;", "bitwise xor"),
    ("return 8 >> 2;", "right shift"),
    ("return 2 << 2;", "left shift"),
    ("return ~0;", "bitwise complement"),
]

@pytest.mark.parametrize("code,desc", BITWISE_TESTS)
def test_bitwise(code, desc):
    ast = parse(code)
    assert ast is not None


UNARY_TESTS = [
    ("return -1;", "unary minus"),
    ("return !0;", "logical not"),
    ("return ~0;", "bitwise complement"),
]

@pytest.mark.parametrize("code,desc", UNARY_TESTS)
def test_unary(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# OPERATOR PRECEDENCE
# =============================================================================

PRECEDENCE_TESTS = [
    # Multiplication before addition
    ("return 1 + 2 * 3;", "mult before add"),

    # Power is right-associative
    ("return 2 ^ 3 ^ 2;", "power right assoc"),

    # Comparison before logical
    ("return 1 < 2 && 3 < 4;", "comparison before logical"),

    # Logical AND before OR
    ("return 1 || 0 && 0;", "and before or"),

    # Unary before binary
    ("return -1 + 2;", "unary before binary"),

    # Assignment is right-associative
    ("a = b = 1;", "assignment right assoc"),
]

@pytest.mark.parametrize("code,desc", PRECEDENCE_TESTS)
def test_precedence(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# CONTROL FLOW
# =============================================================================

IF_TESTS = [
    ("if (1) return 1; endif", "simple if"),
    ("if (1) return 1; else return 0; endif", "if-else"),
    ("if (1) return 1; elseif (0) return 0; endif", "if-elseif"),
    ("if (1) return 1; elseif (0) return 0; else return -1; endif", "if-elseif-else"),
    ("if (1) if (1) return 1; endif endif", "nested if"),
]

@pytest.mark.parametrize("code,desc", IF_TESTS)
def test_if(code, desc):
    ast = parse(code)
    assert ast is not None


FOR_TESTS = [
    ("for i in ({1, 2, 3}) return i; endfor", "for in list"),
    ("for i in (x) return i; endfor", "for in variable"),
    ("for i, j in ({1, 2, 3}) return {i, j}; endfor", "for with index"),
    ("for i in [1..10] return i; endfor", "for in range"),
    ("for i in [1..$] return i; endfor", "for in range with $"),
]

@pytest.mark.parametrize("code,desc", FOR_TESTS)
def test_for(code, desc):
    ast = parse(code)
    assert ast is not None


WHILE_TESTS = [
    ("while (1) return 1; endwhile", "simple while"),
    ("while (x < 10) x = x + 1; endwhile", "while with condition"),
    ("while myloop (1) break myloop; endwhile", "named while"),
]

@pytest.mark.parametrize("code,desc", WHILE_TESTS)
def test_while(code, desc):
    ast = parse(code)
    assert ast is not None


LOOP_CONTROL_TESTS = [
    ("while (1) break; endwhile", "simple break"),
    ("while (1) continue; endwhile", "simple continue"),
    ("while outer (1) while (1) break outer; endwhile endwhile", "labeled break"),
    ("while outer (1) while (1) continue outer; endwhile endwhile", "labeled continue"),
]

@pytest.mark.parametrize("code,desc", LOOP_CONTROL_TESTS)
def test_loop_control(code, desc):
    ast = parse(code)
    assert ast is not None


FORK_TESTS = [
    ("fork (0) return 1; endfork", "simple fork"),
    ("fork task (5) return 1; endfork", "named fork"),
]

@pytest.mark.parametrize("code,desc", FORK_TESTS)
def test_fork(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# ERROR HANDLING
# =============================================================================

TRY_EXCEPT_TESTS = [
    ("try return 1; except (ANY) return 0; endtry", "try-except ANY"),
    ("try return 1; except (E_TYPE) return 0; endtry", "try-except specific error"),
    ("try return 1; except e (E_TYPE) return e; endtry", "try-except with binding"),
    ("try return 1; except (E_TYPE, E_RANGE) return 0; endtry", "try-except multiple errors"),
]

@pytest.mark.parametrize("code,desc", TRY_EXCEPT_TESTS)
def test_try_except(code, desc):
    ast = parse(code)
    assert ast is not None


TRY_FINALLY_TESTS = [
    ("try return 1; finally return 2; endtry", "try-finally"),
]

@pytest.mark.parametrize("code,desc", TRY_FINALLY_TESTS)
def test_try_finally(code, desc):
    ast = parse(code)
    assert ast is not None


CATCH_EXPRESSION_TESTS = [
    ("return `1/0 ! any';", "catch any"),
    ("return `1/0 ! E_DIV';", "catch specific"),
    ("return `1/0 ! E_DIV => 0';", "catch with fallback"),
    ('return `x.prop ! E_PROPNF => "default"\';', "catch property access"),
]

@pytest.mark.parametrize("code,desc", CATCH_EXPRESSION_TESTS)
def test_catch_expression(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# PROPERTY AND VERB ACCESS
# =============================================================================

PROPERTY_TESTS = [
    ("return obj.prop;", "simple property"),
    ('return obj.("prop");', "computed property"),
    ("return $prop;", "dollar property"),
    ("return obj.prop1.prop2;", "chained property"),
]

@pytest.mark.parametrize("code,desc", PROPERTY_TESTS)
def test_property(code, desc):
    ast = parse(code)
    assert ast is not None


VERB_CALL_TESTS = [
    ("obj:verb();", "simple verb call"),
    ('obj:("verb")();', "computed verb call"),
    ("obj:verb(1, 2, 3);", "verb with args"),
    ("$verb();", "dollar verb call"),
    ("$verb(1, 2);", "dollar verb with args"),
    ("obj:verb1():verb2();", "chained verb call"),
]

@pytest.mark.parametrize("code,desc", VERB_CALL_TESTS)
def test_verb_call(code, desc):
    ast = parse(code)
    assert ast is not None


FUNCTION_CALL_TESTS = [
    ("length({1, 2, 3});", "builtin function"),
    ("tostr(123);", "conversion function"),
    ("call_function(\"unknown\", 1, 2);", "call_function"),
]

@pytest.mark.parametrize("code,desc", FUNCTION_CALL_TESTS)
def test_function_call(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# INDEXING AND RANGES
# =============================================================================

INDEX_TESTS = [
    ("return x[1];", "simple index"),
    ("return x[y];", "variable index"),
    ("return x[1][2];", "nested index"),
    ("return x[$];", "dollar index (last)"),
    ("return x[^];", "caret index (first)"),
]

@pytest.mark.parametrize("code,desc", INDEX_TESTS)
def test_index(code, desc):
    ast = parse(code)
    assert ast is not None


RANGE_TESTS = [
    ("return x[1..3];", "simple range"),
    ("return x[1..$];", "range to end"),
    ("return x[^..3];", "range from start"),
    ("return x[^..$];", "full range"),
]

@pytest.mark.parametrize("code,desc", RANGE_TESTS)
def test_range(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# ASSIGNMENT
# =============================================================================

ASSIGNMENT_TESTS = [
    ("x = 1;", "simple assignment"),
    ("x = y = 1;", "chained assignment"),
    ("obj.prop = 1;", "property assignment"),
    ('obj.("prop") = 1;', "computed property assignment"),
    ("x[1] = 1;", "indexed assignment"),
    ("x[1..3] = {4, 5, 6};", "range assignment"),
]

@pytest.mark.parametrize("code,desc", ASSIGNMENT_TESTS)
def test_assignment(code, desc):
    ast = parse(code)
    assert ast is not None


SCATTER_TESTS = [
    ("{a, b, c} = {1, 2, 3};", "simple scatter"),
    ("{a, ?b} = {1};", "scatter with optional"),
    ("{a, @rest} = {1, 2, 3, 4};", "scatter with rest"),
    ("{a, ?b = 0, @rest} = {1};", "scatter with default"),
]

@pytest.mark.parametrize("code,desc", SCATTER_TESTS)
def test_scatter(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# TERNARY AND SPECIAL EXPRESSIONS
# =============================================================================

TERNARY_TESTS = [
    ("return 1 ? 2 | 3;", "simple ternary"),
    ("return x ? y ? 1 | 2 | 3;", "nested ternary"),
]

@pytest.mark.parametrize("code,desc", TERNARY_TESTS)
def test_ternary(code, desc):
    ast = parse(code)
    assert ast is not None


SPLICER_TESTS = [
    ("return {@list};", "spliced list"),
    ("func(@args);", "spliced args"),
    ("{@first, last} = {1, 2, 3};", "splice in scatter"),
]

@pytest.mark.parametrize("code,desc", SPLICER_TESTS)
def test_splicer(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# COMMENTS
# =============================================================================

COMMENT_TESTS = [
    ("/* block comment */ return 1;", "block comment before"),
    ("return 1; /* comment */", "block comment after"),
    ("return /* inline */ 1;", "block comment inline"),
    ("/* multi\nline\ncomment */ return 1;", "multiline block comment"),
    ("// line comment\nreturn 1;", "line comment"),
    ("return 1; // trailing comment", "trailing line comment"),
]

@pytest.mark.parametrize("code,desc", COMMENT_TESTS)
def test_comment(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# ERROR LITERALS
# =============================================================================

ERROR_LITERAL_TESTS = [
    ("return E_NONE;", "E_NONE"),
    ("return E_TYPE;", "E_TYPE"),
    ("return E_DIV;", "E_DIV"),
    ("return E_PERM;", "E_PERM"),
    ("return E_PROPNF;", "E_PROPNF"),
    ("return E_VERBNF;", "E_VERBNF"),
    ("return E_VARNF;", "E_VARNF"),
    ("return E_INVIND;", "E_INVIND"),
    ("return E_RECMOVE;", "E_RECMOVE"),
    ("return E_MAXREC;", "E_MAXREC"),
    ("return E_RANGE;", "E_RANGE"),
    ("return E_ARGS;", "E_ARGS"),
    ("return E_NACC;", "E_NACC"),
    ("return E_INVARG;", "E_INVARG"),
    ("return E_QUOTA;", "E_QUOTA"),
    ("return E_FLOAT;", "E_FLOAT"),
]

@pytest.mark.parametrize("code,desc", ERROR_LITERAL_TESTS)
def test_error_literal(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# WAIF PROPERTIES (ToastStunt extension)
# =============================================================================

WAIF_TESTS = [
    ("return obj.:prop;", "waif property access"),
    ("obj.:prop = 1;", "waif property assignment"),
]

@pytest.mark.parametrize("code,desc", WAIF_TESTS)
def test_waif(code, desc):
    ast = parse(code)
    assert ast is not None


# =============================================================================
# COMPLEX EXPRESSIONS
# =============================================================================

COMPLEX_TESTS = [
    # Multiple statements
    ("x = 1; y = 2; return x + y;", "multiple statements"),

    # Nested structures
    ("if (1) for i in ({1}) return i; endfor endif", "if with nested for"),

    # Complex expressions
    ("return (a + b) * (c - d);", "parenthesized expression"),
    ("return obj:method(x[1], y.prop, z ? 1 | 0);", "complex call"),

    # Empty statements
    (";", "empty statement"),
    (";;", "multiple empty statements"),
]

@pytest.mark.parametrize("code,desc", COMPLEX_TESTS)
def test_complex(code, desc):
    ast = parse(code)
    assert ast is not None
