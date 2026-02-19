"""Tests for MOO AST → Python AST transformer."""

import ast

import pytest

from moo_interp.to_python_ast import moo_to_python_ast


# --- Cycle 1: Scaffold + number literal ---

def test_number_literal():
    py = moo_to_python_ast("return 42;")
    assert isinstance(py, ast.Module)
    ret = py.body[0]
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.Constant)
    assert ret.value.value == 42


# --- Cycle 2: String literal ---

def test_string_literal():
    py = moo_to_python_ast('return "hello";')
    ret = py.body[0]
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.Constant)
    assert ret.value.value == "hello"


# --- Cycle 3: Float literal ---

def test_float_literal():
    py = moo_to_python_ast("return 3.14;")
    ret = py.body[0]
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.Constant)
    assert ret.value.value == 3.14


# --- Cycle 4: Boolean literal ---

def test_boolean_literal_true():
    py = moo_to_python_ast("return true;")
    ret = py.body[0]
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.Constant)
    assert ret.value.value is True


def test_boolean_literal_false():
    py = moo_to_python_ast("return false;")
    ret = py.body[0]
    assert isinstance(ret.value, ast.Constant)
    assert ret.value.value is False


# --- Cycle 5: ObjnumLiteral ---

def test_objnum_literal():
    py = moo_to_python_ast("return #42;")
    ret = py.body[0]
    assert isinstance(ret, ast.Return)
    call = ret.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "ObjNum"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == 42


# --- Cycle 6: Identifier ---

def test_identifier():
    py = moo_to_python_ast("return x;")
    ret = py.body[0]
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.Name)
    assert ret.value.id == "x"


# --- Cycle 7: Binary arithmetic ---

def test_binary_add():
    py = moo_to_python_ast("return 1 + 2;")
    ret = py.body[0]
    binop = ret.value
    assert isinstance(binop, ast.BinOp)
    assert isinstance(binop.left, ast.Constant)
    assert binop.left.value == 1
    assert isinstance(binop.op, ast.Add)
    assert isinstance(binop.right, ast.Constant)
    assert binop.right.value == 2


def test_binary_sub():
    py = moo_to_python_ast("return 3 - 1;")
    assert isinstance(py.body[0].value.op, ast.Sub)


def test_binary_mult():
    py = moo_to_python_ast("return 3 * 2;")
    assert isinstance(py.body[0].value.op, ast.Mult)


def test_binary_div():
    py = moo_to_python_ast("return 6 / 2;")
    assert isinstance(py.body[0].value.op, ast.Div)


def test_binary_mod():
    py = moo_to_python_ast("return 7 % 3;")
    assert isinstance(py.body[0].value.op, ast.Mod)


def test_binary_pow():
    py = moo_to_python_ast("return 2 ^ 3;")
    assert isinstance(py.body[0].value.op, ast.Pow)


# --- Cycle 8: Comparison operators ---

def test_compare_eq():
    py = moo_to_python_ast("return x == 1;")
    ret = py.body[0]
    cmp = ret.value
    assert isinstance(cmp, ast.Compare)
    assert isinstance(cmp.left, ast.Name)
    assert cmp.left.id == "x"
    assert len(cmp.ops) == 1
    assert isinstance(cmp.ops[0], ast.Eq)
    assert len(cmp.comparators) == 1
    assert isinstance(cmp.comparators[0], ast.Constant)
    assert cmp.comparators[0].value == 1


def test_compare_ne():
    py = moo_to_python_ast("return x != 1;")
    assert isinstance(py.body[0].value.ops[0], ast.NotEq)


def test_compare_lt():
    py = moo_to_python_ast("return x < 1;")
    assert isinstance(py.body[0].value.ops[0], ast.Lt)


def test_compare_gt():
    py = moo_to_python_ast("return x > 1;")
    assert isinstance(py.body[0].value.ops[0], ast.Gt)


def test_compare_le():
    py = moo_to_python_ast("return x <= 1;")
    assert isinstance(py.body[0].value.ops[0], ast.LtE)


def test_compare_ge():
    py = moo_to_python_ast("return x >= 1;")
    assert isinstance(py.body[0].value.ops[0], ast.GtE)


def test_compare_in():
    py = moo_to_python_ast("return x in {1, 2};")
    assert isinstance(py.body[0].value.ops[0], ast.In)


# --- Cycle 9: Logical operators ---

def test_logical_and():
    py = moo_to_python_ast("return x && y;")
    ret = py.body[0]
    boolop = ret.value
    assert isinstance(boolop, ast.BoolOp)
    assert isinstance(boolop.op, ast.And)
    assert len(boolop.values) == 2
    assert isinstance(boolop.values[0], ast.Name)
    assert boolop.values[0].id == "x"
    assert isinstance(boolop.values[1], ast.Name)
    assert boolop.values[1].id == "y"


def test_logical_or():
    py = moo_to_python_ast("return x || y;")
    boolop = py.body[0].value
    assert isinstance(boolop, ast.BoolOp)
    assert isinstance(boolop.op, ast.Or)


# --- Cycle 10: Bitwise operators ---

def test_bitwise_or():
    py = moo_to_python_ast("return x |. y;")
    binop = py.body[0].value
    assert isinstance(binop, ast.BinOp)
    assert isinstance(binop.op, ast.BitOr)


def test_bitwise_and():
    py = moo_to_python_ast("return x &. y;")
    assert isinstance(py.body[0].value.op, ast.BitAnd)


def test_bitwise_xor():
    py = moo_to_python_ast("return x ^. y;")
    assert isinstance(py.body[0].value.op, ast.BitXor)


def test_bitwise_lshift():
    py = moo_to_python_ast("return x << y;")
    assert isinstance(py.body[0].value.op, ast.LShift)


def test_bitwise_rshift():
    py = moo_to_python_ast("return x >> y;")
    assert isinstance(py.body[0].value.op, ast.RShift)


# --- Cycle 11: Unary operators ---

def test_unary_neg():
    py = moo_to_python_ast("return -x;")
    ret = py.body[0]
    unary = ret.value
    assert isinstance(unary, ast.UnaryOp)
    assert isinstance(unary.op, ast.USub)
    assert isinstance(unary.operand, ast.Name)
    assert unary.operand.id == "x"


def test_unary_not():
    py = moo_to_python_ast("return !x;")
    unary = py.body[0].value
    assert isinstance(unary, ast.UnaryOp)
    assert isinstance(unary.op, ast.Not)


def test_unary_complement():
    py = moo_to_python_ast("return ~x;")
    unary = py.body[0].value
    assert isinstance(unary, ast.UnaryOp)
    assert isinstance(unary.op, ast.Invert)


# --- Cycle 12: Ternary ---

def test_ternary():
    py = moo_to_python_ast("return x ? 1 | 0;")
    ret = py.body[0]
    ifexp = ret.value
    assert isinstance(ifexp, ast.IfExp)
    assert isinstance(ifexp.test, ast.Name)
    assert ifexp.test.id == "x"
    assert isinstance(ifexp.body, ast.Constant)
    assert ifexp.body.value == 1
    assert isinstance(ifexp.orelse, ast.Constant)
    assert ifexp.orelse.value == 0


# --- Cycle 13: List literal ---

def test_list_literal():
    py = moo_to_python_ast("return {1, 2, 3};")
    ret = py.body[0]
    lst = ret.value
    assert isinstance(lst, ast.List)
    assert len(lst.elts) == 3
    assert all(isinstance(e, ast.Constant) for e in lst.elts)
    assert [e.value for e in lst.elts] == [1, 2, 3]


def test_empty_list():
    py = moo_to_python_ast("return {};")
    ret = py.body[0]
    lst = ret.value
    assert isinstance(lst, ast.List)
    assert len(lst.elts) == 0


# --- Cycle 14: Map literal ---

def test_map_literal():
    py = moo_to_python_ast('return ["a" -> 1, "b" -> 2];')
    ret = py.body[0]
    d = ret.value
    assert isinstance(d, ast.Dict)
    assert len(d.keys) == 2
    assert isinstance(d.keys[0], ast.Constant)
    assert d.keys[0].value == "a"
    assert isinstance(d.values[0], ast.Constant)
    assert d.values[0].value == 1


# --- Cycle 15: Splicer ---

def test_splicer():
    py = moo_to_python_ast("return {@x};")
    ret = py.body[0]
    lst = ret.value
    assert isinstance(lst, ast.List)
    assert len(lst.elts) == 1
    assert isinstance(lst.elts[0], ast.Starred)
    assert isinstance(lst.elts[0].value, ast.Name)
    assert lst.elts[0].value.id == "x"


# --- Cycle 16: Function call ---

def test_function_call():
    py = moo_to_python_ast('return length("hello");')
    ret = py.body[0]
    call = ret.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "length"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == "hello"


# --- Cycle 17: Property access ---

def test_property_access():
    py = moo_to_python_ast("return obj.name;")
    ret = py.body[0]
    attr = ret.value
    assert isinstance(attr, ast.Attribute)
    assert isinstance(attr.value, ast.Name)
    assert attr.value.id == "obj"
    assert attr.attr == "name"


# --- Cycle 18: Dollar property ---

def test_dollar_property():
    py = moo_to_python_ast("return $foo;")
    ret = py.body[0]
    attr = ret.value
    assert isinstance(attr, ast.Attribute)
    assert isinstance(attr.value, ast.Name)
    assert attr.value.id == "system"
    assert attr.attr == "foo"


# --- Cycle 19: Verb call ---

def test_verb_call():
    py = moo_to_python_ast('return obj:tell("hi");')
    ret = py.body[0]
    call = ret.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Attribute)
    assert isinstance(call.func.value, ast.Name)
    assert call.func.value.id == "obj"
    assert call.func.attr == "tell"
    assert len(call.args) == 1
    assert call.args[0].value == "hi"


# --- Cycle 20: Dollar verb call ---

def test_dollar_verb_call():
    py = moo_to_python_ast('return $string_utils:trim("hi");')
    ret = py.body[0]
    call = ret.value
    assert isinstance(call, ast.Call)
    # $string_utils becomes system.string_utils
    assert isinstance(call.func, ast.Attribute)
    assert call.func.attr == "trim"
    # The object should be system.string_utils
    obj = call.func.value
    assert isinstance(obj, ast.Attribute)
    assert isinstance(obj.value, ast.Name)
    assert obj.value.id == "system"
    assert obj.attr == "string_utils"


# --- Cycle 21: Index ---

def test_index():
    py = moo_to_python_ast("return x[1];")
    ret = py.body[0]
    sub = ret.value
    assert isinstance(sub, ast.Subscript)
    assert isinstance(sub.value, ast.Name)
    assert sub.value.id == "x"
    assert isinstance(sub.slice, ast.Constant)
    assert sub.slice.value == 1


# --- Cycle 22: Range ---

def test_range_slice():
    py = moo_to_python_ast("return x[1..3];")
    ret = py.body[0]
    sub = ret.value
    assert isinstance(sub, ast.Subscript)
    assert isinstance(sub.value, ast.Name)
    assert sub.value.id == "x"
    assert isinstance(sub.slice, ast.Slice)
    assert isinstance(sub.slice.lower, ast.Constant)
    assert sub.slice.lower.value == 1
    assert isinstance(sub.slice.upper, ast.Constant)
    assert sub.slice.upper.value == 3


# --- Cycle 23: Simple assignment ---

def test_simple_assignment():
    py = moo_to_python_ast("x = 42;")
    assign = py.body[0]
    assert isinstance(assign, ast.Assign)
    assert len(assign.targets) == 1
    assert isinstance(assign.targets[0], ast.Name)
    assert assign.targets[0].id == "x"
    assert isinstance(assign.targets[0].ctx, ast.Store)
    assert isinstance(assign.value, ast.Constant)
    assert assign.value.value == 42


# --- Cycle 24: Empty statement ---

def test_empty_statement():
    py = moo_to_python_ast(";")
    assert len(py.body) == 1
    assert isinstance(py.body[0], ast.Pass)


# --- Cycle 25: Expression as statement ---

def test_expression_statement():
    py = moo_to_python_ast('length("hi");')
    stmt = py.body[0]
    assert isinstance(stmt, ast.Expr)
    assert isinstance(stmt.value, ast.Call)
    assert isinstance(stmt.value.func, ast.Name)
    assert stmt.value.func.id == "length"


# --- Cycle 26: If/else ---

def test_if_else():
    py = moo_to_python_ast("if (x) return 1; else return 0; endif")
    ifstmt = py.body[0]
    assert isinstance(ifstmt, ast.If)
    assert isinstance(ifstmt.test, ast.Name)
    assert ifstmt.test.id == "x"
    assert len(ifstmt.body) == 1
    assert isinstance(ifstmt.body[0], ast.Return)
    assert ifstmt.body[0].value.value == 1
    assert len(ifstmt.orelse) == 1
    assert isinstance(ifstmt.orelse[0], ast.Return)
    assert ifstmt.orelse[0].value.value == 0


# --- Cycle 27: If/elseif/else ---

def test_if_elseif_else():
    py = moo_to_python_ast("if (x) return 1; elseif (y) return 2; else return 3; endif")
    ifstmt = py.body[0]
    assert isinstance(ifstmt, ast.If)
    assert ifstmt.body[0].value.value == 1
    # elseif becomes nested if in orelse
    assert len(ifstmt.orelse) == 1
    elif_stmt = ifstmt.orelse[0]
    assert isinstance(elif_stmt, ast.If)
    assert isinstance(elif_stmt.test, ast.Name)
    assert elif_stmt.test.id == "y"
    assert elif_stmt.body[0].value.value == 2
    assert len(elif_stmt.orelse) == 1
    assert elif_stmt.orelse[0].value.value == 3


# --- Cycle 28: While loop ---

def test_while_loop():
    py = moo_to_python_ast("while (x) x = x - 1; endwhile")
    whilestmt = py.body[0]
    assert isinstance(whilestmt, ast.While)
    assert isinstance(whilestmt.test, ast.Name)
    assert whilestmt.test.id == "x"
    assert len(whilestmt.body) >= 1


# --- Cycle 29: For-list loop ---

def test_for_list():
    py = moo_to_python_ast("for x in ({1, 2, 3}) return x; endfor")
    forstmt = py.body[0]
    assert isinstance(forstmt, ast.For)
    assert isinstance(forstmt.target, ast.Name)
    assert forstmt.target.id == "x"
    assert isinstance(forstmt.iter, ast.List)
    assert len(forstmt.body) >= 1


# --- Cycle 30: For-range loop ---

def test_for_range():
    py = moo_to_python_ast("for i in [1..10] i; endfor")
    forstmt = py.body[0]
    assert isinstance(forstmt, ast.For)
    assert isinstance(forstmt.target, ast.Name)
    assert forstmt.target.id == "i"
    # for i in [1..10] → for i in range(1, 11)
    assert isinstance(forstmt.iter, ast.Call)
    assert isinstance(forstmt.iter.func, ast.Name)
    assert forstmt.iter.func.id == "range"
    assert len(forstmt.iter.args) == 2
    assert forstmt.iter.args[0].value == 1
    # end is 10 + 1 = 11
    end_arg = forstmt.iter.args[1]
    assert isinstance(end_arg, ast.BinOp)
    assert isinstance(end_arg.op, ast.Add)
    assert end_arg.left.value == 10
    assert end_arg.right.value == 1


# --- Cycle 31: Break/continue ---

def test_break():
    py = moo_to_python_ast("while (1) break; endwhile")
    whilestmt = py.body[0]
    assert isinstance(whilestmt.body[0], ast.Break)


def test_continue():
    py = moo_to_python_ast("while (1) continue; endwhile")
    whilestmt = py.body[0]
    assert isinstance(whilestmt.body[0], ast.Continue)


# --- Cycle 32: Try/except ---

def test_try_except():
    py = moo_to_python_ast("try x = 1; except (ANY) return 0; endtry")
    trystmt = py.body[0]
    assert isinstance(trystmt, ast.Try)
    assert len(trystmt.body) >= 1
    assert len(trystmt.handlers) >= 1
    handler = trystmt.handlers[0]
    assert isinstance(handler, ast.ExceptHandler)


# --- Cycle 33: Try/finally ---

def test_try_finally():
    py = moo_to_python_ast("try x = 1; finally x = 0; endtry")
    trystmt = py.body[0]
    assert isinstance(trystmt, ast.Try)
    assert len(trystmt.body) >= 1
    assert len(trystmt.finalbody) >= 1
    assert len(trystmt.handlers) == 0


# --- Cycle 34: Catch expression ---

def test_catch_expression():
    py = moo_to_python_ast("return `x ! ANY => 0';")
    ret = py.body[0]
    # Catch becomes a try-except wrapped in a lambda or similar
    # The plan says "appropriate Python AST representation"
    # We'll use ast.Call wrapping a helper function
    assert isinstance(ret, ast.Return)
    # The catch expression should produce some valid AST
    assert ret.value is not None


# --- Cycle 35: Fork ---

def test_fork():
    py = moo_to_python_ast("fork (0) x = 1; endfork")
    stmt = py.body[0]
    assert isinstance(stmt, ast.Expr)
    call = stmt.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "fork"


# --- Cycle 36: Scatter assignment ---

def test_scatter_assignment():
    py = moo_to_python_ast("{a, b, c} = {1, 2, 3};")
    assign = py.body[0]
    assert isinstance(assign, ast.Assign)
    assert isinstance(assign.targets[0], ast.Tuple)
    tup = assign.targets[0]
    assert len(tup.elts) == 3
    assert all(isinstance(e, ast.Name) for e in tup.elts)
    assert [e.id for e in tup.elts] == ["a", "b", "c"]
    assert isinstance(assign.value, ast.List)


# --- Integration: ast.fix_missing_locations ---

def test_ast_has_locations():
    """All nodes should have lineno/col_offset after fix_missing_locations."""
    py = moo_to_python_ast("return 42;")
    for node in ast.walk(py):
        if isinstance(node, (ast.expr, ast.stmt)):
            assert hasattr(node, 'lineno'), f"{type(node).__name__} missing lineno"


# --- Integration: ast.unparse ---

def test_unparse_simple():
    """ast.unparse should work on the output."""
    py = moo_to_python_ast("return 1 + 2;")
    result = ast.unparse(py)
    assert "1 + 2" in result


# --- From pre-parsed AST ---

def test_from_moo_ast():
    """moo_to_python_ast should accept pre-parsed MOO AST."""
    from moo_interp.moo_ast import parse
    moo_ast = parse("return 42;")
    py = moo_to_python_ast(moo_ast)
    assert isinstance(py, ast.Module)
    ret = py.body[0]
    assert isinstance(ret, ast.Return)
    assert ret.value.value == 42
