"""Tests for builtin functions."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from lambdamoo_db.database import ObjNum
from moo_interp.builtin_functions import BuiltinFunctions
from moo_interp.errors import MOOError, TYPE_ERR
from moo_interp.moo_ast import compile, parse, run
from moo_interp.moo_types import MOOString


PRIVATE_CALLABLE_NAMES = tuple(
    name
    for name in dir(BuiltinFunctions)
    if name.startswith("_")
    and not name.startswith("__")
    and callable(getattr(BuiltinFunctions, name))
)
REGISTRY_INFRASTRUCTURE_NAMES = (
    "register",
    "get_function_by_name",
    "get_function_by_id",
    "get_function_name_by_id",
    "get_id_by_function",
    "get_id_by_name",
    "raise_error",
)


class TestStrtr:
    """Test strtr builtin function."""

    def test_strtr_with_moostring(self):
        """strtr should handle MOOString inputs without maketrans() errors."""
        bi = BuiltinFunctions()

        # This should not raise: maketrans() argument 2 must be str, not MOOString
        result = bi.strtr(MOOString("hello"), MOOString("el"), MOOString("ip"))

        # Result could be str or MOOString, handle both
        result_str = str(result) if hasattr(result, 'value') else result
        assert result_str == "hippo"

    def test_strtr_basic_transformation(self):
        """Test basic character transformation."""
        bi = BuiltinFunctions()
        result = bi.strtr(MOOString("abcdef"), MOOString("abc"), MOOString("123"))
        result_str = str(result) if hasattr(result, 'value') else result
        assert result_str == "123def"

    def test_strtr_case_insensitive_default(self):
        """By default strtr should be case insensitive."""
        bi = BuiltinFunctions()
        result = bi.strtr(MOOString("HeLLo"), MOOString("el"), MOOString("ip"))
        result_str = str(result) if hasattr(result, 'value') else result
        # Should transform: H->H, e->i, L->P, L->P, o->o = "HiPPo"
        # Case insensitive means uppercase maps to uppercase, lowercase to lowercase
        assert result_str == "HiPPo"


@given(error=st.sampled_from(list(MOOError)))
def test_typeof_preserves_error_type(error):
    assert BuiltinFunctions().typeof(error) == TYPE_ERR


def test_compiled_error_constant_preserves_error_type():
    assert run(compile(parse("return typeof(E_TYPE);"))).result == TYPE_ERR


@given(number=st.integers(min_value=-(2**31), max_value=2**31 - 1))
def test_object_string_conversions_keep_hash_prefix(number):
    builtins = BuiltinFunctions()
    obj = ObjNum(number)
    assert str(builtins.tostr(obj)) == f"#{number}"
    assert str(builtins.toliteral(obj)) == f"#{number}"


@given(name=st.sampled_from(PRIVATE_CALLABLE_NAMES))
def test_private_helpers_are_not_registered_as_builtins(name):
    assert name not in BuiltinFunctions().functions


@given(name=st.sampled_from(REGISTRY_INFRASTRUCTURE_NAMES))
def test_registry_infrastructure_is_not_registered_as_builtins(name):
    builtins = BuiltinFunctions()
    assert name not in builtins.functions
    assert "raise" in builtins.functions
