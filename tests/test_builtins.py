"""Tests for builtin functions."""

import pytest
from moo_interp.builtin_functions import BuiltinFunctions
from moo_interp.moo_types import MOOString


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
