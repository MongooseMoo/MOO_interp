import pytest
from hypothesis import given
from hypothesis import strategies as st

from moo_interp.list import MOOList
from moo_interp.map import MOOMap
from moo_interp.string import MOOString


class TestMOOList:

    @given(st.lists(st.integers()), st.integers(min_value=1), st.integers())
    def test_list_operations(self, data, index, value):
        moo_list = MOOList(*data)
        len_data = len(data)
        index = index if index <= len_data else len_data  # make sure index is valid

        # Test getitem
        assert moo_list[index] == data[index - 1]

        # Test setitem
        moo_list[index] = value
        assert moo_list[index] == value

        # Test insert
        moo_list.insert(index, value)
        assert moo_list[index] == value
        assert len(moo_list) == len_data + 2

        # Test delitem
        del moo_list[index]
        assert len(moo_list) == len_data + 1

    @given(st.lists(st.integers()))
    def test_len(self, data):
        moo_list = MOOList(*data)
        assert len(moo_list) == len(data)

    @given(st.lists(st.integers()))
    def test_repr(self, data):
        moo_list = MOOList(*data)
        assert repr(moo_list) == f"MOOList({data})"

    def test_empty_list(self):
        moo_list = MOOList()
        with pytest.raises(IndexError):
            moo_list[1]  # There is no item at this position


class TestMOOMap:

    @given(st.dictionaries(st.text(), st.integers()), st.text(), st.integers())
    def test_map_operations(self, data, key, value):
        moo_map = MOOMap()
        moo_map.update(data)

        # Test getitem
        if key in data:
            assert moo_map[key] == data[key]

        # Test setitem
        moo_map[key] = value
        assert moo_map[key] == value

        # Test delitem
        if key in moo_map:
            del moo_map[key]
            assert key not in moo_map

    @given(st.dictionaries(st.text(), st.integers()))
    def test_len(self, data):
        moo_map = MOOMap()
        moo_map.update(data)
        assert len(moo_map) == len(data)

    @given(st.dictionaries(st.text(), st.integers()))
    def test_repr(self, data):
        moo_map = MOOMap()
        moo_map.update(data)
        assert repr(moo_map) == f"MOOMap({data})"

    @given(st.dictionaries(st.text(), st.integers()))
    def test_iter(self, data):
        moo_map = MOOMap()
        moo_map.update(data)
        assert set(moo_map) == set(data.keys())


class TestMOOString:

    @given(st.text(min_size=1), st.integers(min_value=1), st.text())
    def test_string_operations(self, data, index, value):
        moo_string = MOOString(data)
        len_data = len(data)
        index = index if index <= len_data else len_data  # make sure index is valid

        # Test getitem
        assert moo_string[index] == data[index - 1]

        # Test setitem
        moo_string[index] = value
        assert moo_string[index] == value

        # Test string representation
        assert str(moo_string) == moo_string.data

        # Test slice operations
        sliced_moo_string = moo_string[1:len_data:2]
        assert str(sliced_moo_string) == data[::2]

    @given(st.text())
    def test_len(self, data):
        moo_string = MOOString(data)
        assert len(moo_string) == len(data)

    @given(st.text())
    def test_repr(self, data):
        moo_string = MOOString(data)
        assert repr(moo_string) == f"MOOString({data})"

    def test_empty_string(self):
        moo_string = MOOString()
        with pytest.raises(IndexError):
            moo_string[1]  # There is no item at this position
