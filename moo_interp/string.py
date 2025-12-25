from collections import UserString

from .list import MOOList

"""The MOO String class.

MOO strings are different from Python strings in several ways:
- They are mutable
- they are one indexed
"""


class MOOString(UserString):

    def __init__(self, seq=None):
        if seq is None:
            seq = ""
        super().__init__(seq)

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = slice(index.start - 1, index.stop, index.step)
        else:
            index -= 1
        return MOOString(super().__getitem__(index))

    def __setitem__(self, index, value):
        # make  sure we are one indexed
        if isinstance(index, slice):
            index = slice(index.start - 1, index.stop, index.step)
        else:
            index -= 1
        # make a copy of the data
        data = list(self.data)
        # Ensure value is a plain string for joining
        data[index] = str(value) if not isinstance(value, str) else value
        self.data = "".join(data)

    def __delitem__(self, index):
        if isinstance(index, slice):
            index = slice(index.start - 1, index.stop, index.step)
        else:
            index -= 1
        # make a copy of the data
        data = list(self.data)
        del data[index]
        self.data = "".join(data)

        def find(self, sub, start=None, end=None):
            if start is None:
                start = 1
            else:
                start -= 1
            if end is None:
                end = len(self)
            else:
                end -= 1
            return super().find(sub, start, end) + 1

        def rfind(self, sub, start=None, end=None):
            if start is None:
                start = 1
            else:
                start -= 1
            if end is None:
                end = len(self)
            else:
                end -= 1
            return super().rfind(sub, start, end) + 1

        def index(self, sub, start=None, end=None):
            if start is None:
                start = 1
            else:
                start -= 1
            if end is None:
                end = len(self)
            else:
                end -= 1
            return super().index(sub, start, end) + 1

        def rindex(self, sub, start=None, end=None):
            if start is None:
                start = 1
            else:
                start -= 1
            if end is None:
                end = len(self)
            else:
                end -= 1
            return super().rindex(sub, start, end) + 1

        def split(self, sep=None, maxsplit=-1):
            return MOOList(*super().split(sep, maxsplit))

        def rsplit(self, sep=None, maxsplit=-1):
            return MOOList(*super().rsplit(sep, maxsplit))

        def splitlines(self, keepends=False):
            return MOOList(*super().splitlines(keepends))
