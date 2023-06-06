from collections import UserString
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
            index = slice(index.start - 1, index.stop , index.step)
        else:
            index -= 1
        return MOOString(super().__getitem__(index))

    def __setitem__(self, index, value):
        # make  sure we are one indexed
        if isinstance(index, slice):
            index = slice(index.start - 1, index.stop , index.step)
        else:
            index -= 1
        # make a copy of the data
        data = list(self.data)
        data[index] = value
        self.data = "".join(data)
