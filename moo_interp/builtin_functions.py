import base64
from functools import reduce
import hashlib
import json
import math
import os
import random
import re
from logging import getLogger
from typing import Union

from lambdamoo_db.database import MooDatabase, MooObject
from .errors import MOOError
from .list import MOOList
from .map import MOOMap
from .string import MOOString
from .moo_types import MOOAny, MOONumber, to_moo

logger = getLogger(__name__)


class BuiltinFunctions:

    ANSI_TAGS = {
        "[red]": "\033[31m", "[green]": "\033[32m", "[yellow]": "\033[33m",
        "[blue]": "\033[34m", "[purple]": "\033[35m", "[cyan]": "\033[36m",
        "[normal]": "\033[0m", "[inverse]": "\033[7m", "[underline]": "\033[4m",
        "[bold]": "\033[1m", "[bright]": "\033[1m", "[unbold]": "\033[22m",
        "[blink]": "\033[5m", "[unblink]": "\033[25m", "[magenta]": "\033[35m",
        "[unbright]": "\033[22m", "[white]": "\033[37m", "[gray]": "\033[1;30m",
        "[grey]": "\033[1;30m", "[beep]": "\a", "[black]": "\033[30m",
        "[b:black]": "\033[40m", "[b:red]": "\033[41m", "[b:green]": "\033[42m",
        "[b:yellow]": "\033[43m", "[b:blue]": "\033[44m", "[b:magenta]": "\033[45m",
        "[b:purple]": "\033[45m", "[b:cyan]": "\033[46m", "[b:white]": "\033[47m",
        "[null]": ""
    }

    RANDOM_CODES = ["\033[31m", "\033[32m",
                    "\033[33m", "\033[34m", "\033[35m", "\033[36m"]

    def __init__(self, db=None, server=None):
        self.functions = {}  # Stores function_name: function pairs
        self.id_to_function = {}  # Stores function_id: function pairs
        self.function_to_id = {}  # Stores function: function_id pairs
        self.current_id = 0
        self.db = db
        self.server = server

        # automatically register functions
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and not attr_name.startswith('__'):
                self(attr)

    def __call__(self, fn):
        if self.current_id > 255:
            raise Exception("Cannot register more than 256 functions.")
        function_id = self.current_id
        self.current_id += 1
        self.functions[fn.__name__] = fn
        self.id_to_function[function_id] = fn
        self.function_to_id[fn] = function_id
        return fn

    # Dictionary-like methods

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_function_by_name(key)
        elif isinstance(key, int):
            return self.get_function_by_id(key)
        else:
            raise KeyError(
                f"Invalid key type. Expected str or int, got {type(key).__name__}")

    def __setitem__(self, key, value):
        self(value)

    def __delitem__(self, key):
        if isinstance(key, str):
            fn = self.get_function_by_name(key)
            if fn is None:
                raise KeyError(f"No function named {key}")
            del self.functions[key]
            id = self.get_id_by_function(fn)
            del self.id_to_function[id]
            del self.function_to_id[fn]
        elif isinstance(key, int):
            fn = self.get_function_by_id(key)
            if fn is None:
                raise KeyError(f"No function with id {key}")
            del self.id_to_function[key]
            name = next(name for name,
                        function in self.functions.items() if function == fn)
            del self.functions[name]
            del self.function_to_id[fn]
        else:
            raise KeyError(
                f"Invalid key type. Expected str or int, got {type(key).__name__}")

    def __contains__(self, item):
        return item in self.functions or item in self.id_to_function

    def __iter__(self):
        return iter(self.functions)

    # Helper methods

    def get_function_by_name(self, name):
        return self.functions.get(name)

    def get_function_by_id(self, id):
        return self.id_to_function.get(id)

    def get_id_by_function(self, fn):
        return self.function_to_id.get(fn)

    def get_id_by_name(self, name):
        fn = self.get_function_by_name(name)
        return self.get_id_by_function(fn) if fn else None

    def to_string(self, value):
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, MOOList):
            return "{list}"
        elif isinstance(value, MOOMap):
            return "[map]"
        elif isinstance(value, MOOString):
            return str(value)
        elif isinstance(value, float):
            return str(value)
        # elif isinstance(value, MOOObj):
            # return "#" + str(value)
        elif isinstance(value, MOOError):
            return self.unparse_error(value)
        elif isinstance(value, bool):
            return "true" if value else "false"
        # elif isinstance(value, MOOAnon):
            # return "*anonymous*"
        else:
            logger.error("TOSTR: Unknown Var type")

    def tostr(self, *args):
        return MOOString("".join(map(self.to_string, args)))

    def toint(self, value):
        if isinstance(value, int):
            return value
        elif isinstance(value, MOOString):
            try:
                return int(value)
            except ValueError:
                return 0
        elif isinstance(value, float):
            return int(value)
        elif isinstance(value,
                        MOOObj):
            return value.id
        elif isinstance(value, MOOError):
            return self.unparse_error(value)
        elif isinstance(value, bool):
            return 1 if value else 0
        # elif isinstance(value, MOOAnon):
            # return 0
        else:
            logger.error("TOINT: Unknown Var type")

    def tofloat(self, value):
        if isinstance(value, int):
            return float(value)
        elif isinstance(value, MOOString):
            return float(value)
        elif isinstance(value, float):
            return value
        elif isinstance(value, MOOObj):
            return value.id
        elif isinstance(value, MOOError):
            return self.unparse_error(value)
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, MOOAnon):
            return 0.0
        else:
            logger.error("TOFLOAT: Unknown Var type")

    _min = min

    def min(self, *args):
        return self._min(*args)

    _max = max

    def max(self, *args):
        return self._max(*args)

    def floor(self, value):
        return float(math.floor(self.tofloat(value)))

    def ceil(self, value):
        return float(math.ceil(self.tofloat(value)))

    def time(self,):
        import time
        return int(time.time())

    def ftime(self,):
        import time
        return float(time.time())

    def ctime(self, value=None):
        import time
        if value is None:
            return MOOString(time.ctime())
        return MOOString(time.ctime(self.tofloat(value)))

    def sin(self, value):
        return math.sin(self.tofloat(value))

    def cos(self, value):
        return math.cos(self.tofloat(value))

    def cosh(self, value):
        return math.cosh(self.tofloat(value))

    def distance(self, l1: MOOList, l2: MOOList) -> float:
        """Return the distance between two lists."""
        if len(l1) != len(l2):
            raise MOOError("distance", "Lists must be the same length")
        return math.sqrt(sum((l1[i] - l2[i]) ** 2 for i in range(len(l1))))

    def floatstr(self, x, precision, scientific=False):
        # Capping the precision
        precision = min(precision, 19)

        # Handling the scientific notation
        if scientific:
            return format(x, f".{precision}e")

        # Regular float to string conversion
        return MOOString(format(x, f".{precision}f"))

    def string_hash(self, string, algo='SHA256'):
        algo = algo.upper()

        if algo not in ['MD5', 'SHA1', 'SHA256']:
            raise ValueError(
                "Unsupported hash algorithm. Please choose either 'MD5', 'SHA1', or 'SHA256'.")

        hash_object = hashlib.new(algo)
        hash_object.update(string.encode())

        return hash_object.hexdigest()

    def exp(self, x):
        return math.exp(self.tofloat(x))

    def trunc(self, x):
        if x < 0:
            return self.ceil(x)
        else:
            return self.floor(x)

    def acos(self, x):
        return math.acos(self.tofloat(x))

    def asin(self, x):
        return math.asin(self.tofloat(x))

    def atan(self, x):
        return math.atan(self.tofloat(x))

    def atan2(self, y, x):
        return math.atan2(self.tofloat(y), self.tofloat(x))

    def log10(self, x):
        return math.log10(self.tofloat(x))

    def sin(self, x):
        return math.sin(self.tofloat(x))

    def sqrt(self, x):
        return math.sqrt(self.tofloat(x))

    def tan(self, x):
        return math.tan(self.tofloat(x))

    def listappend(self, list: MOOList, value):
        """Append value to list."""
        list.append(value)
        return list

    def listdelete(self, list: MOOList, index: int) -> MOOList:
        """Delete index from list."""
        del list[index]
        return list

    def listinsert(self, list: MOOList, index: int, value) -> MOOList:
        """Insert value into list at index."""
        list.insert(index, value)
        return list

    def listset(self, list: MOOList, index: int, value) -> MOOList:
        """Set value at index in list."""
        list[index] = value
        return list

    def all_members(self, value: MOOAny, list: MOOList):
        """Return all indices of value in list."""
        return MOOList([i for i, x in enumerate(list) if x == value])

    def explode(self, string: MOOString, separator: MOOString) -> MOOList:
        """Split string by separator."""
        return MOOList(string.split(separator))

    def bf_reverse(self, list: MOOList) -> MOOList:
        """Reverse list."""
        return MOOList(list[::-1])

    def equal(self, x, y):
        return x == y

    def strcmp(self, str1: MOOString, str2: MOOString):
        if str1 < str2:
            return -1
        elif str1 == str2:
            return 0
        else:
            return 1

    def strtr(self, str1: MOOString, str2: MOOString, str3: MOOString, case_matters=False):
        """
            Transforms the string source by replacing the characters specified by str1 with the corresponding characters specified by str2. All other characters are not transformed. If str2 has fewer characters than str1 the unmatched characters are simply removed from source. By default the transformation is done on both upper and lower case characters no matter the case. If case-matters is provided and true, then case is treated as significant.
        """
        if case_matters:
            return str1.translate(str.maketrans(str2, str3))
        else:
            return str1.translate(str.maketrans(str2, str3, str2.upper() + str2.lower()))

    _chr = chr

    def chr(self, x):
        return self._chr(x)

    def index(self, str1: MOOString, str2: MOOString):
        return str1.find(str2)

    def rindex(self, str1: MOOString, str2: MOOString):
        return str1.rfind(str2)

    def strsub(self, str1: MOOString, str2: MOOString, str3: MOOString):
        return str1.replace(str2, str3)

    def strcmp(self, str1: MOOString, str2: MOOString):
        if str1 < str2:
            return -1
        elif str1 == str2:
            return 0
        else:
            return 1

    _abs = abs

    def abs(self, x):
        return self._abs((x))

    def length(self, x):
        return len(x)

    def toliteral(self, x):
        if isinstance(x, MOOString):
            return MOOString("\"" + x.data + "\"")
        elif isinstance(x, MOONumber):
            return MOOString(str(x))
        elif isinstance(x, MOOList):
            return MOOString("{" + ", ".join([str(self.toliteral(y)) for y in x]) + "}")
        elif isinstance(x, MOOMap):
            return MOOString("[" + ", ".join([self.toliteral(y) + ": " + self.toliteral(z) for y, z in x.items()]) + "]")
        elif isinstance(x, MooObject):
            return MOOString("#" + str(x))
        elif isinstance(x, bool):
            return MOOString(str(x).lower())
        elif isinstance(x, MOOAny):
            return MOOString(str(x))
        else:
            raise (TypeError, "Unknown type: " + str(type(x)))

    def mapkeys(self, x):
        return MOOList(x.keys())

    def mapvalues(self, x):
        return MOOList(x.values())

    def mapdelete(self, x, y):
        """Delete key y from map x."""
        del x[y]
        return x

    def mapinsert(self, x, y, z):
        x[y] = z
        return x

    def eval(self, x):
        """\
        The MOO-code compiler processes <string> as if it were to be the program associated with some verb and, if no errors are found, that fictional verb is invoked.  If the programmer is not, in fact, a programmer, then E_PERM is raised.  The normal result of calling `eval()' is a two element list. The first element is true if there were no compilation errors and false otherwise.  The second element is either the result returned from the fictional verb (if there were no compilation errors) or a list of the compiler's error messages (otherwise).
        When the fictional verb is invoked, the various built-in variables have values as shown below:
        player    the same as in the calling verb
        this      #-1
        caller    the same as the initial value of `this' in the calling verb
        args      {}
        argstr    ""
        verb      ""
        dobjstr   ""
        dobj      #-1
        prepstr   ""
        iobjstr   ""
        iobj      #-1
        The fictional verb runs with the permissions of the programmer and as if its `d' permissions bit were on.
        """
        from .moo_ast import compile, run
        try:
            compiled = compile(x)
            compiled.debug = True
            compiled.this = -1
            compiled.verb = ""
            result = run(x)
        except Exception as e:
            return MOOList([False, MOOList([e])])
        return MOOList([True, result.result])

    def encode_base64(self, x, safe=False):
        if safe:
            return base64.urlsafe_b64encode(x)
        else:
            return base64.b64encode(x)

    def decode_base64(self, x, safe=False):
        if safe:
            return base64.urlsafe_b64decode(x)
        else:
            return base64.b64decode(x)

    def generate_json(self, x):
        return MOOString(json.dumps(x))

    def parse_json(self, x):
        return to_moo(json.loads(x))

    def remove_ansi(self, input_string):
        return reduce(lambda s, tag: s.replace(tag, ''), BuiltinFunctions.ANSI_TAGS, input_string)

    def parse_ansi(self, input_string):
        result_string = input_string
        for tag, code in BuiltinFunctions.ANSI_TAGS.items():
            result_string = result_string.replace(tag, code)
        while "[random]" in result_string:
            random_code = random.choice(BuiltinFunctions.RANDOM_CODES)
            result_string = result_string.replace("[random]", random_code, 1)
        return result_string

    # Fileio functions

    # Note: Need to add permissions checks and further compatibility to these functions

    def file_version(self,):
        return MOOString("FIO/2.0")

    def file_open(self, name: MOOString, mode: MOOString):
        return open(name, mode).fileno()

    def file_close(self, fd: int):
        return os.close(fd)

    def file_readline(self, fd: int):
        open_file = os.fdopen(fd)
        return MOOString(open_file.readline())

    def file_readlines(self, fd: int, start: int, end: int):
        open_file = os.fdopen(fd)
        return MOOList(open_file.readlines()[start:end])

    def file_writeline(self, fd: int, line: MOOString):
        open_file = os.fdopen(fd)
        open_file.write(str(line))

    def file_flush(self, fd: int):
        open_file = os.fdopen(fd)
        open_file.flush()

    def file_seek(self, fd: int, pos: int):
        open_file = os.fdopen(fd)
        open_file.seek(pos)

    def file_size(self, fd: int):
        return os.fstat(fd).st_size

    def file_last_access(self, fd: int):
        return os.fstat(fd).st_atime

    def file_last_modify(self, fd: int):
        return os.fstat(fd).st_mtime

    def file_count_lines(self, fd: int):
        open_file = os.fdopen(fd)
        return len(open_file.readlines())

    def file_tell(self, fd: int):
        open_file = os.fdopen(fd)
        return open_file.tell()

    # property functions

    def property_info(self, obj: MooObject, prop_name: MOOString):
        prop = obj.get_prop(prop_name)
        if prop is None:
            # need to raise E_PROPNF
            raise RuntimeError(
                f"Property {prop_name} does not exist on object {obj.id}")
        return MOOList([prop.owner, prop.perms])

