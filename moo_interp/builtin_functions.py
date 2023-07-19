import base64
import hashlib
import json
from logging import getLogger
import math
import os
import re
from typing import Union

from .errors import MOOError
from .list import MOOList
from .map import MOOMap
from .string import MOOString
from .moo_ast import compile, run
from .vm import VM, MOOAny

logger = getLogger(__name__)


def to_moo(py_obj: Union[str, int, float, bool, list, dict]) -> MOOAny:
    py_type = type(py_obj)
    if py_type is str:
        return MOOString(py_obj)
    elif py_type is int:
        return py_obj
    elif py_type is float:
        return py_obj
    elif py_type is bool:
        return py_obj
    elif py_type is list:
        return MOOList(*[to_moo(item) for item in py_obj])
    elif py_type is dict:
        return MOOMap(**{key: to_moo(value) for key, value in py_obj.items()})
    else:
        raise TypeError(f"Cannot convert {py_type.__name__} to MOO type.")
    
class FunctionRegistry:
    def __init__(self):
        self.functions = {}  # Stores function_name: function pairs
        self.id_to_function = {}  # Stores function_id: function pairs
        self.function_to_id = {}  # Stores function: function_id pairs
        self.current_id = 0

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


BF_REGISTRY = FunctionRegistry()


@BF_REGISTRY
def to_string(value):
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
        return unparse_error(value)
    elif isinstance(value, bool):
        return "true" if value else "false"
    # elif isinstance(value, MOOAnon):
        # return "*anonymous*"
    else:
        logger.error("TOSTR: Unknown Var type")


@BF_REGISTRY
def tostr(*args):
    return MOOString(" ".join(map(to_string, args)))


@BF_REGISTRY
def toint(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, MOOString):
        try:
            return int(value)
        except ValueError:
            return 0
    elif isinstance(value, float):
        return int(value)
    elif isinstance(value, MOOObj):
        return value.id
    elif isinstance(value, MOOError):
        return unparse_error(value)
    elif isinstance(value, bool):
        return 1 if value else 0
    # elif isinstance(value, MOOAnon):
        # return 0
    else:
        logger.error("TOINT: Unknown Var type")


@BF_REGISTRY
def tofloat(value):
    if isinstance(value, int):
        return float(value)
    elif isinstance(value, MOOString):
        return float(value)
    elif isinstance(value, float):
        return value
    elif isinstance(value, MOOObj):
        return value.id
    elif isinstance(value, MOOError):
        return unparse_error(value)
    elif isinstance(value, bool):
        return 1.0 if value else 0.0
    elif isinstance(value, MOOAnon):
        return 0.0
    else:
        logger.error("TOFLOAT: Unknown Var type")


_min = min


@BF_REGISTRY
def min(*rgs):
    return _min(*args)


_max = max


@BF_REGISTRY
def max(*args):
    return _max(*args)


@BF_REGISTRY
def floor(value):
    return float(math.floor(tofloat(value)))


@BF_REGISTRY
def ceil(value):
    return float(math.ceil(tofloat(value)))


@BF_REGISTRY
def time():
    import time
    return int(time.time())


@BF_REGISTRY
def ftime():
    import time
    return float(time.time())


@BF_REGISTRY
def ctime(value=None):
    import time
    if value is None:
        return MOOString(time.ctime())
    return MOOString(time.ctime(tofloat(value)))


@BF_REGISTRY
def sin(value):
    return math.sin(tofloat(value))


@BF_REGISTRY
def cos(value):
    return math.cos(tofloat(value))


@BF_REGISTRY
def cosh(value):
    return math.cosh(tofloat(value))


@BF_REGISTRY
def distance(l1: MOOList, l2: MOOList) -> float:
    """Return the distance between two lists."""
    if len(l1) != len(l2):
        raise MOOError("distance", "Lists must be the same length")
    return math.sqrt(sum((l1[i] - l2[i]) ** 2 for i in range(len(l1))))


@BF_REGISTRY
def floatstr(x, precision, scientific=False):
    # Capping the precision
    precision = min(precision, 19)

    # Handling the scientific notation
    if scientific:
        return format(x, f".{precision}e")

    # Regular float to string conversion
    return MOOString(format(x, f".{precision}f"))


@BF_REGISTRY
def string_hash(string, algo='SHA256'):
    algo = algo.upper()

    if algo not in ['MD5', 'SHA1', 'SHA256']:
        raise ValueError(
            "Unsupported hash algorithm. Please choose either 'MD5', 'SHA1', or 'SHA256'.")

    hash_object = hashlib.new(algo)
    hash_object.update(string.encode())

    return hash_object.hexdigest()


@BF_REGISTRY
def exp(x):
    return math.exp(tofloat(x))


@BF_REGISTRY
def trunc(x):
    if x < 0:
        return ceil(x)
    else:
        return floor(x)


@BF_REGISTRY
def acos(x):
    return math.acos(tofloat(x))


@BF_REGISTRY
def asin(x):
    return math.asin(tofloat(x))


@BF_REGISTRY
def atan(x):
    return math.atan(tofloat(x))


@BF_REGISTRY
def atan2(y, x):
    return math.atan2(tofloat(y), tofloat(x))


@BF_REGISTRY
def log10(x):
    return math.log10(tofloat(x))


@BF_REGISTRY
def sin(x):
    return math.sin(tofloat(x))


@BF_REGISTRY
def sqrt(x):
    return math.sqrt(tofloat(x))


@BF_REGISTRY
def tan(x):
    return math.tan(tofloat(x))


@BF_REGISTRY
def listappend(list: MOOList, value):
    """Append value to list."""
    list.append(value)
    return list


@BF_REGISTRY
def listdelete(list: MOOList, index: int) -> MOOList:
    """Delete index from list."""
    del list[index]
    return list


@BF_REGISTRY
def listinsert(list: MOOList, index: int, value) -> MOOList:
    """Insert value into list at index."""
    list.insert(index, value)
    return list


@BF_REGISTRY
def listset(list: MOOList, index: int, value) -> MOOList:
    """Set value at index in list."""
    list[index] = value
    return list


@BF_REGISTRY
def all_members(value: MOOAny, list: MOOList):
    """Return all indices of value in list."""
    return MOOList([i for i, x in enumerate(list) if x == value])


@BF_REGISTRY
def explode(string: MOOString, separator: MOOString) -> MOOList:
    """Split string by separator."""
    return MOOList(string.split(separator))


@BF_REGISTRY
def bf_reverse(list: MOOList) -> MOOList:
    """Reverse list."""
    return MOOList(list[::-1])


@BF_REGISTRY
def equal(x, y):
    return x == y


@BF_REGISTRY
def strcmp(str1: MOOString, str2: MOOString):
    if str1 < str2:
        return -1
    elif str1 == str2:
        return 0
    else:
        return 1


@BF_REGISTRY
def strtr(str1: MOOString, str2: MOOString, str3: MOOString, case_matters=False):
    """
        Transforms the string source by replacing the characters specified by str1 with the corresponding characters specified by str2. All other characters are not transformed. If str2 has fewer characters than str1 the unmatched characters are simply removed from source. By default the transformation is done on both upper and lower case characters no matter the case. If case-matters is provided and true, then case is treated as significant.
    """
    if case_matters:
        return str1.translate(str.maketrans(str2, str3))
    else:
        return str1.translate(str.maketrans(str2, str3, str2.upper() + str2.lower()))


_chr = chr


@BF_REGISTRY
def chr(x):
    return _chr(x)


@BF_REGISTRY
def index(str1: MOOString, str2: MOOString):
    return str1.find(str2)


@BF_REGISTRY
def rindex(str1: MOOString, str2: MOOString):
    return str1.rfind(str2)


@BF_REGISTRY
def strsub(str1: MOOString, str2: MOOString, str3: MOOString):
    return str1.replace(str2, str3)


@BF_REGISTRY
def strcmp(str1: MOOString, str2: MOOString):
    if str1 < str2:
        return -1
    elif str1 == str2:
        return 0
    else:
        return 1


_abs = abs


@BF_REGISTRY
def abs(x):
    return _abs((x))


@BF_REGISTRY
def length(x):
    return len(x)


@BF_REGISTRY
def toliteral(x):
    return MOOString(x)


@BF_REGISTRY
def mapkeys(x):
    return MOOList(x.keys())


@BF_REGISTRY
def mapvalues(x):
    return MOOList(x.values())


@BF_REGISTRY
def mapdelete(x, y):
    """Delete key y from map x."""
    del x[y]
    return x


@BF_REGISTRY
def mapinsert(x, y, z):
    x[y] = z
    return x


@BF_REGISTRY
def eval(x):
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
    try:
        compiled = compile(x)
        compiled.debug = True
        compiled.this = -1
        compiled.verb = ""
        result = run(x)
    except Exception as e:
        return MOOList([False, MOOList([e])])
    return MOOList([True, result.result])


@BF_REGISTRY
def encode_base64(x, safe=False):
    if safe:
        return base64.urlsafe_b64encode(x)
    else:
        return base64.b64encode(x)


@BF_REGISTRY
def decode_base64(x, safe=False):
    if safe:
        return base64.urlsafe_b64decode(x)
    else:
        return base64.b64decode(x)


@BF_REGISTRY
def generate_json(x):
    return MOOString(json.dumps(x))


@BF_REGISTRY
def parse_json(x):
    return to_moo(json.loads(x))


ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')


@BF_REGISTRY
def strip_ansi(x):
    return MOOString(ANSI_ESCAPE.sub('', x))


# Fileio functions

##  Note: Need to add permissions checks and further compatibility to these functions

@BF_REGISTRY
def file_version():
    return MOOString("FIO/2.0")

@BF_REGISTRY
def file_open(name: MOOString, mode: MOOString):
    return open(name, mode).fileno()

@BF_REGISTRY
def file_close(fd: int):
    return os.close(fd)


@BF_REGISTRY
def file_readline(fd: int):
    open_file = os.fdopen(fd)
    return MOOString(open_file.readline())


@BF_REGISTRY
def file_readlines(fd: int, start: int, end: int):
    open_file = os.fdopen(fd)
    return MOOList(open_file.readlines()[start:end])


@BF_REGISTRY
def file_writeline(fd: int, line: MOOString):
    open_file = os.fdopen(fd)
    open_file.write(str(line))


@BF_REGISTRY
def file_flush(fd: int):
    open_file = os.fdopen(fd)
    open_file.flush()


@BF_REGISTRY
def file_seek(fd: int, pos: int):
    open_file = os.fdopen(fd)
    open_file.seek(pos)


@BF_REGISTRY
def file_size(fd: int):
    return os.fstat(fd).st_size


@BF_REGISTRY
def file_last_access(fd: int):
    return os.fstat(fd).st_atime


@BF_REGISTRY
def file_last_modify(fd: int):
    return os.fstat(fd).st_mtime

@BF_REGISTRY
def file_count_lines(fd: int):
    open_file = os.fdopen(fd)
    return len(open_file.readlines())

@BF_REGISTRY
def file_tell(fd: int):
    open_file = os.fdopen(fd)
    return open_file.tell()
