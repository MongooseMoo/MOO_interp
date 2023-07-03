import hashlib
from logging import getLogger
import math

from .errors import MOOError
from .list import MOOList
from .map import MOOMap
from .string import MOOString

logger = getLogger(__name__)

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
            raise KeyError(f"Invalid key type. Expected str or int, got {type(key).__name__}")

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
            name = next(name for name, function in self.functions.items() if function == fn)
            del self.functions[name]
            del self.function_to_id[fn]
        else:
            raise KeyError(f"Invalid key type. Expected str or int, got {type(key).__name__}")

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
def mapdelete(map: MOOMap, key: MOOString) -> MOOMap:
    """Delete key from map."""
    del map[key]
    return map

@BF_REGISTRY
def mapinsert(map: MOOMap, key: MOOString, value) -> MOOMap:
    """Insert key/value pair into map."""
    map[key] = value
    return map

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
        return int(value)
    elif isinstance(value, float):
        return int(value)
    elif isinstance(value, MOOObj):
        return value.id
    elif isinstance(value, MOOError):
        return unparse_error(value)
    elif isinstance(value, bool):
        return 1 if value else 0
    #elif isinstance(value, MOOAnon):
        #return 0
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
def ctime(value):
    import time
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
def equal(x, y):
    return x == y


@BF_REGISTRY
def strcmp(str1, str2):
    if str1 < str2:
        return -1
    elif str1 == str2:
        return 0
    else:
        return 1

@BF_REGISTRY
def index(str1, str2):
    return str1.find(str2)
@BF_REGISTRY
def rindex(str1, str2):
    return str1.rfind(str2)

