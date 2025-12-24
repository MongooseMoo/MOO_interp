import base64
from functools import reduce
from passlib.context import CryptContext

# Unix crypt compatible context - handles all standard formats
_crypt_context = CryptContext(schemes=[
    "sha512_crypt", "sha256_crypt", "md5_crypt", "des_crypt", "bcrypt"
], default="sha512_crypt")
import hashlib
import hmac
import json
import math
import os
import random
import re
import sys
import urllib.parse
from logging import getLogger
from typing import Union

from lambdamoo_db.database import MooDatabase, MooObject, ObjNum
from .errors import MOOError, MOOException
from .list import MOOList
from .map import MOOMap
from .string import MOOString
from .moo_types import MOOAny, MOONumber, to_moo

logger = getLogger(__name__)


def _verb_name_matches(search_name: str, verb_pattern: str) -> bool:
    """Check if search_name matches a MOO verb pattern.

    MOO verb patterns use * for minimum abbreviation:
    - "co*nnect" matches "co", "con", "conn", "conne", "connec", "connect"
    - "@co*nnect" is an alias (ignore @ prefix)
    - Space-separated patterns are alternatives

    Returns True if search_name matches any alias in verb_pattern.
    """
    search = search_name.lower()
    for alias in verb_pattern.split():
        # Strip @ prefix (alias marker)
        if alias.startswith('@'):
            alias = alias[1:]
        alias_lower = alias.lower()

        if '*' in alias_lower:
            # Has abbreviation marker
            star_pos = alias_lower.index('*')
            prefix = alias_lower[:star_pos]  # Required minimum
            full = alias_lower.replace('*', '')  # Full name without *

            # Match if search starts with prefix and is a prefix of full
            if search.startswith(prefix) and full.startswith(search):
                return True
        else:
            # Exact match required
            if search == alias_lower:
                return True
    return False


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

        # Register aliases (raise is a Python keyword, so we use raise_error internally)
        # The alias 'raise' should map to the same function object and ID as 'raise_error'
        if 'raise_error' in self.functions:
            func = self.functions['raise_error']
            self.functions['raise'] = func
            # Both names now point to the same function, so get_id_by_name will return the same ID

    def __call__(self, fn):
        if self.current_id > 255:
            raise Exception("Cannot register more than 256 functions.")
        function_id = self.current_id
        self.current_id += 1
        self.functions[fn.__name__] = fn
        self.id_to_function[function_id] = fn
        self.function_to_id[fn] = function_id
        return fn


    def register(self, name: str, func):
        """Register an external function with a custom name.
        
        Args:
            name: The name to register the function under
            func: A callable that will be registered as a builtin
        """
        if self.current_id > 255:
            raise Exception("Cannot register more than 256 functions.")
        
        # Create a wrapper to ensure the function has the right __name__
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__name__ = name
        
        function_id = self.current_id
        self.current_id += 1
        self.functions[name] = wrapper
        self.id_to_function[function_id] = wrapper
        self.function_to_id[wrapper] = function_id
        return wrapper

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

    def get_function_name_by_id(self, id):
        func = self.id_to_function.get(id)
        return func.__name__ if func else f"<unknown_id_{id}>"

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

    def callers(self, include_line_numbers: int = 0):
        """Return the call stack as a list of {player, this, verb, programmer, line}.

        For top-level server calls, returns an empty list.
        Args:
            include_line_numbers: If true, include line numbers (not implemented yet)
        Returns:
            MOOList of call stack entries
        """
        # TODO: Implement full call stack tracking
        # For now, return empty list (correct for top-level server calls)
        return MOOList([])

    def caller_perms(self):
        """Return the object whose permissions apply to the current verb call.

        For top-level server calls from #0, returns #0 (which is typically wizard).
        Returns:
            Object ID whose permissions are in effect
        """
        # For top-level server calls, the permissions come from the object
        # the verb is defined on. Since do_login_command is on #0, return #0.
        # TODO: Track actual caller permissions through the call stack
        return 0  # #0 is typically wizard

    def task_perms(self):
        """Return the object whose permissions are in effect for the current task.

        This is the programmer identity for permission checks.
        Returns:
            Object ID of the current task's programmer
        """
        # TODO: Track actual task permissions through the VM
        return 0  # #0 is typically wizard

    def set_task_perms(self, who: int) -> None:
        """Set the programmer identity for the current task.

        This changes whose permissions are used for subsequent operations.
        Args:
            who: Object ID to set as the task's programmer
        """
        # TODO: Actually modify the VM's current frame programmer
        # For now, just accept and ignore (allows code to proceed)
        pass

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

    def listappend(self, list: MOOList, value, position: int = None) -> MOOList:
        """Append value to list, optionally after a specific position.

        listappend(list, value [, position])
        - If position is not provided, append at end
        - If position is provided, insert AFTER that position (1-based)
        """
        if position is None:
            list.append(value)
        else:
            # Insert after the given position (MOO is 1-based)
            # position=1 means insert after index 0, so at index 1
            list._list.insert(position, value)
        return list

    def listdelete(self, list: MOOList, index: int) -> MOOList:
        """Delete index from list."""
        del list[index]
        return list

    def listinsert(self, list: MOOList, value, position: int = None) -> MOOList:
        """Insert value into list before a specific position.

        listinsert(list, value [, position])
        - VALUE comes before POSITION in MOO
        - If position is not provided, insert at beginning (position 1)
        - If position is provided, insert BEFORE that position (1-based)
        """
        if position is None:
            position = 1
        # Insert before the given position (MOO is 1-based)
        # position=1 means insert at index 0
        list._list.insert(position - 1, value)
        return list

    def listset(self, list: MOOList, value, index: int) -> MOOList:
        """Set value at index in list.

        listset(list, value, index)
        - VALUE comes before INDEX in MOO
        - Index is 1-based
        """
        list[index] = value
        return list

    def all_members(self, value: MOOAny, list: MOOList):
        """Return all indices of value in list."""
        return MOOList([i for i, x in enumerate(list) if x == value])

    def explode(self, string: MOOString, separator: MOOString) -> MOOList:
        """Split string by separator."""
        return MOOList(string.split(separator))

    def reverse(self, lst: MOOList) -> MOOList:
        """Reverse a list or string."""
        if isinstance(lst, (str, MOOString)):
            return MOOString(str(lst)[::-1])
        return MOOList(list(lst)[::-1])

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
        # Unwrap MOOString to native str for maketrans
        s1 = str(str1)
        from_chars = str(str2)
        to_chars = str(str3)

        # Build translation table
        # If from is longer than to, extra chars should be deleted
        trans_from = []
        trans_to = []
        trans_delete = []

        for i, c in enumerate(from_chars):
            if i < len(to_chars):
                # Map from[i] to to[i]
                if case_matters or not c.isalpha():
                    # Case-sensitive or non-alphabetic: direct mapping
                    trans_from.append(c)
                    trans_to.append(to_chars[i])
                else:
                    # Case-insensitive alphabetic: map both upper and lower
                    trans_from.append(c.upper())
                    trans_to.append(to_chars[i].upper())
                    trans_from.append(c.lower())
                    trans_to.append(to_chars[i].lower())
            else:
                # from is longer than to, delete this char
                if case_matters or not c.isalpha():
                    trans_delete.append(c)
                else:
                    trans_delete.append(c.upper())
                    trans_delete.append(c.lower())

        table = str.maketrans(
            ''.join(trans_from),
            ''.join(trans_to),
            ''.join(trans_delete)
        )
        result = s1.translate(table)
        # Return MOOString, not native str
        return MOOString(result)

    _chr = chr

    def chr(self, x):
        return self._chr(x)

    def index(self, str1: MOOString, str2: MOOString, case_matters: int = 0):
        """Return 1-based index of str2 in str1, or 0 if not found.

        MOO index() returns 0 for not found, and 1-based positions.
        """
        s1 = str(str1)
        s2 = str(str2)
        if not case_matters:
            s1 = s1.lower()
            s2 = s2.lower()
        pos = s1.find(s2)
        return pos + 1 if pos >= 0 else 0

    def rindex(self, str1: MOOString, str2: MOOString, case_matters: int = 0):
        """Return 1-based index of last occurrence of str2 in str1, or 0 if not found.

        MOO rindex() returns 0 for not found, and 1-based positions.
        """
        s1 = str(str1)
        s2 = str(str2)
        if not case_matters:
            s1 = s1.lower()
            s2 = s2.lower()
        pos = s1.rfind(s2)
        return pos + 1 if pos >= 0 else 0

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
        return MOOList(list(x.keys()))

    def mapvalues(self, x):
        return MOOList(list(x.values()))

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

    def _moo_to_python(self, value):
        """Convert MOO types to native Python for JSON serialization."""
        # Handle MOOString - UserString stores string in .data attribute
        if isinstance(value, MOOString):
            return value.data

        # Handle MOOList - iterate and convert elements
        elif isinstance(value, MOOList):
            return [self._moo_to_python(v) for v in value]

        # Handle MOOMap - convert keys and values
        elif isinstance(value, MOOMap):
            return {self._moo_to_python(k): self._moo_to_python(v) for k, v in value.items()}

        # Handle ObjNum - convert to integer
        elif isinstance(value, ObjNum):
            return int(value)

        # Handle MOOError/MOOException - convert to string
        elif isinstance(value, (MOOError, MOOException)):
            return str(value)

        # Pass through native Python types (int, float, str, bool, None)
        return value

    def generate_json(self, x, mode: MOOString = None):
        """Generate JSON from a MOO value.

        Args:
            x: MOO value to convert to JSON
            mode: Optional mode - "common-subset" (default) or "embedded-types"

        Returns:
            MOOString containing JSON representation
        """
        # For now, both modes behave the same - just convert to JSON
        # In full implementation, "embedded-types" would include type annotations
        # like {"_type": "OBJ", "value": 123}
        native = self._moo_to_python(x)
        return MOOString(json.dumps(native))

    def parse_json(self, x, mode: MOOString = None):
        """Parse JSON string into MOO value.

        Args:
            x: JSON string to parse
            mode: Optional mode - "common-subset" (default) or "embedded-types"

        Returns:
            MOO value parsed from JSON
        """
        # For now, both modes behave the same - just parse JSON
        # In full implementation, "embedded-types" would recognize type annotations
        # like {"_type": "OBJ", "value": 123} and convert to ObjNum(123)
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

    # Miscellaneous functions

    def random(self, x: int):
        return random.randint(0, x)

    def frandom(self) -> float:
        return random.random()

    # =========================================================================
    # NEW BUILTINS - Math functions
    # =========================================================================

    _round = round

    def round(self, x):
        """Round to nearest integer."""
        return float(self._round(self.tofloat(x)))

    def cbrt(self, x):
        """Cube root."""
        val = self.tofloat(x)
        return math.copysign(abs(val) ** (1/3), val)

    def log(self, x):
        """Natural logarithm."""
        return math.log(self.tofloat(x))

    def sinh(self, x):
        """Hyperbolic sine."""
        return math.sinh(self.tofloat(x))

    def tanh(self, x):
        """Hyperbolic tangent."""
        return math.tanh(self.tofloat(x))

    def asinh(self, x):
        """Inverse hyperbolic sine."""
        return math.asinh(self.tofloat(x))

    def acosh(self, x):
        """Inverse hyperbolic cosine."""
        return math.acosh(self.tofloat(x))

    def atanh(self, x):
        """Inverse hyperbolic tangent."""
        return math.atanh(self.tofloat(x))

    def random_bytes(self, n: int) -> MOOString:
        """Generate n cryptographically secure random bytes as hex string."""
        return MOOString(os.urandom(n).hex())

    def reseed_random(self):
        """Reseed the random number generator."""
        random.seed()
        return 0

    def relative_heading(self, p1: MOOList, p2: MOOList) -> float:
        """Calculate heading from p1 to p2 in degrees (0-360, north=0)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return (90 - angle) % 360

    # =========================================================================
    # Type/Utility functions
    # =========================================================================

    # MOO type codes (toaststunt compatible)
    TYPE_INT = 0
    TYPE_OBJ = 1
    TYPE_STR = 2
    TYPE_ERR = 3
    TYPE_LIST = 4
    TYPE_FLOAT = 9
    TYPE_MAP = 10
    TYPE_BOOL = 14

    def typeof(self, x) -> int:
        """Return the MOO type code for a value."""
        if isinstance(x, bool):
            return self.TYPE_BOOL
        elif isinstance(x, ObjNum):
            # ObjNum must be checked before int since it inherits from int
            return self.TYPE_OBJ
        elif isinstance(x, int):
            return self.TYPE_INT
        elif isinstance(x, float):
            return self.TYPE_FLOAT
        elif isinstance(x, (str, MOOString)):
            return self.TYPE_STR
        elif isinstance(x, (list, MOOList)):
            return self.TYPE_LIST
        elif isinstance(x, (dict, MOOMap)):
            return self.TYPE_MAP
        elif isinstance(x, MooObject):
            return self.TYPE_OBJ
        elif isinstance(x, MOOError):
            return self.TYPE_ERR
        else:
            return self.TYPE_INT  # fallback

    def raise_error(self, code, message=None, value=None):
        """Raise an error that can be caught by try/except.

        raise(code [, message [, value]])

        Args:
            code: Error code (E_TYPE, E_PERM, etc.)
            message: Optional error message string
            value: Optional value to include in the error
        """
        if message is None:
            if isinstance(code, MOOError):
                message = code.name
            else:
                message = str(code)
        raise MOOException(code, str(message))

    def ancestors(self, obj, full=False):
        """Return the ancestors of an object.

        ancestors(obj [, full])

        Args:
            obj: Object ID or ObjNum to get ancestors of
            full: If true, return full tree including duplicates

        Returns:
            MOOList of ancestor object IDs
        """
        if self.db is None:
            return MOOList([])

        # Get object ID - handle various types
        if isinstance(obj, ObjNum):
            obj_id = int(str(obj).lstrip('#'))
        elif isinstance(obj, int):
            obj_id = obj
        elif isinstance(obj, (str, MOOString)):
            # Handle string representations like "#10" or "10"
            s = str(obj).strip().lstrip('#')
            try:
                obj_id = int(s)
            except ValueError:
                raise MOOException(MOOError.E_TYPE, f"ancestors() requires an object, got string: {obj}")
        else:
            raise MOOException(MOOError.E_TYPE, "ancestors() requires an object")

        # Check if object exists
        if obj_id not in self.db.objects:
            raise MOOException(MOOError.E_INVARG, f"Invalid object: #{obj_id}")

        ancestors = []
        visited = set()
        to_visit = []

        # Start with direct parents
        db_obj = self.db.objects[obj_id]
        if hasattr(db_obj, 'parents') and db_obj.parents:
            to_visit.extend(db_obj.parents)
        elif hasattr(db_obj, 'parent') and db_obj.parent is not None and db_obj.parent >= 0:
            to_visit.append(db_obj.parent)

        while to_visit:
            parent_id = to_visit.pop(0)
            if parent_id < 0:
                continue
            if not full and parent_id in visited:
                continue
            visited.add(parent_id)
            ancestors.append(ObjNum(parent_id))

            if parent_id in self.db.objects:
                parent_obj = self.db.objects[parent_id]
                if hasattr(parent_obj, 'parents') and parent_obj.parents:
                    to_visit.extend(parent_obj.parents)
                elif hasattr(parent_obj, 'parent') and parent_obj.parent is not None and parent_obj.parent >= 0:
                    to_visit.append(parent_obj.parent)

        return MOOList(ancestors)

    def verbs(self, obj):
        """Return list of verb names defined on an object.

        verbs(obj) => list of strings
        """
        if self.db is None:
            return MOOList([])

        # Get object ID
        if isinstance(obj, ObjNum):
            obj_id = int(str(obj).lstrip('#'))
        elif isinstance(obj, int):
            obj_id = obj
        elif isinstance(obj, (str, MOOString)):
            s = str(obj).strip().lstrip('#')
            try:
                obj_id = int(s)
            except ValueError:
                raise MOOException(MOOError.E_TYPE, f"verbs() requires an object")
        else:
            raise MOOException(MOOError.E_TYPE, "verbs() requires an object")

        if obj_id not in self.db.objects:
            raise MOOException(MOOError.E_INVARG, f"Invalid object: #{obj_id}")

        db_obj = self.db.objects[obj_id]
        verb_names = []
        for v in getattr(db_obj, 'verbs', []):
            name = getattr(v, 'name', '')
            verb_names.append(MOOString(name))
        return MOOList(verb_names)

    def verb_info(self, obj, verb_desc):
        """Return info about a verb: {owner, perms, names}.

        verb_info(obj, verb-desc) => {owner, perms, names}
        """
        if self.db is None:
            return MOOList([])

        # Get object ID
        if isinstance(obj, ObjNum):
            obj_id = int(str(obj).lstrip('#'))
        elif isinstance(obj, int):
            obj_id = obj
        else:
            raise MOOException(MOOError.E_TYPE, "verb_info() requires an object")

        if obj_id not in self.db.objects:
            raise MOOException(MOOError.E_INVARG, f"Invalid object: #{obj_id}")

        db_obj = self.db.objects[obj_id]
        verbs = getattr(db_obj, 'verbs', [])

        # verb_desc can be a name (string) or index (int)
        verb = None
        if isinstance(verb_desc, int):
            idx = verb_desc - 1  # MOO is 1-indexed
            if 0 <= idx < len(verbs):
                verb = verbs[idx]
        else:
            verb_name = str(verb_desc)
            for v in verbs:
                if _verb_name_matches(verb_name, getattr(v, 'name', '')):
                    verb = v
                    break

        if verb is None:
            raise MOOException(MOOError.E_VERBNF, f"Verb not found")

        owner = ObjNum(getattr(verb, 'owner', -1))
        perms = MOOString(getattr(verb, 'perms', 'rd'))
        names = MOOString(getattr(verb, 'name', ''))
        return MOOList([owner, perms, names])

    def verb_args(self, obj, verb_desc):
        """Return argument spec for a verb: {dobj, prep, iobj}.

        verb_args(obj, verb-desc) => {dobj, prep, iobj}
        """
        if self.db is None:
            return MOOList([])

        # Get object ID
        if isinstance(obj, ObjNum):
            obj_id = int(str(obj).lstrip('#'))
        elif isinstance(obj, int):
            obj_id = obj
        else:
            raise MOOException(MOOError.E_TYPE, "verb_args() requires an object")

        if obj_id not in self.db.objects:
            raise MOOException(MOOError.E_INVARG, f"Invalid object: #{obj_id}")

        db_obj = self.db.objects[obj_id]
        verbs = getattr(db_obj, 'verbs', [])

        # verb_desc can be a name (string) or index (int)
        verb = None
        if isinstance(verb_desc, int):
            idx = verb_desc - 1  # MOO is 1-indexed
            if 0 <= idx < len(verbs):
                verb = verbs[idx]
        else:
            verb_name = str(verb_desc)
            for v in verbs:
                if _verb_name_matches(verb_name, getattr(v, 'name', '')):
                    verb = v
                    break

        if verb is None:
            raise MOOException(MOOError.E_VERBNF, f"Verb not found")

        # Extract arg specs from packed perms field
        # Encoding: bits 0-3 = permissions, bits 4-5 = dobj, bits 6-7 = iobj
        DOBJSHIFT = 4
        IOBJSHIFT = 6
        OBJMASK = 0x3

        perms = getattr(verb, 'perms', 0)
        dobj_spec = (perms >> DOBJSHIFT) & OBJMASK
        iobj_spec = (perms >> IOBJSHIFT) & OBJMASK

        # Map arg spec values to strings (0=none, 1=any, 2=this)
        arg_spec_map = {0: 'none', 1: 'any', 2: 'this'}
        dobj = arg_spec_map.get(dobj_spec, 'none')
        iobj = arg_spec_map.get(iobj_spec, 'none')

        # prep is in the separate preps field: -2=any, -1=none, 0+=index
        prep_value = getattr(verb, 'preps', -1)
        if prep_value == -2:
            prep = 'any'
        elif prep_value == -1:
            prep = 'none'
        else:
            # TODO: Look up prep string from index
            prep = 'none'

        return MOOList([MOOString(dobj), MOOString(prep), MOOString(iobj)])

    def toobj(self, x):
        """Convert a value to an object reference."""
        if isinstance(x, MooObject):
            return x
        elif isinstance(x, int):
            return x  # Return int as object number for now
        elif isinstance(x, (str, MOOString)):
            s = str(x).strip()
            if s.startswith('#'):
                s = s[1:]
            try:
                return int(s)
            except ValueError:
                return -1
        elif isinstance(x, float):
            return int(x)
        else:
            return -1

    # =========================================================================
    # Object manipulation builtins
    # =========================================================================

    def parent(self, obj):
        """Return the parent of an object.

        parent(obj) => objnum or E_INVARG

        Note: In MOO, parent() returns a single object. With multiple inheritance,
        this returns the first parent in the parents list.
        """
        if self.db is None:
            raise MOOException(MOOError.E_INVARG, "No database available")

        # Get object ID
        if isinstance(obj, ObjNum):
            obj_id = int(str(obj).lstrip('#'))
        elif isinstance(obj, int):
            obj_id = obj
        else:
            raise MOOException(MOOError.E_TYPE, "parent() requires an object")

        if obj_id not in self.db.objects:
            raise MOOException(MOOError.E_INVARG, f"Invalid object: #{obj_id}")

        db_obj = self.db.objects[obj_id]
        # Check for both 'parent' (old single-inheritance) and 'parents' (new multi-inheritance)
        if hasattr(db_obj, 'parents') and db_obj.parents:
            parent_id = db_obj.parents[0] if db_obj.parents else -1
        elif hasattr(db_obj, 'parent'):
            parent_id = db_obj.parent
            if parent_id is None:
                parent_id = -1
        else:
            parent_id = -1
        return ObjNum(parent_id)

    def children(self, obj):
        """Return the list of children of an object.

        children(obj) => list of objnums
        """
        if self.db is None:
            return MOOList([])

        # Get object ID
        if isinstance(obj, ObjNum):
            obj_id = int(str(obj).lstrip('#'))
        elif isinstance(obj, int):
            obj_id = obj
        else:
            raise MOOException(MOOError.E_TYPE, "children() requires an object")

        if obj_id not in self.db.objects:
            raise MOOException(MOOError.E_INVARG, f"Invalid object: #{obj_id}")

        # Find all objects that have this object as a parent
        children = []
        for oid, child_obj in self.db.objects.items():
            # Check both 'parent' (old) and 'parents' (new) attributes
            if hasattr(child_obj, 'parents') and child_obj.parents:
                if obj_id in child_obj.parents:
                    children.append(ObjNum(oid))
            elif hasattr(child_obj, 'parent'):
                parent_id = child_obj.parent
                if parent_id == obj_id:
                    children.append(ObjNum(oid))

        return MOOList(children)

    def valid(self, obj):
        """Check if an object exists in the database.

        valid(obj) => 1 if valid, 0 otherwise
        """
        if self.db is None:
            return 0

        # Get object ID
        if isinstance(obj, ObjNum):
            obj_id = int(str(obj).lstrip('#'))
        elif isinstance(obj, int):
            obj_id = obj
        else:
            return 0  # Non-object types are not valid objects

        return 1 if obj_id in self.db.objects else 0

    def create(self, parent, owner=None):
        """Create a new object with the given parent.

        create(parent [, owner]) => new object number

        Note: This is a simplified implementation that creates the object
        but doesn't persist it. For testing purposes only.
        """
        if self.db is None:
            raise MOOException(MOOError.E_INVARG, "No database available")

        # Get parent ID
        if isinstance(parent, ObjNum):
            parent_id = int(str(parent).lstrip('#'))
        elif isinstance(parent, int):
            parent_id = parent
        else:
            raise MOOException(MOOError.E_TYPE, "create() requires an object as parent")

        # Validate parent exists
        if parent_id not in self.db.objects:
            raise MOOException(MOOError.E_INVARG, f"Invalid parent object: #{parent_id}")

        # Determine owner (default to parent's owner or #2)
        if owner is None:
            parent_obj = self.db.objects[parent_id]
            owner_id = getattr(parent_obj, 'owner', 2)
        elif isinstance(owner, ObjNum):
            owner_id = int(str(owner).lstrip('#'))
        elif isinstance(owner, int):
            owner_id = owner
        else:
            raise MOOException(MOOError.E_TYPE, "create() owner must be an object")

        # Create new object - find next available ID
        new_id = max(self.db.objects.keys()) + 1 if self.db.objects else 0

        # Create minimal object using MooObject constructor
        # Note: MooObject uses 'parents' (list) not 'parent' (single)
        new_obj = MooObject(
            id=new_id,
            name=MOOString(""),
            flags=0,
            owner=owner_id,
            location=-1,
            parents=[parent_id],
        )

        # Add to database
        self.db.objects[new_id] = new_obj

        return ObjNum(new_id)

    def is_player(self, obj):
        """Check if an object is a player (has player flag set).

        is_player(obj) => 1 if player, 0 otherwise
        """
        if self.db is None:
            return 0

        # Get object ID
        if isinstance(obj, ObjNum):
            obj_id = int(str(obj).lstrip('#'))
        elif isinstance(obj, int):
            obj_id = obj
        else:
            return 0

        if obj_id not in self.db.objects:
            return 0

        db_obj = self.db.objects[obj_id]
        # Check for player flag (bit 3 = 0x08 in flags field)
        flags = getattr(db_obj, 'flags', 0)
        return 1 if (flags & 0x08) else 0

    def maphaskey(self, m: MOOMap, key, case_matters: int = 1) -> int:
        """Check if a map contains a key. Returns 1 if found, 0 otherwise."""
        if case_matters or not isinstance(key, (str, MOOString)):
            return 1 if key in m else 0
        # Case-insensitive string key search
        key_lower = str(key).lower()
        return 1 if any(str(k).lower() == key_lower for k in m.keys()) else 0

    def is_member(self, val, lst: MOOList, case_matters: int = 1) -> int:
        """Return 1-based index of val in lst, or 0 if not found."""
        if case_matters or not isinstance(val, (str, MOOString)):
            try:
                return lst._list.index(val) + 1  # MOO uses 1-based indexing
            except (ValueError, AttributeError):
                try:
                    return list(lst).index(val) + 1
                except ValueError:
                    return 0
        # Case-insensitive string search
        val_lower = str(val).lower()
        for i, item in enumerate(lst):
            if isinstance(item, (str, MOOString)) and str(item).lower() == val_lower:
                return i + 1
        return 0

    def value_bytes(self, x) -> int:
        """Return approximate memory size of a value in bytes."""
        return sys.getsizeof(x)

    # =========================================================================
    # List/Set functions
    # =========================================================================

    def setadd(self, lst: MOOList, val) -> MOOList:
        """Add val to lst if not already present. Returns new list."""
        result = MOOList(list(lst))
        if val not in result:
            result.append(val)
        return result

    def setremove(self, lst: MOOList, val) -> MOOList:
        """Remove first occurrence of val from lst. Returns new list."""
        result = list(lst)
        try:
            result.remove(val)
        except ValueError:
            pass  # Not found, return unchanged
        return MOOList(result)

    _slice = slice  # Save builtin

    def slice(self, lst: MOOList, index: int, length: int = None) -> MOOList:
        """Extract a sublist starting at index (1-based) with optional length."""
        start = index - 1 if index > 0 else index
        if length is None:
            return MOOList(list(lst)[start:])
        return MOOList(list(lst)[start:start + length])

    # =========================================================================
    # Crypto/HMAC functions
    # =========================================================================

    def _get_hash_algo(self, algo: str):
        """Get hashlib algorithm by name."""
        algo = str(algo).upper()
        algos = {'MD5': 'md5', 'SHA1': 'sha1', 'SHA256': 'sha256',
                 'SHA512': 'sha512', 'SHA224': 'sha224', 'SHA384': 'sha384'}
        return algos.get(algo, 'sha256')

    def binary_hash(self, data, algo: str = 'SHA256', binary: int = 0) -> MOOString:
        """Hash binary/string data. Returns hex unless binary=1."""
        h = hashlib.new(self._get_hash_algo(algo))
        h.update(str(data).encode('latin-1'))
        return MOOString(h.digest() if binary else h.hexdigest())

    def value_hash(self, val, algo: str = 'SHA256', binary: int = 0) -> MOOString:
        """Hash any MOO value by converting to literal first."""
        literal = str(self.toliteral(val))
        h = hashlib.new(self._get_hash_algo(algo))
        h.update(literal.encode('utf-8'))
        return MOOString(h.digest() if binary else h.hexdigest())

    def string_hmac(self, data, key, algo: str = 'SHA256', binary: int = 0) -> MOOString:
        """Compute HMAC of string data."""
        h = hmac.new(str(key).encode('utf-8'), str(data).encode('utf-8'),
                     self._get_hash_algo(algo))
        return MOOString(h.digest() if binary else h.hexdigest())

    def binary_hmac(self, data, key, algo: str = 'SHA256', binary: int = 0) -> MOOString:
        """Compute HMAC of binary data."""
        h = hmac.new(str(key).encode('latin-1'), str(data).encode('latin-1'),
                     self._get_hash_algo(algo))
        return MOOString(h.digest() if binary else h.hexdigest())

    def value_hmac(self, val, key, algo: str = 'SHA256', binary: int = 0) -> MOOString:
        """Compute HMAC of any MOO value."""
        literal = str(self.toliteral(val))
        h = hmac.new(str(key).encode('utf-8'), literal.encode('utf-8'),
                     self._get_hash_algo(algo))
        return MOOString(h.digest() if binary else h.hexdigest())

    # =========================================================================
    # Regex functions
    # =========================================================================

    def match(self, subject: MOOString, pattern: MOOString, case_matters: int = 0) -> MOOList:
        """
        Match pattern against subject. Returns {start, end, replacements, subject}
        where start/end are 1-based indices, or empty list if no match.
        """
        flags = 0 if case_matters else re.IGNORECASE
        try:
            m = re.search(str(pattern), str(subject), flags)
            if not m:
                return MOOList([])
            # Build replacements list from groups
            replacements = MOOList([
                MOOList([m.start(i) + 1, m.end(i)])
                for i in range(1, m.lastindex + 1)
            ] if m.lastindex else [])
            return MOOList([m.start() + 1, m.end(), replacements, MOOString(subject)])
        except re.error:
            return MOOList([])

    def rmatch(self, subject: MOOString, pattern: MOOString, case_matters: int = 0) -> MOOList:
        """Match pattern from end of subject."""
        flags = 0 if case_matters else re.IGNORECASE
        try:
            matches = list(re.finditer(str(pattern), str(subject), flags))
            if not matches:
                return MOOList([])
            m = matches[-1]
            replacements = MOOList([
                MOOList([m.start(i) + 1, m.end(i)])
                for i in range(1, m.lastindex + 1)
            ] if m.lastindex else [])
            return MOOList([m.start() + 1, m.end(), replacements, MOOString(subject)])
        except re.error:
            return MOOList([])

    def substitute(self, template: MOOString, subs: MOOList) -> MOOString:
        """
        Apply substitutions from a match result.
        %1-%9 replaced with captured groups, %% is literal %.
        """
        result = str(template)
        if len(subs) >= 4:
            subject = str(subs[3])
            repls = subs[2] if len(subs) > 2 else MOOList([])
            for i, repl in enumerate(repls):
                if isinstance(repl, (list, MOOList)) and len(repl) >= 2:
                    start, end = int(repl[0]) - 1, int(repl[1])
                    if start >= 0:
                        result = result.replace(f'%{i+1}', subject[start:end])
        return MOOString(result.replace('%%', '%'))

    def pcre_match(self, subject: MOOString, pattern: MOOString,
                   options: int = 0, offset: int = 0) -> MOOList:
        """PCRE-style regex match."""
        flags = 0
        if options & 1:
            flags |= re.IGNORECASE
        if options & 2:
            flags |= re.MULTILINE
        if options & 4:
            flags |= re.DOTALL
        try:
            m = re.search(str(pattern), str(subject)[offset:], flags)
            if not m:
                return MOOList([])
            groups = MOOList([MOOString(g or "") for g in m.groups()])
            return MOOList([
                MOOString(m.group(0)),
                MOOList([m.start() + offset + 1, m.end() + offset]),
                groups
            ])
        except re.error:
            return MOOList([])

    def pcre_replace(self, subject: MOOString, pattern: MOOString) -> MOOString:
        """PCRE-style replace. Pattern: s/pattern/replacement/flags"""
        s = str(pattern)
        if not s.startswith('s/'):
            return subject
        parts = s[2:].split('/')
        if len(parts) < 2:
            return subject
        pat, repl = parts[0], parts[1]
        flags_str = parts[2] if len(parts) > 2 else ""
        flags = re.IGNORECASE if 'i' in flags_str else 0
        count = 0 if 'g' in flags_str else 1
        try:
            return MOOString(re.sub(pat, repl, str(subject), count=count, flags=flags))
        except re.error:
            return subject

    # =========================================================================
    # Enhanced sort
    # =========================================================================

    def _natural_key(self, s):
        """Key function for natural sorting (handles numbers in strings)."""
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', str(s))]

    def sort(self, lst: MOOList, keys: MOOList = None, natural: int = 0,
             reverse: int = 0) -> MOOList:
        """
        Sort a list with optional key extraction, natural sort, and reverse.
        keys: 1-based indices to use as sort keys for nested lists
        natural: sort strings naturally ("a2" < "a10")
        reverse: descending order
        """
        items = list(lst)

        def get_key(item):
            if keys:
                if isinstance(item, (list, MOOList)):
                    extracted = tuple(
                        item[k - 1] if 0 < k <= len(item) else None
                        for k in keys
                    )
                else:
                    extracted = (item,)
                if natural:
                    return tuple(self._natural_key(v) for v in extracted)
                return extracted
            elif natural:
                return self._natural_key(item)
            return item

        try:
            sorted_items = sorted(items, key=get_key, reverse=bool(reverse))
        except TypeError:
            sorted_items = sorted(items, key=lambda x: str(x), reverse=bool(reverse))

        return MOOList(sorted_items)

    # =========================================================================
    # Binary encode/decode
    # =========================================================================

    def encode_binary(self, *args) -> MOOString:
        """
        Encode values to MOO binary string format using ~XX escapes.

        Format: printable chars (except ~) stay as-is, others become ~XX.
        Tilde (0x7e) is treated as non-printable and becomes ~7E.

        Args can be: int (0-255), string, or list of ints/strings.
        """
        result = []

        def encode_bytes(data):
            """Encode byte data to binary string format."""
            for byte_val in data:
                if isinstance(byte_val, int):
                    b = byte_val & 0xFF
                else:
                    # String - encode each char
                    for char in str(byte_val):
                        b = ord(char)
                        if 33 <= b <= 126 and b != 0x7e:  # printable except ~
                            result.append(chr(b))
                        else:
                            result.append(f'~{b:02X}')
                    continue

                # Handle integer byte value
                if 32 <= b <= 126 and b != 0x7e:  # printable (space + graph, except ~)
                    result.append(chr(b))
                else:
                    result.append(f'~{b:02X}')

        # Process arguments
        for arg in args:
            if isinstance(arg, (list, MOOList)):
                encode_bytes(arg)
            elif isinstance(arg, (str, MOOString)):
                encode_bytes([arg])
            elif isinstance(arg, int):
                encode_bytes([arg])

        return MOOString(''.join(result))

    def decode_binary(self, data, fully: int = 0) -> MOOList:
        """
        Decode MOO binary string format (~XX escapes) to list.

        Format: ~XX = hex byte, ~~ = literal tilde, else literal char

        If fully=1: returns list of integers (all bytes)
        If fully=0: groups printable chars into strings, non-printable as ints
        """
        s = str(data)
        raw_bytes = []
        i = 0

        # First pass: decode ~XX escapes to raw bytes
        while i < len(s):
            if s[i] == '~' and i + 2 < len(s):
                hex_chars = s[i+1:i+3].upper()
                if all(c in '0123456789ABCDEF' for c in hex_chars):
                    # Valid ~XX escape
                    byte_val = int(hex_chars, 16)
                    raw_bytes.append(byte_val)
                    i += 3
                    continue
            # Regular character (or invalid escape - treat as literal)
            raw_bytes.append(ord(s[i]))
            i += 1

        if fully:
            # Return all as integers
            return MOOList(raw_bytes)

        # Group consecutive printable chars into strings
        result = []
        current_string = []

        for byte_val in raw_bytes:
            # Check if printable (space, tab, or graphic char)
            if byte_val == 32 or byte_val == 9 or (33 <= byte_val <= 126):
                current_string.append(chr(byte_val))
            else:
                # Non-printable: flush string if any, add integer
                if current_string:
                    result.append(MOOString(''.join(current_string)))
                    current_string = []
                result.append(byte_val)

        # Flush remaining string
        if current_string:
            result.append(MOOString(''.join(current_string)))

        return MOOList(result)

    # =========================================================================
    # URL encoding
    # =========================================================================

    def url_encode(self, s: MOOString) -> MOOString:
        """URL-encode a string."""
        return MOOString(urllib.parse.quote(str(s), safe=''))

    def url_decode(self, s: MOOString) -> MOOString:
        """URL-decode a string."""
        return MOOString(urllib.parse.unquote(str(s)))

    # =========================================================================
    # Password hashing (crypt) - Cross-platform, toaststunt-compatible
    # =========================================================================

    def salt(self, method: MOOString = "SHA512", prefix: MOOString = "") -> MOOString:
        """Generate a salt for use with crypt().

        Cross-platform, toaststunt-compatible using passlib.
        Methods: DES, MD5, SHA256, SHA512, BLOWFISH/BCRYPT
        """
        from passlib.hash import sha512_crypt, sha256_crypt, md5_crypt, des_crypt, bcrypt
        method = str(method).upper()

        handlers = {
            'SHA512': sha512_crypt,
            'SHA256': sha256_crypt,
            'MD5': md5_crypt,
            'DES': des_crypt,
            'BLOWFISH': bcrypt,
            'BCRYPT': bcrypt,
        }
        handler = handlers.get(method, sha512_crypt)

        # Generate proper salt using handler's genconfig
        if hasattr(handler, 'genconfig'):
            return MOOString(handler.genconfig())
        else:
            # Fallback: generate hash and extract salt portion
            dummy = handler.hash("")
            if method == 'DES':
                return MOOString(dummy[:2])
            parts = dummy.rsplit('$', 1)
            return MOOString(parts[0] + '$')

    def crypt(self, password: MOOString, salt_str: MOOString = None) -> MOOString:
        """Hash a password using Unix crypt-style hashing.

        Cross-platform, toaststunt-compatible. When called with an existing
        hash as salt, produces the same hash if password matches - this is
        how password verification works in MOO.
        """
        from passlib.hash import sha512_crypt, sha256_crypt, md5_crypt, des_crypt, bcrypt
        password = str(password)

        if salt_str is None:
            # Default: generate new SHA512 hash
            return MOOString(sha512_crypt.hash(password))

        salt_str = str(salt_str)

        # Identify hash type from prefix and use appropriate handler
        # The handler.hash(password, salt=existing_hash) re-hashes with same salt
        if salt_str.startswith('$6$'):
            handler = sha512_crypt
        elif salt_str.startswith('$5$'):
            handler = sha256_crypt
        elif salt_str.startswith('$1$'):
            handler = md5_crypt
        elif salt_str.startswith('$2'):
            handler = bcrypt
        else:
            # DES or unknown - use des_crypt with 2-char salt
            handler = des_crypt
            salt_str = salt_str[:2]

        # Use .using(salt=...) to configure, then hash
        # For verification: this produces same hash if password matches
        try:
            # Parse the existing hash to extract settings, then re-hash
            parsed = handler.from_string(salt_str)
            result = handler.using(
                salt=parsed.salt,
                rounds=getattr(parsed, 'rounds', None)
            ).hash(password) if hasattr(parsed, 'rounds') else handler.using(
                salt=parsed.salt
            ).hash(password)
            return MOOString(result)
        except Exception:
            # Fallback: just hash with whatever salt we got
            try:
                return MOOString(handler.hash(password))
            except Exception:
                return MOOString(sha512_crypt.hash(password))

    # =========================================================================
    # ToastStunt networking builtins
    # =========================================================================

    def connection_name_lookup(self, connection: int, do_lookup: int = 0) -> MOOString:
        """Perform DNS lookup on connection.

        ToastStunt builtin that initiates or returns DNS lookup for a connection.
        In our stub implementation, we return the connection's hostname directly.
        """
        # Stub - in real server this would do async DNS lookup
        # For now just return a placeholder
        return MOOString(f"connection_{connection}")

    def connection_name(self, connection: int, name_lookup: int = 0) -> MOOString:
        """Get the hostname of a connection.

        ToastStunt builtin that returns hostname for connection.
        """
        return MOOString(f"connection_{connection}")

    def buffered_output_length(self, connection: int = None) -> int:
        """Return the amount of output currently buffered for a connection.

        If connection is not provided, returns total for all connections.
        """
        return 0  # Stub

    def connection_option(self, connection: int, option: MOOString) -> int:
        """Get connection option value."""
        return 0  # Stub

    def set_connection_option(self, connection: int, option: MOOString, value) -> int:
        """Set connection option value."""
        return 0  # Stub
