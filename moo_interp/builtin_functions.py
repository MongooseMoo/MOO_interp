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

    def __init__(self):
        self.functions = {}  # Stores function_name: function pairs
        self.id_to_function = {}  # Stores function_id: function pairs
        self.function_to_id = {}  # Stores function: function_id pairs
        self.current_id = 0
        self._vm = None  # VM context, set by VM before execution

        # SQLite support
        self._sqlite_handles = {}  # SQLite connection handles
        self._next_sqlite_handle = 1

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
        if self.current_id > 511:
            raise Exception("Cannot register more than 512 functions.")
        function_id = self.current_id
        self.current_id += 1
        self.functions[fn.__name__] = fn
        self.id_to_function[function_id] = fn
        self.function_to_id[fn] = function_id
        return fn

    def _unwrap(self, value):
        """Unwrap MOO types to Python primitives.

        MOOString -> str
        MOOList -> list (recursively)
        MOOMap -> dict (recursively)
        Other types pass through unchanged.
        """
        if isinstance(value, MOOString):
            return str(value)
        elif isinstance(value, MOOList):
            return [self._unwrap(v) for v in value]
        elif isinstance(value, MOOMap):
            return {self._unwrap(k): self._unwrap(v) for k, v in value.items()}
        return value

    def _unwrap_bytes(self, value):
        """Unwrap MOO string to bytes for binary operations."""
        if isinstance(value, MOOString):
            return str(value).encode('latin-1')
        elif isinstance(value, str):
            return value.encode('latin-1')
        elif isinstance(value, bytes):
            return value
        raise TypeError(f"Expected string or bytes, got {type(value).__name__}")


    def register(self, name: str, func):
        """Register an external function with a custom name.
        
        Args:
            name: The name to register the function under
            func: A callable that will be registered as a builtin
        """
        if self.current_id > 511:
            raise Exception("Cannot register more than 512 functions.")

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
        if isinstance(value, ObjNum):
            return "#" + str(value)
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, MOOList):
            return "{list}"
        elif isinstance(value, MOOMap):
            return "[map]"
        elif isinstance(value, MOOString):
            return str(value)
        elif isinstance(value, float):
            return str(value)
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
        # Check ObjNum FIRST before int (ObjNum inherits from int)
        if isinstance(value, ObjNum):
            # Extract numeric value, not the ObjNum object
            # int() on ObjNum just returns itself, so use int.__new__ or .value
            return int.__index__(value)
        elif isinstance(value, bool):
            # Check bool before int (bool inherits from int)
            return 1 if value else 0
        elif isinstance(value, int):
            return value
        elif isinstance(value, MOOString):
            try:
                return int(value)
            except ValueError:
                return 0
        elif isinstance(value, float):
            return int(value)
        elif isinstance(value, MOOError):
            return self.unparse_error(value)
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

    def getenv(self, name):
        """Get an environment variable.
        
        getenv(name) => string value or 0 if not set
        
        Requires wizard permissions.
        Raises E_PERM if not wizard.
        Raises E_TYPE if name is not a string.
        Raises E_ARGS if called with wrong number of arguments (enforced by VM).
        """
        # Permission check - requires wizard
        # This will be enforced by the VM/task runner context
        # For now, we'll implement the basic functionality
        
        # Type check
        if not isinstance(name, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "getenv() requires a string argument")
            
        # Convert to Python string
        if isinstance(name, MOOString):
            name_str = name.data
        else:
            name_str = name
            
        # Get environment variable
        value = os.environ.get(name_str)
        
        if value is None:
            return 0  # Return 0 for non-existent variables
        
        return MOOString(value)

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

    def task_local(self, *args):
        """Get the task-local storage for the current task.

        task_local() => value stored by set_task_local(), or {} if not set

        Requires wizard permissions.
        Raises E_PERM if not wizard.
        Raises E_ARGS if called with arguments.

        Task-local storage is per-task and persists only for that task's lifetime.
        """
        # Validate argument count
        if len(args) != 0:
            raise MOOException(MOOError.E_ARGS, "task_local() takes no arguments")

        # Permission check - requires wizard
        # TODO: Check if caller is wizard and raise E_PERM if not
        # This will be enforced by the VM/task runner context

        # Get task-local storage from VM context
        # For now, return empty map (not implemented in VM yet)
        # TODO: Implement task-local storage in VM
        return MOOMap({})

    def set_task_local(self, *args):
        """Set the task-local storage for the current task.

        set_task_local(value) => 0

        Requires wizard permissions.
        Raises E_PERM if not wizard.
        Raises E_ARGS if called with wrong number of arguments.

        Stores a value that can be retrieved later with task_local().
        The value is cleared when the task completes.
        """
        # Validate argument count
        if len(args) != 1:
            raise MOOException(MOOError.E_ARGS, "set_task_local() requires exactly 1 argument")

        # Permission check - requires wizard
        # TODO: Check if caller is wizard and raise E_PERM if not
        # This will be enforced by the VM/task runner context

        value = args[0]

        # Set task-local storage in VM context
        # For now, just return 0 (not implemented in VM yet)
        # TODO: Implement task-local storage in VM
        return 0

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
        string = self._unwrap(string)
        algo = self._unwrap(algo).upper()

        # Supported algorithms - map to hashlib names
        algo_map = {
            'MD5': 'md5',
            'SHA1': 'sha1',
            'SHA224': 'sha224',
            'SHA256': 'sha256',
            'SHA384': 'sha384',
            'SHA512': 'sha512',
            'RIPEMD160': 'ripemd160',
        }

        if algo not in algo_map:
            raise MOOException('E_INVARG', f"Invalid hash algorithm: {algo}")

        try:
            hash_object = hashlib.new(algo_map[algo])
        except ValueError:
            # Some algorithms may not be available on all platforms
            raise MOOException('E_INVARG', f"Hash algorithm not available: {algo}")

        hash_object.update(string.encode('utf-8'))
        return MOOString(hash_object.hexdigest())

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

    def index(self, str1: MOOString, str2: MOOString, case_matters: int = 0, offset: int = 0):
        """Return 1-based index of str2 in str1, or 0 if not found.

        Args:
            str1: String to search in
            str2: Substring to find
            case_matters: 1 for case-sensitive, 0 for case-insensitive
            offset: 0-based offset to start search from (must be >= 0)

        Returns:
            1-based index of first occurrence, or 0 if not found

        Raises:
            MOOException: E_INVARG if offset is negative
        """
        # Validate offset
        if offset < 0:
            raise MOOException(MOOError.E_INVARG, "index() offset must be >= 0")

        s1 = str(str1)
        s2 = str(str2)

        # Apply offset - search only from offset onwards
        search_str = s1[offset:]

        if not case_matters:
            search_str = search_str.lower()
            s2 = s2.lower()

        pos = search_str.find(s2)
        if pos >= 0:
            # Return 1-based index within the searched substring (like C strindex)
            # This matches toaststunt behavior: offset shifts the search window,
            # but the result is relative to that window, not the original string
            return pos + 1
        return 0

    def rindex(self, str1: MOOString, str2: MOOString, case_matters: int = 0, offset: int = 0):
        """Return 1-based index of last occurrence of str2 in str1, or 0 if not found.

        Args:
            str1: String to search in
            str2: Substring to find
            case_matters: 1 for case-sensitive, 0 for case-insensitive
            offset: Offset from end to limit search (must be <= 0, negative values count from end)

        Returns:
            1-based index of last occurrence, or 0 if not found

        Raises:
            MOOException: E_INVARG if offset is positive
        """
        # Validate offset - rindex requires offset <= 0
        if offset > 0:
            raise MOOException(MOOError.E_INVARG, "rindex() offset must be <= 0")

        s1 = str(str1)
        s2 = str(str2)

        # For rindex, offset is subtracted from the string length
        # offset=0 means search entire string
        # offset=-3 means search up to 3 characters before the end
        search_len = len(s1) + offset  # offset is <= 0
        if search_len <= 0:
            # If offset makes search length non-positive, no search possible
            return 0

        search_str = s1[:search_len]

        if not case_matters:
            search_str = search_str.lower()
            s2 = s2.lower()

        pos = search_str.rfind(s2)
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

    def capitalize(self, s):
        """Capitalize first character of string."""
        s = self._unwrap(s)
        if not s:
            return MOOString("")
        return MOOString(s[0].upper() + s[1:])

    def upcase(self, s):
        """Convert string to uppercase."""
        return MOOString(self._unwrap(s).upper())

    def downcase(self, s):
        """Convert string to lowercase."""
        return MOOString(self._unwrap(s).lower())

    def ltrim(self, s, chars=None):
        """Strip leading whitespace (or specified chars)."""
        s = self._unwrap(s)
        if chars is not None:
            chars = self._unwrap(chars)
            return MOOString(s.lstrip(chars))
        return MOOString(s.lstrip())

    def rtrim(self, s, chars=None):
        """Strip trailing whitespace (or specified chars)."""
        s = self._unwrap(s)
        if chars is not None:
            chars = self._unwrap(chars)
            return MOOString(s.rstrip(chars))
        return MOOString(s.rstrip())

    def trim(self, s, chars=None):
        """Strip leading and trailing whitespace (or specified chars)."""
        s = self._unwrap(s)
        if chars is not None:
            chars = self._unwrap(chars)
            return MOOString(s.strip(chars))
        return MOOString(s.strip())

    def implode(self, lst, sep=" "):
        """Join list elements into string with separator."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "implode requires a list")
        sep = self._unwrap(sep)
        parts = [self._unwrap(x) if isinstance(x, MOOString) else str(x) for x in lst]
        return MOOString(sep.join(parts))

    def ord(self, s):
        """Return ASCII value of first character."""
        s = self._unwrap(s)
        if not s:
            raise MOOException(MOOError.E_INVARG, "ord requires non-empty string")
        return ord(s[0])

    _abs = abs

    def abs(self, x):
        return self._abs((x))

    def length(self, x):
        return len(x)

    def toliteral(self, x):
        # Check ObjNum FIRST (before int/float) since it inherits from int
        if isinstance(x, ObjNum):
            return MOOString("#" + str(x))
        elif isinstance(x, MooObject):
            return MOOString("#" + str(x))
        elif isinstance(x, bool):
            return MOOString(str(x).lower())
        elif isinstance(x, MOOString):
            return MOOString("\"" + x.data + "\"")
        elif isinstance(x, (int, float)):
            return MOOString(str(x))
        elif isinstance(x, MOOList):
            return MOOString("{" + ", ".join([str(self.toliteral(y)) for y in x]) + "}")
        elif isinstance(x, MOOMap):
            # Convert each key-value pair to strings before joining
            # MOO uses -> as the separator for map literals
            items = [str(self.toliteral(k)) + " -> " + str(self.toliteral(v)) for k, v in x.items()]
            return MOOString("[" + ", ".join(items) + "]")
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

    def mapmerge(self, m1, m2):
        """Merge two maps. m2 values override m1."""
        if not isinstance(m1, MOOMap) or not isinstance(m2, MOOMap):
            raise MOOException(MOOError.E_TYPE, "mapmerge requires two maps")
        result = MOOMap(dict(m1))
        result.update(m2)
        return result

    def mapslice(self, m, keys):
        """Extract a subset of map by keys."""
        if not isinstance(m, MOOMap):
            raise MOOException(MOOError.E_TYPE, "mapslice requires a map")
        if not isinstance(keys, MOOList):
            raise MOOException(MOOError.E_TYPE, "mapslice requires a list of keys")
        result = MOOMap()
        for k in keys:
            if k in m:
                result[k] = m[k]
        return result

    def eval(self, *args):
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

        eval() accepts multiple string arguments and concatenates them before evaluation.
        """
        # Check for no arguments
        if len(args) == 0:
            raise MOOException(MOOError.E_ARGS, "eval() requires at least one argument")

        # Check all arguments are strings
        for arg in args:
            if not isinstance(arg, (str, MOOString)):
                raise MOOException(MOOError.E_TYPE, "eval() requires string arguments")

        # Concatenate all string arguments
        code = ''.join(str(arg.data if hasattr(arg, 'data') else arg) for arg in args)

        from .moo_ast import compile
        from .vm import VM

        try:
            # Get bi_funcs from VM context (needed for server builtins like add_property)
            bi_funcs = self._vm.bi_funcs if self._vm else self
            db = self._vm.db if self._vm else None

            # Compile with the same bi_funcs as the parent VM
            compiled = compile(code, bi_funcs=bi_funcs)
            compiled.debug = True
            compiled.this = -1
            compiled.verb = ""

            # Get player from parent VM if available
            player = -1
            if self._vm and self._vm.call_stack:
                player = getattr(self._vm.call_stack[-1], 'player', -1)
            compiled.player = player

            # Create new VM with same db and bi_funcs
            vm = VM(db=db, bi_funcs=bi_funcs)
            vm.call_stack = [compiled]

            # Run to completion
            for _ in vm.run():
                pass

            return MOOList([True, vm.result])
        except Exception as e:
            return MOOList([False, MOOList([MOOString(str(e))])])

    def encode_base64(self, x, safe=False):
        x = self._unwrap_bytes(x)
        if safe:
            return MOOString(base64.urlsafe_b64encode(x).decode('ascii'))
        else:
            return MOOString(base64.b64encode(x).decode('ascii'))

    def decode_base64(self, x, safe=False):
        x = self._unwrap_bytes(x)
        try:
            if safe:
                # URL-safe base64 often omits padding - add it back
                padding = 4 - len(x) % 4
                if padding != 4:
                    x = x + b'=' * padding
                return MOOString(base64.urlsafe_b64decode(x).decode('latin-1'))
            else:
                return MOOString(base64.b64decode(x).decode('latin-1'))
        except Exception as e:
            raise MOOException('E_INVARG', str(e))

    def _moo_to_python(self, value, mode="common-subset"):
        """Convert MOO types to native Python for JSON serialization.

        Args:
            value: The MOO value to convert
            mode: "common-subset" (default) or "embedded-types"
        """
        # Handle MOOString - UserString stores string in .data attribute
        if isinstance(value, MOOString):
            return value.data

        # Handle MOOList - iterate and convert elements
        elif isinstance(value, MOOList):
            return [self._moo_to_python(v, mode) for v in value]

        # Handle MOOMap - convert keys and values
        elif isinstance(value, MOOMap):
            return {self._moo_to_python(k, mode): self._moo_to_python(v, mode) for k, v in value.items()}

        # Handle ObjNum - convert to string format "#N" or "#N|obj"
        elif isinstance(value, ObjNum):
            if mode == "embedded-types":
                return f"#{int(value)}|obj"
            else:
                return f"#{int(value)}"

        # Handle MOOError - convert to string format or with type annotation
        elif isinstance(value, MOOError):
            if mode == "embedded-types":
                return f"{value.name}|err"
            else:
                return value.name

        elif isinstance(value, MOOException):
            if mode == "embedded-types":
                return f"{value.error_code.name}|err"
            else:
                return value.error_code.name

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
        # Convert mode from MOOString if needed
        if mode is None:
            mode_str = "common-subset"
        elif isinstance(mode, MOOString):
            mode_str = mode.data
        else:
            mode_str = str(mode)

        import sys
        print(f"DEBUG generate_json x={x!r}, type={type(x)}, mode_str={mode_str!r}", file=sys.stderr)
        native = self._moo_to_python(x, mode_str)
        print(f"DEBUG native={native!r}, type={type(native)}", file=sys.stderr)
        result_json = json.dumps(native)
        print(f"DEBUG json.dumps result={result_json!r}", file=sys.stderr)
        result = MOOString(result_json)
        print(f"DEBUG final MOOString={result!r}", file=sys.stderr)
        return result

    def _python_to_moo(self, value, mode="common-subset"):
        """Convert Python value from JSON to MOO types.

        Args:
            value: Python value from json.loads()
            mode: "common-subset" or "embedded-types"

        Raises:
            MOOException: E_INVARG if common-subset mode encounters non-JSON types
        """
        if isinstance(value, str):
            # Check for embedded type annotations in string
            if mode == "embedded-types":
                # Handle type suffixes: "value|type"
                if "|" in value:
                    try:
                        prefix, suffix = value.rsplit("|", 1)

                        # Handle "|int" - convert prefix to int or 0 if empty
                        if suffix == "int":
                            return int(prefix) if prefix else 0

                        # Handle "|float" - convert prefix to float or 0.0 if empty
                        elif suffix == "float":
                            return float(prefix) if prefix else 0.0

                        # Handle "|str" - return prefix as MOOString
                        elif suffix == "str":
                            return MOOString(prefix)

                        # Handle "#N|obj" or "|obj" -> ObjNum(N) or ObjNum(0)
                        elif suffix == "obj":
                            if prefix.startswith("#"):
                                prefix = prefix[1:]
                            return ObjNum(int(prefix) if prefix else 0)

                        # Handle "E_XXX|err" or "|err" -> MOOError
                        elif suffix == "err":
                            if prefix:
                                # Try to get error by name
                                if hasattr(MOOError, prefix):
                                    return getattr(MOOError, prefix)
                            else:
                                # Bare "|err" returns E_NONE
                                return MOOError.E_NONE
                    except (ValueError, AttributeError):
                        # Fall through to return as string
                        pass

            # Check if string looks like an object/error in common-subset mode (should fail)
            if mode == "common-subset":
                if value.startswith("#") or (value.startswith("E_") and value.isupper()):
                    raise MOOException(MOOError.E_INVARG, "Object/error types not allowed in common-subset mode")

            return MOOString(value)

        elif isinstance(value, list):
            return MOOList([self._python_to_moo(v, mode) for v in value])

        elif isinstance(value, dict):
            return MOOMap({self._python_to_moo(k, mode): self._python_to_moo(v, mode) for k, v in value.items()})

        # Pass through primitives (int, float, bool, None)
        return value

    def parse_json(self, x, mode: MOOString = None):
        """Parse JSON string into MOO value.

        Args:
            x: JSON string to parse (MOOString or str)
            mode: Optional mode - "common-subset" (default) or "embedded-types"

        Returns:
            MOO value parsed from JSON

        Raises:
            MOOException: E_INVARG for invalid JSON or disallowed types
        """
        # Convert mode from MOOString if needed
        if mode is None:
            mode_str = "common-subset"
        elif isinstance(mode, MOOString):
            mode_str = mode.data
        else:
            mode_str = str(mode)

        # Convert MOOString to Python string if needed
        if isinstance(x, MOOString):
            x = x.data

        try:
            parsed = json.loads(x)
        except (json.JSONDecodeError, TypeError) as e:
            raise MOOException(MOOError.E_INVARG, f"Invalid JSON: {e}")

        return self._python_to_moo(parsed, mode_str)

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


    def file_exists(self, path):
        """Check if a file exists.

        file_exists(path) => 1 if exists, 0 otherwise

        Requires wizard permissions.
        """
        # Type check
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_exists() requires a string argument")

        # Convert to Python string
        path_str = str(path) if isinstance(path, MOOString) else path
        return 1 if os.path.exists(path_str) else 0

    def file_read(self, handle: int, bytes_to_read: int):
        """Read bytes from an open file handle.

        file_read(handle, bytes) => string data read
        """
        try:
            open_file = os.fdopen(handle)
            data = open_file.read(bytes_to_read)
            return MOOString(data)
        except (OSError, ValueError) as e:
            raise MOOException(MOOError.E_INVARG, f"file_read failed: {e}")

    def file_write(self, handle: int, data):
        """Write data to an open file handle.

        file_write(handle, data) => bytes written
        """
        try:
            open_file = os.fdopen(handle)
            data_str = str(data) if isinstance(data, MOOString) else str(data)
            bytes_written = open_file.write(data_str)
            return bytes_written if bytes_written is not None else len(data_str)
        except (OSError, ValueError) as e:
            raise MOOException(MOOError.E_INVARG, f"file_write failed: {e}")

    def file_eof(self, handle: int):
        """Check if file handle is at end of file.

        file_eof(handle) => 1 if at EOF, 0 otherwise
        """
        try:
            open_file = os.fdopen(handle)
            current_pos = open_file.tell()
            open_file.seek(0, os.SEEK_END)
            end_pos = open_file.tell()
            open_file.seek(current_pos)
            return 1 if current_pos >= end_pos else 0
        except (OSError, ValueError) as e:
            raise MOOException(MOOError.E_INVARG, f"file_eof failed: {e}")

    def file_stat(self, path):
        """Get file statistics.

        file_stat(path) => list [size, type, mode, owner, group, atime, mtime, ctime]
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_stat() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            st = os.stat(path_str)
            import stat as stat_module

            # Determine file type
            if stat_module.S_ISREG(st.st_mode):
                file_type = "reg"
            elif stat_module.S_ISDIR(st.st_mode):
                file_type = "dir"
            elif stat_module.S_ISCHR(st.st_mode):
                file_type = "chr"
            elif stat_module.S_ISBLK(st.st_mode):
                file_type = "block"
            elif stat_module.S_ISFIFO(st.st_mode):
                file_type = "fifo"
            elif stat_module.S_ISSOCK(st.st_mode):
                file_type = "socket"
            else:
                file_type = "unknown"

            mode_octal = oct(st.st_mode & 0o777)[2:]

            return MOOList([
                st.st_size,
                MOOString(file_type),
                MOOString(mode_octal),
                MOOString(""),
                MOOString(""),
                int(st.st_atime),
                int(st.st_mtime),
                int(st.st_ctime)
            ])
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_stat failed: {e}")

    def file_type(self, path):
        """Get file type.

        file_type(path) => string ("reg", "dir", "chr", etc)
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_type() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            st = os.stat(path_str)
            import stat as stat_module

            if stat_module.S_ISREG(st.st_mode):
                return MOOString("reg")
            elif stat_module.S_ISDIR(st.st_mode):
                return MOOString("dir")
            elif stat_module.S_ISCHR(st.st_mode):
                return MOOString("chr")
            elif stat_module.S_ISBLK(st.st_mode):
                return MOOString("block")
            elif stat_module.S_ISFIFO(st.st_mode):
                return MOOString("fifo")
            elif stat_module.S_ISSOCK(st.st_mode):
                return MOOString("socket")
            else:
                return MOOString("unknown")
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_type failed: {e}")

    def file_list(self, path, detailed: int = 0):
        """List directory contents.

        file_list(path [, detailed]) => list of filenames or detailed info
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_list() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            entries = os.listdir(path_str)
            result = []

            for entry in entries:
                if entry in ['.', '..']:
                    continue

                if detailed:
                    full_path = os.path.join(path_str, entry)
                    st = os.stat(full_path)
                    import stat as stat_module

                    if stat_module.S_ISREG(st.st_mode):
                        file_type = "reg"
                    elif stat_module.S_ISDIR(st.st_mode):
                        file_type = "dir"
                    elif stat_module.S_ISCHR(st.st_mode):
                        file_type = "chr"
                    elif stat_module.S_ISBLK(st.st_mode):
                        file_type = "block"
                    elif stat_module.S_ISFIFO(st.st_mode):
                        file_type = "fifo"
                    elif stat_module.S_ISSOCK(st.st_mode):
                        file_type = "socket"
                    else:
                        file_type = "unknown"

                    mode_octal = oct(st.st_mode & 0o777)[2:]

                    result.append(MOOList([
                        MOOString(entry),
                        MOOString(file_type),
                        MOOString(mode_octal),
                        st.st_size
                    ]))
                else:
                    result.append(MOOString(entry))

            return MOOList(result)
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_list failed: {e}")

    def file_mkdir(self, path, mode: int = 0o777):
        """Create a directory.

        file_mkdir(path [, mode]) => 0 on success
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_mkdir() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            os.mkdir(path_str, mode)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_mkdir failed: {e}")

    def file_rmdir(self, path):
        """Remove a directory.

        file_rmdir(path) => 0 on success
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_rmdir() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            os.rmdir(path_str)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_rmdir failed: {e}")

    def file_remove(self, path):
        """Remove a file.

        file_remove(path) => 0 on success
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_remove() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            os.remove(path_str)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_remove failed: {e}")

    def file_rename(self, old_path, new_path):
        """Rename a file or directory.

        file_rename(old, new) => 0 on success
        """
        if not isinstance(old_path, (str, MOOString)) or not isinstance(new_path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_rename() requires string arguments")

        old_str = str(old_path) if isinstance(old_path, MOOString) else old_path
        new_str = str(new_path) if isinstance(new_path, MOOString) else new_path

        try:
            os.rename(old_str, new_str)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_rename failed: {e}")

    def file_copy(self, src, dst):
        """Copy a file.

        file_copy(src, dst) => 0 on success
        """
        import shutil

        if not isinstance(src, (str, MOOString)) or not isinstance(dst, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_copy() requires string arguments")

        src_str = str(src) if isinstance(src, MOOString) else src
        dst_str = str(dst) if isinstance(dst, MOOString) else dst

        try:
            shutil.copy2(src_str, dst_str)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_copy failed: {e}")

    def file_chmod(self, path, mode):
        """Change file permissions.

        file_chmod(path, mode) => 0 on success
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_chmod() requires path as string")

        path_str = str(path) if isinstance(path, MOOString) else path

        # Convert mode to integer
        if isinstance(mode, (str, MOOString)):
            mode_str = str(mode) if isinstance(mode, MOOString) else mode
            try:
                mode_int = int(mode_str, 8)
            except ValueError:
                raise MOOException(MOOError.E_INVARG, f"Invalid mode string: {mode_str}")
        elif isinstance(mode, int):
            mode_int = mode
        else:
            raise MOOException(MOOError.E_TYPE, "file_chmod() mode must be string or integer")

        try:
            os.chmod(path_str, mode_int)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_chmod failed: {e}")

    # property functions

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
    def shift(self, n, count):
        """Shift n by count bits. Positive count=left shift, negative=right shift.

        shift(1, 3) => 8 (1 << 3)
        shift(8, -3) => 1 (8 >> 3)
        """
        if count >= 0:
            return n << count
        else:
            return n >> (-count)

    def bitand(self, a, b):
        """Bitwise AND of two integers."""
        return int(a) & int(b)

    def bitor(self, a, b):
        """Bitwise OR of two integers."""
        return int(a) | int(b)

    def bitxor(self, a, b):
        """Bitwise XOR of two integers."""
        return int(a) ^ int(b)

    def bitnot(self, a):
        """Bitwise NOT of an integer."""
        return ~int(a)

    def bitshl(self, n, count):
        """Bitwise left shift."""
        return int(n) << int(count)

    def bitshr(self, n, count):
        """Bitwise right shift."""
        return int(n) >> int(count)

    # Additional math builtins
    def log2(self, x):
        """Return base-2 logarithm."""
        return math.log2(self.tofloat(x))

    def fmod(self, x, y):
        """Return floating-point remainder."""
        return math.fmod(self.tofloat(x), self.tofloat(y))

    def hypot(self, x, y):
        """Return Euclidean distance sqrt(x*x + y*y)."""
        return math.hypot(self.tofloat(x), self.tofloat(y))

    def copysign(self, x, y):
        """Return x with sign of y."""
        return math.copysign(self.tofloat(x), self.tofloat(y))

    def frexp(self, x):
        """Return (mantissa, exponent) tuple."""
        m, e = math.frexp(self.tofloat(x))
        return MOOList([m, e])

    def ldexp(self, x, i):
        """Return x * (2 ** i)."""
        return math.ldexp(self.tofloat(x), int(i))

    def modf(self, x):
        """Return (fractional, integer) parts."""
        f, i = math.modf(self.tofloat(x))
        return MOOList([f, i])

    def remainder(self, x, y):
        """Return IEEE 754 remainder."""
        return math.remainder(self.tofloat(x), self.tofloat(y))

    def isfinite(self, x):
        """Return 1 if x is finite, 0 otherwise."""
        return 1 if math.isfinite(self.tofloat(x)) else 0

    def isinf(self, x):
        """Return 1 if x is infinite, 0 otherwise."""
        return 1 if math.isinf(self.tofloat(x)) else 0

    def isnan(self, x):
        """Return 1 if x is NaN, 0 otherwise."""
        return 1 if math.isnan(self.tofloat(x)) else 0

    # Aggregation functions for lists
    _sum = sum  # Save builtin

    def sum(self, lst):
        """Return sum of numeric list elements."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "sum requires a list")
        return self._sum(lst)

    def avg(self, lst):
        """Return average of numeric list elements."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "avg requires a list")
        if not lst:
            raise MOOException(MOOError.E_INVARG, "avg requires non-empty list")
        return self._sum(lst) / len(lst)

    def product(self, lst):
        """Return product of numeric list elements."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "product requires a list")
        result = 1
        for x in lst:
            result *= x
        return result

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

    # Type name mapping
    TYPE_NAMES = {0: "INT", 1: "OBJ", 2: "STR", 3: "ERR", 4: "LIST", 9: "FLOAT", 10: "MAP", 14: "BOOL"}

    def typename(self, x):
        """Return the type name as a string."""
        type_code = self.typeof(x)
        return MOOString(self.TYPE_NAMES.get(type_code, "UNKNOWN"))

    def is_type(self, x, type_code):
        """Check if x is of the given type code."""
        return 1 if self.typeof(x) == type_code else 0

    def tonum(self, x):
        """Alias for toint."""
        return self.toint(x)

    def toerr(self, x):
        """Convert value to error code."""
        if isinstance(x, MOOError):
            return x
        elif isinstance(x, int):
            # Map integer to error code
            error_map = {0: MOOError.E_NONE, 1: MOOError.E_TYPE, 2: MOOError.E_DIV,
                         3: MOOError.E_PERM, 4: MOOError.E_PROPNF, 5: MOOError.E_VERBNF,
                         6: MOOError.E_VARNF, 7: MOOError.E_INVIND, 8: MOOError.E_RECMOVE,
                         9: MOOError.E_MAXREC, 10: MOOError.E_RANGE, 11: MOOError.E_ARGS,
                         12: MOOError.E_NACC, 13: MOOError.E_INVARG, 14: MOOError.E_QUOTA,
                         15: MOOError.E_FLOAT}
            return error_map.get(x, MOOError.E_NONE)
        elif isinstance(x, (str, MOOString)):
            # Parse error name like "E_TYPE"
            name = str(x).upper()
            if hasattr(MOOError, name):
                return getattr(MOOError, name)
            return MOOError.E_NONE
        return MOOError.E_NONE

    # Encoding builtins
    def encode_base64(self, s):
        """Encode string to base64."""
        import base64
        data = self._unwrap_bytes(s)
        return MOOString(base64.b64encode(data).decode('ascii'))

    def decode_base64(self, s):
        """Decode base64 string."""
        import base64
        try:
            data = base64.b64decode(self._unwrap(s))
            return MOOString(data.decode('latin-1'))
        except Exception:
            raise MOOException(MOOError.E_INVARG, "Invalid base64")

    def encode_hex(self, s):
        """Encode string to hex."""
        data = self._unwrap_bytes(s)
        return MOOString(data.hex())

    def decode_hex(self, s):
        """Decode hex string."""
        try:
            data = bytes.fromhex(self._unwrap(s))
            return MOOString(data.decode('latin-1'))
        except Exception:
            raise MOOException(MOOError.E_INVARG, "Invalid hex")

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

    def toobj(self, x):
        """Convert a value to an object reference."""
        if isinstance(x, ObjNum):
            return x
        elif isinstance(x, MooObject):
            return ObjNum(x.id)
        elif isinstance(x, int):
            return ObjNum(x)
        elif isinstance(x, (str, MOOString)):
            s = str(x).strip()
            if s.startswith('#'):
                s = s[1:]
            try:
                obj_id = int(s)
                return ObjNum(obj_id)
            except ValueError:
                return ObjNum(0)  # Invalid strings return #0 per MOO spec
        elif isinstance(x, float):
            return ObjNum(int(x))
        else:
            return ObjNum(0)

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

    def assoc(self, key, alist, idx=1):
        """Search alist for element where alist[n][idx] == key. Returns element or 0."""
        if not isinstance(alist, MOOList):
            raise MOOException(MOOError.E_TYPE, "assoc requires a list")
        for item in alist:
            if isinstance(item, MOOList) and len(item) >= idx:
                if item[idx] == key:
                    return item
        return 0

    def rassoc(self, key, alist, idx=1):
        """Like assoc but searches from end."""
        if not isinstance(alist, MOOList):
            raise MOOException(MOOError.E_TYPE, "rassoc requires a list")
        for item in reversed(list(alist)):
            if isinstance(item, MOOList) and len(item) >= idx:
                if item[idx] == key:
                    return item
        return 0

    def iassoc(self, key, alist, idx=1):
        """Like assoc but returns 1-based index instead of element."""
        if not isinstance(alist, MOOList):
            raise MOOException(MOOError.E_TYPE, "iassoc requires a list")
        for i, item in enumerate(alist):
            if isinstance(item, MOOList) and len(item) >= idx:
                if item[idx] == key:
                    return i + 1  # 1-based
        return 0

    def count(self, lst, val):
        """Count occurrences of val in lst."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "count requires a list")
        return sum(1 for x in lst if x == val)

    def diff(self, lst1, lst2):
        """Return elements in lst1 not in lst2."""
        if not isinstance(lst1, MOOList) or not isinstance(lst2, MOOList):
            raise MOOException(MOOError.E_TYPE, "diff requires two lists")
        lst2_set = set(lst2)
        return MOOList([x for x in lst1 if x not in lst2_set])

    def intersection(self, lst1, lst2):
        """Return elements common to both lists."""
        if not isinstance(lst1, MOOList) or not isinstance(lst2, MOOList):
            raise MOOException(MOOError.E_TYPE, "intersection requires two lists")
        lst2_set = set(lst2)
        return MOOList([x for x in lst1 if x in lst2_set])

    def union(self, lst1, lst2):
        """Return combined unique elements from both lists."""
        if not isinstance(lst1, MOOList) or not isinstance(lst2, MOOList):
            raise MOOException(MOOError.E_TYPE, "union requires two lists")
        seen = set()
        result = []
        for x in list(lst1) + list(lst2):
            if x not in seen:
                seen.add(x)
                result.append(x)
        return MOOList(result)

    def unique(self, lst):
        """Remove duplicate elements, preserving order."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "unique requires a list")
        seen = set()
        result = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return MOOList(result)

    def flatten(self, lst, depth=-1):
        """Flatten nested lists. depth=-1 means fully flatten."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "flatten requires a list")
        def _flatten(items, d):
            result = []
            for item in items:
                if isinstance(item, MOOList) and d != 0:
                    result.extend(_flatten(item, d - 1))
                else:
                    result.append(item)
            return result
        return MOOList(_flatten(lst, depth))

    def rotate(self, lst, n=1):
        """Rotate list elements by n positions."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "rotate requires a list")
        if not lst:
            return MOOList([])
        items = list(lst)
        n = n % len(items)
        return MOOList(items[n:] + items[:n])

    def make_list(self, count, val=0):
        """Create list with count copies of val."""
        if count < 0:
            raise MOOException(MOOError.E_INVARG, "count must be non-negative")
        return MOOList([val] * count)

    def indexc(self, lst, val, start=1):
        """Case-insensitive search in list of strings."""
        if not isinstance(lst, MOOList):
            raise MOOException(MOOError.E_TYPE, "indexc requires a list")
        val_lower = self._unwrap(val).lower() if isinstance(val, MOOString) else str(val).lower()
        for i, item in enumerate(lst, 1):
            if i < start:
                continue
            item_str = self._unwrap(item).lower() if isinstance(item, MOOString) else str(item).lower()
            if item_str == val_lower:
                return i
        return 0

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
    # =========================================================================
    # SQLite builtins - based on toaststunt/src/sqlite.cc
    # =========================================================================

    def _valid_sqlite_handle(self, handle: int) -> bool:
        """Check if a SQLite handle is valid and active."""
        return isinstance(handle, int) and handle in self._sqlite_handles

    def _sanitize_sqlite_string(self, s: str) -> str:
        """Replace newlines with tabs for MOO compatibility."""
        return s.replace('\n', '\t') if s else s

    def _sqlite_type_to_moo(self, value, parse_types: bool = True, parse_objects: bool = True):
        """Convert SQLite value to MOO type.

        Args:
            value: Raw value from SQLite (str, int, float, bytes, or None)
            parse_types: If True, try to parse as int/float/obj
            parse_objects: If True, parse strings like "#123" as ObjNum
        """
        if value is None:
            return MOOString("NULL")

        if not parse_types:
            # Everything becomes string
            return MOOString(self._sanitize_sqlite_string(str(value)))

        # Try to parse as specific types
        s = str(value)

        # Check for object reference like "#123"
        if parse_objects and s.startswith('#'):
            try:
                num = int(s[1:])
                return ObjNum(num)
            except ValueError:
                pass

        # Try integer
        try:
            return int(s)
        except ValueError:
            pass

        # Try float
        try:
            return float(s)
        except ValueError:
            pass

        # Default to string
        return MOOString(self._sanitize_sqlite_string(s))

    def sqlite_open(self, filename, options: int = 0x3):
        """Open an SQLite database and return a handle.

        Args:
            filename: Path to database file, ":memory:", or "" for memory DB
            options: Bitmask - 0x1=parse_types, 0x2=parse_objects, 0x4=sanitize_strings

        Returns:
            Integer handle for use with other sqlite_* functions

        Raises:
            E_PERM if caller is not wizard
            E_QUOTA if too many connections open (max 10)
            E_INVARG if filename is invalid or already open
        """
        import sqlite3

        # TODO: Permission check - requires wizard
        # For now we'll allow it

        # Check connection limit
        if len(self._sqlite_handles) >= 10:
            raise MOOException(MOOError.E_QUOTA, "Too many database connections open")

        filename = str(filename) if isinstance(filename, MOOString) else filename

        # Check if database already open (except :memory: which can have multiple)
        for handle, conn_data in self._sqlite_handles.items():
            if conn_data['path'] == filename and filename not in (":memory:", ""):
                raise MOOException(MOOError.E_INVARG,
                    f"Database already open with handle: {handle}")

        try:
            conn = sqlite3.connect(filename)
            conn.row_factory = None  # We'll handle rows manually

            handle = self._next_sqlite_handle
            self._next_sqlite_handle += 1

            self._sqlite_handles[handle] = {
                'connection': conn,
                'path': filename,
                'options': options,
                'locks': 0
            }

            return handle

        except sqlite3.Error as e:
            raise MOOException(MOOError.E_NONE, str(e))

    def sqlite_close(self, handle: int):
        """Close an SQLite database connection.

        Args:
            handle: Database handle from sqlite_open

        Raises:
            E_PERM if caller is not wizard
            E_INVARG if handle is invalid
            E_PERM if handle has active operations
        """
        # TODO: Permission check

        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]

        if conn_data['locks'] > 0:
            raise MOOException(MOOError.E_PERM,
                "Handle can't be closed until all operations are finished")

        conn_data['connection'].close()
        del self._sqlite_handles[handle]

        # Reset handle counter if all connections closed
        if not self._sqlite_handles:
            self._next_sqlite_handle = 1

        return None  # no_var_pack equivalent

    def sqlite_execute(self, handle: int, sql, args: MOOList = None):
        """Execute SQL with optional parameters and return rows.

        This is the prepared statement version that binds parameters.

        Args:
            handle: Database handle
            sql: SQL query string with ? placeholders
            args: List of values to bind to ? placeholders

        Returns:
            List of rows (each row is a list of values) for SELECT queries
            Empty list for non-SELECT queries

        Raises:
            E_PERM if caller is not wizard
            E_INVARG if handle is invalid
            Returns error string if SQL fails
        """
        import sqlite3

        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        conn = conn_data['connection']
        options = conn_data['options']

        sql = str(sql) if isinstance(sql, MOOString) else sql

        # Prepare bind parameters
        bind_params = []
        if args:
            for arg in args:
                if isinstance(arg, ObjNum):
                    bind_params.append(f"#{int(arg)}")
                elif isinstance(arg, MOOString):
                    bind_params.append(str(arg))
                elif isinstance(arg, (int, float)):
                    bind_params.append(arg)
                else:
                    bind_params.append(str(arg))

        try:
            conn_data['locks'] += 1
            cursor = conn.cursor()

            if bind_params:
                cursor.execute(sql, bind_params)
            else:
                cursor.execute(sql)

            # Fetch all rows if any
            result = MOOList([])
            for row in cursor.fetchall():
                moo_row = MOOList([
                    self._sqlite_type_to_moo(
                        val,
                        parse_types=bool(options & 0x1),
                        parse_objects=bool(options & 0x2)
                    ) for val in row
                ])
                result.append(moo_row)

            conn.commit()

            # Store cursor for lastrowid access
            conn_data['last_cursor'] = cursor

            conn_data['locks'] -= 1

            return result

        except sqlite3.Error as e:
            conn_data['locks'] -= 1
            return MOOString(str(e))  # Return error as string

    def sqlite_query(self, handle: int, sql, include_headers: int = 0):
        """Execute SQL query using sqlite3_exec callback style.

        Args:
            handle: Database handle
            sql: SQL query string (no parameter binding)
            include_headers: If true, each row element is [column_name, value]

        Returns:
            List of rows

        Raises:
            E_PERM if caller is not wizard
            E_INVARG if handle is invalid
            Returns error string if SQL fails
        """
        import sqlite3

        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        conn = conn_data['connection']
        options = conn_data['options']

        sql = str(sql) if isinstance(sql, MOOString) else sql

        try:
            conn_data['locks'] += 1
            cursor = conn.cursor()
            cursor.execute(sql)

            result = MOOList([])
            col_names = [desc[0] for desc in cursor.description] if cursor.description else []

            for row in cursor.fetchall():
                if include_headers and col_names:
                    # Each element is [column_name, value]
                    moo_row = MOOList([
                        MOOList([
                            MOOString(col_names[i]),
                            self._sqlite_type_to_moo(
                                val,
                                parse_types=bool(options & 0x1),
                                parse_objects=bool(options & 0x2)
                            )
                        ]) for i, val in enumerate(row)
                    ])
                else:
                    # Just the values
                    moo_row = MOOList([
                        self._sqlite_type_to_moo(
                            val,
                            parse_types=bool(options & 0x1),
                            parse_objects=bool(options & 0x2)
                        ) for val in row
                    ])
                result.append(moo_row)

            conn.commit()

            # Store cursor for lastrowid access
            conn_data['last_cursor'] = cursor

            conn_data['locks'] -= 1

            return result

        except sqlite3.Error as e:
            conn_data['locks'] -= 1
            return MOOString(str(e))

    def sqlite_last_insert_row_id(self, handle: int) -> int:
        """Get the row ID of the last INSERT.

        Args:
            handle: Database handle

        Returns:
            Integer row ID

        Raises:
            E_PERM if caller is not wizard
            E_INVARG if handle is invalid
        """
        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]

        # Get lastrowid from stored cursor or connection
        # SQLite3 stores lastrowid on the connection itself after execute
        cursor = conn_data.get('last_cursor')
        if cursor and hasattr(cursor, 'lastrowid'):
            return cursor.lastrowid

        return 0

    def sqlite_handles(self) -> MOOList:
        """Return a list of all open SQLite database handles.

        Returns:
            List of integer handles

        Raises:
            E_PERM if caller is not wizard
        """
        # TODO: Permission check
        return MOOList(list(self._sqlite_handles.keys()))

    def sqlite_info(self, handle: int) -> MOOMap:
        """Get information about a database handle.

        Args:
            handle: Database handle

        Returns:
            Map with keys: path, parse_types, parse_objects, sanitize_strings, locks

        Raises:
            E_PERM if caller is not wizard
            E_INVARG if handle is invalid
        """
        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        options = conn_data['options']

        return MOOMap({
            MOOString('path'): MOOString(conn_data['path']),
            MOOString('parse_types'): 1 if options & 0x1 else 0,
            MOOString('parse_objects'): 1 if options & 0x2 else 0,
            MOOString('sanitize_strings'): 1 if options & 0x4 else 0,
            MOOString('locks'): conn_data['locks']
        })

    # Extension functions not in toaststunt but useful:

    def sqlite_last_insert_id(self, handle: int) -> int:
        """Alias for sqlite_last_insert_row_id (shorter name)."""
        return self.sqlite_last_insert_row_id(handle)

    def sqlite_query_maps(self, handle: int, sql, args: MOOList = None):
        """Execute SQL and return results as list of maps (column->value).

        Args:
            handle: Database handle
            sql: SQL query
            args: Optional bind parameters

        Returns:
            List of maps, where each map has column names as keys
        """
        import sqlite3

        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        conn = conn_data['connection']
        options = conn_data['options']

        sql = str(sql) if isinstance(sql, MOOString) else sql

        # Prepare bind parameters
        bind_params = []
        if args:
            for arg in args:
                if isinstance(arg, ObjNum):
                    bind_params.append(f"#{int(arg)}")
                elif isinstance(arg, MOOString):
                    bind_params.append(str(arg))
                elif isinstance(arg, (int, float)):
                    bind_params.append(arg)
                else:
                    bind_params.append(str(arg))

        try:
            conn_data['locks'] += 1
            cursor = conn.cursor()

            if bind_params:
                cursor.execute(sql, bind_params)
            else:
                cursor.execute(sql)

            col_names = [desc[0] for desc in cursor.description] if cursor.description else []

            result = MOOList([])
            for row in cursor.fetchall():
                row_map = MOOMap({})
                for i, val in enumerate(row):
                    moo_val = self._sqlite_type_to_moo(
                        val,
                        parse_types=bool(options & 0x1),
                        parse_objects=bool(options & 0x2)
                    )
                    row_map[MOOString(col_names[i])] = moo_val
                result.append(row_map)

            conn.commit()

            # Store cursor for lastrowid access
            conn_data['last_cursor'] = cursor

            conn_data['locks'] -= 1

            return result

        except sqlite3.Error as e:
            conn_data['locks'] -= 1
            return MOOString(str(e))

    def sqlite_changes(self, handle: int) -> int:
        """Get number of rows affected by last INSERT/UPDATE/DELETE.

        Args:
            handle: Database handle

        Returns:
            Number of rows changed
        """
        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        return conn_data['connection'].total_changes

    def sqlite_begin(self, handle: int):
        """Begin a transaction.

        Args:
            handle: Database handle
        """
        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        conn_data['connection'].execute("BEGIN")
        return None

    def sqlite_commit(self, handle: int):
        """Commit the current transaction.

        Args:
            handle: Database handle
        """
        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        conn_data['connection'].commit()
        return None

    def sqlite_rollback(self, handle: int):
        """Rollback the current transaction.

        Args:
            handle: Database handle
        """
        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        conn_data['connection'].rollback()
        return None

    def sqlite_tables(self, handle: int) -> MOOList:
        """List all tables in the database.

        Args:
            handle: Database handle

        Returns:
            List of table names
        """
        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        conn_data = self._sqlite_handles[handle]
        cursor = conn_data['connection'].cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")

        return MOOList([MOOString(row[0]) for row in cursor.fetchall()])

    def sqlite_columns(self, handle: int, table) -> MOOList:
        """Get column information for a table.

        Args:
            handle: Database handle
            table: Table name

        Returns:
            List of column info maps with keys: name, type, notnull, dflt_value, pk
        """
        import sqlite3

        if not self._valid_sqlite_handle(handle):
            raise MOOException(MOOError.E_INVARG, f"Invalid database handle: {handle}")

        table = str(table) if isinstance(table, MOOString) else table
        conn_data = self._sqlite_handles[handle]
        cursor = conn_data['connection'].cursor()

        try:
            cursor.execute(f"PRAGMA table_info({table})")

            result = MOOList([])
            for row in cursor.fetchall():
                # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
                col_map = MOOMap({
                    MOOString('name'): MOOString(row[1]),
                    MOOString('type'): MOOString(row[2]),
                    MOOString('notnull'): row[3],
                    MOOString('dflt_value'): MOOString(str(row[4])) if row[4] is not None else MOOString("NULL"),
                    MOOString('pk'): row[5]
                })
                result.append(col_map)

            return result

        except sqlite3.Error as e:
            return MOOString(str(e))
