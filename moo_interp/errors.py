from enum import IntEnum


class MOOError(IntEnum):
    """MOO error codes as integers, matching LambdaMOO's enum order."""
    E_NONE = 0
    E_TYPE = 1
    E_DIV = 2
    E_PERM = 3
    E_PROPNF = 4
    E_VERBNF = 5
    E_VARNF = 6
    E_INVIND = 7
    E_RECMOVE = 8
    E_MAXREC = 9
    E_RANGE = 10
    E_ARGS = 11
    E_NACC = 12
    E_INVARG = 13
    E_QUOTA = 14
    E_FLOAT = 15
    E_FILE = 16
    E_EXEC = 17
    E_INTRPT = 18


# Dictionary mapping error name strings to MOOError enum members for runtime lookup
ERROR_CODES = {e.name: e for e in MOOError}

# MOO type codes (toaststunt compatible)
# These are the canonical values - BuiltinFunctions.typeof() returns these
TYPE_INT = 0
TYPE_OBJ = 1
TYPE_STR = 2
TYPE_ERR = 3
TYPE_LIST = 4
TYPE_FLOAT = 9
TYPE_MAP = 10
TYPE_ANON = 12
TYPE_WAIF = 13
TYPE_BOOL = 14

# Mapping from type code to canonical name (used by typename())
TYPE_NAMES = {
    TYPE_INT: "INT",
    TYPE_OBJ: "OBJ",
    TYPE_STR: "STR",
    TYPE_ERR: "ERR",
    TYPE_LIST: "LIST",
    TYPE_FLOAT: "FLOAT",
    TYPE_MAP: "MAP",
    TYPE_ANON: "ANON",
    TYPE_WAIF: "WAIF",
    TYPE_BOOL: "BOOL",
}

# Reverse mapping: type name string to code (used by exec_push for MOO constants)
# Includes both "NUM" (MOO convention) and "INT" (internal name)
TYPE_CODES = {name: code for code, name in TYPE_NAMES.items()}
TYPE_CODES["NUM"] = TYPE_INT  # MOO uses NUM, not INT


class MOOException(Exception):
    """Exception raised for MOO runtime errors that can be caught by try/except."""
    def __init__(self, error_code: MOOError, message: str = ''):
        self.error_code = error_code
        self.message = message or error_code.value
        super().__init__(self.message)


class SuspendException(Exception):
    """Exception raised by builtins to signal task suspension.

    This is NOT a MOO error - it's a control flow mechanism.
    When a builtin raises this, the VM should:
    1. Stop execution
    2. Set state to OUTCOME_BLOCKED
    3. Store the suspend info for the task scheduler

    Attributes:
        seconds: How long to suspend (0 = just yield to other tasks)
    """
    def __init__(self, seconds: float = 0):
        self.seconds = seconds
        super().__init__(f"Suspend for {seconds} seconds")


class ExecSuspendException(SuspendException):
    """Exception raised by exec() to suspend task until subprocess completes.

    The task should be suspended and the subprocess should be monitored.
    When the subprocess completes, the task should be resumed with the result
    as a list [exit_code, stdout, stderr].

    Attributes:
        process: The subprocess.Popen object
        stdin_bytes: Bytes to send to stdin
        display_cmd: Command name for queued_tasks() display
    """
    def __init__(self, process, stdin_bytes: bytes = b"", display_cmd: str = ""):
        super().__init__(seconds=0)  # Suspend indefinitely until process completes
        self.process = process
        self.stdin_bytes = stdin_bytes
        self.display_cmd = display_cmd
