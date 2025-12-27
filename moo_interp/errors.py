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

# Dictionary mapping type name strings to their integer codes for runtime lookup
# Used by typeof() comparisons like: typeof(x) == NUM
# ToastStunt type codes match these values
TYPE_CODES = {
    # Primary names
    "NUM": 0,       # Also called INT - integer type
    "INT": 0,       # Alias for NUM
    "OBJ": 1,       # Object reference
    "STR": 2,       # String
    "ERR": 3,       # Error type
    "LIST": 4,      # List/array
    "FLOAT": 9,     # Floating point
    "MAP": 10,      # Map/dictionary (ToastStunt extension)
    "ANON": 12,     # Anonymous object (ToastStunt extension)
    "WAIF": 13,     # Waif object (ToastStunt extension)
    "BOOL": 14,     # Boolean (ToastStunt extension)
}


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
