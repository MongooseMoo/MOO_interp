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


# Dictionary mapping error name strings to MOOError enum members for runtime lookup
ERROR_CODES = {e.name: e for e in MOOError}


class MOOException(Exception):
    """Exception raised for MOO runtime errors that can be caught by try/except."""
    def __init__(self, error_code: MOOError, message: str = ''):
        self.error_code = error_code
        self.message = message or error_code.value
        super().__init__(self.message)
