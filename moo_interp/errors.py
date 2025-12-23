from enum import Enum


class MOOError(Enum):
    E_TYPE = 'E_TYPE'
    E_DIV = 'E_DIV'
    E_PERM = 'E_PERM'
    E_PROPNF = 'E_PROPNF'
    E_VERBNF = 'E_VERBNF'
    E_VARNF = 'E_VARNF'
    E_INVIND = 'E_INVIND'
    E_RECMOVE = 'E_RECMOVE'
    E_MAXREC = 'E_MAXREC'
    E_RANGE = 'E_RANGE'
    E_ARGS = 'E_ARGS'
    E_NACC = 'E_NACC'
    E_INVARG = 'E_INVARG'
    E_QUOTA = 'E_QUOTA'
    E_FLOAT = 'E_FLOAT'
    E_FILE = 'E_FILE'
    E_EXEC = 'E_EXEC'


class MOOException(Exception):
    """Exception raised for MOO runtime errors that can be caught by try/except."""
    def __init__(self, error_code: MOOError, message: str = ''):
        self.error_code = error_code
        self.message = message or error_code.value
        super().__init__(self.message)
