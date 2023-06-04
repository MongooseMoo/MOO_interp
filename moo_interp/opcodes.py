"""LambdaMOO Opcodes"""

from enum import Enum

NUM_READY_VARS = 32


class Extended_Opcode(Enum):
    EOP_RANGESET = 1
    EOP_LENGTH = 2  # EOP_LENGTH is retired
    EOP_FIRST = 3
    EOP_LAST = 4
    EOP_PUSH_LABEL = 5
    EOP_END_CATCH = 6
    EOP_END_EXCEPT = 7
    EOP_END_FINALLY = 8
    EOP_CONTINUE = 9
    EOP_CATCH = 10  # ops after this point cost one tick
    EOP_TRY_EXCEPT = 11
    EOP_TRY_FINALLY = 12
    EOP_WHILE_ID = 13
    EOP_EXIT = 14
    EOP_EXIT_ID = 15
    EOP_SCATTER = 16
    EOP_EXP = 17
    EOP_FOR_LIST_1 = 18
    EOP_FOR_LIST_2 = 19
    EOP_BITOR = 20  # bitwise operators
    EOP_BITAND = 21
    EOP_BITXOR = 22
    EOP_BITSHL = 23
    EOP_BITSHR = 24
    EOP_COMPLEMENT = 25
    Last_Extended_Opcode = 255


class Opcode(Enum):
    OP_IF = 1  # control/statement constructs with 1 tick:
    OP_WHILE = 2
    OP_EIF = 3
    OP_FORK = 4
    OP_FORK_WITH_ID = 5
    OP_FOR_LIST = 6  # retired
    OP_FOR_RANGE = 7
    OP_INDEXSET = 8  # expr-related opcodes with 1 tick:
    OP_PUSH_GET_PROP = 9
    OP_GET_PROP = 10
    OP_CALL_VERB = 11
    OP_PUT_PROP = 12
    OP_BI_FUNC_CALL = 13
    OP_IF_QUES = 14
    OP_REF = 15
    OP_RANGE_REF = 16
    OP_MAKE_SINGLETON_LIST = 17  # arglist-related opcodes with 1 tick:
    OP_CHECK_LIST_FOR_SPLICE = 18
    OP_MULT = 19  # arith binary ops -- 1 tick:
    OP_DIV = 20
    OP_MOD = 21
    OP_ADD = 22
    OP_MINUS = 23
    OP_EQ = 24  # comparison binary ops -- 1 tick:
    OP_NE = 25
    OP_LT = 26
    OP_LE = 27
    OP_GT = 28
    OP_GE = 29
    OP_IN = 30
    OP_AND = 31  # logic binary ops -- 1 tick:
    OP_OR = 32
    OP_UNARY_MINUS = 33  # unary ops -- 1 tick:
    OP_NOT = 34
    OP_PUT = 35  # assignments, 1 tick:
    OP_G_PUT = OP_PUT + NUM_READY_VARS  # variable references, no tick:
    OP_PUSH = 36
    OP_G_PUSH = OP_PUSH + NUM_READY_VARS
    OP_PUSH_CLEAR = 37  # final variable references, no tick:
    OP_G_PUSH_CLEAR = OP_PUSH_CLEAR + NUM_READY_VARS
    OP_IMM = 38  # expr-related opcodes with no tick:
    OP_MAKE_EMPTY_LIST = 39
    OP_LIST_ADD_TAIL = 40
    OP_LIST_APPEND = 41
    OP_PUSH_REF = 42
    OP_PUT_TEMP = 43
    OP_PUSH_TEMP = 44
    OP_JUMP = 45  # control/statement constructs with no ticks:
    OP_RETURN = 46
    OP_RETURN0 = 47
    OP_DONE = 48
    OP_POP = 49
    OP_EXTENDED = 50  # Used to add more opcodes
    OP_MAP_CREATE = 51
    OP_MAP_INSERT = 52
    OPTIM_NUM_START = 53
    Last_Opcode = 255

# OPTIM_NUM_LOW = -10
# OPTIM_NUM_HI = Opcode.Last_Opcode - Opcode.OPTIM_NUM_START + OPTIM_NUM_LOW
