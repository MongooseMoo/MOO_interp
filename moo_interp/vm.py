import inspect
import warnings
from enum import Enum
from functools import wraps
from logging import basicConfig, getLogger
from typing import (Any, Callable, Dict, List, Mapping, Optional, Tuple, Union,
                    cast)

from attr import define, field

from moo_interp.string import MOOString

from .list import MOOList
from .map import MOOMap
from .opcodes import Extended_Opcode, Opcode

basicConfig(level="DEBUG")
logger = getLogger(__name__)


""" LambdaMOO Virtual Machine

    This module implements the LambdaMOO Virtual Machine, which is
    responsible for executing the bytecode generated by the compiler.
    The VM is a stack machine with a single stack of values and a
    separate call stack.  The VM is also responsible for handling
    errors, which are represented by Python exceptions.
"""


class VMError(Exception):
    """Base class for all VM errors"""
    pass


class FinallyReason(Enum):
    FIN_FALL_THRU = 0
    FIN_RAISE = 1
    FIN_UNCAUGHT = 2
    FIN_RETURN = 3
    FIN_ABORT = 4
    FIN_EXIT = 5


@define
class Instruction:
    """Represents a single bytecode instruction"""
    opcode: Opcode
    operand: Optional[int] = None
    numbytes_label: int = field(default=0)
    numbytes_literal: int = field(default=0)
    numbytes_fork: int = field(default=0)
    numbytes_var_name: int = field(default=0)
    numbytes_stack: int = field(default=0)
    size: int = field(default=0)
    max_stack: int = field(default=0)


@define
class Program:
    first_lineno: int = field(default=0)
    literals: List[Any] = field(factory=list)
    fork_vectors: List[int] = field(factory=list)
    var_names: List[str] = field(factory=list)


@define
class StackFrame:
    """Represents a single frame in the call stack"""
    func_id: int
    prog: Program
    ip: int
    stack: List[Any] = field(factory=list)
    rt_env: List[Any] = field(factory=list)
    bf_ip: int = field(default=0)
    temp: int = field(default=0)
    this: int = field(default=0)
    player: int = field(default=0)
    verb: str = field(default="")
    verb_name: str = field(default="")
    debug: bool = field(default=False)
    threaded: bool = field(default=False)


def operator(opcode):
    """Operator decorator.
    """
    def decorator(func):
        sig = inspect.signature(func)
        func_params = list(sig.parameters.values())[1:]  # Skip 'self'

        num_args = len(func_params)

        unannotated_params = [
            param.name for param in func_params if param.annotation is inspect._empty]

        if unannotated_params:
            warnings.warn(
                f"Parameter(s) {', '.join(unannotated_params)} of {func.__name__} are not annotated, will be considered as 'Any'")

        @wraps(func)
        def wrapper(self, *args):
            # Check for stack underflow
            if isinstance(opcode, Opcode) and opcode != Opcode.OP_PUSH:
                if len(self.stack) < num_args:
                    raise VMError(
                        f"Stack underflow in opcode {opcode} in function {func.__name__}")

            # Check types match annotations
            for (param, arg) in zip(func_params, args):
                if param.annotation not in {inspect._empty, Any}:
                    if not isinstance(arg, param.annotation):
                        raise VMError(
                            f"Argument {arg} in function {func.__name__} is not of expected type {param.annotation}")

            # Call the function
            return func(self, *args)

        # Store the opcode and number of arguments on the function itself
        if isinstance(opcode, Opcode):
            wrapper.opcode = opcode
        elif isinstance(opcode, Extended_Opcode):
            wrapper.eopcode = opcode
        wrapper.num_args = num_args

        return wrapper
    return decorator


@define
class VM:
    """The virtual machine"""

    stack: List[Any] = field(factory=list)
    call_stack: List[StackFrame] = field(factory=list)
    opcode_handlers: Dict[Union[Opcode, Extended_Opcode],
                          Callable] = field(factory=dict)

    def __init__(self):
        super().__init__()
        self.stack = []
        self.call_stack = []
        self.opcode_handlers = {}
        handled_opcodes = set()

        # Register all opcode handlers
        for name in dir(self):
            method = getattr(self, name, None)
            if hasattr(method, 'opcode') or hasattr(method, 'eopcode'):
                opcode = method.opcode if hasattr(
                    method, 'opcode') else method.eopcode
                self.opcode_handlers[opcode] = method
                handled_opcodes.add(opcode)

        # Set of all opcodes
        all_opcodes = set(Opcode)
        unhandled_opcodes = all_opcodes - handled_opcodes

        if unhandled_opcodes:
            warnings.warn(
                f"The following opcodes are not implemented: {[opcode.name for opcode in unhandled_opcodes]}", UserWarning)

        # Set of all extended opcodes
        all_eopcodes = set(Extended_Opcode)
        unhandled_eopcodes = all_eopcodes - handled_opcodes

        if unhandled_eopcodes:
            warnings.warn(
                f"The following extended opcodes are not implemented: {[opcode.name for opcode in unhandled_eopcodes]}", UserWarning)

    def push(self, value: Any) -> None:
        """Push a value onto the stack"""
        self.stack.append(value)

    def pop(self) -> Any:
        """Pop a value off the stack"""
        try:
            return self.stack.pop()
        except IndexError:
            raise VMError("Stack underflow")

    def step(self) -> None:
        """Execute the next instruction in the current stack frame."""
        if not self.call_stack:
            return

        frame = self.call_stack[-1]

        if frame.ip >= len(frame.stack):
            self.call_stack.pop()
            return

        instr = frame.stack[frame.ip]
        handler = self.opcode_handlers.get(instr.opcode)
        if not handler:
            # Handle extended opcode
            if instr.opcode == Opcode.OP_EXTENDED:
                handler = self.opcode_handlers.get(
                    Extended_Opcode(instr.operand))
                if not handler:
                    raise VMError(f"Unknown extended opcode {instr.operand}")
            if not handler:
                raise VMError(f"Unknown opcode {instr.opcode}")
        logger.debug(f"Executing {instr.opcode} {instr.operand}")
        args = []
        if instr.opcode == Opcode.OP_PUSH:
            args = [instr.operand]
        elif handler.num_args:
            args = self.stack[-handler.num_args:]

        logger.debug(f"Args: {args}")

        try:
            if instr.opcode != Opcode.OP_POP:
                result = handler(*args)

                if handler.num_args and instr.opcode != Opcode.OP_PUSH:
                    del self.stack[-handler.num_args:]

                self.stack.append(result)
        except Exception as e:
            raise VMError(f"Error executing opcode: {e}")

        frame.ip += 1
        # pop the stack frame if we've reached the end of the stack
        if frame.ip >= len(frame.stack):
            self.call_stack.pop()

    # Basic opcode implementations

    @operator(Opcode.OP_PUSH)
    def handle_push(self, value: Any):
        return value

    @operator(Opcode.OP_POP)
    def handle_pop(self):
        return self.pop()

    @operator(Opcode.OP_ADD)
    def handle_add(self, op1, op2):
        return op1 + op2

    @operator(Opcode.OP_MINUS)
    def handle_subtract(self, op1, op2):
        return op1 - op2

    @operator(Opcode.OP_MULT)
    def handle_multiply(self, op1, op2):
        return op1 * op2

    @operator(Opcode.OP_DIV)
    def handle_divide(self, op1, op2):
        try:
            return op1 / op2
        except ZeroDivisionError:
            raise VMError("Division by zero")

    @operator(Opcode.OP_MOD)
    def handle_mod(self, op1, op2):
        try:
            return op1 % op2
        except ZeroDivisionError:
            raise VMError("Division by zero")

    @operator(Opcode.OP_EQ)
    def handle_eq(self, op1, op2):
        return op1 == op2

    @operator(Opcode.OP_IN)
    def handle_in(self, rhs, lhs):
        # either 0 if not in the list or the index if it is
        index = rhs.find(lhs)
        if index == -1:
            return 0
        return index

    @operator(Opcode.OP_NE)
    def handle_ne(self, op1, op2):
        return op1 != op2

    @operator(Opcode.OP_LT)
    def handle_lt(self, op1, op2):
        return op1 < op2

    @operator(Opcode.OP_LE)
    def handle_le(self, op1, op2):
        return op1 <= op2

    @operator(Opcode.OP_GT)
    def handle_gt(self, op1, op2):
        return op1 > op2

    @operator(Opcode.OP_GE)
    def handle_ge(self, op1, op2):
        return op1 >= op2

    @operator(Opcode.OP_AND)
    def handle_and(self, op1, op2):
        return op1 and op2

    @operator(Opcode.OP_OR)
    def handle_or(self, op1, op2):
        return op1 or op2

    @operator(Opcode.OP_NOT)
    def handle_not(self, op1):
        return not op1

    @operator(Opcode.OP_UNARY_MINUS)
    def handle_unary_minus(self, op1):
        return -op1

    # Extended opcode implementations

    @operator(Extended_Opcode.EOP_BITOR)
    def handle_bitor(self, op1, op2):
        return op1 | op2

    @operator(Extended_Opcode.EOP_BITAND)
    def handle_bitand(self, op1: int, op2: int):
        return op1 & op2

    @operator(Extended_Opcode.EOP_BITXOR)
    def handle_bitxor(self, op1, op2):
        return op1 ^ op2

    @operator(Extended_Opcode.EOP_BITSHL)
    def handle_bitshl(self, op1, op2):
        return op1 << op2

    @operator(Extended_Opcode.EOP_BITSHR)
    def handle_bitshr(self, op1, op2):
        return op1 >> op2

    @operator(Extended_Opcode.EOP_EXP)
    def handle_exp(self, lhs,   rhs):
        return lhs ** rhs

    # List operations

    @operator(Opcode.OP_MAKE_EMPTY_LIST)
    def handle_make_empty_list(self) -> MOOList:
        return MOOList()

    @operator(Opcode.OP_LIST_ADD_TAIL)
    def handle_list_add_tail(self, tail, lst: MOOList) -> MOOList:
        if not isinstance(lst, MOOList):
            raise VMError("Expected list")
        lst.append(tail)
        return lst

    @operator(Opcode.OP_LIST_APPEND)  # extend in Python
    def handle_list_append(self, lst1: MOOList, lst2: MOOList) -> MOOList:
        if not isinstance(lst1, MOOList) or not isinstance(lst2, MOOList):
            raise VMError("Expected list")
        return lst1 + lst2

    @operator(Opcode.OP_MAKE_SINGLETON_LIST)
    def handle_make_singleton_list(self, value) -> MOOList:
        return MOOList([value])

    # Map operations

    @operator(Opcode.OP_MAP_CREATE)
    def handle_make_empty_map(self) -> MOOMap:
        return MOOMap()

    @operator(Opcode.OP_MAP_INSERT)
    def handle_map_insert(self, key: MOOString, value, mapping: MOOMap) -> MOOMap:
        if not isinstance(mapping, MOOMap):
            raise VMError("Expected map")
        mapping[key] = value
        return mapping

    # Return Operations

    @operator(Opcode.OP_RETURN)
    def handle_return(self, value):
        self.call_stack.pop()
        return value

    @operator(Opcode.OP_RETURN0)
    def handle_return0(self):
        self.call_stack.pop()
        return 0

    @operator(Opcode.OP_DONE)
    def handle_done(self):
        return 0
