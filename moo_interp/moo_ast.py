import sys
from dataclasses import dataclass
from typing import List, Union

from lark import Lark, Transformer, ast_utils, v_args
from lark.tree import Meta

from .vm import VM, Instruction, Program, StackFrame
from .opcodes import Opcode, Extended_Opcode
from .parser import parser
from .string import MOOString
from .builtin_functions import BuiltinFunctions


this_module = sys.modules[__name__]

# operator to opcode mapping

binary_opcodes = {
    '+': Opcode.OP_ADD,
    '-': Opcode.OP_MINUS,
    "*": Opcode.OP_MULT,
    "/": Opcode.OP_DIV,
    '%': Opcode.OP_MOD,
    '==': Opcode.OP_EQ,
    '!=': Opcode.OP_NE,
    '<': Opcode.OP_LT,
    '>': Opcode.OP_GT,
    '<=': Opcode.OP_LE,
    '>=': Opcode.OP_GE,
    'in': Opcode.OP_IN,
    '&&': Opcode.OP_AND,
    '||': Opcode.OP_OR,
    '^': Extended_Opcode.EOP_EXP,
}

unary_opcodes = {
    '-': Opcode.OP_UNARY_MINUS,
    '!': Opcode.OP_NOT,
}

#
# Define AST
#


class _Ast(ast_utils.Ast):

    def emit_byte(self, opcode: Opcode, operand: Union[int, float, MOOString, None]):
        return Instruction(opcode=opcode, operand=operand)

    def emit_extended_byte(self, opcode: Extended_Opcode):
        return Instruction(opcode=Opcode.OP_EXTENDED, operand=opcode)


class _Expression(_Ast):
    pass


class _Statement(_Ast):

    def to_bytecode(self, program: Program):
        raise NotImplementedError(
            f"to_bytecode not implemented for {self.__class__.__name__}")


@dataclass
class Identifier(_Expression):
    value: str

    def to_bytecode(self, program: Program):
        return [self.emit_byte(Opcode.OP_PUSH, self.value)]


@dataclass
class Splicer(_Expression):
    expression: _Expression

    def to_bytecode(self, program: Program):
        return self.expression.to_bytecode(program) + [self.emit_byte(Opcode.OP_CHECK_LIST_FOR_SPLICE, None)]


@dataclass
class _Literal(_Ast):
    value: any

    def to_bytecode(self, program: Program):
        return [self.emit_byte(Opcode.OP_IMM, self.value)]


@dataclass
class StringLiteral(_Literal, _Expression):
    value: str


@dataclass
class BooleanLiteral(_Literal, _Expression):
    value: bool


@dataclass
class NumberLiteral(_Literal, _Expression):
    value: int

    def to_bytecode(self, program: Program):
        if self.value > -1 and self.value < 256:
            return [self.emit_byte(113+self.value, None)]
        return [self.emit_byte(Opcode.OP_IMM, self.value)]


@dataclass
class FloatLiteral(_Literal, _Expression):
    value: float

    def to_bytecode(self, program: Program):
        return [self.emit_byte(Opcode.OP_IMM, self.value)]


@dataclass
class ObjnumLiteral(_Literal, _Expression):
    value: int

    def to_bytecode(self, program: Program):
        return [self.emit_byte(Opcode.OP_IMM, self.value)]


@dataclass
class _List(_Expression):
    value: list

    def to_bytecode(self, program: Program):
        if (not self.value):
            # empty list
            return [Instruction(opcode=Opcode.OP_MAKE_EMPTY_LIST)]
        result = []
        for index, element in enumerate(self.value):
            result += element.to_bytecode(program)
            if index == 0:
                result += [Instruction(opcode=Opcode.OP_MAKE_SINGLETON_LIST)]
            else:
                result += [Instruction(opcode=Opcode.OP_LIST_ADD_TAIL)]
        return result


@dataclass
class Map(_Expression):
    value: List

    def to_bytecode(self, program: Program):
        result = [Instruction(opcode=Opcode.OP_MAP_CREATE)]
        for child in self.value:
            key, value = child.children
            result += key.to_bytecode(program)
            result += value.to_bytecode(program)
            result += [Instruction(opcode=Opcode.OP_MAP_INSERT)]
            result += [Instruction(opcode=Opcode.OP_POP, operand=2)]
        return result


@dataclass
class UnaryExpression(_Expression):
    operator: str
    operand: _Expression

    def to_bytecode(self, program: Program):
        operand_bc = self.operand.to_bytecode(program)
        return operand_bc + [Instruction(opcode=unary_opcodes[self.operator])]


@dataclass
class BinaryExpression(_Expression):
    left: _Expression
    operator: str
    right: _Expression

    def to_bytecode(self, program: Program):
        left_bc = self.left.to_bytecode(program)
        right_bc = self.right.to_bytecode(program)
        return left_bc + right_bc + [Instruction(opcode=binary_opcodes[self.operator])]


@dataclass
class _Assign(_Statement):
    target: Identifier
    value: _Expression

    def to_bytecode(self, program: Program):
        value_bc = self.value.to_bytecode(program)
        if isinstance(self.target, Identifier):
            return value_bc + [Instruction(opcode=Opcode.OP_PUT, operand=self.target.value)]
        elif isinstance(self.target, _Property):
            return value_bc + self.target.object.to_bytecode(program) + self.target.name.to_bytecode(program) + [Instruction(opcode=Opcode.OP_PUT_PROP)]


@dataclass
class ElseIfClause(_Ast):
    condition: _Expression
    then_block: List[_Statement]


@dataclass
class _IfStatement(_Statement):
    condition: _Expression
    then_block: List[_Statement]
    elseif_clauses: List[ElseIfClause]
    else_block: Union[List[_Statement], None]

    def to_bytecode(self, program: Program):
        condition_bc = self.condition.to_bytecode(program)
        then_block_bc = []
        for stmt in self.then_block.children:
            then_block_bc += stmt.to_bytecode(program)
        # Add IF/EIF bytecode instruction
        if_then_bc = condition_bc + \
            [Instruction(opcode=Opcode.OP_IF, operand=len(then_block_bc) + 1)]

        # Add then block and JUMP bytecode instruction
        # Operand for JUMP will be filled in later
        if_then_bc += then_block_bc + \
            [Instruction(opcode=Opcode.OP_JUMP, operand=None)]

        done_bc = []  # Placeholder for 'done' bytecode, will be filled in later
        else_block_bc = []

        # Add bytecode for ELSE block if it exists
        if self.else_block is not None:
            else_block_bc = []
            for stmt in self.else_block.children:
                else_block_bc += stmt.to_bytecode(program)
            done_bc = else_block_bc

        # Add bytecode for ELSEIF clauses if they exist
        for elseif in self.elseif_clauses:
            elseif_bc = elseif.to_bytecode(program)
            if_then_bc += elseif_bc
            done_bc = elseif_bc  # Update 'done' bytecode

        # Now that we know the length of the bytecode for the 'done' block, we can fill in the operand for the JUMP instruction
        if_then_bc[-1].operand = len(done_bc)

        # Combine all the bytecode together and return it
        return if_then_bc + done_bc


@dataclass
class _FunctionCall(_Expression):
    name: str
    arguments: List[_Expression]

    def to_bytecode(self, program: Program):
        result = self.arguments.to_bytecode(program)
        builtin_id = BuiltinFunctions().get_id_by_name(self.name)
        result += [Instruction(opcode=Opcode.OP_BI_FUNC_CALL,
                               operand=builtin_id)]
        return result


@dataclass
class _Property(_Expression):
    object: _Expression
    name: str

    def to_bytecode(self, program: Program):
        result = self.object.to_bytecode(program)
        result += self.name.to_bytecode(program)
        result += [Instruction(opcode=Opcode.OP_GET_PROP)]
        return result


@dataclass
class _VerbCall(_Expression):
    object: _Expression
    name: _Expression
    arguments: List[_Expression]

    def to_bytecode(self, program: Program):
        result = self.object.to_bytecode(program)
        result += self.name.to_bytecode(program)
        result += self.arguments.to_bytecode(program)
        result += [Instruction(opcode=Opcode.OP_CALL_VERB)]
        return result


@dataclass
class ReturnStatement(_Statement):
    value: _Expression = None

    def to_bytecode(self, program: Program):
        if self.value is None:
            return [Instruction(opcode=Opcode.OP_RETURN0)]
        value_bc = self.value.to_bytecode(program)
        return value_bc + [Instruction(opcode=Opcode.OP_RETURN)]


@dataclass
class WhileStatement(_Statement):
    condition: _Expression
    body: List[_Statement]


@dataclass
class WhileStatement(_Statement):
    condition: _Expression
    body: List[_Statement]

    def to_bytecode(self, program: Program):
        # First generate bytecode for condition and body
        condition_bc = self.condition.to_bytecode(program)
        body_bc = []
        for stmt in self.body.children:
            body_bc += stmt.to_bytecode(program)

        # Calculate the relative jump points
        # +1 accounts for the OP_JUMP instruction at the end
        jump_over_body = len(body_bc) + 1
        # +1 accounts for the OP_WHILE instruction
        jump_to_start = -(len(condition_bc) + len(body_bc) + 1)

        # Generate bytecode
        return (
            condition_bc
            + [Instruction(opcode=Opcode.OP_WHILE, operand=jump_over_body)]
            + body_bc
            + [Instruction(opcode=Opcode.OP_JUMP, operand=jump_to_start)]
        )


class ToAst(Transformer):
    def BOOLEAN(self, b):
        return BooleanLiteral(b == "true")

    def IDENTIFIER(self, s):
        if s.value == "true" or s.value == "false":
            return BooleanLiteral(s.value == "true")
        return Identifier(s.value)

    def ESCAPED_STRING(self, s):
        return StringLiteral(s[1:-1])

    def NUMBER(self, n):
        return NumberLiteral(int(n))

    def FLOAT(self, n):
        return FloatLiteral(float(n))

    def objnum(self, n):
        return ObjnumLiteral(int(n[0].value))

    def assign(self, assignment):
        target, value = assignment
        return _Assign(target=target, value=value)

    def list(self, args):
        if len(args) == 1 and args[0] == None:
            return _List([])
        return _List(args)

    def dict(self, entries):
        return Map(entries)

    def if_statement(self, if_clause):
        condition, then_block = if_clause[0].children
        if len(if_clause) > 2:
            elseif_clauses = [ElseIfClause(clause.children[0], clause.children[1].children)
                              for clause in if_clause[1:-1]]
        else:
            elseif_clauses = []
        else_clause = if_clause[-1].children[0] if len(if_clause) > 1 else None
        return _IfStatement(condition, then_block, elseif_clauses, else_clause)

    def function_call(self, call):
        name, args = call
        return _FunctionCall(name.value, _List(args.children))

    def property(self, access):
        object, name = access
        # the original /* Treat foo.bar like foo.("bar") for simplicity */ so we need to convert to string
        name = StringLiteral(name.value)
        return _Property(object, name)

    def verb_call(self, call):
        object, name, args = call
        return _VerbCall(object, name, _List(args.children))

    @v_args(inline=True)
    def start(self, x):
        return x


#
# Define Parser
#
transformer = ast_utils.create_transformer(this_module, ToAst())


def parse(text):
    if isinstance(text, list):
        text = "".join(text)
    tree = parser.parse(text)
    return transformer.transform(tree)

#
# Test
#


def compile(tree):
    if isinstance(tree, list):
        tree = parse("".join(tree))
    elif isinstance(tree, str):
        tree = parse(tree)
    bc = []
    for node in tree.children:
        bc += node.to_bytecode(None)
    bc = bc + [Instruction(opcode=Opcode.OP_DONE)]
    prog = Program()
    frame = StackFrame(func_id=0, prog=prog, ip=0, stack=bc)
    return frame


def disassemble(frame: StackFrame):
    bc = frame.stack
    for instruction in bc:
        if isinstance(instruction.opcode, int) and instruction.opcode >= 113:
            print(f"Num            {instruction.opcode-113}")
            continue
        print(
            f"{instruction.opcode.value} {instruction.opcode.name} {type(instruction.operand).__name__} {instruction.operand}")


def run(frame: StackFrame, debug=True, player=-1, this=-1):
    if isinstance(frame, str):
        frame = compile(frame)
    frame.debug = debug
    frame.player = player
    frame.this = this
    vm = VM()
    vm.call_stack = [frame]
    try:
        for top in vm.run():
            print(vm.stack)
            pass
    except Exception as e:
        raise VMRunError(vm, e)
    return vm


class VMRunError(Exception):

    def __init__(self, vm, message):
        self.vm = vm
        self.message = message


if __name__ == '__main__':
    print(parse("""
        if (1==1) player:tell("hello"); endif
    """))
