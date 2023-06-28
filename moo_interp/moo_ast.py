import sys
from dataclasses import dataclass
from typing import List, Union

from lark import Lark, Transformer, ast_utils, v_args
from lark.tree import Meta

from .vm import Instruction, Program, StackFrame
from .opcodes import Opcode, Extended_Opcode
from .parser import parser
from .string import MOOString

this_module = sys.modules[__name__]

# operator to opcode mapping

binary_opcodes = {
    '+': Opcode.OP_ADD,
    '-': Opcode.OP_MINUS,
    '==': Opcode.OP_EQ,
    '!=': Opcode.OP_NE,
    '<': Opcode.OP_LT,
    '>': Opcode.OP_GT,
    '<=': Opcode.OP_LE,
    '>=': Opcode.OP_GE,
    'in': Opcode.OP_IN,
    '&&': Opcode.OP_AND,
    '||': Opcode.OP_OR,

}

unary_opcodes = {
    '-': Opcode.OP_UNARY_MINUS,
    '!': Opcode.OP_NOT,
}

#
# Define AST
#


class _Ast(ast_utils.Ast):
    pass


class _Expression(_Ast):
    pass


class _Statement(_Ast):

    def to_bytecode(self, program: Program):
        raise NotImplementedError(
            f"to_bytecode not implemented for {self.__class__.__name__}")


@dataclass
class Identifier(_Expression):
    value: str


@dataclass
class StringLiteral(_Expression):
    value: str


def to_bytecode(self, program: Program):
    return [Instruction(opcode=Opcode.PUSH, operand=MOOString(self.value))]


@dataclass
class BooleanLiteral(_Expression):
    value: bool

    def to_bytecode(self, program: Program):
        return [Instruction(opcode=Opcode.OP_PUSH, operand=self.value)]


@dataclass
class NumberLiteral(_Expression):
    value: int

    def to_bytecode(self, program: Program):
        return [Instruction(opcode=Opcode.OP_PUSH, operand=self.value)]


@dataclass
class FloatLiteral(_Expression):
    value: float

    def to_bytecode(self, program: Program):
        return [Instruction(opcode=Opcode.PUSH, operand=self.value)]


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
        left_bc = []
        for child in self.left.children:
            left_bc += child.to_bytecode(program)
        right_bc = []
        for child in self.right.children:
            right_bc += child.to_bytecode(program)
        return left_bc + right_bc + [Instruction(opcode=binary_opcodes[self.operator])]


@dataclass
class Assignment(_Statement):
    target: Identifier
    value: _Expression

    def to_bytecode(self, program: Program):
        value_bc = self.value.to_bytecode(program)
        return value_bc + [Instruction(opcode=Opcode.OP_PUT, operand=self.target.value)]


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
class ReturnStatement(_Statement):
    value: _Expression

    def to_bytecode(self, program: Program):
        value_bc = self.value.to_bytecode(program)
        return value_bc + [Instruction(opcode=Opcode.OP_RETURN)]


@dataclass
class WhileStatement(_Statement):
    condition: _Expression
    body: List[_Statement]

    def to_bytecode(self, program: Program):
        condition_bc = self.condition.to_bytecode(program)
        body_bc = []
        for stmt in self.body.children:
            body_bc += stmt.to_bytecode(program)
        return condition_bc + [Instruction(opcode=Opcode.OP_WHILE, operand=None)] + body_bc + [Instruction(opcode=Opcode.OP_JUMP, operand=-len(body_bc) - 1)]


class ToAst(Transformer):
    def BOOLEAN(self, b):
        return BooleanLiteral(b == "true")

    def IDENTIFIER(self, s):
        if s.value == "true" or s.value == "false":
            return BooleanLiteral(s.value == "true")
        return Identifier(s.value)

    def STRING(self, s):
        return StringLiteral(s[1:-1])

    def NUMBER(self, n):
        return NumberLiteral(int(n))

    def FLOAT(self, n):
        return FloatLiteral(float(n))

    def assignment(self, target, _, value):
        return Assignment(target, value)

    def if_statement(self, if_clause, *elseif_clauses, else_clause=None):
        condition, then_block = if_clause[0].children
        if elseif_clauses:
            elseif_clauses = [ElseIfClause(*clause)
                              for clause in elseif_clauses]
        else_block = else_clause or None  # in case else_clause is not provided
        return _IfStatement(condition, then_block, elseif_clauses, else_block)

    # ... other transformation functions ...

    @v_args(inline=True)
    def start(self, x):
        return x


#
# Define Parser
#
transformer = ast_utils.create_transformer(this_module, ToAst())


def parse(text):
    tree = parser.parse(text)
    return transformer.transform(tree)

#
# Test
#


def compile(tree):
    bc = []
    for node in tree.children:
        bc += node.to_bytecode(None)
    bc = bc + [Instruction(opcode=Opcode.OP_DONE)]
    prog = Program()
    frame = StackFrame(prog, 0, ip=0, stack=bc)
    return frame


def disassemble(bc: List[Instruction]):
    for instruction in bc:
        print(
            f"{instruction.opcode.value} {instruction.opcode.name} {type(instruction.operand).__name__} {instruction.operand}")


if __name__ == '__main__':
    print(parse("""
        if (1==1) player:tell("hello"); endif
    """))
