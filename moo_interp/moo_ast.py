import sys
from dataclasses import dataclass, field
from typing import List, Optional, Union
import lark
from lark import Lark, Transformer, ast_utils, v_args
from lark.tree import Meta

from .builtin_functions import BuiltinFunctions
from .moo_types import to_moo
from .opcodes import Extended_Opcode, Opcode
from .parser import parser
from .string import MOOString
from .vm import VM, Instruction, Program, StackFrame

this_module = sys.modules[__name__]


@dataclass
class CompilerState:
    last_label = 0

    def next_label(self):
        self.last_label += 1
        return self.last_label


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

    def emit_byte(self, opcode: Opcode, operand: Union[int, float, MOOString, None], label: str = None):
        return Instruction(opcode=opcode, operand=operand, label=label)

    def emit_extended_byte(self, opcode: Extended_Opcode, label: str = None):
        return Instruction(opcode=Opcode.OP_EXTENDED, operand=opcode, label=label)

    def to_moo(self) -> str:
        raise NotImplementedError(
            f"to_moo not implemented for {self.__class__.__name__}")


@dataclass
class VerbCode(_Ast):
    children: List[_Ast]

    def to_bytecode(self, state: CompilerState, program: Program):
        result = []
        for node in self.children:
            result += node.to_bytecode(state, program)
        return result

    def to_moo(self) -> str:
        return "\n".join([node.to_moo() for node in self.children])


class _Expression(_Ast):
    pass


class _Statement(_Ast):

    def to_bytecode(self, state: CompilerState, program: Program):
        raise NotImplementedError(
            f"to_bytecode not implemented for {self.__class__.__name__}")


@dataclass
class SingleStatement(_Ast):
    statement: _Statement

    def to_bytecode(self, state: CompilerState, program: Program):
        return self.statement.to_bytecode(state, program)

    def to_moo(self) -> str:
        return self.statement.to_moo() + ";"


@dataclass(init=False)
class _Body(_Ast):
    statements: List[_Statement]

    def __init__(self, *statements: List[_Statement]):
        self.statements = statements

    def to_bytecode(self, state: CompilerState, program: Program):
        result = []
        for stmt in self.statements:
            result += stmt.to_bytecode(state, program)
        return result

    def to_moo(self) -> str:
        return "\n".join([stmt.to_moo() for stmt in self.statements])


@dataclass
class Identifier(_Expression):
    value: str

    def to_bytecode(self, state: CompilerState, program: Program):
        return [self.emit_byte(Opcode.OP_PUSH, MOOString(self.value))]

    def to_moo(self):
        return self.value


@dataclass
class Splicer(_Expression):
    expression: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        return self.expression.to_bytecode(state, program) + [self.emit_byte(Opcode.OP_CHECK_LIST_FOR_SPLICE, None)]

    def to_moo(self) -> str:
        return '@' + self.expression.to_moo()


@dataclass
class _Literal(_Ast):
    value: any

    def to_bytecode(self, state: CompilerState, program: Program):
        return [self.emit_byte(Opcode.OP_IMM, to_moo(self.value))]

    def to_moo(self) -> MOOString:
        return str(self.value)


@dataclass
class StringLiteral(_Literal, _Expression):
    value: str


@dataclass
class BooleanLiteral(_Literal, _Expression):
    value: bool

    def to_moo(self):
        return str(self.value).lower()


@dataclass
class NumberLiteral(_Literal, _Expression):
    value: int

    def to_bytecode(self, state: CompilerState, program: Program):
        if self.value > -1 and self.value < 256:
            return [self.emit_byte(113+self.value, None)]
        return [self.emit_byte(Opcode.OP_IMM, self.value)]


@dataclass
class FloatLiteral(_Literal, _Expression):
    value: float

    def to_bytecode(self, state: CompilerState, program: Program):
        return [self.emit_byte(Opcode.OP_IMM, self.value)]


@dataclass
class ObjnumLiteral(_Literal, _Expression):
    value: int

    def to_bytecode(self, state: CompilerState, program: Program):
        return [self.emit_byte(Opcode.OP_IMM, self.value)]

    def to_moo(self) -> str:
        return "#" + str(self.value)


@dataclass
class _List(_Expression):
    value: list

    def to_bytecode(self, state: CompilerState, program: Program):
        if (not self.value):
            # empty list
            return [Instruction(opcode=Opcode.OP_MAKE_EMPTY_LIST)]
        result = []
        for index, element in enumerate(self.value):
            result += element.to_bytecode(state, program)
            if index == 0:
                result += [Instruction(opcode=Opcode.OP_MAKE_SINGLETON_LIST)]
            else:
                result += [Instruction(opcode=Opcode.OP_LIST_ADD_TAIL)]
        return result

    def to_moo(self) -> str:
        return "{" + ", ".join([element.to_moo() for element in self.value]) + "}"


@dataclass
class Map(_Expression):
    value: List

    def to_bytecode(self, state: CompilerState, program: Program):
        result = [Instruction(opcode=Opcode.OP_MAP_CREATE)]
        for child in self.value:
            key, value = child.children
            result += key.to_bytecode(state, program)
            result += value.to_bytecode(state, program)
            result += [Instruction(opcode=Opcode.OP_MAP_INSERT)]
            result += [Instruction(opcode=Opcode.OP_POP, operand=2)]
        return result

    def to_moo(self) -> str:
        return "[" + ", ".join([f"{key.to_moo()}: {value.to_moo()}" for key, value in self.value]) + "]"


@dataclass
class UnaryExpression(_Expression):
    operator: str
    operand: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        operand_bc = self.operand.to_bytecode(state, program)
        return operand_bc + [Instruction(opcode=unary_opcodes[self.operator])]

    def to_moo(self) -> str:
        return f"{self.operator}{self.operand.to_moo()}"


@dataclass
class BinaryExpression(_Expression):
    left: _Expression
    operator: str
    right: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        left_bc = self.left.to_bytecode(state, program)
        right_bc = self.right.to_bytecode(state, program)
        return left_bc + right_bc + [Instruction(opcode=binary_opcodes[self.operator])]

    def to_moo(self) -> str:
        return f"{self.left.to_moo()} {self.operator} {self.right.to_moo()}"


@dataclass
class _Assign(_Statement):
    target: Identifier
    value: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        value_bc = self.value.to_bytecode(state, program)
        if isinstance(self.target, Identifier):
            return value_bc + [Instruction(opcode=Opcode.OP_PUT, operand=self.target.value)]
        elif isinstance(self.target, _Property):
            return value_bc + self.target.object.to_bytecode(state, program) + self.target.name.to_bytecode(state, program) + [Instruction(opcode=Opcode.OP_PUT_PROP)]

    def to_moo(self) -> str:
        return f"{self.target.to_moo()} = {self.value.to_moo()}"


@dataclass
class ElseifClause(_Ast):
    condition: _Expression
    then_block:         _Body

    def to_bytecode(self, state: CompilerState, program: Program):
        condition_bc = self.condition.to_bytecode(state, program)
        then_block_bc = self.then_block.to_bytecode(state, program)
        return condition_bc + [Instruction(opcode=Opcode.OP_EIF, operand=len(then_block_bc) + 1)] + then_block_bc

    def to_moo(self) -> str:
        result = f"elseif ({self.condition.to_moo()})\n"
        if self.then_block is not None:
            result = result + self.then_block.to_moo()
        return result


@dataclass
class IfClause(_Ast):
    condition: _Expression
    then_block: _Body

    def to_bytecode(self, state: CompilerState, program: Program):
        condition_bc = self.condition.to_bytecode(state, program)
        then_block_bc = self.then_block.to_bytecode(state, program)
        return condition_bc + [Instruction(opcode=Opcode.OP_IF, operand=len(then_block_bc) + 1)] + then_block_bc

    def to_moo(self) -> str:
        result = f"if ({self.condition.to_moo()})\n"
        if self.then_block is not None:
            result += self.then_block.to_moo()
        return result


@dataclass
class ElseClause(_Ast):
    then_block: _Body

    def to_bytecode(self, state: CompilerState, program: Program):
        then_block_bc = []
        if self.then_block is not None:
            then_block_bc = self.then_block.to_bytecode(state, program)
        return then_block_bc

    def to_moo(self) -> str:
        result = "else\n"
        if self.then_block is not None:
            result += self.then_block.to_moo()
        return result


@dataclass
class _IfStatement(_Statement):
    if_clause: IfClause
    elseif_clauses: List[ElseifClause] = field(default_factory=list)
    else_clause: Optional[ElseClause] = None

    def to_bytecode(self, state: CompilerState, program: Program):
        end_label = state.next_label()
        if_clause_bc = self.if_clause.to_bytecode(
            state, program) + [Instruction(opcode=Opcode.OP_JUMP, operand=end_label)]

        elseif_clauses_bc = []
        for elseif in self.elseif_clauses:
            elseif_clauses_bc += elseif.to_bytecode(state, program)
            elseif_clauses_bc += [Instruction(
                opcode=Opcode.OP_JUMP, operand=end_label)]
        else_clause_bc = []
        if self.else_clause is not None:
            else_clause_bc = self.else_clause.to_bytecode(state, program)
        full_bc = if_clause_bc + elseif_clauses_bc + else_clause_bc
        # update end label for jumps relative to the end of the if statement
        # (i.e. jumps to the end of the if statement before the elseif or else clauses)
        for index, instruction in enumerate(full_bc):
            if instruction.opcode == Opcode.OP_JUMP:
                instruction.operand = len(full_bc) - index
        return full_bc 

    def to_moo(self) -> str:
        nl = "\n"
        result = self.if_clause.to_moo()
        result += "\n".join([elseif.to_moo()
                             for elseif in self.elseif_clauses])
        if self.else_clause is not None:
            result += f"\n{self.else_clause.to_moo()}"
        result += "\nendif"
        return result


@dataclass
class _FunctionCall(_Expression):
    name: str
    arguments: List[_Expression]

    def to_bytecode(self, state: CompilerState, program: Program):
        result = self.arguments.to_bytecode(state, program)
        builtin_id = BuiltinFunctions().get_id_by_name(self.name)
        result += [Instruction(opcode=Opcode.OP_BI_FUNC_CALL,
                               operand=builtin_id)]
        return result

    def to_moo(self) -> str:
        arguments = ", ".join([arg.to_moo() for arg in self.arguments.value])
        return f"{self.name}({arguments})"


@dataclass
class _Property(_Expression):
    object: _Expression
    name: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        result = self.object.to_bytecode(state, program)
        result += self.name.to_bytecode(state, program)
        result += [Instruction(opcode=Opcode.OP_GET_PROP)]
        return result

    def to_moo(self) -> str:
        return f"{self.object.to_moo()}.{self.name.to_moo()}"


@dataclass
class DollarProperty(_Ast):
    name: StringLiteral

    def to_bytecode(self, state: CompilerState, program: Program):
        # $prop means #0.prop
        result = [Instruction(opcode=Opcode.OP_IMM, operand=0)]
        result += self.name.to_bytecode(state, program)
        result += [Instruction(opcode=Opcode.OP_GET_PROP)]
        return result

    def to_moo(self) -> str:
        return f"${self.name.to_moo()}"


@dataclass
class DollarVerbCall(_Ast):
    name = StringLiteral
    arguments = List[_Expression]

    def to_bytecode(self, state: CompilerState, program: Program):
        # $verb() means #0:verb()
        result = [Instruction(opcode=Opcode.OP_IMM, operand=0)]
        result += self.name.to_bytecode(state, program)
        result += self.arguments.to_bytecode(state, program)
        result += [Instruction(opcode=Opcode.OP_CALL_VERB)]
        return result

    def to_moo(self) -> str:
        arguments = ", ".join([arg.to_moo() for arg in self.arguments.value])
        return f"${self.name.to_moo()}({arguments})"


@dataclass
class _VerbCall(_Expression):
    object: _Expression
    name: _Expression
    arguments: List[_Expression]

    def to_bytecode(self, state: CompilerState, program: Program):
        result = self.object.to_bytecode(state, program)
        result += self.name.to_bytecode(state, program)
        result += self.arguments.to_bytecode(state, program)
        result += [Instruction(opcode=Opcode.OP_CALL_VERB)]
        return result

    def to_moo(self) -> str:
        arguments = ", ".join([arg.to_moo() for arg in self.arguments.value])
        return f"{self.object.to_moo()}:{self.name.to_moo()}({arguments})"


@dataclass
class Index(_Expression):
    object: _Expression
    index: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        result = self.object.to_bytecode(state, program)
        result += self.index.to_bytecode(state, program)
        result += self.emit_extended_byte(Extended_Opcode.EOP_INDEX)
        return result

    def to_moo(self) -> str:
        return f"{self.object.to_moo()}[{self.index.to_moo()}]"


@dataclass
class ReturnStatement(_Statement):
    value: _Expression = None

    def to_bytecode(self, state: CompilerState, program: Program):
        if self.value is None:
            return [Instruction(opcode=Opcode.OP_RETURN0)]
        value_bc = self.value.to_bytecode(state, program)
        return value_bc + [Instruction(opcode=Opcode.OP_RETURN)]

    def to_moo(self) -> str:
        return f"return {self.value.to_moo()}"


@dataclass
class WhileStatement(_Statement):
    condition: _Expression
    body: List[_Statement]

    def to_bytecode(self, state: CompilerState, program: Program):
        # First generate bytecode for condition and body
        condition_bc = self.condition.to_bytecode(state, program)
        body_bc = []
        for stmt in self.body.children:
            body_bc += stmt.to_bytecode(state, program)

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

    def to_moo(self) -> str:
        return f"while ({self.condition.to_moo()}) {self.body.to_moo()} endwhile"


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

    def if_statement(self, args):
        if_clause = args[0]
        extra_clauses = args[1:]
        else_clause = None
        if extra_clauses and isinstance(extra_clauses[-1], ElseClause):
            else_clause = extra_clauses.pop()
        return _IfStatement(if_clause, extra_clauses, else_clause)

    def body(self, block):
        if len(block) == 1 and isinstance(block[0], lark.tree.Tree) and block[0].children == []:
            return _Body()
        return _Body(*block)

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
        return VerbCode(x.children)


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
    state = CompilerState()
    for node in tree.children:
        bc += node.to_bytecode(state, None)
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
