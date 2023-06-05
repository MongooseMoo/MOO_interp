import sys
from dataclasses import dataclass
from typing import List, Union

from lark import Lark, Transformer, ast_utils, v_args
from lark.tree import Meta
from .opcodes import Opcode
from .parser import parser

this_module = sys.modules[__name__]

#
# Define AST
#


class _Ast(ast_utils.Ast):
    pass


class _Expression(_Ast):
    pass


class _Statement(_Ast):
    pass


@dataclass
class Identifier(_Expression):
    value: str


@dataclass
class StringLiteral(_Expression):
    value: str


@dataclass
class NumberLiteral(_Expression):
    value: int


@dataclass
class FloatLiteral(_Expression):
    value: float


@dataclass
class BinaryExpression(_Expression):
    operator: str
    left: _Expression
    right: _Expression


@dataclass
class Assignment(_Statement):
    target: Identifier
    value: _Expression


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


@dataclass
class WhileStatement(_Statement):
    condition: _Expression
    body: List[_Statement]

# ... other classes ...

#
# Transformer
#


class ToAst(Transformer):
    def IDENTIFIER(self, s):
        return Identifier(s)

    def STRING(self, s):
        return StringLiteral(s[1:-1])

    def NUMBER(self, n):
        return NumberLiteral(int(n))

    def FLOAT(self, n):
        return FloatLiteral(float(n))

    def assignment(self, target, _, value):
        return Assignment(target, value)

    def binary_expression(self, left, operator, right):
        return BinaryExpression(operator, left, right)

    def if_statement(self, if_clause, *elseif_clauses, else_clause=None):
        print("if_statement", if_clause, elseif_clauses, else_clause)
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


if __name__ == '__main__':
    print(parse("""
        if (1==1) player:tell("hello"); endif
    """))
