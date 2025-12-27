import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import lark
from lark import Lark, Transformer, ast_utils, v_args
from lark.tree import Meta

from .builtin_functions import BuiltinFunctions
from .moo_types import MOOAny, to_moo
from .opcodes import Extended_Opcode, Opcode, optim_num_to_opcode, in_optim_num_range, OPTIM_NUM_LOW, OPTIM_NUM_HI
from .parser import parser
from .string import MOOString
from .vm import VM, Instruction, Program, StackFrame

this_module = sys.modules[__name__]


@dataclass
class CompilerState:
    last_label = 0
    bi_funcs = None  # Optional BuiltinFunctions instance for compilation
    var_names: List[MOOString] = None  # Track local variable names
    indexed_object = None  # Track the object being indexed for $ references

    def __init__(self, bi_funcs=None):
        self.bi_funcs = bi_funcs
        self.last_label = 0
        self.var_names = []
        self.indexed_object = None

    def next_label(self):
        self.last_label += 1
        return self.last_label

    def add_var(self, name: str) -> int:
        """Register a variable and return its index.

        If variable already exists, returns existing index.
        """
        var_name = MOOString(name)
        if var_name not in self.var_names:
            self.var_names.append(var_name)
        return self.var_names.index(var_name)


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
    # Bitwise operators (ToastStunt syntax with .)
    '|.': Extended_Opcode.EOP_BITOR,
    '&.': Extended_Opcode.EOP_BITAND,
    '^.': Extended_Opcode.EOP_BITXOR,
    # Shift operators
    '<<': Extended_Opcode.EOP_BITSHL,
    '>>': Extended_Opcode.EOP_BITSHR,
}

unary_opcodes = {
    '-': Opcode.OP_UNARY_MINUS,
    '!': Opcode.OP_NOT,
    '~': Extended_Opcode.EOP_COMPLEMENT,
}

# utility functions


def walk_ast(node: '_AstNode'):
    if isinstance(node, _AstNode):
        yield node
        for child in node.get_children():
            yield from walk_ast(child)

#
# Define AST
#


class _AstNode(ast_utils.Ast):

    def emit_byte(self, opcode: Union[Opcode, int], operand: Optional[MOOAny] = None):
        return Instruction(opcode=opcode, operand=operand)

    def emit_extended_byte(self, opcode: Extended_Opcode):
        return Instruction(opcode=Opcode.OP_EXTENDED, operand=opcode)

    @abstractmethod
    def to_bytecode(self, state: CompilerState, program: Program):
        raise NotImplementedError(
            f"to_bytecode not implemented for {self.__class__.__name__}")

    @abstractmethod
    def to_moo(self) -> str:
        raise NotImplementedError(
            f"to_moo not implemented for {self.__class__.__name__}")

    def get_children(node: '_AstNode'):
        for field, value in node.__dict__.items():
            if isinstance(value, tuple) or isinstance(value, list):
                for item in value:
                    if isinstance(item, _AstNode):
                        yield item
            elif isinstance(value, _AstNode):
                yield value


@dataclass
class VerbCode(_AstNode):
    children: List[_AstNode]

    def to_bytecode(self, state: CompilerState, program: Program):
        result = []
        for node in self.children:
            result += node.to_bytecode(state, program)
        return result

    def to_moo(self) -> str:
        return "\n".join([node.to_moo() for node in self.children])

    def rename_property(self, old_name: str, new_name: str):
        for node in walk_ast(self):
            if isinstance(node, _Property) and hasattr(node.name, 'value') and node.name.value == old_name:
                node.name.value = new_name

    def rename_variable(self, old_name: str, new_name: str):
        for node in walk_ast(self):
            if isinstance(node, Identifier) and node.value == old_name:
                node.value = new_name

    def rename_verb(self, old_name: str, new_name: str):
        for node in walk_ast(self):
            if isinstance(node, _VerbCall) and hasattr(node.name, 'value') and node.name.value == old_name:
                node.name.value = new_name


class _Expression(_AstNode):
    pass


class _Statement(_AstNode):

    def to_bytecode(self, state: CompilerState, program: Program):
        raise NotImplementedError(
            f"to_bytecode not implemented for {self.__class__.__name__}")


@dataclass
class _SingleStatement(_AstNode):
    """Single statement ending with semicolon. Underscore prefix avoids ast_utils conflict."""
    statement: _Statement  # Can also be _Expression when expression is used as statement

    def to_bytecode(self, state: CompilerState, program: Program):
        bc = self.statement.to_bytecode(state, program)
        # If the statement is an expression (not a proper statement like return/break/etc),
        # its value should be discarded - add a POP
        if isinstance(self.statement, _Expression):
            bc.append(Instruction(opcode=Opcode.OP_POP, operand=1))
        return bc

    def to_moo(self) -> str:
        return self.statement.to_moo() + ";"


@dataclass
class _EmptyStatement(_Statement):
    """Empty statement (just a semicolon)."""

    def to_bytecode(self, state: CompilerState, program: Program):
        return []  # No bytecode for empty statement

    def to_moo(self) -> str:
        return ";"


@dataclass(init=False)
class _Body(_AstNode):
    statements: Tuple[_Statement]

    def __init__(self, *statements: _Statement):
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
        return self.expression.to_bytecode(state, program) + [self.emit_byte(Opcode.OP_CHECK_LIST_FOR_SPLICE)]

    def to_moo(self) -> str:
        return '@' + self.expression.to_moo()


@dataclass
class _Literal(_AstNode):
    value: Any

    def to_bytecode(self, state: CompilerState, program: Program):
        return [self.emit_byte(Opcode.OP_IMM, to_moo(self.value))]

    def to_moo(self) -> str:
        return str(self.value)


@dataclass
class StringLiteral(_Literal, _Expression):
    value: str

    def to_moo(self) -> str:
        # Escape backslashes first, then quotes
        escaped = self.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'


@dataclass
class BooleanLiteral(_Literal, _Expression):
    value: bool

    def to_moo(self):
        return str(self.value).lower()


@dataclass
class NumberLiteral(_Literal, _Expression):
    value: int

    def to_bytecode(self, state: CompilerState, program: Program):
        if in_optim_num_range(self.value):
            return [self.emit_byte(optim_num_to_opcode(self.value), None)]
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
        from lambdamoo_db.database import ObjNum
        return [self.emit_byte(Opcode.OP_IMM, ObjNum(self.value))]

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
            is_splice = isinstance(element, Splicer)
            result += element.to_bytecode(state, program)
            if index == 0:
                if is_splice:
                    # Splice: the list is already on stack, use as-is
                    pass
                else:
                    # Regular element: wrap in singleton list
                    result += [Instruction(opcode=Opcode.OP_MAKE_SINGLETON_LIST)]
            else:
                if is_splice:
                    # Splice: extend with the list (concatenate)
                    result += [Instruction(opcode=Opcode.OP_LIST_APPEND)]
                else:
                    # Regular element: append single item
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
            result += value.to_bytecode(state, program)
            result += key.to_bytecode(state, program)
            result += [Instruction(opcode=Opcode.OP_MAP_INSERT)]
        return result

    def to_moo(self) -> str:
        return "[" + ", ".join([f"{key.to_moo()}: {value.to_moo()}" for key, value in self.value]) + "]"


@dataclass
class _UnaryExpression(_Expression):
    """Unary expression like -x, !x, ~x. Underscore prefix avoids ast_utils conflict."""
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

        # Handle short-circuit operators with proper MOO semantics
        # In MOO: a || b returns a if truthy, else b
        #         a && b returns a if falsy, else b
        if self.operator == '||':
            # Pattern: [a] PUT_TEMP IF_QUES(skip to b) PUSH_TEMP JUMP(end) [b]
            # - [a] evaluates left side
            # - PUT_TEMP saves a (peek, doesn't pop)
            # - IF_QUES pops and jumps if false
            # - If true: PUSH_TEMP pushes a's value, JUMP skips b
            # - If false: falls through to [b]
            result = left_bc
            result += [Instruction(opcode=Opcode.OP_PUT_TEMP)]
            # IF_QUES: if false, skip 2 instructions (PUSH_TEMP and JUMP) to evaluate b
            result += [Instruction(opcode=Opcode.OP_IF_QUES, operand=2)]
            result += [Instruction(opcode=Opcode.OP_PUSH_TEMP)]
            result += [Instruction(opcode=Opcode.OP_JUMP, operand=len(right_bc) + 1)]
            result += right_bc
            return result

        elif self.operator == '&&':
            # Pattern: [a] PUT_TEMP NOT IF_QUES(skip to b) PUSH_TEMP JUMP(end) [b]
            # - [a] evaluates left side
            # - PUT_TEMP saves a (peek, doesn't pop)
            # - NOT inverts for the conditional
            # - IF_QUES pops and jumps if false (i.e., if a was truthy)
            # - If a was falsy: PUSH_TEMP pushes a's value, JUMP skips b
            # - If a was truthy: falls through to [b]
            result = left_bc
            result += [Instruction(opcode=Opcode.OP_PUT_TEMP)]
            result += [Instruction(opcode=Opcode.OP_NOT)]
            result += [Instruction(opcode=Opcode.OP_IF_QUES, operand=2)]
            result += [Instruction(opcode=Opcode.OP_PUSH_TEMP)]
            result += [Instruction(opcode=Opcode.OP_JUMP, operand=len(right_bc) + 1)]
            result += right_bc
            return result

        return left_bc + right_bc + [Instruction(opcode=binary_opcodes[self.operator])]

    def to_moo(self) -> str:
        return f"{self.left.to_moo()} {self.operator} {self.right.to_moo()}"


@dataclass
class _Ternary(_Expression):
    """Ternary conditional: condition ? true_value | false_value."""
    condition: _Expression
    true_value: _Expression
    false_value: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        # Bytecode structure:
        # 1. condition_bc (pushes condition)
        # 2. OP_IF_QUES with jump to false_value (pops condition, jumps if false)
        # 3. true_value_bc
        # 4. OP_JUMP to end
        # 5. false_value_bc
        condition_bc = self.condition.to_bytecode(state, program)
        true_bc = self.true_value.to_bytecode(state, program)
        false_bc = self.false_value.to_bytecode(state, program)

        # OP_IF_QUES: if condition is false, jump past true_bc + OP_JUMP
        # operand is how many instructions to skip
        if_ques_jump = len(true_bc) + 1  # +1 for the OP_JUMP instruction

        # OP_JUMP: after evaluating true_value, jump past false_bc
        end_jump = len(false_bc)

        result = condition_bc
        result.append(Instruction(opcode=Opcode.OP_IF_QUES, operand=if_ques_jump))
        result.extend(true_bc)
        result.append(Instruction(opcode=Opcode.OP_JUMP, operand=end_jump))
        result.extend(false_bc)
        return result

    def to_moo(self) -> str:
        return f"({self.condition.to_moo()} ? {self.true_value.to_moo()} | {self.false_value.to_moo()})"


@dataclass
class _Assign(_Expression):
    target: Identifier
    value: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        value_bc = self.value.to_bytecode(state, program)
        if isinstance(self.target, Identifier):
            # Register variable with compiler state
            state.add_var(self.target.value)
            # OP_PUT peeks (doesn't pop) - leaves value on stack for expression result
            # _SingleStatement will add OP_POP when assignment is used as statement
            return value_bc + [
                Instruction(opcode=Opcode.OP_PUT, operand=MOOString(self.target.value))
            ]
        elif isinstance(self.target, _Property):
            # OP_PUT_PROP expects stack: [obj, propname, value] with value on top
            # C code: rhs = POP(); propname = POP(); obj = POP();
            obj_bc = self.target.object.to_bytecode(state, program)
            name_bc = self.target.name.to_bytecode(state, program)
            # OP_PUT_PROP pops all 3 and pushes result (value assigned)
            return obj_bc + name_bc + value_bc + [
                Instruction(opcode=Opcode.OP_PUT_PROP)
            ]
        elif isinstance(self.target, _Index):
            # Indexed assignment: obj[index] = value
            #
            # C compiler pattern for x[i1][i2]...[iN] = value:
            # 1. push_lvalue: push base, for each index push index then PUSH_REF (except last)
            # 2. generate_expr: push value
            # 3. PUT_TEMP: save value to temp
            # 4. For each index level: INDEXSET
            # 5. Store to base: PUT var or PUT_PROP
            # 6. POP: remove modified container
            # 7. PUSH_TEMP: push saved value (expression result)

            # Flatten the _Index chain to get base object and list of indices
            indices = []
            current = self.target
            while isinstance(current, _Index):
                indices.append(current.index)
                current = current.object
            indices.reverse()  # Now indices[0] is outermost, indices[-1] is innermost
            base = current  # The base object (Identifier, _Property, etc.)

            result = []

            # Step 1: Generate lvalue (push_lvalue equivalent)
            if isinstance(base, Identifier):
                # Push the variable value
                result += [Instruction(opcode=Opcode.OP_PUSH, operand=MOOString(base.value))]
            elif isinstance(base, _Property):
                # For property base like obj.prop[i] = v:
                # Push obj, push propname, then PUSH_GET_PROP (keeps obj/propname on stack)
                result += base.object.to_bytecode(state, program)
                result += base.name.to_bytecode(state, program)
                result += [Instruction(opcode=Opcode.OP_PUSH_GET_PROP)]
            elif isinstance(base, DollarProperty):
                # For $prop[i] = v (which is #0.prop[i] = v):
                # Push #0, push propname, then PUSH_GET_PROP (keeps obj/propname on stack)
                result += [Instruction(opcode=Opcode.OP_IMM, operand=0)]
                if isinstance(base.name, Identifier):
                    prop_name = MOOString(base.name.value)
                else:
                    prop_name = MOOString(str(base.name.value))
                result += [Instruction(opcode=Opcode.OP_IMM, operand=prop_name)]
                result += [Instruction(opcode=Opcode.OP_PUSH_GET_PROP)]
            else:
                # Other base types - just evaluate normally
                result += base.to_bytecode(state, program)

            # Set indexed_object for ^ and $ operators to work correctly on maps
            old_indexed = state.indexed_object
            state.indexed_object = base

            # For each index except the last, push index and PUSH_REF
            for idx in indices[:-1]:
                result += idx.to_bytecode(state, program)
                result += [Instruction(opcode=Opcode.OP_PUSH_REF)]

            # Push final index (no PUSH_REF for the last one)
            result += indices[-1].to_bytecode(state, program)

            # Restore indexed_object
            state.indexed_object = old_indexed

            # Step 2: Push value
            result += value_bc

            # Step 3: Save value to temp
            result += [Instruction(opcode=Opcode.OP_PUT_TEMP)]

            # Step 4: Chain INDEXSET for each index level
            for _ in indices:
                result += [Instruction(opcode=Opcode.OP_INDEXSET)]

            # Step 5: Store modified container back to base
            if isinstance(base, Identifier):
                state.add_var(base.value)
                result += [Instruction(opcode=Opcode.OP_PUT, operand=MOOString(base.value))]
            elif isinstance(base, _Property):
                # For property: stack has [obj, propname, modified_propvalue]
                # PUT_PROP expects [obj, propname, value] and stores obj.propname = value
                result += [Instruction(opcode=Opcode.OP_PUT_PROP)]
            elif isinstance(base, DollarProperty):
                # For $prop: stack has [obj, propname, modified_propvalue]
                # PUT_PROP expects [obj, propname, value] and stores obj.propname = value
                result += [Instruction(opcode=Opcode.OP_PUT_PROP)]
            # else: other base types - no store needed (result is on stack)

            # Step 6 & 7: Pop result, push original value for expression result
            result += [Instruction(opcode=Opcode.OP_POP, operand=1)]
            result += [Instruction(opcode=Opcode.OP_PUSH_TEMP)]

            return result
        elif isinstance(self.target, _Range):
            # Range assignment: obj[start..end] = value
            # Stack layout for EOP_RANGESET: [base, start, end, value]
            #
            # For simple case: x[1..2] = value
            # 1. Push base variable value
            # 2. Push start
            # 3. Push end
            # 4. Push value
            # 5. PUT_TEMP to save value
            # 6. EOP_RANGESET (consumes 4, pushes result)
            # 7. Store modified container back to base
            # 8. Pop result, push original value

            base = self.target.object
            result = []

            # For nested index+range like l[3][2..$] = "u", we need to:
            # 1. Push outer container and index for later OP_INDEXSET
            # 2. Push current value at that index as base for EOP_RANGESET
            # 3. After EOP_RANGESET, use OP_INDEXSET to store back

            if isinstance(base, _Index):
                # Nested case: x[idx][start..end] = value
                # For x = variable: Stack flow:
                # 1. Push x value (for final OP_PUT to x)
                # 2. Push idx (for OP_INDEXSET)
                # 3. Push x[idx] value (for EOP_RANGESET)
                # 4. Push start, end, value
                # 5. EOP_RANGESET -> modified inner value
                # 6. OP_INDEXSET -> modified x
                # 7. OP_PUT x -> store modified x back to variable
                # 8. Pop, push original value

                outer_base = base.object

                if isinstance(outer_base, Identifier):
                    # Push outer variable value (for OP_INDEXSET)
                    result += [Instruction(opcode=Opcode.OP_PUSH, operand=MOOString(outer_base.value))]
                    # Push outer index
                    result += base.index.to_bytecode(state, program)
                    # Push current value at that index (for EOP_RANGESET)
                    result += [Instruction(opcode=Opcode.OP_PUSH, operand=MOOString(outer_base.value))]
                    result += base.index.to_bytecode(state, program)
                    result += [Instruction(opcode=Opcode.OP_REF)]
                else:
                    # Non-variable outer base - just evaluate
                    result += outer_base.to_bytecode(state, program)
                    result += base.index.to_bytecode(state, program)
                    result += outer_base.to_bytecode(state, program)
                    result += base.index.to_bytecode(state, program)
                    result += [Instruction(opcode=Opcode.OP_REF)]

                # Set indexed_object for ^ and $ operators
                old_indexed = state.indexed_object
                state.indexed_object = base

                # Push start and end
                result += self.target.start.to_bytecode(state, program)
                result += self.target.end.to_bytecode(state, program)

                # Restore indexed_object
                state.indexed_object = old_indexed

                # Push value and save to temp
                result += value_bc
                result += [Instruction(opcode=Opcode.OP_PUT_TEMP)]

                # EOP_RANGESET: [inner_value, start, end, value] -> [modified_inner]
                result += [Instruction(opcode=Opcode.OP_EXTENDED,
                                       operand=Extended_Opcode.EOP_RANGESET.value)]

                # Stack now: [outer_value, index, modified_inner]
                # OP_INDEXSET: outer_value[index] = modified_inner -> modified_outer
                result += [Instruction(opcode=Opcode.OP_INDEXSET)]

                # Store modified outer back to variable
                if isinstance(outer_base, Identifier):
                    state.add_var(outer_base.value)
                    result += [Instruction(opcode=Opcode.OP_PUT, operand=MOOString(outer_base.value))]

                # Pop modified value, push original assigned value
                result += [Instruction(opcode=Opcode.OP_POP, operand=1)]
                result += [Instruction(opcode=Opcode.OP_PUSH_TEMP)]

                return result

            # Step 1: Push base
            if isinstance(base, Identifier):
                result += [Instruction(opcode=Opcode.OP_PUSH, operand=MOOString(base.value))]
            elif isinstance(base, _Property):
                result += base.object.to_bytecode(state, program)
                result += base.name.to_bytecode(state, program)
                result += [Instruction(opcode=Opcode.OP_PUSH_GET_PROP)]
            elif isinstance(base, DollarProperty):
                # For $prop[start..end] = v (which is #0.prop[start..end] = v):
                # Push #0, push propname, then PUSH_GET_PROP (keeps obj/propname on stack)
                result += [Instruction(opcode=Opcode.OP_IMM, operand=0)]
                if isinstance(base.name, Identifier):
                    prop_name = MOOString(base.name.value)
                else:
                    prop_name = MOOString(str(base.name.value))
                result += [Instruction(opcode=Opcode.OP_IMM, operand=prop_name)]
                result += [Instruction(opcode=Opcode.OP_PUSH_GET_PROP)]
            else:
                result += base.to_bytecode(state, program)

            # Set indexed_object for ^ and $ operators
            old_indexed = state.indexed_object
            state.indexed_object = base

            # Step 2 & 3: Push start and end
            result += self.target.start.to_bytecode(state, program)
            result += self.target.end.to_bytecode(state, program)

            # Restore indexed_object
            state.indexed_object = old_indexed

            # Step 4: Push value
            result += value_bc

            # Step 5: Save value to temp
            result += [Instruction(opcode=Opcode.OP_PUT_TEMP)]

            # Step 6: EOP_RANGESET
            result += [Instruction(opcode=Opcode.OP_EXTENDED,
                                   operand=Extended_Opcode.EOP_RANGESET.value)]

            # Step 7: Store modified container back to base
            if isinstance(base, Identifier):
                state.add_var(base.value)
                result += [Instruction(opcode=Opcode.OP_PUT, operand=MOOString(base.value))]
            elif isinstance(base, _Property):
                result += [Instruction(opcode=Opcode.OP_PUT_PROP)]
            elif isinstance(base, DollarProperty):
                # For $prop: stack has [obj, propname, modified_propvalue]
                # PUT_PROP expects [obj, propname, value] and stores obj.propname = value
                result += [Instruction(opcode=Opcode.OP_PUT_PROP)]

            # Step 8: Pop result, push original value for expression result
            result += [Instruction(opcode=Opcode.OP_POP, operand=1)]
            result += [Instruction(opcode=Opcode.OP_PUSH_TEMP)]

            return result
        elif isinstance(self.target, _List):
            # Destructuring assignment: {a, b, c} = list
            # Build scatter pattern from list items
            scatter_pattern = []
            for item in self.target.value:
                if isinstance(item, Identifier):
                    state.add_var(item.value)
                    scatter_pattern.append((item.value, False, False, None))
                elif isinstance(item, Splicer) and isinstance(item.expression, Identifier):
                    # @rest - rest variable
                    state.add_var(item.expression.value)
                    scatter_pattern.append((item.expression.value, False, True, None))
                # TODO: Handle optional with defaults (?var = default)

            result = value_bc
            result.append(Instruction(
                opcode=Opcode.OP_EXTENDED,
                operand=Extended_Opcode.EOP_SCATTER.value,
                scatter_pattern=scatter_pattern
            ))
            return result
        elif isinstance(self.target, DollarProperty):
            # $prop = value means #0.prop = value
            # OP_PUT_PROP expects stack: [obj, propname, value] with value on top
            obj_bc = [Instruction(opcode=Opcode.OP_IMM, operand=0)]
            if isinstance(self.target.name, Identifier):
                prop_name = MOOString(self.target.name.value)
            else:
                prop_name = MOOString(str(self.target.name.value))
            name_bc = [Instruction(opcode=Opcode.OP_IMM, operand=prop_name)]
            return obj_bc + name_bc + value_bc + [
                Instruction(opcode=Opcode.OP_PUT_PROP)
            ]
        else:
            # Unknown target type - return empty (will cause error)
            return []

    def to_moo(self) -> str:
        return f"{self.target.to_moo()} = {self.value.to_moo()}"


@dataclass
class ElseifClause(_AstNode):
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
class IfClause(_AstNode):
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
class ElseClause(_AstNode):
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
            if instruction.opcode == Opcode.OP_JUMP and instruction.operand == end_label:
                instruction.operand = len(full_bc) - index
        return full_bc

    def to_moo(self) -> str:
        result = self.if_clause.to_moo()
        result += "\n".join([elseif.to_moo()
                             for elseif in self.elseif_clauses])
        if self.else_clause is not None:
            result += f"\n{self.else_clause.to_moo()}"
        result += "\nendif"
        return result


@dataclass
class _ForClause(_AstNode):
    id: Identifier
    index: Optional[Identifier]
    iterable: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        # Register loop variable(s) with compiler state
        state.add_var(self.id.value)
        if self.index is not None:
            state.add_var(self.index.value)

        iterable_bc = self.iterable.to_bytecode(state, program)
        result = iterable_bc + \
            [Instruction(opcode=Opcode.OP_IMM, operand=None)]
        opcode = Extended_Opcode.EOP_FOR_LIST_1
        var = MOOString(self.id.value)
        index = None
        if self.index is not None:
            opcode = Extended_Opcode.EOP_FOR_LIST_2
            index = MOOString(self.index.value)
        instruction = Instruction(opcode=Opcode.OP_EXTENDED, operand=opcode)
        instruction.loop_index = index
        instruction.loop_var = var
        result += [instruction]
        return result

    def to_moo(self) -> str:
        index = ""
        if self.index is not None:
            index = f", {self.index.to_moo()}"
        return f"for {self.id.to_moo()}{index} in ({self.iterable.to_moo()})"


@dataclass
class _ForRangeClause(_AstNode):
    """For-range loop clause: for i in [start..end]"""
    id: Identifier
    start: _Expression
    end: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        # Register loop variable with compiler state
        state.add_var(self.id.value)

        # Emit bytecode for start and end expressions
        # Stack will have: [start, end]
        start_bc = self.start.to_bytecode(state, program)
        end_bc = self.end.to_bytecode(state, program)
        result = start_bc + end_bc

        # Emit OP_FOR_RANGE with loop variable name
        var = MOOString(self.id.value)
        instruction = Instruction(opcode=Opcode.OP_FOR_RANGE, operand=None)
        instruction.loop_var = var
        result += [instruction]
        return result

    def to_moo(self) -> str:
        return f"for {self.id.to_moo()} in [{self.start.to_moo()}..{self.end.to_moo()}]"


@dataclass
class ContinueStatement(_Statement):
    id: Optional[Identifier] = None

    def to_moo(self) -> str:
        if self.id is not None:
            return f"continue {self.id.to_moo()}"
        return "continue"

    def to_bytecode(self, state: CompilerState, program: Program):
        """Continue to next iteration of current loop."""
        return [Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_CONTINUE)]


@dataclass
class BreakStatement(_Statement):
    id: Optional[Identifier] = None

    def to_moo(self) -> str:
        if self.id is not None:
            return f"break {self.id.to_moo()}"
        return "break"

    def to_bytecode(self, state: CompilerState, program: Program):
        """Break out of current loop."""
        return [Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_EXIT)]


@dataclass
class ForStatement(_Statement):
    condition: Union[_ForClause, _ForRangeClause]
    body: _Body

    def to_bytecode(self, state: CompilerState, program: Program):
        result = self.condition.to_bytecode(state, program)
        body = self.body.to_bytecode(state, program)
        result = result + body
        result += [Instruction(opcode=Opcode.OP_JUMP,
                               operand=-(len(body) + 1))]
        return result

    def to_moo(self) -> str:
        # Indent body statements
        body_lines = self.body.to_moo().split("\n")
        indented_body = "\n".join("  " + line for line in body_lines if line)
        return f"{self.condition.to_moo()}\n{indented_body}\nendfor"


@dataclass
class _ScatterItem(_AstNode):
    """Single item in a scatter pattern.

    Types:
    - required: var_name, is_optional=False, is_rest=False, default=None
    - optional: var_name, is_optional=True, is_rest=False, default=expression
    - rest: var_name, is_optional=False, is_rest=True, default=None
    """
    var_name: str
    is_optional: bool = False
    is_rest: bool = False
    default: Optional[_Expression] = None

    def to_moo(self) -> str:
        if self.is_rest:
            return f"@{self.var_name}"
        elif self.is_optional:
            if self.default:
                return f"?{self.var_name} = {self.default.to_moo()}"
            return f"?{self.var_name}"
        return self.var_name


@dataclass
class _ScatterTarget(_AstNode):
    """List of items to scatter into: {a, ?b, @rest}."""
    items: List[_ScatterItem]

    def to_moo(self) -> str:
        return "{" + ", ".join(item.to_moo() for item in self.items) + "}"


@dataclass
class _ScatterAssignment(_Statement):
    """Scatter assignment: {a, ?b, @rest} = expr."""
    target: _ScatterTarget
    value: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        # Register all variables first
        for item in self.target.items:
            state.add_var(item.var_name)

        # Build scatter pattern: [(var_name, is_optional, is_rest, default_bc)]
        scatter_pattern = []
        for item in self.target.items:
            default_bc = item.default.to_bytecode(state, program) if item.default else None
            scatter_pattern.append((item.var_name, item.is_optional, item.is_rest, default_bc))

        # Compile value expression
        result = self.value.to_bytecode(state, program)

        # Emit scatter instruction with pattern
        result.append(Instruction(
            opcode=Opcode.OP_EXTENDED,
            operand=Extended_Opcode.EOP_SCATTER.value,
            scatter_pattern=scatter_pattern
        ))
        return result

    def to_moo(self) -> str:
        return f"{self.target.to_moo()} = {self.value.to_moo()}"


@dataclass
class _FunctionCall(_Expression):
    name: str
    arguments: _List

    def to_bytecode(self, state: CompilerState, program: Program):
        result = self.arguments.to_bytecode(state, program)
        # Use the bi_funcs from state if available, otherwise create a new instance
        bi_funcs = state.bi_funcs if state.bi_funcs else BuiltinFunctions()
        builtin_id = bi_funcs.get_id_by_name(self.name)
        result += [Instruction(opcode=Opcode.OP_BI_FUNC_CALL,
                               operand=builtin_id)]
        return result

    def to_moo(self) -> str:
        arguments = self.arguments.to_moo()[1:-1]
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
        # Extract raw property name from StringLiteral (without quotes)
        # Property names are identifiers, not string expressions
        if isinstance(self.name, StringLiteral):
            prop_name = self.name.value
        else:
            prop_name = self.name.to_moo()

        # Use $name shorthand when accessing properties on #0
        if isinstance(self.object, ObjnumLiteral) and self.object.value == 0:
            return f"${prop_name}"
        return f"{self.object.to_moo()}.{prop_name}"


@dataclass
class _WaifProperty(_Expression):
    """Waif property access: obj.:prop (ToastStunt extension). Underscore avoids ast_utils conflict."""
    object: _Expression
    name: str

    def to_bytecode(self, state: CompilerState, program: Program):
        # TODO: Implement waif property bytecode when needed
        raise NotImplementedError("Waif property bytecode not yet implemented")

    def to_moo(self) -> str:
        return f"{self.object.to_moo()}.:{self.name}"


@dataclass
class DollarProperty(_AstNode):
    name: StringLiteral

    def to_bytecode(self, state: CompilerState, program: Program):
        # $prop means #0.prop - push object #0 and property name as string
        result = [Instruction(opcode=Opcode.OP_IMM, operand=0)]
        # Get the property name as a string (works for both Identifier and StringLiteral)
        if isinstance(self.name, Identifier):
            prop_name = MOOString(self.name.value)
        else:
            prop_name = MOOString(str(self.name.value))
        result += [Instruction(opcode=Opcode.OP_IMM, operand=prop_name)]
        result += [Instruction(opcode=Opcode.OP_GET_PROP)]
        return result

    def to_moo(self) -> str:
        return f"${self.name.to_moo()}"


@dataclass
class _CallArgs(_AstNode):
    """Internal class for call arguments - renamed to avoid ast_utils conflict with call_arguments rule."""
    arguments: Optional[List[_Expression]] = None

    def to_moo(self) -> str:
        if self.arguments is None:
            return ""
        return ", ".join([arg.to_moo() for arg in self.arguments])


@dataclass
class DollarVerbCall(_AstNode):
    name: StringLiteral
    arguments: List[_Expression]

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


class _FirstIndex(_Expression):
    """Represents ^ (first element) in an index context. Prefixed with _ to avoid ast_utils auto-registration."""
    def to_bytecode(self, state: CompilerState, program: Program):
        # EOP_FIRST_INDEX needs the container on stack to determine index/key
        if state.indexed_object is not None:
            result = state.indexed_object.to_bytecode(state, program)
            result += [self.emit_extended_byte(Extended_Opcode.EOP_FIRST_INDEX)]
            return result
        else:
            # No context - just return 1 for lists/strings
            return [Instruction(opcode=Opcode.OP_IMM, operand=1)]

    def to_moo(self) -> str:
        return "^"


class _LastIndex(_Expression):
    """Represents $ (last element) in an index context. Prefixed with _ to avoid ast_utils auto-registration."""
    def to_bytecode(self, state: CompilerState, program: Program):
        # EOP_LAST_INDEX needs the container on stack to determine last index/key
        if state.indexed_object is not None:
            result = state.indexed_object.to_bytecode(state, program)
            result += [self.emit_extended_byte(Extended_Opcode.EOP_LAST_INDEX)]
            return result
        else:
            # Fallback - no context available
            # This shouldn't happen in well-formed code, but return a safe default
            return [Instruction(opcode=Opcode.OP_IMM, operand=1)]

    def to_moo(self) -> str:
        return "$"


@dataclass
class _Index(_Expression):
    """Renamed to avoid ast_utils conflict with index rule."""
    object: _Expression
    index: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        result = self.object.to_bytecode(state, program)
        # Set indexed_object context so $ knows which object to reference
        old_indexed = state.indexed_object
        state.indexed_object = self.object
        result += self.index.to_bytecode(state, program)
        state.indexed_object = old_indexed
        result += [self.emit_byte(Opcode.OP_REF)]
        return result

    def to_moo(self) -> str:
        return f"{self.object.to_moo()}[{self.index.to_moo()}]"


@dataclass
class _Range(_Expression):
    """Range expression: obj[start..end]"""
    object: _Expression
    start: _Expression
    end: _Expression

    def to_bytecode(self, state: CompilerState, program: Program):
        result = self.object.to_bytecode(state, program)
        # Set indexed_object context so $ knows which object to reference
        old_indexed = state.indexed_object
        state.indexed_object = self.object
        result += self.start.to_bytecode(state, program)
        result += self.end.to_bytecode(state, program)
        state.indexed_object = old_indexed
        result += [self.emit_byte(Opcode.OP_RANGE_REF)]
        return result

    def to_moo(self) -> str:
        return f"{self.object.to_moo()}[{self.start.to_moo()} .. {self.end.to_moo()}]"


@dataclass
class ReturnStatement(_Statement):
    value: Optional[_Expression] = None

    def to_bytecode(self, state: CompilerState, program: Program):
        if self.value is None:
            return [Instruction(opcode=Opcode.OP_RETURN0)]
        value_bc = self.value.to_bytecode(state, program)
        return value_bc + [Instruction(opcode=Opcode.OP_RETURN)]

    def to_moo(self) -> str:
        if self.value is None:
            return "return"
        return f"return {self.value.to_moo()}"


@dataclass
class WhileStatement(_Statement):
    condition: _Expression
    body: _Body

    def to_bytecode(self, state: CompilerState, program: Program):
        # First generate bytecode for condition and body
        condition_bc = self.condition.to_bytecode(state, program)
        body_bc = self.body.to_bytecode(state, program)
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
        res = f"while ({self.condition.to_moo()})\n"
        res += self.body.to_moo()
        res += "\nendwhile\n"
        return res


@dataclass
class _ForkStatement(_Statement):
    """fork [id] (delay) body endfork"""
    delay: _Expression  # Delay in seconds before fork executes
    body: _Body
    var_id: str = None  # Optional variable to store task ID (for fork id ...)

    def to_bytecode(self, state: CompilerState, program: Program):
        # Generate bytecode for the delay expression
        delay_bc = self.delay.to_bytecode(state, program)

        # Compile fork body separately - this becomes a fork vector
        body_bc = self.body.to_bytecode(state, program)

        # Add fork body to program's fork_vectors and get its index
        f_index = len(program.fork_vectors)
        program.fork_vectors.append(body_bc)

        # Create the fork instruction
        if self.var_id is not None:
            # OP_FORK_WITH_ID: operand is tuple (f_index, var_id)
            var_index = state.add_var(self.var_id)
            fork_instr = Instruction(
                opcode=Opcode.OP_FORK_WITH_ID,
                operand=(f_index, var_index)
            )
        else:
            # OP_FORK: operand is just f_index
            fork_instr = Instruction(
                opcode=Opcode.OP_FORK,
                operand=f_index
            )

        return delay_bc + [fork_instr]

    def to_moo(self) -> str:
        if self.var_id:
            res = f"fork {self.var_id} ({self.delay.to_moo()})\n"
        else:
            res = f"fork ({self.delay.to_moo()})\n"
        res += self.body.to_moo()
        res += "\nendfork\n"
        return res


@dataclass
class _ExceptClause:
    """An except clause: except [var] (codes) body. Underscore prefix avoids ast_utils conflict."""
    codes: list  # List of error codes to catch (or 'ANY')
    var: str = None  # Optional variable to bind error to
    body: "_Body" = None


@dataclass
class _TryExceptStatement(_Statement):
    """try ... except ... endtry. Underscore prefix avoids ast_utils conflict."""
    try_body: _Body
    except_clauses: list  # List of _ExceptClause

    def to_bytecode(self, state: CompilerState, program: Program):
        from moo_interp.opcodes import Extended_Opcode

        # Compile try body
        try_bc = self.try_body.to_bytecode(state, program)

        # Compile except clauses
        except_bcs = []
        for clause in self.except_clauses:
            clause_bc = clause.body.to_bytecode(state, program) if clause.body else []
            except_bcs.append((clause, clause_bc))

        result = []

        # EOP_TRY_EXCEPT instruction - stores handler offset
        # handler_offset = distance from TRY_EXCEPT instruction to first handler
        # Format: try_bc + 2 (for JUMP instruction + 1 to get PAST the jump)
        handler_offset = len(try_bc) + 2  # +1 for JUMP, +1 to get past it
        try_except_instr = Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_TRY_EXCEPT.value)
        try_except_instr.handler_offset = handler_offset
        try_except_instr.error_codes = [code for clause in self.except_clauses for code in clause.codes]
        # Store error variable names for binding (one per clause, None if no binding)
        try_except_instr.error_vars = [clause.var for clause in self.except_clauses]
        result.append(try_except_instr)
        result.extend(try_bc)

        # Calculate total size of except handlers
        handlers_size = sum(len(bc) + 2 for _, bc in except_bcs)  # +2 for EOP_END_EXCEPT and nop/jump

        # After try body, jump past all except handlers (no error case)
        result.append(Instruction(opcode=Opcode.OP_JUMP, operand=handlers_size))

        # Except handlers
        for i, (clause, clause_bc) in enumerate(except_bcs):
            # Each handler: execute body, end
            result.extend(clause_bc)
            # Jump to end of try/except after handler
            remaining = sum(len(bc) + 2 for _, bc in except_bcs[i+1:])
            result.append(Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_END_EXCEPT.value))
            if remaining > 0:
                result.append(Instruction(opcode=Opcode.OP_JUMP, operand=remaining))
            else:
                result.append(Instruction(opcode=Opcode.OP_POP, operand=0))

        return result


@dataclass
class _TryFinallyStatement(_Statement):
    """try ... finally ... endtry. Underscore prefix avoids ast_utils conflict."""
    try_body: _Body
    finally_body: _Body

    def to_bytecode(self, state: CompilerState, program: Program):
        from moo_interp.opcodes import Extended_Opcode

        try_bc = self.try_body.to_bytecode(state, program)
        finally_bc = self.finally_body.to_bytecode(state, program)

        result = []
        result.append(Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_TRY_FINALLY.value))
        result.extend(try_bc)
        result.append(Instruction(opcode=Opcode.OP_EXTENDED, operand=Extended_Opcode.EOP_END_FINALLY.value))
        result.extend(finally_bc)

        return result


@dataclass
class _Catch(_Expression):
    """Inline catch expression: `expr ! codes => default'."""
    expr: _Expression
    codes: list
    default: _Expression = None

    def to_bytecode(self, state: CompilerState, program: Program):
        from moo_interp.opcodes import Extended_Opcode

        expr_bc = self.expr.to_bytecode(state, program)
        default_bc = self.default.to_bytecode(state, program) if self.default else []

        result = []

        # 1. Push error codes as list
        result.append(Instruction(opcode=Opcode.OP_IMM, operand=self.codes))

        # 2. EOP_PUSH_LABEL - handler offset (where to jump on error)
        # handler is after: CATCH + expr_bc + END_CATCH
        handler_offset = len(expr_bc) + 2
        result.append(Instruction(
            opcode=Opcode.OP_EXTENDED,
            operand=Extended_Opcode.EOP_PUSH_LABEL.value,
            handler_offset=handler_offset
        ))

        # 3. EOP_CATCH - set up catch handler
        result.append(Instruction(
            opcode=Opcode.OP_EXTENDED,
            operand=Extended_Opcode.EOP_CATCH.value,
            error_codes=self.codes
        ))

        # 4. Main expression
        result.extend(expr_bc)

        # 5. EOP_END_CATCH - success path, jump past default
        jump_dist = len(default_bc) + 1 if default_bc else 2
        result.append(Instruction(
            opcode=Opcode.OP_EXTENDED,
            operand=Extended_Opcode.EOP_END_CATCH.value,
            handler_offset=jump_dist
        ))

        # 6. Handler: pop exception tuple, evaluate default
        result.append(Instruction(opcode=Opcode.OP_POP, operand=1))
        if default_bc:
            result.extend(default_bc)
        else:
            result.append(Instruction(opcode=Opcode.OP_IMM, operand=0))

        return result


class ToAst(Transformer):

    def BOOLEAN(self, b):
        return BooleanLiteral(b == "true")

    # MOO type constants
    MOO_TYPE_CONSTANTS = {
        'INT': 0,
        'OBJ': 1,
        'STR': 2,
        'ERR': 3,
        'LIST': 4,
        'FLOAT': 9,
        'MAP': 10,
        'BOOL': 12,
        'WAIF': 13,
        'ANON': 14,
    }

    def IDENTIFIER(self, s):
        if s.value == "true" or s.value == "false":
            return BooleanLiteral(s.value == "true")
        # Handle MOO type constants (INT, OBJ, STR, LIST, etc.)
        if s.value in self.MOO_TYPE_CONSTANTS:
            return NumberLiteral(self.MOO_TYPE_CONSTANTS[s.value])
        return Identifier(s.value)

    def ESCAPED_STRING(self, s):
        # Strip quotes and unescape the content
        raw_content = s[1:-1]
        # Python's decode handles standard escape sequences: \n, \t, \", \\, etc.
        unescaped = raw_content.encode('utf-8').decode('unicode_escape')
        return StringLiteral(unescaped)

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

    def for_clause(self, args):
        # for-list uses IN terminal (passed as Token), for-range uses "in" literal (NOT passed)
        # for-list: [id, IN_token, expr] or [id, idx, IN_token, expr]
        # for-range: [id, start, end] or [id, idx, start, end]
        from lark import Token
        if len(args) == 4:
            # Could be for-list with index OR for-range with index
            if isinstance(args[2], Token):
                # for-list with index: [id, idx, IN_token, expr]
                [identifier, index, token, lst] = args
                return _ForClause(identifier, index, lst)
            else:
                # for-range with index: [id, idx, start, end]
                [identifier, index, start, end] = args
                return _ForRangeClause(identifier, start, end)
        elif len(args) == 3:
            # Could be for-list without index OR for-range without index
            if isinstance(args[1], Token):
                # for-list without index: [id, IN_token, expr]
                [identifier, token, lst] = args
                return _ForClause(identifier, None, lst)
            else:
                # for-range without index: [id, start, end]
                [identifier, start, end] = args
                return _ForRangeClause(identifier, start, end)
        else:
            # Shouldn't happen, but fallback
            raise ValueError(f"Unexpected for_clause args: {args}")

    def fork_clause(self, args):
        """Parse fork clause: 'fork' IDENTIFIER? '(' expression ')'

        Returns (var_id, delay_expr) tuple.
        """
        if len(args) == 2:
            # fork with id: [identifier, expression]
            var_id = args[0].value if hasattr(args[0], 'value') else str(args[0])
            delay = args[1]
            return (var_id, delay)
        else:
            # fork without id: [expression]
            delay = args[0]
            return (None, delay)

    def fork_statement(self, args):
        """Parse fork statement: fork_clause statement* 'endfork'

        Due to `?` on fork_clause in grammar:
        - fork (expr): args = [expr, stmts...] (fork_clause is transparent)
        - fork id (expr): args = [(var_id, delay), stmts...] (fork_clause returns tuple)
        """
        import lark
        first = args[0]

        # fork_clause may be a Tree if it wasn't transformed yet, or a tuple if it was
        if isinstance(first, lark.tree.Tree) and first.data == 'fork_clause':
            # Manually extract from the Tree
            fc_children = first.children
            if len(fc_children) == 2:
                # fork id (expr): [identifier, expression]
                var_id = fc_children[0].value if hasattr(fc_children[0], 'value') else str(fc_children[0])
                delay = fc_children[1]
            else:
                # fork (expr): [expression]
                var_id = None
                delay = fc_children[0]
            body_stmts = args[1:]
        elif isinstance(first, tuple):
            # Fork with ID: first arg is tuple from fork_clause
            var_id, delay = first
            body_stmts = args[1:]
        else:
            # Fork without ID: first arg is the delay expression
            var_id = None
            delay = first
            body_stmts = args[1:]

        body = _Body(*body_stmts)
        return _ForkStatement(delay=delay, body=body, var_id=var_id)

    def function_call(self, call):
        name, args = call
        return _FunctionCall(name.value, args)

    def call_arguments(self, args):
        # Collect all arguments into a _List
        return _List(list(args))

    def property(self, access):
        object, name = access
        # Handle both obj.prop (Identifier) and obj.(expr) (any expression)
        if isinstance(name, Identifier):
            # Treat foo.bar like foo.("bar") for simplicity
            name = StringLiteral(name.value)
        # else: name is already an expression from obj.(expr)
        return _Property(object, name)

    def verb_call(self, call):
        object, name, args = call
        # Handle both obj:verb (Identifier) and obj:(expr) (any expression)
        if isinstance(name, Identifier):
            # Treat this:helper like this:("helper") for correct bytecode
            name = StringLiteral(name.value)
        # args is now a _List from call_arguments transformer
        if isinstance(args, _List):
            return _VerbCall(object, name, args)
        # Fallback for edge cases
        return _VerbCall(object, name, _List(list(args) if args else []))

    def index(self, args):
        # args = [object, index_target]
        obj = args[0]
        idx = args[1]
        return _Index(object=obj, index=idx)

    def range(self, args):
        # args = [object, range_start, range_end]
        obj = args[0]
        start = args[1]
        end = args[2]
        return _Range(object=obj, start=start, end=end)

    def first_index(self, args):
        return _FirstIndex()

    def last_index(self, args):
        return _LastIndex()

    def dollar(self, args):
        """Handle $ in expression context (e.g., inside function calls like min($, 3))."""
        return _LastIndex()

    def unary_expression(self, args):
        # args = [operator_token, operand]
        op = str(args[0])
        operand = args[1]
        return _UnaryExpression(operator=op, operand=operand)

    def binary_expression(self, args):
        # args = [left, operator_token, right]
        left = args[0]
        op = str(args[1])
        right = args[2]
        return BinaryExpression(left=left, operator=op, right=right)

    def single_statement(self, args):
        # Empty statement (just ';') has no children
        if not args:
            return _EmptyStatement()
        # Normal statement with content
        return _SingleStatement(statement=args[0])

    def waif_property(self, args):
        # args = [object, name_identifier]
        obj = args[0]
        name = args[1].value if hasattr(args[1], 'value') else str(args[1])
        return _WaifProperty(object=obj, name=name)

    def catch(self, args):
        # catch: "`" expression "!" exception_codes ("=>" expression)? "'"
        # args = [expr, exception_codes_tree, optional_default_expr]
        expr = args[0]

        # Parse exception codes
        codes = []
        codes_tree = args[1]
        if isinstance(codes_tree, lark.tree.Tree):
            for child in codes_tree.children:
                if isinstance(child, lark.tree.Tree) and child.data == 'exception_code':
                    code_id = child.children[0]
                    if isinstance(code_id, Identifier):
                        codes.append(code_id.value)
                    else:
                        codes.append(str(code_id))
                elif isinstance(child, Identifier):
                    codes.append(child.value)
                elif hasattr(child, 'value'):
                    codes.append(child.value)
        if not codes:
            codes = ['ANY']

        # Optional default expression
        default = args[2] if len(args) > 2 else None

        return _Catch(expr=expr, codes=codes, default=default)

    def ternary(self, args):
        """ternary: logical_or "?" expression "|" expression."""
        # args = [condition, true_value, false_value]
        condition = args[0]
        true_value = args[1]
        false_value = args[2]
        return _Ternary(condition=condition, true_value=true_value, false_value=false_value)

    def scatter(self, args):
        """scatter: "{" scattering_target "}" "=" expression."""
        # args = [scattering_target_tree, expression]
        target_tree = args[0]
        value_expr = args[1]

        # Parse target_tree into _ScatterTarget
        if isinstance(target_tree, _ScatterTarget):
            target = target_tree
        else:
            # Should be a Tree - transform its items
            items = []
            for item in target_tree.children:
                items.append(self.scattering_target_item([item] if not hasattr(item, 'children') else item.children))
            target = _ScatterTarget(items=items)

        return _ScatterAssignment(target=target, value=value_expr)

    def scattering_target(self, args):
        """scattering_target: scattering_target_item ("," scattering_target_item)*."""
        items = []
        for arg in args:
            if isinstance(arg, _ScatterItem):
                items.append(arg)
            elif hasattr(arg, 'children'):
                # Tree node - transform it
                items.append(self.scattering_target_item(arg.children))
            else:
                # Token - it's an identifier
                items.append(_ScatterItem(var_name=str(arg), is_optional=False, is_rest=False))
        return _ScatterTarget(items=items)

    def scattering_target_item(self, args):
        """scattering_target_item: IDENTIFIER | "?" IDENTIFIER ("=" expression)? | "@" IDENTIFIER."""
        if not args:
            return _ScatterItem(var_name="", is_optional=False, is_rest=False)

        def get_var_name(arg):
            """Extract variable name from Token or Identifier."""
            if isinstance(arg, Identifier):
                return arg.value
            elif hasattr(arg, 'value'):
                return str(arg.value)
            else:
                return str(arg)

        first = args[0]
        # Get string representation for type check
        first_str = str(first.type) if hasattr(first, 'type') else get_var_name(first)

        if first_str == "QMARK" or (hasattr(first, '__str__') and str(first) == "?"):
            # Optional: QMARK IDENTIFIER [EQUALS expression]
            # args[0] = QMARK, args[1] = IDENTIFIER, args[2] = EQUALS (if present), args[3] = expression (if present)
            var_name = get_var_name(args[1]) if len(args) > 1 else ""
            # Skip EQUALS token (args[2]) and get the default expression (args[3])
            default = args[3] if len(args) > 3 else None
            return _ScatterItem(var_name=var_name, is_optional=True, is_rest=False, default=default)
        elif first_str == "AT" or (hasattr(first, '__str__') and str(first) == "@"):
            # Rest: @IDENTIFIER
            var_name = get_var_name(args[1]) if len(args) > 1 else ""
            return _ScatterItem(var_name=var_name, is_optional=False, is_rest=True)
        else:
            # Required: IDENTIFIER
            var_name = get_var_name(first)
            return _ScatterItem(var_name=var_name, is_optional=False, is_rest=False)

    def try_except_statement(self, args):
        # args = [statement*, except_statement+]
        # Split into try body statements and except clauses
        try_stmts = []
        except_clauses = []
        for arg in args:
            if isinstance(arg, _ExceptClause):
                except_clauses.append(arg)
            elif isinstance(arg, lark.tree.Tree):
                if arg.data == 'except_statement':
                    # Parse except_statement tree into _ExceptClause
                    clause = self._parse_except_clause(arg)
                    except_clauses.append(clause)
                else:
                    # Some other tree - might be a statement
                    try_stmts.append(arg)
            else:
                try_stmts.append(arg)

        # Wrap single statement in _Body if needed
        if len(try_stmts) == 1 and isinstance(try_stmts[0], _SingleStatement):
            try_body = try_stmts[0]
        else:
            try_body = _Body(*try_stmts) if try_stmts else _Body()

        return _TryExceptStatement(try_body=try_body, except_clauses=except_clauses)

    def _parse_except_clause(self, tree):
        # except_statement: except_clause statement*
        # except_clause: "except" [IDENTIFIER] "(" exception_codes ")"
        children = tree.children
        except_clause_tree = children[0]
        body_stmts_raw = children[1:]

        # Transform body statements - they may still be Tree objects
        body_stmts = []
        for stmt in body_stmts_raw:
            if isinstance(stmt, lark.tree.Tree):
                # Skip empty statement trees
                if not stmt.children:
                    continue
                # Transform the tree to get the AST node
                transformed = transformer.transform(stmt)
                body_stmts.append(transformed)
            else:
                body_stmts.append(stmt)

        # Parse except_clause
        var = None
        codes = []
        for child in except_clause_tree.children:
            if isinstance(child, lark.tree.Tree) and child.data == 'exception_codes':
                for code_tree in child.children:
                    if isinstance(code_tree, lark.tree.Tree) and code_tree.data == 'exception_code':
                        code_id = code_tree.children[0]
                        if isinstance(code_id, Identifier):
                            codes.append(code_id.value)
                        else:
                            codes.append(str(code_id))
            elif isinstance(child, Identifier):
                var = child.value

        return _ExceptClause(codes=codes, var=var, body=_Body(*body_stmts))

    def try_finally_statement(self, args):
        # args = [statement*, finally_statement]
        # The last arg is the finally_statement tree
        finally_tree = args[-1]
        try_stmts = args[:-1]

        # Parse finally body
        if isinstance(finally_tree, lark.tree.Tree) and finally_tree.data == 'finally_statement':
            finally_stmts = finally_tree.children
        else:
            finally_stmts = [finally_tree]

        return _TryFinallyStatement(try_body=_Body(*try_stmts), finally_body=_Body(*finally_stmts))

    def except_statement(self, args):
        # Let try_except_statement handle this
        return lark.tree.Tree('except_statement', args)

    def finally_statement(self, args):
        # Let try_finally_statement handle this
        return lark.tree.Tree('finally_statement', args)

    @v_args(inline=True)
    def start(self, x):
        # Filter out non-AST nodes (empty Tree from parser on empty input)
        children = [c for c in x.children if isinstance(c, _AstNode)]
        return VerbCode(children)


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


def compile(tree, bi_funcs=None, context_vars=None):
    """Compile MOO code to bytecode.

    Args:
        tree: AST tree, string, or list of strings to compile
        bi_funcs: Optional BuiltinFunctions instance to use for builtin lookups
        context_vars: Optional list of context variable names to pre-register.
                     These are registered FIRST so they get stable indices.
                     MOO context vars: player, this, caller, verb, args, argstr,
                                      dobj, dobjstr, prepstr, iobj, iobjstr

    Returns:
        StackFrame ready for execution
    """
    if isinstance(tree, list):
        tree = parse("".join(tree))
    elif isinstance(tree, str):
        tree = parse(tree)
    bc = []
    state = CompilerState(bi_funcs=bi_funcs)

    # Create program early so fork_vectors can be populated during compilation
    prog = Program(var_names=[])

    # Pre-register context variables BEFORE compiling code
    # This ensures they get stable indices (0-10 for standard MOO context)
    if context_vars:
        for var_name in context_vars:
            state.add_var(var_name)

    for node in tree.children:
        bc += node.to_bytecode(state, prog)
    bc = bc + [Instruction(opcode=Opcode.OP_DONE)]

    # Update program with final variable names
    prog.var_names = list(state.var_names)

    # Create frame with rt_env initialized for all tracked variables
    frame = StackFrame(func_id=0, prog=prog, ip=0, stack=bc)
    # Initialize rt_env with None/0 for each tracked variable
    frame.rt_env = [0] * len(state.var_names)
    return frame


def disassemble(frame: StackFrame):
    bc = frame.stack
    for instruction in bc:
        if isinstance(instruction.opcode, int) and instruction.opcode >= 53:
            from .opcodes import opcode_to_optim_num
            print(f"Num            {opcode_to_optim_num(instruction.opcode)}")
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
