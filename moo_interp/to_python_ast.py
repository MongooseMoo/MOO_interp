"""Transform MOO AST nodes into Python ast module nodes for static analysis."""

import ast as pyast

from . import moo_ast


# Binary operator mapping: MOO operator string → Python AST operator node
_BINOP_MAP = {
    '+': pyast.Add,
    '-': pyast.Sub,
    '*': pyast.Mult,
    '/': pyast.Div,
    '%': pyast.Mod,
    '^': pyast.Pow,
    '|.': pyast.BitOr,
    '&.': pyast.BitAnd,
    '^.': pyast.BitXor,
    '<<': pyast.LShift,
    '>>': pyast.RShift,
}

# Comparison operator mapping
_CMPOP_MAP = {
    '==': pyast.Eq,
    '!=': pyast.NotEq,
    '<': pyast.Lt,
    '>': pyast.Gt,
    '<=': pyast.LtE,
    '>=': pyast.GtE,
    'in': pyast.In,
}

# Logical operator mapping
_LOGICOP_MAP = {
    '&&': pyast.And,
    '||': pyast.Or,
}

# Unary operator mapping
_UNARYOP_MAP = {
    '-': pyast.USub,
    '!': pyast.Not,
    '~': pyast.Invert,
}


class MooPythonTransformer:
    """Transforms MOO AST nodes into Python ast module nodes."""

    def transform(self, node):
        """Dispatch to the appropriate transform method based on node type."""
        method_name = f'transform_{type(node).__name__}'
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(
                f"No transform method for {type(node).__name__}")
        return method(node)

    def transform_body(self, body_nodes):
        """Transform a list of statement nodes into a list of Python AST statements."""
        result = []
        for node in body_nodes:
            transformed = self.transform(node)
            if isinstance(transformed, list):
                result.extend(transformed)
            else:
                result.append(transformed)
        return result

    # --- Top-level ---

    def transform_VerbCode(self, node):
        body = self.transform_body(node.children)
        if not body:
            body = [pyast.Pass()]
        return pyast.Module(body=body, type_ignores=[])

    # --- Statements ---

    def transform__SingleStatement(self, node):
        inner = node.statement
        if isinstance(inner, moo_ast._Expression):
            # Expression used as statement
            if isinstance(inner, moo_ast._Assign):
                return self._transform_assign_as_statement(inner)
            transformed = self.transform(inner)
            return pyast.Expr(value=transformed)
        else:
            # Proper statement (return, break, etc.)
            return self.transform(inner)

    def transform__EmptyStatement(self, node):
        return pyast.Pass()

    def transform_ReturnStatement(self, node):
        if node.value is None:
            return pyast.Return(value=None)
        return pyast.Return(value=self.transform(node.value))

    def transform_BreakStatement(self, node):
        return pyast.Break()

    def transform_ContinueStatement(self, node):
        return pyast.Continue()

    def transform__IfStatement(self, node):
        # Build the if clause
        test = self.transform(node.if_clause.condition)
        body = self._transform_body_node(node.if_clause.then_block)

        # Build orelse from elseif + else clauses
        orelse = self._build_orelse(node.elseif_clauses, node.else_clause)

        return pyast.If(test=test, body=body, orelse=orelse)

    def _build_orelse(self, elseif_clauses, else_clause):
        if elseif_clauses:
            first = elseif_clauses[0]
            rest = elseif_clauses[1:]
            nested_if = pyast.If(
                test=self.transform(first.condition),
                body=self._transform_body_node(first.then_block),
                orelse=self._build_orelse(rest, else_clause),
            )
            return [nested_if]
        elif else_clause is not None:
            return self._transform_body_node(else_clause.then_block)
        return []

    def transform_WhileStatement(self, node):
        test = self.transform(node.condition)
        body = self._transform_body_node(node.body)
        return pyast.While(test=test, body=body, orelse=[])

    def transform_ForStatement(self, node):
        condition = node.condition
        if isinstance(condition, moo_ast._ForClause):
            target = pyast.Name(id=condition.id.value, ctx=pyast.Store())
            iter_expr = self.transform(condition.iterable)
            body = self._transform_body_node(node.body)
            return pyast.For(target=target, iter=iter_expr, body=body, orelse=[])
        elif isinstance(condition, moo_ast._ForRangeClause):
            target = pyast.Name(id=condition.id.value, ctx=pyast.Store())
            start = self.transform(condition.start)
            end = self.transform(condition.end)
            # MOO [1..10] is inclusive, Python range is exclusive, so range(start, end+1)
            end_plus_one = pyast.BinOp(
                left=end, op=pyast.Add(), right=pyast.Constant(value=1))
            iter_expr = pyast.Call(
                func=pyast.Name(id='range', ctx=pyast.Load()),
                args=[start, end_plus_one],
                keywords=[],
            )
            body = self._transform_body_node(node.body)
            return pyast.For(target=target, iter=iter_expr, body=body, orelse=[])
        raise NotImplementedError(f"Unknown for clause type: {type(condition).__name__}")

    def transform__ForkStatement(self, node):
        delay = self.transform(node.delay)
        body_stmts = self._transform_body_node(node.body)
        # Represent fork as fork(delay, lambda: body)
        # Simplified: just fork(delay) as a call
        return pyast.Expr(value=pyast.Call(
            func=pyast.Name(id='fork', ctx=pyast.Load()),
            args=[delay],
            keywords=[],
        ))

    def transform__TryExceptStatement(self, node):
        body = self._transform_body_node(node.try_body)
        handlers = []
        for clause in node.except_clauses:
            handler_body = self._transform_body_node(clause.body) if clause.body else [pyast.Pass()]
            handler = pyast.ExceptHandler(
                type=pyast.Name(id='Exception', ctx=pyast.Load()),
                name=clause.var,
                body=handler_body,
            )
            handlers.append(handler)
        return pyast.Try(body=body, handlers=handlers, orelse=[], finalbody=[])

    def transform__TryFinallyStatement(self, node):
        body = self._transform_body_node(node.try_body)
        finalbody = self._transform_body_node(node.finally_body)
        return pyast.Try(body=body, handlers=[], orelse=[], finalbody=finalbody)

    def transform__ScatterAssignment(self, node):
        elts = []
        for item in node.target.items:
            elts.append(pyast.Name(id=item.var_name, ctx=pyast.Store()))
        target = pyast.Tuple(elts=elts, ctx=pyast.Store())
        value = self.transform(node.value)
        return pyast.Assign(targets=[target], value=value)

    # --- Expressions ---

    def transform_NumberLiteral(self, node):
        return pyast.Constant(value=node.value)

    def transform_StringLiteral(self, node):
        return pyast.Constant(value=node.value)

    def transform_FloatLiteral(self, node):
        return pyast.Constant(value=node.value)

    def transform_BooleanLiteral(self, node):
        return pyast.Constant(value=node.value)

    def transform_ObjnumLiteral(self, node):
        return pyast.Call(
            func=pyast.Name(id='ObjNum', ctx=pyast.Load()),
            args=[pyast.Constant(value=node.value)],
            keywords=[],
        )

    def transform_Identifier(self, node):
        return pyast.Name(id=node.value, ctx=pyast.Load())

    def transform_BinaryExpression(self, node):
        op = node.operator
        if op in _LOGICOP_MAP:
            return pyast.BoolOp(
                op=_LOGICOP_MAP[op](),
                values=[self.transform(node.left), self.transform(node.right)],
            )
        if op in _CMPOP_MAP:
            return pyast.Compare(
                left=self.transform(node.left),
                ops=[_CMPOP_MAP[op]()],
                comparators=[self.transform(node.right)],
            )
        if op in _BINOP_MAP:
            return pyast.BinOp(
                left=self.transform(node.left),
                op=_BINOP_MAP[op](),
                right=self.transform(node.right),
            )
        raise NotImplementedError(f"Unknown binary operator: {op}")

    def transform__UnaryExpression(self, node):
        op = node.operator
        if op not in _UNARYOP_MAP:
            raise NotImplementedError(f"Unknown unary operator: {op}")
        return pyast.UnaryOp(
            op=_UNARYOP_MAP[op](),
            operand=self.transform(node.operand),
        )

    def transform__Ternary(self, node):
        return pyast.IfExp(
            test=self.transform(node.condition),
            body=self.transform(node.true_value),
            orelse=self.transform(node.false_value),
        )

    def transform__List(self, node):
        elts = []
        for elem in node.value:
            if isinstance(elem, moo_ast.Splicer):
                elts.append(pyast.Starred(
                    value=self.transform(elem.expression),
                    ctx=pyast.Load(),
                ))
            else:
                elts.append(self.transform(elem))
        return pyast.List(elts=elts, ctx=pyast.Load())

    def transform_Map(self, node):
        keys = []
        values = []
        for entry in node.value:
            key, value = entry.children
            keys.append(self.transform(key))
            values.append(self.transform(value))
        return pyast.Dict(keys=keys, values=values)

    def transform_Splicer(self, node):
        return pyast.Starred(
            value=self.transform(node.expression),
            ctx=pyast.Load(),
        )

    def transform__FunctionCall(self, node):
        args = [self.transform(arg) for arg in node.arguments.value]
        return pyast.Call(
            func=pyast.Name(id=node.name, ctx=pyast.Load()),
            args=args,
            keywords=[],
        )

    def transform__Property(self, node):
        obj = self.transform(node.object)
        if isinstance(node.name, moo_ast.StringLiteral):
            return pyast.Attribute(value=obj, attr=node.name.value, ctx=pyast.Load())
        else:
            # Computed property access
            return pyast.Call(
                func=pyast.Name(id='getattr', ctx=pyast.Load()),
                args=[obj, self.transform(node.name)],
                keywords=[],
            )

    def transform_DollarProperty(self, node):
        # $foo → system.foo
        if isinstance(node.name, moo_ast.Identifier):
            prop_name = node.name.value
        elif isinstance(node.name, moo_ast.StringLiteral):
            prop_name = node.name.value
        else:
            prop_name = str(node.name.value)
        return pyast.Attribute(
            value=pyast.Name(id='system', ctx=pyast.Load()),
            attr=prop_name,
            ctx=pyast.Load(),
        )

    def transform__VerbCall(self, node):
        obj = self.transform(node.object)
        if isinstance(node.name, moo_ast.StringLiteral):
            func = pyast.Attribute(value=obj, attr=node.name.value, ctx=pyast.Load())
        elif isinstance(node.name, moo_ast.Identifier):
            func = pyast.Attribute(value=obj, attr=node.name.value, ctx=pyast.Load())
        else:
            # Computed verb name: obj:(expr)(args) → getattr(obj, expr)(args)
            func = pyast.Call(
                func=pyast.Name(id='getattr', ctx=pyast.Load()),
                args=[obj, self.transform(node.name)],
                keywords=[],
            )
        args = [self.transform(arg) for arg in node.arguments.value]
        return pyast.Call(func=func, args=args, keywords=[])

    def transform_DollarVerbCall(self, node):
        # $obj:verb() → system.obj.verb()
        # DollarVerbCall has name (the verb name) and arguments
        # But the "object" is implicit $obj, so we need to figure out the structure.
        # Looking at moo_ast: DollarVerbCall is parsed from $verb() or $obj:verb()
        # Actually, $string_utils:trim("hi") is parsed as:
        #   _VerbCall(object=DollarProperty(name="string_utils"), name="trim", args=...)
        # Let me check...
        # From the grammar: dollar_verb_call is $verb() which means #0:verb()
        # So DollarVerbCall.name is the verb name, DollarVerbCall.arguments is args
        # Object is implicit #0 (system)
        system_ref = pyast.Name(id='system', ctx=pyast.Load())
        if isinstance(node.name, moo_ast.StringLiteral):
            func = pyast.Attribute(value=system_ref, attr=node.name.value, ctx=pyast.Load())
        elif isinstance(node.name, moo_ast.Identifier):
            func = pyast.Attribute(value=system_ref, attr=node.name.value, ctx=pyast.Load())
        else:
            # Computed verb name: $(expr)(args) → getattr(system, expr)(args)
            func = pyast.Call(
                func=pyast.Name(id='getattr', ctx=pyast.Load()),
                args=[system_ref, self.transform(node.name)],
                keywords=[],
            )
        args = [self.transform(arg) for arg in node.arguments.value]
        return pyast.Call(func=func, args=args, keywords=[])

    def transform__Index(self, node):
        return pyast.Subscript(
            value=self.transform(node.object),
            slice=self.transform(node.index),
            ctx=pyast.Load(),
        )

    def transform__Range(self, node):
        return pyast.Subscript(
            value=self.transform(node.object),
            slice=pyast.Slice(
                lower=self.transform(node.start),
                upper=self.transform(node.end),
            ),
            ctx=pyast.Load(),
        )

    def transform__Assign(self, node):
        # When used as expression, return the value side
        # (statement-level assign is handled by _transform_assign_as_statement)
        return self.transform(node.value)

    def _transform_assign_as_statement(self, node):
        """Transform _Assign node as a statement (ast.Assign)."""
        if isinstance(node.target, moo_ast.Identifier):
            target = pyast.Name(id=node.target.value, ctx=pyast.Store())
        elif isinstance(node.target, moo_ast._List):
            # Destructuring: {a, b} = ...
            elts = []
            for elem in node.target.value:
                if isinstance(elem, moo_ast.Identifier):
                    elts.append(pyast.Name(id=elem.value, ctx=pyast.Store()))
                elif isinstance(elem, moo_ast.Splicer) and isinstance(elem.expression, moo_ast.Identifier):
                    elts.append(pyast.Starred(
                        value=pyast.Name(id=elem.expression.value, ctx=pyast.Store()),
                        ctx=pyast.Store(),
                    ))
            target = pyast.Tuple(elts=elts, ctx=pyast.Store())
        elif isinstance(node.target, moo_ast._Property):
            obj = self.transform(node.target.object)
            if isinstance(node.target.name, moo_ast.StringLiteral):
                target = pyast.Attribute(value=obj, attr=node.target.name.value, ctx=pyast.Store())
            else:
                target = pyast.Subscript(
                    value=obj, slice=self.transform(node.target.name), ctx=pyast.Store())
        elif isinstance(node.target, moo_ast._Index):
            target = pyast.Subscript(
                value=self.transform(node.target.object),
                slice=self.transform(node.target.index),
                ctx=pyast.Store(),
            )
        else:
            target = pyast.Name(id=str(node.target), ctx=pyast.Store())

        value = self.transform(node.value)
        return pyast.Assign(targets=[target], value=value)

    def transform__Catch(self, node):
        # `expr ! codes => default` → moo_catch(lambda: expr, default)
        # Simplified representation as a function call
        return pyast.Call(
            func=pyast.Name(id='moo_catch', ctx=pyast.Load()),
            args=[self.transform(node.expr), self.transform(node.default)] if node.default else [self.transform(node.expr)],
            keywords=[],
        )

    # --- Helpers ---

    def _transform_body_node(self, body_node):
        """Transform a _Body node into a list of Python AST statements."""
        if body_node is None:
            return [pyast.Pass()]
        if isinstance(body_node, moo_ast._Body):
            stmts = self.transform_body(body_node.statements)
        elif isinstance(body_node, moo_ast._SingleStatement):
            stmts = [self.transform(body_node)]
        else:
            stmts = [self.transform(body_node)]
        return stmts if stmts else [pyast.Pass()]


def moo_to_python_ast(source):
    """Convert MOO source code or pre-parsed MOO AST to a Python ast.Module.

    Args:
        source: Either a string of MOO source code, or a pre-parsed
                moo_ast.VerbCode node.

    Returns:
        ast.Module with Python AST nodes.
    """
    if isinstance(source, str):
        moo_tree = moo_ast.parse(source)
    else:
        moo_tree = source

    transformer = MooPythonTransformer()
    py_module = transformer.transform(moo_tree)
    pyast.fix_missing_locations(py_module)
    return py_module
