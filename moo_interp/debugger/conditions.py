"""Conditional expression parser and evaluator for breakpoints."""

import re
import fnmatch
from typing import Any, Dict, Optional


class ConditionNode:
    """Base class for condition AST nodes."""

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition in the given context.

        Args:
            context: Dictionary with available variables (verb, return_value, args, etc.)

        Returns:
            True if condition is satisfied
        """
        raise NotImplementedError


class BinaryOp(ConditionNode):
    """Binary operation (and, or)."""

    def __init__(self, left: ConditionNode, op: str, right: ConditionNode):
        self.left = left
        self.op = op.lower()
        self.right = right

    def evaluate(self, context: Dict[str, Any]) -> bool:
        if self.op == 'and':
            return self.left.evaluate(context) and self.right.evaluate(context)
        elif self.op == 'or':
            return self.left.evaluate(context) or self.right.evaluate(context)
        else:
            raise ValueError(f"Unknown binary operator: {self.op}")


class Comparison(ConditionNode):
    """Comparison operation (==, !=, matches)."""

    def __init__(self, left: str, op: str, right: Any):
        self.left = left  # variable name
        self.op = op.lower()
        self.right = right  # literal value

    def evaluate(self, context: Dict[str, Any]) -> bool:
        # Get left-hand value from context
        left_value = self._get_value(self.left, context)

        if self.op == '==':
            return left_value == self.right
        elif self.op == '!=':
            return left_value != self.right
        elif self.op == 'matches':
            # Glob pattern matching
            if left_value is None:
                return False
            return fnmatch.fnmatch(str(left_value), str(self.right))
        else:
            raise ValueError(f"Unknown comparison operator: {self.op}")

    def _get_value(self, name: str, context: Dict[str, Any]) -> Any:
        """Extract value from context, handling array indices."""
        # Handle arg[N] notation
        match = re.match(r'(\w+)\[(\d+)\]', name)
        if match:
            var_name, index = match.groups()
            container = context.get(var_name)
            if container is None:
                return None
            try:
                return container[int(index)]
            except (IndexError, KeyError, TypeError):
                return None

        # Simple variable lookup
        return context.get(name)


class ConditionParser:
    """Parser for conditional breakpoint expressions.

    Supports:
    - verb == "name"
    - verb matches "pattern*"
    - return_value == VALUE
    - arg[0] == "value"
    - Combination with 'and', 'or'
    """

    def parse(self, expression: str) -> ConditionNode:
        """Parse a condition expression into an AST.

        Args:
            expression: Condition string

        Returns:
            Root node of condition AST
        """
        # Tokenize
        tokens = self._tokenize(expression)

        # Parse
        return self._parse_or(tokens)

    def _tokenize(self, expression: str) -> list:
        """Split expression into tokens."""
        # Pattern matches: keywords first, then operators, then variables, strings, numbers
        # Order matters - check keywords before variables since 'matches', 'and', 'or' look like vars
        pattern = r'''
            (?P<KEYWORD>\b(?:and|or)\b)|                   # Boolean operators (word boundary)
            (?P<OP>matches|==|!=)|                         # Comparison operators (matches before ==)
            (?P<VAR>[a-zA-Z_][a-zA-Z0-9_]*(?:\[\d+\])?)|  # Variables (with optional [N])
            (?P<STR>"[^"]*"|'[^']*')|                      # Strings
            (?P<NUM>-?\d+(?:\.\d+)?)                       # Numbers
        '''

        tokens = []
        for match in re.finditer(pattern, expression, re.VERBOSE | re.IGNORECASE):
            kind = match.lastgroup
            value = match.group()

            if kind == 'KEYWORD':
                tokens.append(('KEYWORD', value.lower()))
            elif kind == 'OP':
                tokens.append(('OP', value.lower()))
            elif kind == 'VAR':
                tokens.append(('VAR', value))
            elif kind == 'STR':
                # Remove quotes
                tokens.append(('STR', value[1:-1]))
            elif kind == 'NUM':
                # Convert to int or float
                if '.' in value:
                    tokens.append(('NUM', float(value)))
                else:
                    tokens.append(('NUM', int(value)))

        return tokens

    def _parse_or(self, tokens: list) -> ConditionNode:
        """Parse OR expressions (lowest precedence)."""
        left = self._parse_and(tokens)

        while tokens and tokens[0][0] == 'KEYWORD' and tokens[0][1] == 'or':
            tokens.pop(0)  # consume 'or'
            right = self._parse_and(tokens)
            left = BinaryOp(left, 'or', right)

        return left

    def _parse_and(self, tokens: list) -> ConditionNode:
        """Parse AND expressions (higher precedence)."""
        left = self._parse_comparison(tokens)

        while tokens and tokens[0][0] == 'KEYWORD' and tokens[0][1] == 'and':
            tokens.pop(0)  # consume 'and'
            right = self._parse_comparison(tokens)
            left = BinaryOp(left, 'and', right)

        return left

    def _parse_comparison(self, tokens: list) -> ConditionNode:
        """Parse comparison expressions (highest precedence)."""
        if not tokens:
            raise ValueError("Unexpected end of expression")

        # Expect: VAR OP (STR|NUM)
        if tokens[0][0] != 'VAR':
            raise ValueError(f"Expected variable, got {tokens[0]}")

        var_name = tokens.pop(0)[1]

        if not tokens or tokens[0][0] != 'OP':
            raise ValueError(f"Expected operator after {var_name}")

        op = tokens.pop(0)[1]

        if not tokens or tokens[0][0] not in ('STR', 'NUM'):
            raise ValueError(f"Expected value after {op}")

        value = tokens.pop(0)[1]

        return Comparison(var_name, op, value)


def parse_condition(expression: str) -> ConditionNode:
    """Parse a condition expression.

    Args:
        expression: Condition string like 'verb == "name"'

    Returns:
        Parsed condition AST
    """
    parser = ConditionParser()
    return parser.parse(expression)


def evaluate_condition(expression: str, context: Dict[str, Any]) -> bool:
    """Parse and evaluate a condition expression.

    Args:
        expression: Condition string
        context: Variable context

    Returns:
        True if condition is satisfied
    """
    condition = parse_condition(expression)
    return condition.evaluate(context)
