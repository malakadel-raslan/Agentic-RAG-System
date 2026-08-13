"""
Tool #3: calculate
For questions that need arithmetic on numbers found in the documents
(e.g. "what's the percentage difference between X and Y figures?").
Uses a restricted eval — only numbers and basic math operators are allowed.
"""
import ast
import operator
from langchain_core.tools import tool

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression, e.g. "(120 - 95) / 95 * 100".
    Supports +, -, *, /, %, ** and parentheses. Use this for any math
    (percentages, sums, differences, averages) needed to answer the question
    — never do arithmetic in your head, always call this tool.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
