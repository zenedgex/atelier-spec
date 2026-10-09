"""Parameter expressions in contracts and machines: `2*DW + clog2(R)`, `4 + len / LANES`, `len % 64 == 0`.

Arithmetic, comparisons, `and`/`or`/`not`, conditional expressions, and the functions below. Names are
parameters or command fields. Division of integers is exact when it divides, else a float.
"""

from __future__ import annotations

import ast
import math
import operator

FUNCS = {
    "clog2": lambda x: max(0, math.ceil(math.log2(x))) if x > 0 else 0,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
    "min": min,
    "max": max,
    "abs": abs,
    "int": int,
}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow, ast.LShift: operator.lshift,
        ast.RShift: operator.rshift, ast.BitAnd: operator.and_, ast.BitOr: operator.or_}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge}


class ExprError(ValueError):
    pass


def parse(text: str) -> ast.Expression:
    try:
        tree = ast.parse(str(text), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot parse {text!r}: {e.msg}") from None
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
                               ast.Call, ast.Name, ast.Constant, ast.Load, ast.operator, ast.unaryop,
                               ast.boolop, ast.cmpop))
        if not ok:
            raise ExprError(f"{text!r}: {type(node).__name__} is not allowed in an expression")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in FUNCS):
            raise ExprError(f"{text!r}: unknown function")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise ExprError(f"{text!r}: only numbers are allowed as constants")
    return tree


def names(text: str) -> set[str]:
    """The free names of an expression (function names excluded)."""
    tree = parse(text)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - calls


def evaluate(text: str, env: dict):
    return _ev(parse(text).body, env, text)


def _ev(n, env, text):
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.Name):
        if n.id not in env:
            raise ExprError(f"{text!r}: {n.id} is not defined")
        return env[n.id]
    if isinstance(n, ast.BinOp):
        a, b = _ev(n.left, env, text), _ev(n.right, env, text)
        if isinstance(n.op, ast.Div):
            q = a / b
            return int(q) if isinstance(a, int) and isinstance(b, int) and a % b == 0 else q
        return _BIN[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp):
        v = _ev(n.operand, env, text)
        return {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_,
                ast.Invert: operator.invert}[type(n.op)](v)
    if isinstance(n, ast.BoolOp):
        vals = (_ev(v, env, text) for v in n.values)
        return all(vals) if isinstance(n.op, ast.And) else any(vals)
    if isinstance(n, ast.Compare):
        left = _ev(n.left, env, text)
        for op, right in zip(n.ops, n.comparators):
            r = _ev(right, env, text)
            if not _CMP[type(op)](left, r):
                return False
            left = r
        return True
    if isinstance(n, ast.IfExp):
        return _ev(n.body, env, text) if _ev(n.test, env, text) else _ev(n.orelse, env, text)
    if isinstance(n, ast.Call):
        return FUNCS[n.func.id](*(_ev(a, env, text) for a in n.args))
    raise ExprError(f"{text!r}: unsupported")
