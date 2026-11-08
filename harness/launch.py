"""Whitelisted expression evaluator for launch geometry and SOL formulas.

Kernel manifests contain small arithmetic expressions (`ceildiv(N,256)`,
`8*N`) that must be evaluated without `eval`: only integer literals, names bound
from `shape`, `+ - * // / %`, parentheses and the functions `min max ceildiv
sqrt`. Anything else raises :class:`LaunchExprError`.
"""

from __future__ import annotations

import ast
from typing import Any


class LaunchExprError(ValueError):
    """Raised for any expression outside the whitelist."""


_FUNCS = {
    "min": min,
    "max": max,
    "ceildiv": lambda a, b: -(-int(a) // int(b)),
    "sqrt": lambda a: int(int(a) ** 0.5),
}

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}

_UNOPS = {ast.USub: lambda a: -a, ast.UAdd: lambda a: a}


def _eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise LaunchExprError(f"non-numeric constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise LaunchExprError(f"unknown name: {node.id}")
        return env[node.id]
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise LaunchExprError(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        op = _UNOPS.get(type(node.op))
        if op is None:
            raise LaunchExprError(f"unary operator not allowed: {type(node.op).__name__}")
        return op(_eval(node.operand, env))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise LaunchExprError("only whitelisted function calls are allowed")
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise LaunchExprError(f"function not allowed: {node.func.id}")
        if node.keywords:
            raise LaunchExprError("keyword arguments are not allowed")
        return fn(*[_eval(a, env) for a in node.args])
    raise LaunchExprError(f"expression not allowed: {type(node).__name__}")


def evaluate(expr: str | int, env: dict[str, Any] | None = None) -> int:
    """Evaluate a whitelisted integer expression."""
    if isinstance(expr, int):
        return expr
    env = dict(env or {})
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise LaunchExprError(f"invalid expression: {expr!r}") from exc
    value = _eval(tree, env)
    if isinstance(value, float):
        if value != int(value):
            raise LaunchExprError(f"expression is not integral: {expr!r}")
        return int(value)
    return int(value)


def eval_all(exprs: dict[str, Any], env: dict[str, Any] | None = None) -> dict[str, int]:
    """Evaluate every value of a mapping, threading results into the environment."""
    out: dict[str, int] = {}
    e = dict(env or {})
    for k, v in exprs.items():
        r = evaluate(v, e)
        out[k] = r
        e[k] = r
    return out
