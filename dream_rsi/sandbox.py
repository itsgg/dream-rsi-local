"""Shared guards for running LLM-written candidate programs: a static check on the source, then a subprocess
that binds what the scorer needs before the candidate runs (so it cannot forge its score by monkeypatching
numpy), restricts builtins and imports, and caps CPU and memory.

A hurdle, not a sandbox: the subprocess still runs as your user.
"""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = {"numpy", "np", "math", "itertools", "functools", "collections", "typing"}

DISALLOWED_CALLS = {"open", "exec", "eval", "__import__", "input", "getattr", "globals", "vars"}


def static_check(src: str, fn_name: str = "priority") -> str | None:
    """Return an error string if the program is not acceptable, else None."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    has_fn = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import)
                     else [(node.module or "").split(".")[0]])
            for n in names:
                if n not in ALLOWED_IMPORTS:
                    return f"disallowed import: {n}"
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            has_fn = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALLS:
            return f"disallowed call: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"disallowed dunder attribute: {node.attr}"
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            return f"disallowed dunder name: {node.id}"
    if not has_fn:
        return f"no function named `{fn_name}` defined"
    return None


# Prepended to every task's runner. The `_`-prefixed handles are what the scorer must use: looking numpy up
# again after the candidate has run is exactly the hole this closes.
RUNNER_PREAMBLE = r'''
import json, sys, math, builtins, resource, numpy as np
_asarray, _nonzero, _argmax, _argmin, _isfinite, _all, _full, _mean, _arange, _allclose, _zeros = (
    np.asarray, np.nonzero, np.argmax, np.argmin, np.isfinite, np.all, np.full, np.mean, np.arange,
    np.allclose, np.zeros)
_ceil, _sum, _len, _int, _float, _enumerate = math.ceil, sum, len, int, float, enumerate
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
except (ValueError, OSError):
    pass
SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs min max sum len range enumerate zip map filter sorted reversed list dict set tuple float int bool str "
    "round pow divmod any all isinstance ValueError TypeError ZeroDivisionError IndexError KeyError Exception "
    "True False None print").split()}
_ALLOWED = {"numpy", "math", "itertools", "functools", "collections", "typing"}
_real_import = builtins.__import__
def _guarded_import(name, *a, **k):
    if name.split(".")[0] not in _ALLOWED:
        raise ImportError(f"import of {name!r} is not allowed")
    return _real_import(name, *a, **k)
SAFE_BUILTINS["__import__"] = _guarded_import

def load_candidate(path, fn_name="priority"):
    src = open(path).read()
    g = {"np": np, "math": math, "__name__": "candidate", "__builtins__": SAFE_BUILTINS}
    exec(compile(src, "candidate.py", "exec"), g)
    return g[fn_name]
'''
