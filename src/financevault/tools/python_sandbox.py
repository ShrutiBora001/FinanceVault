"""The calculator tool: arithmetic on figures the agent has already retrieved.

**Scope of the guarantee, stated plainly.** This is process isolation, not a security
sandbox. It runs untrusted code in a separate interpreter with imports denied, network and
filesystem access removed, memory and CPU capped, and a wall-clock timeout. That is enough
for the threat model here — a language model doing arithmetic, which may produce wrong or
runaway code but is not an adversary with an exploit chain. It is *not* enough to run code
from an actual attacker; a determined escape from CPython is not hard.

If this ever accepts code from untrusted users rather than from the model, it must move into
a container with seccomp, or out of process entirely. That is noted for MVP2.4, where the
service gains real users.

The design keeps the blast radius small on purpose: no imports at all. Financial arithmetic
needs `+ - * / **`, `round`, `sum`, `min`, `max`, `abs`. Anything needing numpy is a sign the
work belongs in the SQL tool.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap

from pydantic import BaseModel, Field

from financevault.tools.registry import REGISTRY, Tool, ToolContext, ToolResult

TIMEOUT_S = 5
MAX_MEMORY_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_CHARS = 4000

# Node types that have no place in arithmetic and are rejected before execution, so the
# agent gets a clear message rather than a confusing NameError from the stripped builtins.
BANNED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
)
BANNED_NAMES = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "dir",
    "exit",
    "quit",
    "breakpoint",
    "memoryview",
    "object",
    "super",
    "type",
    "help",
}

PRELUDE = """
import sys, json, resource

# Limits are applied best-effort and individually. macOS does not honour RLIMIT_AS and
# raises "current limit exceeds maximum limit", and a hard failure there would take out the
# limits that *do* apply on this platform. The subprocess timeout is the backstop that holds
# everywhere, so a skipped rlimit degrades the guarantee rather than removing it.
for _name, _limit in (
    ("RLIMIT_AS", ({mem}, {mem})),
    ("RLIMIT_CPU", ({cpu}, {cpu})),
    ("RLIMIT_NOFILE", (16, 16)),
    ("RLIMIT_FSIZE", (0, 0)),
):
    try:
        resource.setrlimit(getattr(resource, _name), _limit)
    except (ValueError, OSError, AttributeError):
        pass

ALLOWED = {{
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len,
    "sorted": sorted, "range": range, "enumerate": enumerate, "zip": zip,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "print": print, "pow": pow, "divmod": divmod, "any": any, "all": all,
    "True": True, "False": False, "None": None,
}}

env = {{"__builtins__": ALLOWED}}
code = json.loads(sys.stdin.read())

try:
    exec(compile(code, "<agent>", "exec"), env)
except Exception as exc:
    print(f"__ERROR__{{type(exc).__name__}}: {{exc}}", file=sys.stderr)
    sys.exit(1)

result = env.get("result")
if result is not None:
    print(f"__RESULT__{{json.dumps(result, default=str)}}")
"""


class PythonInput(BaseModel):
    code: str = Field(
        description=(
            "Python arithmetic. No imports are available. Assign the final value to a "
            "variable named `result`. Example: "
            "`revenue = 416161000000; prior = 391035000000; "
            "result = round((revenue - prior) / prior * 100, 2)`"
        )
    )


def static_check(code: str) -> str | None:
    """Reject code that cannot be arithmetic, before spawning an interpreter."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"

    for node in ast.walk(tree):
        if isinstance(node, BANNED_NODES):
            return (
                f"{type(node).__name__} is not available; this tool runs plain arithmetic "
                "with no imports"
            )
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"{node.id!r} is not available"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute access is not available"
    return None


def python_handler(args: PythonInput, ctx: ToolContext) -> ToolResult:
    reason = static_check(args.code)
    if reason:
        return ToolResult(ok=False, error=reason)

    prelude = PRELUDE.format(mem=MAX_MEMORY_BYTES, cpu=TIMEOUT_S)
    try:
        proc = subprocess.run(
            # -I: isolated mode. Ignores PYTHON* env vars and keeps cwd off sys.path, so the
            # child cannot import anything from this project or the user's environment.
            [sys.executable, "-I", "-c", textwrap.dedent(prelude)],
            input=json.dumps(args.code),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env={"PATH": "", "HOME": "/nonexistent"},
            cwd="/",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error=f"execution exceeded {TIMEOUT_S}s")

    if proc.returncode != 0:
        error = proc.stderr.strip()
        marker = error.rfind("__ERROR__")
        return ToolResult(
            ok=False, error=error[marker + len("__ERROR__") :] if marker >= 0 else error[:500]
        )

    stdout, result = [], None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result = json.loads(line[len("__RESULT__") :])
        else:
            stdout.append(line)

    if result is None:
        return ToolResult(
            ok=False,
            error="code produced no `result`; assign the final value to a variable named `result`",
        )

    return ToolResult(
        ok=True, data={"result": result, "stdout": "\n".join(stdout)[:MAX_OUTPUT_CHARS]}
    )


REGISTRY.register(
    Tool(
        name="python",
        description=(
            "Compute a derived figure from numbers already retrieved -- growth rates, "
            "margins, ratios, differences. No imports. Assign the answer to `result`."
        ),
        input_model=PythonInput,
        handler=python_handler,
    )
)
