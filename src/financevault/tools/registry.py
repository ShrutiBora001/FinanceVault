"""Tool contracts and dispatch.

Each tool is declared once, as a Pydantic input model plus a handler. That single
declaration drives three consumers:

1. the tool specification sent to the model,
2. the runtime dispatcher that validates and executes a call,
3. the `s1` tool-validity signal, which scores whether a step's call was well-formed.

Keeping them from one source is what makes `s1` meaningful. If the verifier had its own idea
of what a valid call looks like, it would be scoring agreement between two hand-written
schemas rather than scoring the agent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from financevault.env.asof import AsOf
from financevault.runtime.budget import Ledger


@dataclass(slots=True)
class ToolContext:
    """Everything a tool needs that the model does not supply.

    `as_of` is here rather than in the tool arguments deliberately: the horizon is a property
    of the run, not a choice the agent makes. An agent that could set its own `as_of` could
    grant itself lookahead.
    """

    as_of: AsOf
    ledger: Ledger
    cik: str | None = None
    ticker: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None

    def to_json(self) -> dict:
        return {"ok": self.ok, "data": self.data, "error": self.error}


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], ToolResult]
    terminal: bool = False

    def spec(self) -> dict:
        """Anthropic tool-use specification, generated from the Pydantic model."""
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, only: list[str] | None = None) -> list[dict]:
        """Specs for the model. `only` restricts the toolset to an execution path's subset."""
        chosen = self._tools.values() if only is None else [self._tools[n] for n in only]
        return [t.spec() for t in chosen]

    def validate(self, name: str, args: dict) -> tuple[BaseModel | None, str | None]:
        """Parse arguments without executing. This is exactly what `s1` scores."""
        tool = self._tools.get(name)
        if tool is None:
            return None, f"unknown tool {name!r}; available: {', '.join(self.names())}"
        try:
            return tool.input_model.model_validate(args), None
        except ValidationError as exc:
            return None, "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in exc.errors()
            )

    def dispatch(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        """Validate then execute. Handler exceptions become results, not crashes.

        A tool that raises is a normal event in an agent loop -- a malformed query, an empty
        result -- and the agent should get the chance to recover from it on the next step.
        """
        parsed, error = self.validate(name, args)
        if error is not None:
            return ToolResult(ok=False, error=error)
        assert parsed is not None
        try:
            return self._tools[name].handler(parsed, ctx)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent as an observation
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


REGISTRY = Registry()
