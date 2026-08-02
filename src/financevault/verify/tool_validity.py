"""s1 — tool validity.

Was the call well-formed, and did it name a real tool available on this path?

Scored against the same Pydantic contract the dispatcher uses, so this measures the agent
rather than agreement between two hand-written schemas. Partial credit distinguishes the
failure modes, because they need different fixes: hallucinating a tool is a prompting
problem, malformed arguments is usually a schema-clarity problem, and a tool that ran but
errored is often the agent behaving correctly against bad data.
"""

from __future__ import annotations

from financevault.tools import REGISTRY
from financevault.verify.signals import Signal, na


def score(
    tool: str | None, args: dict, observation: dict, allowed: list[str] | None = None
) -> Signal:
    if tool is None:
        return Signal(0.0, "no tool call: the model answered in prose instead of acting")

    if tool not in REGISTRY:
        return Signal(0.0, f"hallucinated tool {tool!r}; available: {', '.join(REGISTRY.names())}")

    if allowed is not None and tool not in allowed:
        return Signal(
            0.25, f"tool {tool!r} is not available on this execution path ({', '.join(allowed)})"
        )

    _, error = REGISTRY.validate(tool, args)
    if error is not None:
        return Signal(0.25, f"arguments failed validation: {error}")

    # The call was well-formed but the tool reported a problem. That is a weaker signal than
    # success and a much stronger one than a malformed call: asking a valid question and
    # getting "no such tag" is a reasonable step in a search.
    if not observation.get("ok", False):
        return Signal(0.5, f"well-formed call, tool returned an error: {observation.get('error')}")

    return Signal(1.0, "well-formed call to an available tool, returned successfully")


def score_step(step: dict, allowed: list[str] | None = None) -> Signal:
    """Convenience wrapper over a `steps` row."""
    return score(step.get("tool"), step.get("args") or {}, step.get("obs") or {}, allowed)


__all__ = ["na", "score", "score_step"]
