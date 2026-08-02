"""The terminal tool.

Citations are a required, structured argument rather than a request in the prompt. That is
what makes `s2` (citation support) and `D2` (unsupported-claim rate) computable at all: a
claim can only be traced to a source if the source was named in a machine-readable way.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from financevault.tools.registry import REGISTRY, Tool, ToolContext, ToolResult


class Citation(BaseModel):
    kind: str = Field(description="'fact' for an XBRL figure, 'passage' for filing text.")
    ref: str = Field(
        description=(
            "For a fact: the US-GAAP tag, e.g. 'NetIncomeLoss'. "
            "For a passage: the chunk_id returned by retrieve_filings."
        )
    )
    accession: str | None = Field(default=None, description="Source filing accession number.")


class FinishInput(BaseModel):
    answer: str = Field(description="The final answer. State figures with their units.")
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "One entry per figure or claim in the answer. An answer stating a number with "
            "no citation will be scored as unsupported."
        ),
    )
    value: float | None = Field(
        default=None,
        description=(
            "If the answer is a single number, its numeric value in base units -- "
            "416161000000, not '416.16 billion'. Used for exact-match scoring."
        ),
    )
    unit: str | None = Field(default=None, description="Unit of `value`, e.g. 'USD', 'percent'.")


def finish_handler(args: FinishInput, ctx: ToolContext) -> ToolResult:
    return ToolResult(
        ok=True,
        data={
            "answer": args.answer,
            "citations": [c.model_dump() for c in args.citations],
            "value": args.value,
            "unit": args.unit,
        },
    )


REGISTRY.register(
    Tool(
        name="finish",
        description=(
            "Give the final answer and stop. Cite every figure. If the answer is a single "
            "number, also provide it in `value` with its `unit`, in base units."
        ),
        input_model=FinishInput,
        handler=finish_handler,
        terminal=True,
    )
)
