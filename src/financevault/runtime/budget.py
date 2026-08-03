"""The budget ledger.

Every step debits tokens, dollars, wall-clock time and depth. A run that breaches any ceiling
raises `BudgetExceeded` and stops, rather than being asked politely in a prompt to stay cheap.

This is the mechanism behind the E-group metrics: cost per correct answer is meaningful only
when cost is measured at the step level and enforced, not estimated afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Per-million-token prices (input, output) in USD. Kept explicit rather than fetched so a
# run's cost stays reproducible from its journal months later, when list prices may have moved.
#
# List prices, deliberately. Sonnet 5 carries introductory pricing of $2/$10 through
# 2026-08-31; billing at list slightly overstates cost during that window rather than
# understating it, and the project runs past the expiry. Cost per correct answer is a
# headline metric, so it errs pessimistic.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
}
# Local models cost GPU time, not tokens; they are priced at zero here and accounted for
# separately in the efficiency benchmark.
LOCAL_PREFIXES = ("qwen", "local/", "hf/")


class BudgetExceeded(RuntimeError):
    """Raised when a run breaches a ceiling. Carries what was spent and what the limit was."""

    def __init__(self, resource: str, spent: float, limit: float) -> None:
        super().__init__(f"budget exceeded: {resource} {spent:.6g} > {limit:.6g}")
        self.resource = resource
        self.spent = spent
        self.limit = limit


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    """Cost in USD for one call. Unknown remote models raise rather than silently costing 0."""
    if any(model.lower().startswith(p) for p in LOCAL_PREFIXES):
        return 0.0
    if model not in PRICES:
        raise KeyError(
            f"no price for model {model!r}; add it to PRICES so cost accounting stays honest"
        )
    rate_in, rate_out = PRICES[model]
    return (tokens_in / 1e6) * rate_in + (tokens_out / 1e6) * rate_out


@dataclass(slots=True)
class Ledger:
    """Accumulates spend for a single run and enforces its ceilings."""

    max_usd: float
    max_steps: int
    max_seconds: float = 300.0

    # `usd` is what this run actually spent; a journal hit costs nothing and is the whole
    # point of replay. `list_usd` is what the same run would cost on a cold journal.
    #
    # Both are needed, and conflating them corrupts a metric. E1 (cost per correct answer) is
    # a property of the *policy* and must use list cost, or re-running an already-journalled
    # sweep reports every policy as free. F2 (sweep cost with replay versus without) is a
    # property of the *harness* and must use actual cost, or the replay saving disappears.
    usd: float = 0.0
    list_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    steps: int = 0
    started: float = field(default_factory=time.monotonic)

    # Per-step detail, so the report can show where a run spent its budget.
    entries: list[dict] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.usd)

    def check(self) -> None:
        """Raise if any ceiling is already breached. Called before starting a step."""
        if self.usd > self.max_usd:
            raise BudgetExceeded("usd", self.usd, self.max_usd)
        if self.steps > self.max_steps:
            raise BudgetExceeded("steps", self.steps, self.max_steps)
        if self.elapsed > self.max_seconds:
            raise BudgetExceeded("seconds", self.elapsed, self.max_seconds)

    def would_exceed(self, usd: float) -> bool:
        """Whether a call of the given cost would breach the dollar ceiling."""
        return self.usd + usd > self.max_usd

    def charge(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        *,
        latency_ms: int = 0,
        cached: bool = False,
        label: str = "",
    ) -> float:
        """Record a model call and return its cost.

        A journal hit costs nothing, which is the entire point of replay: the same
        trajectory re-runs for $0.00 and the ledger says so rather than being told so.
        """
        list_cost = price(model, tokens_in, tokens_out)
        cost = 0.0 if cached else list_cost
        self.usd += cost
        self.list_usd += list_cost
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.entries.append(
            {
                "label": label,
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "latency_ms": latency_ms,
                "cached": cached,
            }
        )
        self.check()
        return cost

    def step(self) -> None:
        """Advance the step counter, raising if it takes the run past its ceiling."""
        self.steps += 1
        self.check()

    def summary(self) -> dict:
        cached = sum(1 for e in self.entries if e["cached"])
        return {
            "usd": round(self.usd, 6),
            "list_usd": round(self.list_usd, 6),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "steps": self.steps,
            "calls": len(self.entries),
            "cached_calls": cached,
            "cache_hit_rate": round(cached / len(self.entries), 4) if self.entries else 0.0,
            "elapsed_s": round(self.elapsed, 3),
        }
