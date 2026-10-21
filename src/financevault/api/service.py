"""What the API does, kept apart from HTTP so it can be tested and scripted directly.

A question runs in a worker thread: route, execute, persist, verify. Each stage publishes
into the run's stream as it happens. Three rules shape it:

**Replay is the default, and it is free.** A replayed question resolves every model call
from the journal, so it costs $0.00 and cannot spend money even by mistake -- a journal miss
is an error, never a fallback to a live call. Live mode needs the API key.

**API runs are stored under their own policy name.** `api:b2-tools`, not `b2-tools`. The
journal key does not include the policy, so replay is unaffected, but the benchmark queries
select on the bare policy name: B2 calibration samples the most recent `b2-tools` runs
(`docs/open_issues.md` #2), and demo traffic must not become its sample.

**Verification uses the harness's own budget,** so the judged signals of a replayed run hit
the journal entries the eval sweep wrote.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from financevault import verify
from financevault.config import settings
from financevault.env.asof import AsOf
from financevault.runtime import executor, llm
from financevault.runtime.budget import Ledger
from financevault.runtime.journal import ReplayMiss
from financevault.store import pg
from financevault.verify import citation, numeric, tool_validity
from financevault.verify.signals import NAMES, Signal

from .streams import STREAMS, Stream

CATALOG_PATH = Path(__file__).with_name("catalog.json")

# The policies the API offers, and the model each runs on. Mirrors `eval/baselines.py`
# exactly -- a different model argument would change the journal key and miss on replay.
POLICIES = ("b2-tools", "b3-frontier")


def policy_model(policy: str) -> str | None:
    return settings().frontier_model if policy == "b3-frontier" else None


class Rejected(Exception):
    """A request refused before it cost anything. Carries the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- catalog


def catalog() -> list[dict]:
    """Questions known to replay, written by `scripts/demo_catalog.py`. Empty if not built."""
    if not CATALOG_PATH.exists():
        return []
    return json.loads(CATALOG_PATH.read_text())["questions"]


def catalog_entry(question: str, as_of: str, policy: str) -> dict | None:
    """The catalog entry this request matches, if any -- the source of its gold answer."""
    day = as_of[:10]
    for q in catalog():
        if q["question"] == question and q["as_of"][:10] == day and policy in q["policies"]:
            return q
    return None


# ---------------------------------------------------------------- spend


class DailySpend:
    """Live list-dollars spent today by this process. Resets at UTC midnight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = datetime.now(UTC).date()
        self._usd = 0.0

    def _roll(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._day:
            self._day, self._usd = today, 0.0

    def remaining(self) -> float:
        with self._lock:
            self._roll()
            return max(0.0, settings().api_daily_usd - self._usd)

    def add(self, usd: float) -> None:
        with self._lock:
            self._roll()
            self._usd += usd


SPEND = DailySpend()


# ---------------------------------------------------------------- running a question


@dataclass(slots=True)
class Request:
    question: str
    as_of: str
    ticker: str | None
    cik: str | None
    policy: str
    live: bool
    max_usd: float
    expected_value: float | None = None


_pool: ThreadPoolExecutor | None = None
_slots: threading.BoundedSemaphore | None = None
_init_lock = threading.Lock()


def _workers() -> tuple[ThreadPoolExecutor, threading.BoundedSemaphore]:
    global _pool, _slots
    with _init_lock:
        if _pool is None:
            n = settings().api_max_concurrent
            _pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="fv-run")
            _slots = threading.BoundedSemaphore(n)
    assert _slots is not None
    return _pool, _slots


def submit(req: Request) -> str:
    """Admit a question and start it. Returns the run id; raises `Rejected` if refused."""
    if req.policy not in POLICIES:
        raise Rejected(400, f"unknown policy {req.policy!r}; choose one of {', '.join(POLICIES)}")
    try:
        AsOf.parse(req.as_of)
    except ValueError as exc:
        raise Rejected(400, f"as_of: {exc}") from exc

    if req.live:
        if not settings().anthropic_api_key:
            raise Rejected(503, "live mode is unavailable: no model API key is configured")
        if SPEND.remaining() < req.max_usd:
            raise Rejected(429, "today's live budget is spent; replayed questions still work")

    if req.expected_value is None:
        entry = catalog_entry(req.question, req.as_of, req.policy)
        if entry is not None:
            req.expected_value = entry["expected_value"]
            req.cik = req.cik or entry.get("cik")

    pool, slots = _workers()
    if not slots.acquire(blocking=False):
        raise Rejected(429, "all run slots are busy; try again in a few seconds")

    run_id = str(uuid.uuid4())
    stream = STREAMS.create(run_id)

    def task() -> None:
        try:
            _run(run_id, req, stream)
        finally:
            slots.release()

    pool.submit(task)
    return run_id


def _signal_json(sig: Signal) -> dict:
    return {
        "score": round(sig.score, 4),
        "passed": sig.passed,
        "applicable": sig.applicable,
        "reason": sig.reason,
    }


def _verdicts_json(verdicts: list[verify.StepVerdict]) -> list[dict]:
    return [
        {
            "idx": v.idx,
            "tool": v.tool,
            "passed": v.passed,
            "signals": {k: _signal_json(s) for k, s in v.signals.items()},
        }
        for v in verdicts
    ]


def _run(run_id: str, req: Request, stream: Stream) -> None:
    mode = "live" if req.live else "replay"
    stream.publish(
        "accepted",
        {
            "run_id": run_id,
            "mode": mode,
            "policy": req.policy,
            "question": req.question,
            "as_of": req.as_of,
            "ticker": req.ticker,
            "has_gold": req.expected_value is not None,
        },
    )
    spent = 0.0
    try:
        with llm.replay_mode(not req.live):
            run = executor.execute(
                req.question,
                as_of=AsOf.parse(req.as_of),
                cik=req.cik,
                ticker=req.ticker,
                policy=f"api:{req.policy}",
                model=policy_model(req.policy),
                # Replay keeps the sweep's own ceiling: a different budget can end the run at
                # a different step, which is a different trajectory and a journal miss.
                max_usd=req.max_usd if req.live else None,
                run_id=run_id,
                on_event=stream.publish,
            )
            spent += run.ledger.list_usd if run.ledger and req.live else 0.0

            verdicts: list[verify.StepVerdict] = []
            verify_error = None
            vledger = Ledger(max_usd=0.05, max_steps=40)
            try:
                verdicts = verify.verify_run(
                    run_id, cik=req.cik, expected_value=req.expected_value, ledger=vledger
                )
            except ReplayMiss:
                verify_error = (
                    "the judged signals for this trajectory are not in the journal, so "
                    "verification needs live mode"
                )
            spent += vledger.list_usd if req.live else 0.0
            stream.publish(
                "verified",
                {"steps": _verdicts_json(verdicts), "names": NAMES, "error": verify_error},
            )

        correct = verify.answered_correctly(verdicts) if req.expected_value is not None else None
        stream.publish(
            "done",
            {
                **run.summary(),
                "citations": run.citations,
                "correct": correct,
                "expected_value": req.expected_value,
                "steps_passed": sum(1 for v in verdicts if v.passed),
                "mode": mode,
            },
        )
    except ReplayMiss:
        stream.publish(
            "error",
            {
                "kind": "replay_miss",
                "message": (
                    "This question is not in the journal, so it cannot be replayed for free. "
                    "Pick one from the catalog, or run it live with an API key."
                ),
            },
        )
    except Exception as exc:  # noqa: BLE001 - every failure must reach the stream
        traceback.print_exc()
        stream.publish("error", {"kind": type(exc).__name__, "message": str(exc)})
    finally:
        if spent:
            SPEND.add(spent)


# ---------------------------------------------------------------- reading stored runs


def stored_run(run_id: str) -> dict | None:
    row = pg.fetch_one(
        """
        SELECT id, question, as_of, policy, path, outcome, answer, citations, n_steps,
               cost_usd, latency_ms, created_at
        FROM runs WHERE id = %s
        """,
        (run_id,),
    )
    return _jsonable(row) if row else None


def stored_steps(run_id: str) -> list[dict]:
    rows = pg.fetch_all(
        """
        SELECT idx, thought, tool, args, obs, tokens_in, tokens_out, cost_usd, latency_ms,
               s1, s2, s3, s4, s5, s_reasons
        FROM steps WHERE run_id = %s ORDER BY idx
        """,
        (run_id,),
    )
    return [_jsonable(r) for r in rows]


def _jsonable(row: dict) -> dict:
    return json.loads(json.dumps(row, default=str))


# ---------------------------------------------------------------- seeded errors


def mutations() -> dict[str, str]:
    """Error classes the demo can inject, with the signal expected to catch each."""
    from bench.verifier_calib import MUTATIONS  # noqa: PLC0415 - bench is not in the wheel

    return {name: signal for name, (_, signal) in MUTATIONS.items()}


def _programmatic(
    answer: str, value: float | None, citations: list[dict], steps: list[dict], cik: str | None
) -> dict[str, dict]:
    """The three signals that need no judge, with reasons. Same calls as B2's `score_case`,
    but with the run's own company rather than a fixed one."""
    calls = [s for s in steps if s.get("tool")]
    s1 = min(
        (tool_validity.score_step(s) for s in calls),
        key=lambda sig: sig.score,
        default=Signal(1.0, "no tool calls"),
    )
    s2 = citation.score(answer, citations, steps, ledger=None, evidence_text="")  # type: ignore[arg-type]
    s3 = numeric.score(
        answer, evidence=numeric.collect_evidence(steps), cik=cik, stated_value=value
    )
    return {"s1": _signal_json(s1), "s2": _signal_json(s2), "s3": _signal_json(s3)}


def seed_error(run_id: str, kind: str, cik: str | None) -> dict:
    """Inject one known error into a stored trajectory and score it before and after.

    Nothing is written: the mutated trajectory exists only for this response. Only the
    programmatic signals are scored, so this never calls a model and costs nothing.
    """
    from bench.verifier_calib import MUTATIONS, Case  # noqa: PLC0415

    if kind not in MUTATIONS:
        raise Rejected(400, f"unknown error class {kind!r}")
    run = pg.fetch_one("SELECT id, question, answer FROM runs WHERE id = %s", (run_id,))
    if run is None:
        raise Rejected(404, "no such run")
    steps = pg.fetch_all(
        "SELECT idx, tool, args, obs FROM steps WHERE run_id = %s ORDER BY idx", (run_id,)
    )
    terminal = next((s for s in reversed(steps) if s["tool"] == "finish"), None)
    if terminal is None or not run["answer"]:
        raise Rejected(409, "this run has no final answer to corrupt")
    args: dict[str, Any] = terminal.get("args") or {}
    case = Case(
        run_id=run_id,
        question=run["question"],
        answer=run["answer"],
        steps=steps,
        citations=args.get("citations") or [],
        value=args.get("value"),
    )
    mutate, signal = MUTATIONS[kind]
    bad = mutate(case)
    if bad is None:
        raise Rejected(409, f"this trajectory cannot carry a {kind} error")

    before = _programmatic(case.answer, case.value, case.citations, case.steps, cik)
    after = _programmatic(bad.answer, bad.value, bad.citations, bad.steps, cik)
    return {
        "kind": kind,
        "expected_signal": signal,
        "caught": not after[signal]["passed"],
        "original": {"answer": case.answer, "value": case.value, "signals": before},
        "mutated": {"answer": bad.answer, "value": bad.value, "signals": after},
    }


# ---------------------------------------------------------------- benchmark results

RESULTS = Path("eval/results")


def results() -> dict:
    """The committed benchmark numbers the report is built from."""
    out: dict[str, Any] = {}
    for name in ("mvp2.2_dev", "b2_seeded", "a1_leak"):
        path = RESULTS / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out
