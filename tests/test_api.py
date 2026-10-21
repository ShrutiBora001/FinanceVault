"""The HTTP API: admission, the event stream, and the replay switch.

The agent and verifier are replaced with stand-ins here, so these tests need no database and
no model. What they check is the API's own contract: who may spend money, what a stream
contains and in what order, and that replay can be forced per request.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from financevault.api import app as app_module
from financevault.api import service
from financevault.api.streams import Registry, Stream
from financevault.config import settings
from financevault.runtime import llm
from financevault.runtime.budget import Ledger
from financevault.runtime.journal import ReplayMiss
from financevault.verify import StepVerdict
from financevault.verify.signals import Signal, na

QUESTION = {
    "question": "What was AAPL's net income in fiscal year 2025?",
    "as_of": "2025-11-01",
    "ticker": "AAPL",
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


@pytest.fixture
def fake_agent(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace execute and verify_run; record what they were called with."""
    seen: dict = {}

    def execute(question: str, **kw: object) -> SimpleNamespace:
        seen["execute"] = kw
        seen["replaying"] = llm.replaying()
        on_event = kw["on_event"]
        on_event("routed", {"path": "P2", "tools": ["lookup_fact", "finish"], "reason": "P2"})
        on_event("step", {"idx": 0, "tool": "lookup_fact", "args": {}, "observation": {"ok": True}})
        on_event("step", {"idx": 1, "tool": "finish", "args": {}, "observation": {"ok": True}})
        ledger = Ledger(max_usd=0.25, max_steps=12)
        return SimpleNamespace(
            ledger=ledger,
            citations=[],
            summary=lambda: {"run_id": kw["run_id"], "outcome": "ok", "answer": "$112.01B"},
        )

    def verify_run(run_id: str, **kw: object) -> list[StepVerdict]:
        seen["verify"] = kw
        ok = Signal(1.0, "fine")
        return [
            StepVerdict(
                0, "lookup_fact", {"s1": ok, "s2": na("x"), "s3": na("x"), "s4": ok, "s5": na("x")}
            ),
            StepVerdict(1, "finish", {"s1": ok, "s2": ok, "s3": ok, "s4": ok, "s5": ok}),
        ]

    monkeypatch.setattr(service.executor, "execute", execute)
    monkeypatch.setattr(service.verify, "verify_run", verify_run)
    return seen


def events(client: TestClient, run_id: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    kind = None
    with client.stream("GET", f"/v1/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event:"):
                kind = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and kind:
                out.append((kind, json.loads(line.split(":", 1)[1])))
    return out


# ---------------------------------------------------------------- streams


def test_stream_delivers_everything_to_a_late_subscriber() -> None:
    stream = Stream("r")
    stream.publish("step", {"idx": 0})
    stream.publish("done", {})
    batch, closed = stream.wait(0, timeout=0.01)
    assert [k for k, _ in batch] == ["step", "done"]
    assert closed


def test_stream_ignores_events_after_it_closes() -> None:
    stream = Stream("r")
    stream.publish("error", {"message": "x"})
    stream.publish("step", {"idx": 9})
    assert [k for k, _ in stream.events] == ["error"]


def test_stream_wakes_a_waiting_subscriber() -> None:
    stream = Stream("r")
    threading.Timer(0.05, stream.publish, ("step", {"idx": 0})).start()
    batch, closed = stream.wait(0, timeout=5)
    assert batch == [("step", {"idx": 0})]
    assert not closed


def test_registry_keeps_open_streams_when_trimming() -> None:
    reg = Registry()
    first = reg.create("open")
    for i in range(300):
        reg.create(f"r{i}").publish("done", {})
    assert reg.get("open") is first


# ---------------------------------------------------------------- replay switch


def test_replay_mode_overrides_the_setting_and_restores_it() -> None:
    default = settings().replay
    with llm.replay_mode(not default):
        assert llm.replaying() is (not default)
    assert llm.replaying() is default


# ---------------------------------------------------------------- admission


def test_live_requires_the_api_key(client: TestClient) -> None:
    resp = client.post("/v1/questions", json={**QUESTION, "mode": "live"})
    assert resp.status_code == 401


def test_live_rejects_a_wrong_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings(), "api_key", "right")
    resp = client.post(
        "/v1/questions",
        json={**QUESTION, "mode": "live"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_live_refused_once_the_daily_budget_is_spent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings(), "api_key", "k")
    monkeypatch.setattr(settings(), "anthropic_api_key", "sk-test")
    monkeypatch.setattr(service.SPEND, "remaining", lambda: 0.0)
    resp = client.post(
        "/v1/questions", json={**QUESTION, "mode": "live"}, headers={"Authorization": "Bearer k"}
    )
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_unknown_policy_is_rejected(client: TestClient) -> None:
    resp = client.post("/v1/questions", json={**QUESTION, "policy": "b9-magic"})
    assert resp.status_code == 422


def test_naive_garbage_as_of_is_rejected(client: TestClient) -> None:
    resp = client.post("/v1/questions", json={**QUESTION, "as_of": "not-a-date"})
    assert resp.status_code == 400


# ---------------------------------------------------------------- a run, end to end


def test_replayed_run_streams_in_order(client: TestClient, fake_agent: dict) -> None:
    resp = client.post("/v1/questions", json=QUESTION)
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    kinds = [k for k, _ in events(client, run_id)]
    assert kinds == ["accepted", "routed", "step", "step", "verified", "done"]

    assert fake_agent["replaying"] is True
    # Stored under its own policy name so demo traffic never enters a benchmark sample.
    assert fake_agent["execute"]["policy"] == "api:b2-tools"
    # Replay keeps the sweep's budget; a different ceiling would be a different trajectory.
    assert fake_agent["execute"]["max_usd"] is None


def test_done_event_reports_verdicts(client: TestClient, fake_agent: dict) -> None:
    run_id = client.post("/v1/questions", json=QUESTION).json()["run_id"]
    got = dict(events(client, run_id))
    assert got["verified"]["steps"][1]["signals"]["s3"]["passed"] is True
    assert got["done"]["steps_passed"] == 2


def test_frontier_policy_uses_the_frontier_model(client: TestClient, fake_agent: dict) -> None:
    run_id = client.post("/v1/questions", json={**QUESTION, "policy": "b3-frontier"}).json()[
        "run_id"
    ]
    events(client, run_id)
    assert fake_agent["execute"]["model"] == settings().frontier_model


def test_journal_miss_ends_the_stream_with_an_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def execute(question: str, **kw: object) -> None:
        raise ReplayMiss("no entry")

    monkeypatch.setattr(service.executor, "execute", execute)
    run_id = client.post("/v1/questions", json=QUESTION).json()["run_id"]
    got = events(client, run_id)
    assert got[-1][0] == "error"
    assert got[-1][1]["kind"] == "replay_miss"


def test_events_for_an_unknown_run_are_404(client: TestClient) -> None:
    assert client.get("/v1/runs/00000000-0000-0000-0000-000000000000/events").status_code == 404
    assert client.get("/v1/runs/not-a-uuid/events").status_code == 404


def test_page_and_config_are_served(client: TestClient) -> None:
    assert "FinanceVault" in client.get("/").text
    cfg = client.get("/v1/config").json()
    assert set(cfg["policies"]) == {"b2-tools", "b3-frontier"}
    assert cfg["mutations"]["off_by_1000"] == "s3"
