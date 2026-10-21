"""Metrics: the endpoint, a run's outcome, and live vs cached model calls. No database, no model."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from financevault.config import settings
from financevault.runtime import journal, llm
from financevault.runtime.budget import Ledger
from tests.test_api import QUESTION, client, events, fake_agent  # noqa: F401 - fixtures


def _value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_serves_prometheus_text(client: TestClient) -> None:  # noqa: F811
    client.get("/healthz")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert 'fv_http_request_duration_seconds_count{method="GET",route="/healthz"' in resp.text


def test_a_finished_run_counts_its_outcome(client: TestClient, fake_agent: dict) -> None:  # noqa: F811
    labels = {"policy": "b2-tools", "mode": "replay", "outcome": "ok"}
    before = _value("fv_runs_total", **labels)
    run_id = client.post("/v1/questions", json={**QUESTION, "policy": "b2-tools"}).json()["run_id"]
    events(client, run_id)  # returns once the run's stream closes
    # The counter is incremented just after the stream closes, on the worker thread.
    for _ in range(100):
        if _value("fv_runs_total", **labels) == before + 1:
            break
        time.sleep(0.01)
    assert _value("fv_runs_total", **labels) == before + 1


def test_a_journal_hit_counts_as_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    model = settings().judge_model
    entry = journal.Entry("h", model, {}, {"content": []}, 1, 1, 0.0)
    monkeypatch.setattr(journal, "get", lambda _hash: entry)
    before = _value("fv_model_calls_total", model=model, cached="true")
    spent = _value("fv_model_spend_usd_total", model=model)
    llm.call(model, [{"role": "user", "content": "x"}], ledger=Ledger(1.0, 5))
    assert _value("fv_model_calls_total", model=model, cached="true") == before + 1
    assert _value("fv_model_spend_usd_total", model=model) == spent
