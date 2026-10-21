"""The HTTP API and the demo page.

Run it with `make serve`, or in the container image. This is the single-machine version of
section 4 of `distributed_plan.md`: the same routes, with runs executing in this process's
thread pool instead of on separate workers.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from financevault import obs
from financevault.config import settings
from financevault.store import pg

from . import service
from .streams import STREAMS

STATIC = Path(__file__).with_name("static")
FIGURES = Path("report/figures")

app = FastAPI(
    title="FinanceVault",
    version="0.2.0",
    description="Point-in-time-correct financial tool use, with every step verified.",
)


obs.setup_tracing(app)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def _timed(request: Request, call_next):
    started = time.monotonic()
    response = await call_next(request)
    # The route template, not the raw path: one series per endpoint, not per run id.
    route = getattr(request.scope.get("route"), "path", "unmatched")
    obs.HTTP_SECONDS.labels(request.method, route, response.status_code).observe(
        time.monotonic() - started
    )
    return response


@app.exception_handler(service.Rejected)
async def _rejected(_: Request, exc: service.Rejected) -> JSONResponse:
    headers = {"Retry-After": "10"} if exc.status in (429, 503) else None
    return JSONResponse({"detail": exc.message}, status_code=exc.status, headers=headers)


# ---------------------------------------------------------------- health


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/readyz")
def readyz() -> JSONResponse:
    try:
        row = pg.fetch_one("SELECT count(*) AS n FROM journal")
    except Exception as exc:  # noqa: BLE001 - any database failure means not ready
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)
    return JSONResponse({"ok": True, "journal_entries": row["n"] if row else 0})


# ---------------------------------------------------------------- configuration


def _live_authorised(authorization: str | None) -> bool:
    key = settings().api_key
    if not key or not authorization or not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization.removeprefix("Bearer ").strip(), key)


@app.get("/v1/config")
def config() -> dict:
    cfg = settings()
    return {
        "live_available": bool(cfg.api_key and cfg.anthropic_api_key),
        "live_remaining_usd": round(service.SPEND.remaining(), 4),
        "max_usd": cfg.max_usd,
        "policies": {p: service.policy_model(p) or cfg.analyst_model for p in service.POLICIES},
        "mutations": service.mutations(),
    }


@app.get("/v1/catalog")
def catalog() -> dict:
    return {"questions": service.catalog()}


@app.get("/v1/results")
def results() -> dict:
    return service.results()


# ---------------------------------------------------------------- questions and runs


class QuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # Required: there is no implicit "now", the same rule as the CLI.
    as_of: str = Field(min_length=10, max_length=40)
    ticker: str | None = Field(default=None, max_length=10)
    cik: str | None = Field(default=None, max_length=10)
    policy: Literal["b2-tools", "b3-frontier"] = "b2-tools"
    mode: Literal["replay", "live"] = "replay"
    max_usd: float = Field(default=0.25, gt=0, le=1.0)


@app.post("/v1/questions", status_code=202)
def ask(body: QuestionIn, authorization: str | None = Header(default=None)) -> dict:
    live = body.mode == "live"
    if live and not _live_authorised(authorization):
        raise HTTPException(401, "live mode needs a valid API key (Authorization: Bearer ...)")
    run_id = service.submit(
        service.Request(
            question=body.question.strip(),
            as_of=body.as_of,
            ticker=body.ticker,
            cik=body.cik,
            policy=body.policy,
            live=live,
            max_usd=min(body.max_usd, settings().max_usd),
        )
    )
    return {
        "run_id": run_id,
        "links": {
            "events": f"/v1/runs/{run_id}/events",
            "run": f"/v1/runs/{run_id}",
            "steps": f"/v1/runs/{run_id}/steps",
        },
    }


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = service.stored_run(_uuid(run_id))
    if run is None:
        stream = STREAMS.get(run_id)
        if stream is not None and not stream.closed:
            return {"id": run_id, "outcome": "running"}
        raise HTTPException(404, "no such run")
    return run


@app.get("/v1/runs/{run_id}/steps")
def get_steps(run_id: str) -> dict:
    return {"steps": service.stored_steps(_uuid(run_id))}


@app.get("/v1/runs/{run_id}/events")
async def events(run_id: str, request: Request) -> EventSourceResponse:
    stream = STREAMS.get(_uuid(run_id))
    if stream is None:
        raise HTTPException(404, "no live stream for this run; read /v1/runs/{id}/steps instead")

    async def gen():
        cursor = 0
        while True:
            if await request.is_disconnected():
                return
            batch, closed = await asyncio.to_thread(stream.wait, cursor, 15.0)
            for kind, payload in batch:
                cursor += 1
                yield {"event": kind, "data": json.dumps(payload, default=str)}
            if closed and cursor >= len(stream.events):
                return

    return EventSourceResponse(gen(), ping=10)


class SeedIn(BaseModel):
    kind: str = Field(max_length=40)
    cik: str | None = Field(default=None, max_length=10)


@app.post("/v1/runs/{run_id}/seed-error")
def seed_error(run_id: str, body: SeedIn) -> dict:
    return service.seed_error(_uuid(run_id), body.kind, body.cik)


def _uuid(value: str) -> str:
    import uuid  # noqa: PLC0415

    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise HTTPException(404, "no such run") from exc


# ---------------------------------------------------------------- the page

if FIGURES.is_dir():
    app.mount("/figures", StaticFiles(directory=FIGURES), name="figures")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
