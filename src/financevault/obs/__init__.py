"""Metrics and traces.

Metrics are Prometheus counters and histograms, served at `/metrics` by the API. Traces are
OpenTelemetry spans: HTTP request -> run -> model call / tool call, each run span carrying
`run_id`. Until `setup_tracing` installs an exporter, the spans are the API's no-op default,
so the eval harness and the tests pay nothing for them.

Label values are kept to small fixed sets (policy, mode, outcome, model, tool name). A
question or run id as a label would create one time series per run.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from prometheus_client import Counter, Histogram

tracer = trace.get_tracer("financevault")

HTTP_SECONDS = Histogram(
    "fv_http_request_duration_seconds", "HTTP request latency", ["method", "route", "status"]
)
RUNS = Counter("fv_runs_total", "API runs finished", ["policy", "mode", "outcome"])
# Runs are bounded at 300 s; the buckets cover replay (milliseconds) through live (minutes).
RUN_SECONDS = Histogram(
    "fv_run_duration_seconds",
    "API run wall time, execute + verify",
    ["policy", "mode"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
MODEL_CALLS = Counter("fv_model_calls_total", "Model calls", ["model", "cached"])
MODEL_USD = Counter("fv_model_spend_usd_total", "Dollars actually spent on model calls", ["model"])
TOOL_CALLS = Counter("fv_tool_calls_total", "Tool calls", ["tool", "ok"])
TOOL_SECONDS = Histogram("fv_tool_duration_seconds", "Tool call latency", ["tool"])


def setup_tracing(app) -> bool:
    """Export spans over OTLP when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Returns whether it did.

    The exporter reads the endpoint and `OTEL_SERVICE_NAME` from the environment itself.
    """
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
    return True
