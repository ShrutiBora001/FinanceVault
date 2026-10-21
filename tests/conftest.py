"""Suite-wide pytest hooks.

Tests that need the corpus skip when it is missing, which is the right behaviour on a fresh
checkout. In CI it is the wrong one: the tests that skip are exactly the ones guarding the
point-in-time guarantee and replay determinism, and a run where they all skipped would report
green while checking none of them. `FV_FORBID_SKIPS=1` turns every skip into a failure, so the
integration job cannot pass by not running.
"""

from __future__ import annotations

import os

import pytest

FORBID_SKIPS = os.environ.get("FV_FORBID_SKIPS", "").lower() in ("1", "true", "yes")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> object:
    outcome = yield
    report = outcome.get_result()
    if FORBID_SKIPS and report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
        report.outcome = "failed"
        report.longrepr = f"skipped under FV_FORBID_SKIPS: {reason}"
