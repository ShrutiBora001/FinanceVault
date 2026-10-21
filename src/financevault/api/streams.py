"""Per-run event buffers, so a browser can watch a run that executes in another thread.

A run publishes `routed`, `step`, `verified`, `done` and `error` events into its stream.
A subscriber first receives everything already published, then waits for the rest, so a page
that connects late -- or reconnects -- still sees the whole trajectory in order.

The buffers live in this process's memory. That is the deliberate limit of the single-machine
deployment: one API process, runs executing in its thread pool. `distributed_plan.md` replaces
this with Redis pub/sub when runs move to separate workers. A stream for a run this process
did not execute does not exist here; the API falls back to the stored rows for those.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

TERMINAL = ("done", "error")

# Finished streams kept for late subscribers. Older ones are dropped; their runs are still
# readable from Postgres.
KEEP = 256


@dataclass
class Stream:
    run_id: str
    events: list[tuple[str, dict]] = field(default_factory=list)
    closed: bool = False
    created: float = field(default_factory=time.monotonic)
    _cond: threading.Condition = field(default_factory=threading.Condition)

    def publish(self, kind: str, payload: dict) -> None:
        with self._cond:
            if self.closed:
                return
            self.events.append((kind, payload))
            if kind in TERMINAL:
                self.closed = True
            self._cond.notify_all()

    def wait(self, cursor: int, timeout: float) -> tuple[list[tuple[str, dict]], bool]:
        """Events after `cursor`, blocking up to `timeout` for at least one. Returns
        `(events, closed)`."""
        with self._cond:
            if cursor >= len(self.events) and not self.closed:
                self._cond.wait(timeout)
            return self.events[cursor:], self.closed


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: OrderedDict[str, Stream] = OrderedDict()

    def create(self, run_id: str) -> Stream:
        with self._lock:
            stream = Stream(run_id)
            self._streams[run_id] = stream
            while len(self._streams) > KEEP:
                oldest_id, oldest = next(iter(self._streams.items()))
                if not oldest.closed:
                    break
                del self._streams[oldest_id]
            return stream

    def get(self, run_id: str) -> Stream | None:
        with self._lock:
            return self._streams.get(run_id)

    def active(self) -> int:
        with self._lock:
            return sum(1 for s in self._streams.values() if not s.closed)


STREAMS = Registry()
