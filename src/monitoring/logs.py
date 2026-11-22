"""Run-scoped log collector.

Rows are buffered in memory during a run and written to the Feature Store
in one insert per log type at the end (`flush`). Batching matters on
Hopsworks: every insert starts a materialization job.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager

import pandas as pd

from src.utils.logger import get_logger
from src.utils.timeutil import utcnow

log = get_logger(__name__)


def _now_ts() -> int:
    return int(utcnow().timestamp())


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class RunLog:
    def __init__(self, pipeline: str, run_id: str | None = None):
        self.pipeline = pipeline
        self.run_id = run_id or f"{pipeline}-{utcnow().strftime('%Y%m%dT%H%M%S')}-{new_id()[:6]}"
        self.rows: dict[str, list[dict]] = defaultdict(list)
        self.started = time.time()

    # ------------------------------------------------------------- writers
    def api_failure(self, source, endpoint, error, attempts, params=None):
        self.rows["api_failures"].append({
            "log_id": new_id(), "ts": _now_ts(), "run_id": self.run_id, "source": source,
            "endpoint": endpoint, "params": json.dumps(params or {}), "error": str(error)[:2000],
            "attempts": float(attempts)})

    def system(self, log_type, component, status, message, payload=None):
        """Generic step log -> pipeline_runs."""
        self.rows["pipeline_runs"].append({
            "log_id": new_id(), "ts": _now_ts(), "run_id": self.run_id, "pipeline": self.pipeline,
            "log_type": log_type, "component": component, "status": status,
            "message": str(message)[:2000], "payload": json.dumps(payload or {}, default=str)[:5000],
            "duration_s": float(time.time() - self.started)})

    def agent(self, agent, status, duration_s, details):
        self.rows["agent_runs"].append({
            "log_id": new_id(), "ts": _now_ts(), "run_id": self.run_id, "agent": agent,
            "status": status, "duration_s": float(duration_s),
            "details": json.dumps(details, default=str)[:5000]})

    def llm_call(self, agent, model, prompt, response, latency_s, status, error=""):
        row = {"log_id": new_id(), "ts": _now_ts(), "run_id": self.run_id, "agent": agent,
               "model": model, "prompt": prompt[:8000], "response": (response or "")[:8000],
               "latency_s": float(latency_s), "status": status, "error": str(error)[:2000]}
        self.rows["llm_calls"].append(row)
        if status != "success":
            self.rows["llm_failures"].append(dict(row, log_id=new_id()))

    @contextmanager
    def timed_agent(self, agent: str):
        """Usage: with run_log.timed_agent('forecast_agent') as details: details[...] = ..."""
        t0, details = time.time(), {}
        try:
            yield details
            self.agent(agent, details.pop("_status", "success"), time.time() - t0, details)
        except Exception as exc:
            details["error"] = f"{type(exc).__name__}: {exc}"
            self.agent(agent, "failed", time.time() - t0, details)
            raise

    # ------------------------------------------------------------- flush
    def add_rows(self, fg_name: str, rows: list[dict]):
        self.rows[fg_name].extend(rows)

    def flush(self, store) -> dict[str, int]:
        written = {}
        for fg, rows in list(self.rows.items()):
            if not rows:
                continue
            try:
                store.insert(fg, pd.DataFrame(rows))
                written[fg] = len(rows)
            except Exception as exc:  # noqa: BLE001 - logging must never crash the pipeline
                log.error("Could not write %d rows to %s: %s", len(rows), fg, exc)
        self.rows.clear()
        try:  # backends that buffer writes (Hugging Face) persist everything here
            store.commit(f"{self.pipeline} run {self.run_id}")
        except Exception as exc:  # noqa: BLE001
            log.error("Store commit failed: %s", exc)
            raise
        return written
