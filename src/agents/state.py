"""Shared LangGraph state + the run context object.

`ForecastState` is what flows between the four agent nodes in one run.
The Forecast Ledger (in the Feature Store) is what persists BETWEEN runs:
`load_context` reads it into `state["ledger"]` at the start of every run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class ForecastState(TypedDict, total=False):
    run_id: str
    run_ts: int                    # origin hour (epoch s, UTC)
    ledger: dict[str, Any]         # context loaded from the persistent ledger/logs
    # Forecast Agent
    forecast: list[dict]           # 24 rows, one per hour ahead
    forecast_status: str           # success | stale_data | no_model | no_data | failed
    model_version: int | None
    model_components: dict[str, str]
    latest_obs: dict[str, Any]
    recent_obs: list[dict]         # last 24h observed (for "changes" in the report)
    # Alert Agent
    alerts: list[dict]
    new_significant_alerts: list[dict]
    # Weather Report Agent
    report_reason: str             # daily | significant_alert | none
    report: dict | None
    # Drift Agent
    drift: dict[str, Any]
    retrain: dict[str, Any]
    errors: list[str]


@dataclass
class AgentContext:
    """Non-serialisable dependencies shared by all nodes (kept out of the state)."""

    store: Any
    run_log: Any
    llm: Any
    bundle: Any = None
    frame: Any = None              # hourly features used this run
    extra: dict = field(default_factory=dict)
