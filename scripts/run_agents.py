"""SCRIPT 3 - forecasting + LangGraph agents (alerts, report, drift, retraining trigger).

Run:  python -m scripts.run_agents
Schedule: GitHub Actions, right after each successful feature-pipeline run.
"""
from __future__ import annotations

import json
import sys

from src.agents.graph import build_graph
from src.agents.llm import LLMClient
from src.agents.state import AgentContext
from src.monitoring.logs import RunLog
from src.store import get_store
from src.utils.logger import get_logger

log = get_logger("agents")


def main() -> int:
    store = get_store()
    run_log = RunLog("agents")
    ctx = AgentContext(store=store, run_log=run_log, llm=LLMClient(run_log))
    try:
        final = build_graph(ctx).invoke({"errors": []})
    except Exception as exc:  # noqa: BLE001
        log.exception("Agent graph crashed")
        run_log.system("run", "agents", "failed", f"{type(exc).__name__}: {exc}")
        run_log.flush(store)
        return 1
    log.info("Run summary: %s", json.dumps({
        "forecast_status": final.get("forecast_status"), "model_version": final.get("model_version"),
        "alerts": [f"{a['metric']}:{a['level']}" for a in final.get("alerts", [])],
        "report": final.get("report_reason"), "drift": (final.get("drift") or {}).get("drifted"),
        "retrain": final.get("retrain")}, default=str))
    return 0 if final.get("forecast_status") in ("success", "stale_data") else 1


if __name__ == "__main__":
    sys.exit(main())
