"""The LangGraph workflow: four agents as nodes in ONE graph.

START -> load_context -> forecast_agent -> alert_agent
      -> [report needed?] -> weather_report_agent -> drift_agent
                          \\-------------------------> drift_agent
      -> [drift?] -> trigger_retraining -> persist_logs -> END
                 \\------------------------> persist_logs -> END

Shared state between nodes: `ForecastState`. Shared state between RUNS:
the Forecast Ledger and logs in the Feature Store (read by load_context,
written by persist_logs).
"""
from __future__ import annotations

from functools import partial

from src.agents.alert_agent import alert_node
from src.agents.drift_agent import drift_node, retrain_node
from src.agents.forecast_agent import forecast_node
from src.agents.report_agent import report_decision, report_node
from src.agents.state import AgentContext, ForecastState
from src.monitoring.ledger import load_context
from src.utils.timeutil import floor_hour, to_ts, utcnow


def load_context_node(ctx: AgentContext, state: dict) -> dict:
    now_ts = to_ts(floor_hour(utcnow()))
    return {"ledger": load_context(ctx.store, now_ts), "run_id": ctx.run_log.run_id,
            "run_ts": now_ts, "errors": []}


def persist_node(ctx: AgentContext, state: dict) -> dict:
    checks = (state.get("drift") or {}).get("checks", [])
    if checks:
        ctx.run_log.add_rows("drift_log", [{**c, "drifted": int(bool(c["drifted"]))} for c in checks])
    ctx.run_log.system("run", "agents", state.get("forecast_status", "unknown"), "graph finished",
                       {"alerts": len(state.get("alerts", [])), "report": state.get("report_reason"),
                        "drift": bool((state.get("drift") or {}).get("drifted")),
                        "retrain": state.get("retrain")})
    written = ctx.run_log.flush(ctx.store)
    return {"errors": state.get("errors", []) + ([] if written else ["nothing written"])}


def route_after_alert(state: dict) -> str:
    return "report" if report_decision(state) != "none" else "skip"


def route_after_drift(state: dict) -> str:
    return "retrain" if (state.get("drift") or {}).get("drifted") else "done"


NODES = [("load_context", load_context_node), ("forecast_agent", forecast_node),
         ("alert_agent", alert_node), ("weather_report_agent", report_node),
         ("drift_agent", drift_node), ("trigger_retraining", retrain_node),
         ("persist_logs", persist_node)]


def build_graph(ctx: AgentContext):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(ForecastState)
    for name, fn in NODES:
        g.add_node(name, partial(fn, ctx))
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "forecast_agent")
    g.add_edge("forecast_agent", "alert_agent")
    g.add_conditional_edges("alert_agent", route_after_alert,
                            {"report": "weather_report_agent", "skip": "drift_agent"})
    g.add_edge("weather_report_agent", "drift_agent")
    g.add_conditional_edges("drift_agent", route_after_drift,
                            {"retrain": "trigger_retraining", "done": "persist_logs"})
    g.add_edge("trigger_retraining", "persist_logs")
    g.add_edge("persist_logs", END)
    return g.compile()


def run_sequential(ctx: AgentContext) -> dict:
    """Same routing without LangGraph - used ONLY by the offline unit tests."""
    state: dict = {}
    fns = dict(NODES)
    for name in ["load_context", "forecast_agent", "alert_agent"]:
        state.update(fns[name](ctx, state))
    if route_after_alert(state) == "report":
        state.update(fns["weather_report_agent"](ctx, state))
    state.update(fns["drift_agent"](ctx, state))
    if route_after_drift(state) == "retrain":
        state.update(fns["trigger_retraining"](ctx, state))
    state.update(fns["persist_logs"](ctx, state))
    return state
