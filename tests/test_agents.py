import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from src.agents import drift_agent
from src.agents.alert_agent import detect_hazards, llm_messages
from src.agents.graph import run_sequential
from src.agents.llm import LLMClient
from src.agents.state import AgentContext
from src.models.train import run_training
from src.monitoring.logs import RunLog


def _fc(**over):
    base = {"run_ts": 0, "horizon": np.arange(1, 25), "target_ts": 3600 * np.arange(1, 25),
            "aqi": 60.0, "temp_c": 25.0, "precip_mm": 0.0, "rain_prob_pct": 5.0, "humidity_pct": 50.0,
            "uvi": 2.0, "wind_speed": 3.0, "wind_dir_deg": 180.0}
    base.update(over)
    return pd.DataFrame(base)


def test_hazard_rules_are_deterministic():
    aqi = np.full(24, 80.0)
    aqi[5:8] = 210
    alerts = detect_hazards(_fc(aqi=aqi, uvi=np.r_[np.zeros(12), np.full(12, 9.0)]))
    by = {a["metric"]: a for a in alerts}
    assert by["aqi"]["level"] == "danger" and by["aqi"]["n_hours"] == 3 and by["aqi"]["peak_value"] == 210
    assert by["uvi"]["level"] == "warning"
    assert "temp_c" not in by and detect_hazards(_fc()) == []


class StubLLM:
    def __init__(self, text):
        self.text = text

    def chat(self, *a, **k):
        return self.text


def test_llm_alert_text_is_validated():
    alerts = detect_hazards(_fc(aqi=np.full(24, 160.0)))
    good = {"0": {a: "Air is unhealthy tonight, limit outdoor time." for a in alerts[0]["audiences"]}}
    assert llm_messages(StubLLM(json.dumps(good)), alerts) == good
    assert llm_messages(StubLLM("Sorry, I cannot help"), alerts) is None        # -> template fallback
    assert llm_messages(StubLLM(json.dumps({"0": {"public": "x"}})), alerts) is None


def test_llm_failure_is_logged_separately():
    log = RunLog("t")
    assert LLMClient(log).chat("alert_agent", "sys", "user") is None   # no key in tests
    assert len(log.rows["llm_calls"]) == 1 and len(log.rows["llm_failures"]) == 1


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    props = np.histogram(ref, edges)[0] / len(ref)
    assert drift_agent.psi(edges, props, rng.normal(0, 1, 500)) < 0.1
    assert drift_agent.psi(edges, props, rng.normal(2, 1, 500)) > 0.25


def test_full_agent_run_offline(filled_store):
    """Forecast -> Alert -> (Report) -> Drift -> persist, with the LLM unavailable (template fallback)."""
    run_training(filled_store, families={"classical"})
    log = RunLog("agents")
    ctx = AgentContext(store=filled_store, run_log=log, llm=LLMClient(log))
    st = run_sequential(ctx)
    assert st["forecast_status"] == "success" and len(st["forecast"]) == 24
    s = filled_store
    assert len(s.read("forecasts")) == 24
    assert len(s.read("forecast_ledger")) == 24 * 8           # 8 output variables x 24 horizons
    runs = s.read("agent_runs")
    assert {"forecast_agent", "alert_agent", "drift_agent"} <= set(runs["agent"])
    assert len(s.read("drift_log")) > 0
    if st.get("report"):
        assert st["report"]["llm_used"] == 0 and len(st["report"]["report_text"]) > 50


def test_model_drift_and_retrain_cooldown(filled_store, monkeypatch):
    run_training(filled_store, families={"classical"})
    log = RunLog("agents")
    ctx = AgentContext(store=filled_store, run_log=log, llm=LLMClient(log))
    run_sequential(ctx)
    # fake a badly wrong ledger for the past 48h -> model drift must fire
    now = int(pd.Timestamp.now(tz="UTC").floor("h").timestamp())
    rows = [{"ledger_id": f"bad_{t}", "ts": t - 3600, "target_ts": t, "metric": "temp_c", "metric_group": "T",
             "horizon": 1.0, "predicted": 99.0, "model_name": "x", "model_version": "1", "status": "success"}
            for t in range(now - 47 * 3600, now + 1, 3600)]
    filled_store.insert("forecast_ledger", pd.DataFrame(rows))
    checks = drift_agent.model_drift(filled_store, ctx.bundle, ctx.frame, now)
    temp = [c for c in checks if c["metric"] == "temp_c"][0]
    assert temp["drifted"] and temp["score"] > 1.5

    st = {"drift": {"drifted": True, "event_id": "e1", "checks": [dict(temp)]},
          "ledger": {"last_retrain_trigger_ts": now - 3600}}        # triggered 1h ago -> cooldown
    out = drift_agent.retrain_node(ctx, st)
    assert out["retrain"]["action"] == "cooldown_skip"
    st["ledger"]["last_retrain_trigger_ts"] = 0
    out = drift_agent.retrain_node(ctx, st)
    assert out["retrain"]["action"] == "dispatch_failed"           # no GitHub token offline - logged, not crashed


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_langgraph_graph_runs(filled_store):
    from src.agents.graph import build_graph

    run_training(filled_store, families={"classical"})
    log = RunLog("agents")
    ctx = AgentContext(store=filled_store, run_log=log, llm=LLMClient(log))
    final = build_graph(ctx).invoke({"errors": []})
    assert final["forecast_status"] == "success" and len(final["forecast"]) == 24
