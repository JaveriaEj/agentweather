"""Agent 2 - Alert Agent.

Hazard DETECTION is 100% deterministic (config/thresholds.yaml).
The LLM is only used to phrase the already-decided alerts for three
audiences; its output is validated, and fixed templates are used if the
LLM fails or returns something unusable. The LLM can never add, remove or
change an alert, a level or a threshold.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.processing.features import heat_index_c
from src.utils.config import settings, thresholds
from src.utils.logger import get_logger

log = get_logger(__name__)
AUDIENCE_NAMES = {"public": "General public", "sensitive": "Sensitive groups", "outdoor": "Outdoor workers"}
UNITS = {"aqi": "", "heat_index_c": "°C", "temp_c": "°C", "precip_mm": "mm/h", "uvi": "",
         "wind_speed": "m/s", "humidity_pct": "%"}


def _local(ts: int) -> str:
    tz = settings()["city"]["timezone"]
    return pd.Timestamp(int(ts), unit="s", tz="UTC").tz_convert(tz).strftime("%a %H:%M")


def detect_hazards(fc: pd.DataFrame) -> list[dict]:
    """Apply the rules. Returns one alert per metric (its highest firing level)."""
    cfg = thresholds()
    order = {lvl: i for i, lvl in enumerate(cfg["levels_order"])}
    fc = fc.copy()
    fc["heat_index_c"] = heat_index_c(fc["temp_c"], fc["humidity_pct"])
    alerts = []
    for metric, rules in cfg["rules"].items():
        if metric not in fc:
            continue
        vals = fc[metric].astype(float)
        best = None
        for rule in rules:
            if "min" in rule:
                hit = vals >= rule["min"]
                thr, peak = rule["min"], vals.max()
            else:
                hit = vals <= rule["max"]
                thr, peak = rule["max"], vals.min()
            if hit.any() and (best is None or order[rule["level"]] > order[best[0]["level"]]):
                best = (rule, hit, thr, peak)
        if best:
            rule, hit, thr, peak = best
            hours = fc.loc[hit, "target_ts"].astype(int)
            alerts.append({
                "metric": metric, "level": rule["level"], "label": rule["label"],
                "threshold": float(thr), "peak_value": round(float(peak), 1),
                "comparison": ">=" if "min" in rule else "<=",
                "unit": UNITS.get(metric, ""), "audiences": list(rule["audiences"]),
                "start_ts": int(hours.min()), "end_ts": int(hours.max()), "n_hours": int(hit.sum()),
                "time_window": f"{_local(hours.min())} - {_local(hours.max())}"})
    return alerts


def template_message(a: dict, audience: str) -> str:
    adv = thresholds()["advice"][audience]
    return (f"{a['label'].upper()} ({a['level']}): {a['metric'].replace('_', ' ')} expected to reach "
            f"{a['peak_value']}{a['unit']} (threshold {a['comparison']} {a['threshold']}{a['unit']}) "
            f"during {a['time_window']} ({a['n_hours']} h). {adv}")


def llm_messages(llm, alerts: list[dict]) -> dict | None:
    """Ask the LLM to phrase the alerts. Returns {index: {audience: text}} or None."""
    facts = [{k: a[k] for k in ("metric", "level", "label", "peak_value", "unit", "threshold",
                                "comparison", "time_window", "audiences")} for a in alerts]
    system = ("You write short public-safety weather alerts. You MUST NOT change any level, "
              "number, threshold, time window or add new hazards. Only rephrase the facts given. "
              "Answer with JSON only.")
    user = ("City: " + settings()["city"]["name"] + "\nAlerts (deterministic facts):\n"
            + json.dumps(facts, indent=1)
            + "\n\nFor each alert index, write one clear message (max 45 words) for each audience "
              "listed in its 'audiences' field (public=general public, sensitive=sensitive groups, "
              "outdoor=outdoor workers), with practical advice. Return JSON like "
              '{"0": {"public": "...", "sensitive": "..."}}')
    text = llm.chat("alert_agent", system, user, max_tokens=900)
    if not text:
        return None
    try:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else None
    except (json.JSONDecodeError, AttributeError):
        data = None
    if not isinstance(data, dict):
        return None
    # validation: every alert/audience must have a usable message, otherwise use templates
    for i, a in enumerate(alerts):
        block = data.get(str(i))
        if not isinstance(block, dict):
            return None
        for aud in a["audiences"]:
            msg = block.get(aud)
            if not isinstance(msg, str) or len(msg) < 10:
                return None
    return data


def alert_node(ctx, state: dict) -> dict:
    with ctx.run_log.timed_agent("alert_agent") as details:
        if not state.get("forecast"):
            details.update(_status="skipped", reason="no forecast")
            return {"alerts": [], "new_significant_alerts": []}
        fc = pd.DataFrame(state["forecast"])
        alerts = detect_hazards(fc)
        cfg = thresholds()
        recent = set(state.get("ledger", {}).get("recent_alert_keys", []))
        msgs = llm_messages(ctx.llm, alerts) if alerts else None
        rows, new_sig = [], []
        run_ts = int(state["run_ts"])
        for i, a in enumerate(alerts):
            key = f"{a['metric']}:{a['level']}"
            a["is_new"] = key not in recent
            a["significant"] = a["level"] in cfg["significant_levels"]
            if a["is_new"] and a["significant"]:
                new_sig.append(a)
            for aud in a["audiences"]:
                llm_text = (msgs or {}).get(str(i), {}).get(aud) if msgs else None
                rows.append({
                    "alert_id": f"{run_ts}_{a['metric']}_{aud}", "ts": run_ts, "metric": a["metric"],
                    "level": a["level"], "label": a["label"], "audience": AUDIENCE_NAMES[aud],
                    "peak_value": a["peak_value"], "threshold": a["threshold"], "time_window": a["time_window"],
                    "start_ts": a["start_ts"], "end_ts": a["end_ts"], "n_hours": float(a["n_hours"]),
                    "message": llm_text or template_message(a, aud),
                    "llm_used": int(bool(llm_text)), "is_new": int(a["is_new"]),
                    "significant": int(a["significant"]),
                    "rule": f"{a['metric']} {a['comparison']} {a['threshold']}"})
        ctx.run_log.add_rows("alerts", rows)
        details.update(n_alerts=len(alerts), n_new_significant=len(new_sig),
                       llm_used=bool(msgs), metrics=[a["metric"] for a in alerts])
        return {"alerts": alerts, "new_significant_alerts": new_sig}
