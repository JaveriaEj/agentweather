"""Forecast Ledger: the persistent memory shared by all agents across runs.

* `ledger_rows` turns a forecast into one ledger row per metric x horizon.
* `load_context` reads what previous runs did (recent alerts, last report,
  last retraining trigger) so agents can deduplicate and respect cooldowns.
"""
from __future__ import annotations

import pandas as pd

from src.models.fuse_outputs import METRIC_GROUPS
from src.utils.config import settings, thresholds

HOUR = 3600
METRIC_TO_VAR = {"aqi": "aqi", "temp_c": "temp_c", "precip_mm": "precip_mm", "rain_prob_pct": "rain_prob",
                 "humidity_pct": "humidity_pct", "uvi": "uvi", "wind_speed": "wind_u", "wind_dir_deg": "wind_u"}


def ledger_rows(fc: pd.DataFrame, model_version, components: dict, status: str) -> list[dict]:
    rows = []
    for group, metrics in METRIC_GROUPS.items():
        for metric in metrics:
            model_name = components.get(METRIC_TO_VAR[metric], "unknown")
            for _, r in fc.iterrows():
                rows.append({
                    "ledger_id": f"{int(r['run_ts'])}_{metric}_{int(r['horizon']):02d}",
                    "ts": int(r["run_ts"]), "target_ts": int(r["target_ts"]),
                    "metric_group": group, "metric": metric, "horizon": float(r["horizon"]),
                    "predicted": float(r[metric]), "model_name": model_name,
                    "model_version": str(model_version), "status": status})
    return rows


def load_context(store, now_ts: int) -> dict:
    dedup = thresholds()["dedup_hours"] * HOUR
    cooldown = settings()["drift"]["retrain_cooldown_hours"] * HOUR
    alerts = store.read("alerts", now_ts - dedup, now_ts)
    reports = store.read("weather_reports", now_ts - 48 * HOUR, now_ts)
    drift = store.read("drift_log", now_ts - max(cooldown, 48 * HOUR), now_ts)
    runs = store.read("forecast_ledger", now_ts - 2 * HOUR, now_ts)

    recent_alert_keys = sorted({f"{r.metric}:{r.level}" for r in alerts.itertuples()}) if len(alerts) else []
    last_report_ts = int(reports["ts"].max()) if len(reports) else 0
    trig = drift[drift["action"] == "retraining_triggered"] if len(drift) else drift
    last_retrain_ts = int(trig["ts"].max()) if len(trig) else 0
    return {"recent_alert_keys": recent_alert_keys, "last_report_ts": last_report_ts,
            "last_retrain_trigger_ts": last_retrain_ts,
            "previous_forecast_runs": int(runs["ts"].nunique()) if len(runs) else 0}
