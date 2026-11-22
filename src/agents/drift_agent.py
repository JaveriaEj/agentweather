"""Agent 4 - Drift Detection Agent.

Data drift : Population Stability Index (PSI) of each monitored feature,
             last 7 days vs the training distribution stored in the model
             bundle (+ a KS test p-value for information).
Model drift: RMSE of recent forecasts (from the Forecast Ledger) against
             what was actually observed, divided by the validation RMSE.
If any check crosses its threshold -> drift event -> (graph) trigger
retraining node, which respects a cooldown to avoid retraining loops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.retrain_trigger import dispatch_training, notify_team
from src.monitoring.logs import new_id
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.timeutil import floor_hour, to_ts, utcnow

log = get_logger(__name__)
HOUR = 3600
LEDGER_TO_ACTUAL = {"aqi": "aqi", "temp_c": "temp_c", "precip_mm": "precip_mm",
                    "humidity_pct": "humidity_pct", "uvi": "uvi", "wind_speed": "wind_speed"}


def psi(ref_edges, ref_props, live: np.ndarray, eps: float = 1e-4) -> float:
    edges = np.asarray(ref_edges, float).copy()
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(live, bins=edges)
    a = np.clip(counts / max(counts.sum(), 1), eps, None)
    e = np.clip(np.asarray(ref_props, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_pvalue(sample, live) -> float:
    try:
        from scipy.stats import ks_2samp

        return float(ks_2samp(sample, live).pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def data_drift(bundle, frame: pd.DataFrame) -> list[dict]:
    cfg = settings()["drift"]
    ref = bundle.meta.get("reference_stats", {})
    recent = frame.tail(cfg["psi_window_hours"])
    out = []
    for feat, st in ref.items():
        if feat not in recent:
            continue
        live = recent[feat].dropna().to_numpy(float)
        if len(live) < cfg["min_live_rows"]:
            continue
        score = psi(st["edges"], st["props"], live)
        out.append({"metric": feat, "drift_type": "data", "score": round(score, 4),
                    "threshold": cfg["psi_threshold"], "drifted": score > cfg["psi_threshold"],
                    "details": f"PSI; KS p={ks_pvalue(st['sample'], live):.4f}; live mean={live.mean():.2f} "
                               f"vs train {st['mean']:.2f}; n={len(live)}"})
    return out


def model_drift(store, bundle, frame: pd.DataFrame, now_ts: int) -> list[dict]:
    cfg = settings()["drift"]
    win = cfg["model_drift_window_hours"] * HOUR
    led = store.read("forecast_ledger", now_ts - win - 24 * HOUR, now_ts)
    if led.empty:
        return []
    led = led[(led["target_ts"].astype("int64") <= now_ts) & (led["target_ts"].astype("int64") > now_ts - win)]
    actual = frame[["ts"] + [c for c in LEDGER_TO_ACTUAL.values() if c in frame]].rename(columns={"ts": "target_ts"})
    comps = bundle.meta["components"]
    ref_rmse = {m: comps[m]["val_rmse"] for m in LEDGER_TO_ACTUAL if m in comps}
    if "wind_u" in comps and "wind_v" in comps:
        ref_rmse["wind_speed"] = float(np.hypot(comps["wind_u"]["val_rmse"], comps["wind_v"]["val_rmse"]))
    out = []
    for metric, col in LEDGER_TO_ACTUAL.items():
        sub = led[led["metric"] == metric][["target_ts", "predicted"]].copy()
        if sub.empty or col not in actual or metric not in ref_rmse:
            continue
        sub["target_ts"] = sub["target_ts"].astype("int64")
        j = sub.merge(actual[["target_ts", col]], on="target_ts", how="inner").dropna()
        if len(j) < cfg["min_error_samples"]:
            continue
        rmse = float(np.sqrt(((j["predicted"] - j[col]) ** 2).mean()))
        ratio = rmse / max(ref_rmse[metric], 1e-6)
        out.append({"metric": metric, "drift_type": "model", "score": round(ratio, 4),
                    "threshold": cfg["model_error_ratio_threshold"],
                    "drifted": ratio > cfg["model_error_ratio_threshold"],
                    "details": f"live RMSE={rmse:.3f} vs validation RMSE={ref_rmse[metric]:.3f}; n={len(j)}"})
    return out


def drift_node(ctx, state: dict) -> dict:
    with ctx.run_log.timed_agent("drift_agent") as details:
        if ctx.bundle is None or ctx.frame is None:
            details.update(_status="skipped", reason="no model or data")
            return {"drift": {"drifted": False, "checks": []}}
        now_ts = to_ts(floor_hour(utcnow()))
        checks = data_drift(ctx.bundle, ctx.frame) + model_drift(ctx.store, ctx.bundle, ctx.frame, now_ts)
        drifted = [c for c in checks if c["drifted"]]
        event_id = new_id()
        for c in checks:
            c.update({"event_id": f"{event_id}_{c['drift_type']}_{c['metric']}", "ts": now_ts,
                      "action": "drift_detected" if c["drifted"] else "none",
                      "retrain_status": "pending" if c["drifted"] else "not_required",
                      "model_version": str(ctx.bundle.version)})
        details.update(n_checks=len(checks), n_drifted=len(drifted),
                       drifted=[f"{c['drift_type']}:{c['metric']}" for c in drifted])
        return {"drift": {"drifted": bool(drifted), "checks": checks, "event_id": event_id}}


def retrain_node(ctx, state: dict) -> dict:
    """Called only when drift was detected. Applies the cooldown, dispatches training, notifies."""
    with ctx.run_log.timed_agent("retrain_trigger") as details:
        cfg = settings()["drift"]
        drift = state["drift"]
        now_ts = to_ts(floor_hour(utcnow()))
        last = state.get("ledger", {}).get("last_retrain_trigger_ts", 0)
        drifted = [c for c in drift["checks"] if c["drifted"]]
        names = ", ".join(f"{c['drift_type']}:{c['metric']}({c['score']})" for c in drifted)
        if now_ts - last < cfg["retrain_cooldown_hours"] * HOUR:
            action, status = "cooldown_skip", f"last trigger {(now_ts - last) / HOUR:.1f}h ago"
        else:
            ok, msg = dispatch_training("drift", drift["event_id"])
            action = "retraining_triggered" if ok else "dispatch_failed"
            status = msg
            notify_team(f"Drift detected: {names[:120]}",
                        f"Drift event `{drift['event_id']}` at {utcnow()}.\n\nChecks over threshold:\n"
                        + "\n".join(f"- {c['drift_type']} / {c['metric']}: score {c['score']} > "
                                    f"{c['threshold']} ({c['details']})" for c in drifted)
                        + f"\n\nRetraining dispatch: {action} ({msg})")
        for c in drift["checks"]:
            if c["drifted"]:
                c["action"], c["retrain_status"] = action, status[:500]
        # one summary row that load_context uses for the cooldown
        drift["checks"].append({"event_id": f"{drift['event_id']}_trigger", "ts": now_ts, "metric": "all",
                                "drift_type": "trigger", "score": float(len(drifted)), "threshold": 1.0,
                                "drifted": True, "action": action, "retrain_status": status[:500],
                                "details": names[:1000]})
        details.update(action=action, status=status)
        return {"retrain": {"action": action, "status": status}, "drift": drift}
