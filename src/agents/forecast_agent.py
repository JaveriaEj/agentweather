"""Agent 1 - Forecast Agent.

Reads the latest features from the Feature Store, loads the production
model bundle from the Model Registry, forecasts all six metric groups for
the next 24 hours, saves the forecast and records it in the Forecast Ledger.
"""
from __future__ import annotations

import numpy as np

from src.models.bundle import ModelBundle
from src.models.dataset import contiguous
from src.monitoring.ledger import ledger_rows
from src.utils.config import settings, target_vars
from src.utils.logger import get_logger
from src.utils.timeutil import floor_hour, to_ts, utcnow

log = get_logger(__name__)
HOUR = 3600
LOOKBACK_HOURS = 400   # enough for lags, LSTM window (48h) and SARIMAX filter (336h)
MAX_DATA_AGE_HOURS = 2


def load_bundle(store):
    latest = store.latest_model(settings()["registry"]["model_name"])
    if not latest:
        return None
    return ModelBundle.load(latest[0], version=latest[1])


def forecast_node(ctx, state: dict) -> dict:
    run_log = ctx.run_log
    with run_log.timed_agent("forecast_agent") as details:
        now_ts = to_ts(floor_hour(utcnow()))
        feats = ctx.store.read("hourly_features", now_ts - LOOKBACK_HOURS * HOUR, now_ts)
        if feats.empty:
            details.update(_status="failed", reason="no features in store")
            return {"forecast_status": "no_data", "forecast": [], "errors": ["no features"]}
        frame = contiguous(feats)
        ctx.frame = frame

        ok = frame[target_vars()].notna().all(axis=1).to_numpy()
        if not ok.any():
            details.update(_status="failed", reason="no complete feature row")
            return {"forecast_status": "no_data", "forecast": [], "errors": ["no complete row"]}
        pos = int(np.flatnonzero(ok)[-1])
        origin_ts = int(frame.loc[pos, "ts"])
        age_h = (now_ts - origin_ts) / HOUR
        status = "success" if age_h <= MAX_DATA_AGE_HOURS else "stale_data"

        if ctx.bundle is None:
            ctx.bundle = load_bundle(ctx.store)
        if ctx.bundle is None:
            details.update(_status="failed", reason="no registered model - run training first")
            return {"forecast_status": "no_model", "forecast": [], "errors": ["no model"]}
        bundle = ctx.bundle

        fc = bundle.forecast(frame, pos)
        comps = {v: c["model"] for v, c in bundle.meta["components"].items()}
        fc["model_version"] = str(bundle.version)
        fc["status"] = status
        fc["forecast_id"] = [f"{origin_ts}_{h:02d}" for h in fc["horizon"]]
        fc["ts"] = origin_ts
        fc["models"] = "; ".join(f"{k}={v}" for k, v in comps.items())

        run_log.add_rows("forecasts", fc.to_dict("records"))
        run_log.add_rows("forecast_ledger", ledger_rows(fc, bundle.version, comps, status))

        last = frame.loc[pos]
        latest_obs = {c: (None if np.isnan(last.get(c, np.nan)) else float(last[c]))
                      for c in ["temp_c", "humidity_pct", "precip_mm", "uvi", "wind_speed", "wind_deg",
                                "aqi", "aqi_aqicn", "pm2_5", "pm10", "pressure_hpa"]
                      if c in frame.columns and isinstance(last.get(c), (int, float, np.floating))}
        latest_obs["obs_ts"] = origin_ts
        recent = frame.loc[max(0, pos - 23):pos, ["ts"] + [c for c in ["temp_c", "aqi", "precip_mm",
                                                                       "humidity_pct", "uvi", "wind_speed"]
                                                           if c in frame]]
        details.update(model_version=bundle.version, origin_ts=origin_ts, data_age_h=age_h,
                       status=status, components=comps)
        if status != "success":
            details["_status"] = status
        log.info("Forecast generated from origin %s with model v%s (%s)", origin_ts, bundle.version, status)
        return {"run_ts": origin_ts, "forecast": fc.to_dict("records"), "forecast_status": status,
                "model_version": bundle.version, "model_components": comps,
                "latest_obs": latest_obs, "recent_obs": recent.to_dict("records")}
