"""Feature engineering + target creation.

LEAKAGE RULE: every feature at hour t is computed from data at hours <= t.
`shift(k)` with k>0 looks back; rolling windows end at t; nothing uses
shift(-k) except the TARGETS, which live in separate columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import horizon, settings, target_vars

CALENDAR_COLS = ["hour", "dayofweek", "month", "dayofyear",
                 "hour_sin", "hour_cos", "doy_sin", "doy_cos", "solar_elev_deg"]


def heat_index_c(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """NOAA Rothfusz heat index (falls back to temperature below ~27°C)."""
    t = temp_c * 9 / 5 + 32
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
          - 6.83783e-3 * t ** 2 - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t ** 2 * rh
          + 8.5282e-4 * t * rh ** 2 - 1.99e-6 * t ** 2 * rh ** 2)
    hi = np.where(t < 80, t, hi)
    return pd.Series((hi - 32) * 5 / 9, index=temp_c.index)


def feature_columns() -> list[str]:
    f = settings()["features"]
    cols = list(CALENDAR_COLS) + ["heat_index_c", "rain_flag"]
    for v in f["base_vars"]:
        cols.append(v)
        cols += [f"{v}_lag{k}" for k in f["lags"]]
        cols += [f"{v}_roll{w}" for w in f["rolling_windows"]]
        cols += [f"{v}_diff{k}" for k in f["diffs"]]
    return cols


def build_features(fused: pd.DataFrame) -> pd.DataFrame:
    """Input: contiguous hourly frame from fuse_sources. Output: ts + features."""
    cfg = settings()
    f = cfg["features"]
    df = fused.sort_values("ts").reset_index(drop=True).copy()
    local = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(cfg["city"]["timezone"])

    out = pd.DataFrame({"ts": df["ts"].astype("int64")})
    out["event_time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    out["hour"] = local.dt.hour.astype("float64")
    out["dayofweek"] = local.dt.dayofweek.astype("float64")
    out["month"] = local.dt.month.astype("float64")
    out["dayofyear"] = local.dt.dayofyear.astype("float64")
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["solar_elev_deg"] = df["solar_elev_deg"].astype("float64")
    out["heat_index_c"] = heat_index_c(df["temp_c"], df["humidity_pct"])
    rain_thr = cfg["forecast"]["rain_threshold_mm"]
    out["rain_flag"] = (df["precip_mm"] > rain_thr).astype("float64").where(df["precip_mm"].notna())

    new_cols = {}
    for v in f["base_vars"]:
        s = df[v].astype("float64")
        new_cols[v] = s
        for k in f["lags"]:
            new_cols[f"{v}_lag{k}"] = s.shift(k)
        for w in f["rolling_windows"]:
            new_cols[f"{v}_roll{w}"] = s.rolling(w, min_periods=w).mean()
        for k in f["diffs"]:
            new_cols[f"{v}_diff{k}"] = s - s.shift(k)
    out = pd.concat([out, pd.DataFrame(new_cols)], axis=1)

    # carry useful non-feature context columns
    for c in ["wind_speed", "wind_deg", "aqi_aqicn", "aqi_source", "n_imputed", "n_missing"]:
        if c in df and c not in out:
            out[c] = df[c].values
    return out


def target_columns() -> list[str]:
    return [f"{v}_h{h:02d}" for v in target_vars() for h in range(1, horizon() + 1)]


def build_targets(fused: pd.DataFrame) -> pd.DataFrame:
    """Targets: value of each variable h hours AFTER t (h = 1..24).

    The newest 24 rows have incomplete targets -> they are forecast-only
    and are dropped from training by `complete_target_mask`.
    """
    df = fused.sort_values("ts").reset_index(drop=True)
    cols = {"ts": df["ts"].astype("int64"),
            "event_time": pd.to_datetime(df["ts"], unit="s", utc=True)}
    for v in target_vars():
        s = df[v].astype("float64")
        for h in range(1, horizon() + 1):
            cols[f"{v}_h{h:02d}"] = s.shift(-h)
    return pd.DataFrame(cols)


def complete_target_mask(targets: pd.DataFrame) -> pd.Series:
    return targets[target_columns()].notna().all(axis=1)
