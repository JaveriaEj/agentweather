"""Fuse weather + air-quality sources into ONE row per hour, then impute.

Source-priority strategy (documented in README):
* Weather: OpenWeather One Call (primary) > OpenWeather 2.5 current (fallback, no UV).
* AQI target: US-EPA AQI computed from OpenWeather pollutant concentrations
  (primary - the SAME method is available for history and live data, so
  training and serving are consistent). If OpenWeather pollutants are missing
  for an hour, the AQICN station AQI is used (aqi_source = 'aqicn').
  AQICN's own value is always kept in `aqi_aqicn` for display/validation.

Imputation strategy (leak-free - only past values are used):
1. Forward-fill gaps of up to `ffill_limit_hours` (default 3h).
2. UV index at night (sun below horizon) is set to 0.
3. Longer gaps stay NaN; those rows are excluded from training and flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing.aqi_calc import compute_aqi_frame
from src.processing.validate import validate_frame
from src.utils.config import settings

WEATHER_COLS = ["temp_c", "humidity_pct", "pressure_hpa", "dew_point_c", "clouds_pct",
                "uvi", "wind_speed", "wind_deg", "precip_mm"]
AIR_COLS = ["pm2_5", "pm10", "o3", "no2", "so2", "co"]


def solar_elevation_deg(ts: pd.Series, lat: float, lon: float) -> pd.Series:
    """Approximate solar elevation angle (degrees). Good enough for day/night + UV."""
    t = pd.to_datetime(ts.astype("int64"), unit="s", utc=True)
    doy = t.dt.dayofyear.to_numpy()
    hour = (t.dt.hour + t.dt.minute / 60.0).to_numpy()
    decl = np.radians(23.44) * np.sin(np.radians(360.0 / 365.0 * (284 + doy)))
    solar_time = hour + lon / 15.0
    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    phi = np.radians(lat)
    sin_elev = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    return pd.Series(np.degrees(np.arcsin(np.clip(sin_elev, -1, 1))), index=ts.index)


def wind_to_uv(speed: pd.Series, deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Meteorological convention: `deg` is where the wind blows FROM."""
    rad = np.radians(deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def uv_to_wind(u, v):
    speed = np.hypot(u, v)
    deg = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, deg


def fuse_sources(weather: pd.DataFrame, air: pd.DataFrame,
                 aqicn: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Return an hourly, contiguous, imputed frame + a quality report."""
    cfg = settings()
    lat, lon = cfg["city"]["lat"], cfg["city"]["lon"]
    limit = int(cfg["imputation"]["ffill_limit_hours"])

    frames = []
    w, inv_w = validate_frame(weather[["ts"] + [c for c in WEATHER_COLS if c in weather]]) \
        if len(weather) else (pd.DataFrame(columns=["ts"]), {})
    a, inv_a = validate_frame(air[["ts"] + [c for c in AIR_COLS if c in air]]) \
        if len(air) else (pd.DataFrame(columns=["ts"]), {})
    frames = [w, a]
    if aqicn is not None and len(aqicn):
        q, _ = validate_frame(aqicn[["ts", "aqi_aqicn"]])
        frames.append(q)

    all_ts = pd.concat([f["ts"] for f in frames if len(f)], ignore_index=True)
    if all_ts.empty:
        return pd.DataFrame(), {"rows": 0}
    full = pd.DataFrame({"ts": np.arange(all_ts.min(), all_ts.max() + 3600, 3600, dtype="int64")})
    for f in frames:
        if len(f):
            full = full.merge(f, on="ts", how="left")
    for c in WEATHER_COLS + AIR_COLS + ["aqi_aqicn"]:
        if c not in full:
            full[c] = np.nan
        full[c] = pd.to_numeric(full[c], errors="coerce").astype("float64")

    raw_missing = full[WEATHER_COLS + AIR_COLS].isna()
    full["solar_elev_deg"] = solar_elevation_deg(full["ts"], lat, lon)

    # --- imputation (past-only) ---
    full["uvi"] = full["uvi"].where(~(full["uvi"].isna() & (full["solar_elev_deg"] <= 0)), 0.0)
    for c in WEATHER_COLS + AIR_COLS:
        full[c] = full[c].ffill(limit=limit)
    full["aqi_aqicn"] = full["aqi_aqicn"].ffill(limit=limit)
    still_missing = full[WEATHER_COLS + AIR_COLS].isna()
    full["n_imputed"] = (raw_missing & ~still_missing).sum(axis=1).astype("int64")
    full["n_missing"] = still_missing.sum(axis=1).astype("int64")

    # --- derived physical variables ---
    full["wind_u"], full["wind_v"] = wind_to_uv(full["wind_speed"], full["wind_deg"])
    full["aqi"] = compute_aqi_frame(full)
    full["aqi_source"] = np.where(full["aqi"].notna(), "openweather_calc",
                                  np.where(full["aqi_aqicn"].notna(), "aqicn", "missing"))
    full["aqi"] = full["aqi"].fillna(full["aqi_aqicn"])

    report = {
        "rows": int(len(full)),
        "invalid_values": {**inv_w, **inv_a},
        "imputed_cells": int(full["n_imputed"].sum()),
        "rows_with_missing": int((full["n_missing"] > 0).sum()),
    }
    return full, report
