"""US EPA AQI from pollutant concentrations (OpenWeather gives µg/m³).

Breakpoints: EPA AQI technical assistance document (PM2.5 values updated in 2024).
Overall AQI = max of the individual pollutant sub-indices.

Documented approximations (no official 24h averages are available hourly):
* PM2.5 / PM10 / NO2 / SO2 use the hourly concentration.
* O3 and CO use a rolling 8-hour mean of past values (no future data -> no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (C_low, C_high, I_low, I_high)
BREAKPOINTS = {
    "pm2_5": [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
              (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "o3_ppb": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
               (86, 105, 151, 200), (106, 200, 201, 300), (201, 604, 301, 500)],
    "no2_ppb": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
                (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "so2_ppb": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
                (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)],
    "co_ppm": [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
               (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)],
}
MOLAR_MASS = {"o3": 48.00, "no2": 46.01, "so2": 64.07, "co": 28.01}


def ugm3_to_ppb(c, gas: str):
    return c * 24.45 / MOLAR_MASS[gas]


def sub_index(c: float, pollutant: str) -> float:
    if c is None or (isinstance(c, float) and np.isnan(c)) or c < 0:
        return np.nan
    for c_lo, c_hi, i_lo, i_hi in BREAKPOINTS[pollutant]:
        if c <= c_hi:
            c_eff = max(c, c_lo)
            return round((i_hi - i_lo) / (c_hi - c_lo) * (c_eff - c_lo) + i_lo)
    return 500.0  # beyond the scale


def compute_aqi_frame(df: pd.DataFrame) -> pd.Series:
    """Vectorised overall AQI for an hourly, time-sorted frame with µg/m³ columns."""
    parts = {}
    if "pm2_5" in df:
        parts["pm2_5"] = df["pm2_5"].map(lambda c: sub_index(c, "pm2_5"))
    if "pm10" in df:
        parts["pm10"] = df["pm10"].map(lambda c: sub_index(c, "pm10"))
    if "o3" in df:
        o3_8h = ugm3_to_ppb(df["o3"], "o3").rolling(8, min_periods=1).mean()
        parts["o3"] = o3_8h.map(lambda c: sub_index(c, "o3_ppb"))
    if "no2" in df:
        parts["no2"] = ugm3_to_ppb(df["no2"], "no2").map(lambda c: sub_index(c, "no2_ppb"))
    if "so2" in df:
        parts["so2"] = ugm3_to_ppb(df["so2"], "so2").map(lambda c: sub_index(c, "so2_ppb"))
    if "co" in df:
        co_8h = (ugm3_to_ppb(df["co"], "co") / 1000.0).rolling(8, min_periods=1).mean()
        parts["co"] = co_8h.map(lambda c: sub_index(c, "co_ppm"))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    sub = pd.DataFrame(parts)
    # PM2.5 or PM10 must be present for a meaningful AQI
    has_pm = sub[[c for c in ("pm2_5", "pm10") if c in sub]].notna().any(axis=1)
    return sub.max(axis=1, skipna=True).where(has_pm)
