"""Physical range checks. Out-of-range values become NaN (and are counted)."""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_RANGES = {
    "temp_c": (-50, 60), "humidity_pct": (0, 100), "pressure_hpa": (850, 1100),
    "dew_point_c": (-60, 40), "clouds_pct": (0, 100), "uvi": (0, 20),
    "wind_speed": (0, 75), "wind_deg": (0, 360), "precip_mm": (0, 300),
    "pm2_5": (0, 2000), "pm10": (0, 3000), "o3": (0, 2000), "no2": (0, 2000),
    "so2": (0, 3000), "co": (0, 100000), "aqi": (0, 999), "aqi_aqicn": (0, 999),
}


def validate_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    invalid: dict[str, int] = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals.notna() & ((vals < lo) | (vals > hi))
        if bad.any():
            invalid[col] = int(bad.sum())
        df[col] = vals.where(~bad, np.nan)
    if "ts" in df:
        df = df.dropna(subset=["ts"]).copy()
        df["ts"] = df["ts"].astype("int64")
        df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts")
    return df, invalid
