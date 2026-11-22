"""Turn raw per-variable predictions into the six metric groups' outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing.fuse import uv_to_wind

METRIC_GROUPS = {
    "AQI": ["aqi"],
    "Temperature": ["temp_c"],
    "Precipitation": ["precip_mm", "rain_prob_pct"],
    "Humidity": ["humidity_pct"],
    "UV Index": ["uvi"],
    "Wind": ["wind_speed", "wind_dir_deg"],
}
OUTPUT_COLUMNS = [c for cols in METRIC_GROUPS.values() for c in cols]


def to_output_frame(raw: dict[str, np.ndarray], origin_ts: int) -> pd.DataFrame:
    H = len(next(iter(raw.values())))
    df = pd.DataFrame({
        "run_ts": origin_ts,
        "horizon": np.arange(1, H + 1),
        "target_ts": origin_ts + 3600 * np.arange(1, H + 1),
    })
    df["aqi"] = raw.get("aqi", np.full(H, np.nan))
    df["temp_c"] = raw.get("temp_c", np.full(H, np.nan))
    df["precip_mm"] = raw.get("precip_mm", np.full(H, np.nan))
    df["rain_prob_pct"] = 100 * np.asarray(raw.get("rain_prob", np.full(H, np.nan)))
    df["humidity_pct"] = raw.get("humidity_pct", np.full(H, np.nan))
    df["uvi"] = raw.get("uvi", np.full(H, np.nan))
    if "wind_u" in raw and "wind_v" in raw:
        df["wind_speed"], df["wind_dir_deg"] = uv_to_wind(np.asarray(raw["wind_u"]), np.asarray(raw["wind_v"]))
    else:
        df["wind_speed"], df["wind_dir_deg"] = np.nan, np.nan
    return df
