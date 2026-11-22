"""SYNTHETIC data generator - FOR AUTOMATED TESTS ONLY.

Never used by the production pipelines or dashboard. It produces raw rows
in exactly the shape the API parsers return, so the full system can be
tested offline without API keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing.fuse import solar_elevation_deg


def make_raw(days: int = 60, end_ts: int | None = None, seed: int = 0,
             lat: float = 24.86, lon: float = 67.0, shift_level: float = 0.0):
    rng = np.random.default_rng(seed)
    n = days * 24
    end_ts = end_ts or (int(pd.Timestamp.now(tz="UTC").floor("h").timestamp()) - 3600)
    ts = np.arange(end_ts - (n - 1) * 3600, end_ts + 1, 3600, dtype="int64")
    t = pd.Series(ts)
    hour = ((ts // 3600) + 5) % 24  # local hour (UTC+5)
    diurnal = np.sin(2 * np.pi * (hour - 9) / 24)
    trend = np.sin(2 * np.pi * np.arange(n) / (24 * 30))

    temp = 28 + 5 * diurnal + 2 * trend + rng.normal(0, 0.8, n) + shift_level
    hum = np.clip(65 - 15 * diurnal + rng.normal(0, 4, n), 5, 100)
    pres = 1008 + 3 * np.sin(2 * np.pi * np.arange(n) / (24 * 7)) + rng.normal(0, 0.5, n)
    wspd = np.clip(4 + 2 * diurnal + rng.gamma(2, 0.8, n), 0, 30)
    wdeg = (220 + 30 * np.sin(2 * np.pi * np.arange(n) / 96) + rng.normal(0, 15, n)) % 360
    rain_event = rng.random(n) < 0.04
    precip = np.where(rain_event, rng.gamma(1.5, 2.0, n), 0.0)
    elev = solar_elevation_deg(t, lat, lon).to_numpy()
    uvi = np.clip(11 * np.sin(np.radians(np.clip(elev, 0, 90))) + rng.normal(0, 0.3, n), 0, None)
    pm25 = np.clip(np.exp(3.4 + 0.3 * np.cos(2 * np.pi * hour / 24) + 0.4 * trend
                          + rng.normal(0, 0.2, n)), 1, None)

    weather = pd.DataFrame({
        "ts": ts, "temp_c": temp, "humidity_pct": hum, "pressure_hpa": pres,
        "dew_point_c": temp - (100 - hum) / 5, "clouds_pct": np.clip(rng.normal(30, 20, n), 0, 100),
        "uvi": uvi, "wind_speed": wspd, "wind_deg": wdeg, "precip_mm": precip,
        "weather_main": np.where(rain_event, "Rain", "Clear"), "source": "synthetic"})
    air = pd.DataFrame({
        "ts": ts, "pm2_5": pm25, "pm10": pm25 * 1.8, "o3": np.clip(60 + 30 * diurnal, 1, None),
        "no2": np.clip(25 - 10 * diurnal + rng.normal(0, 3, n), 1, None), "so2": 8.0, "co": 400.0,
        "ow_aqi_index": 3.0})
    return weather, air
