"""Open-Meteo weather client (free, no API key, no card).

Used as the WEATHER source for both history (backfill) and live hourly data,
so training and serving data come from the same model (no train/serve skew).
* Recent data (last ~3 months + today): https://api.open-meteo.com/v1/forecast
* Older data:  https://historical-forecast-api.open-meteo.com/v1/forecast
Both return hourly values incl. UV index; we request UTC unix timestamps
and wind speed in m/s so the columns match the OpenWeather parser exactly.
Docs: https://open-meteo.com/en/docs and https://open-meteo.com/en/docs/historical-forecast-api
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.ingestion.retry import ApiError, get_json

SOURCE = "open_meteo"
RECENT_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl", "cloud_cover",
          "precipitation", "uv_index", "wind_speed_10m", "wind_direction_10m"]
COLUMN_MAP = {"temperature_2m": "temp_c", "relative_humidity_2m": "humidity_pct", "dew_point_2m": "dew_point_c",
              "pressure_msl": "pressure_hpa", "cloud_cover": "clouds_pct", "precipitation": "precip_mm",
              "uv_index": "uvi", "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_deg"}
RECENT_DAYS = 85          # use the regular forecast API inside this window
CHUNK_DAYS = 31


def parse_hourly(payload: dict, start_ts: int, end_ts: int) -> list[dict[str, Any]]:
    h = payload.get("hourly") or {}
    times = h.get("time") or []
    rows = []
    for i, t in enumerate(times):
        t = int(t)
        if t < start_ts or t > end_ts:
            continue
        row = {"ts": t - t % 3600, "source": SOURCE, "weather_main": None}
        for api_name, col in COLUMN_MAP.items():
            vals = h.get(api_name) or []
            row[col] = vals[i] if i < len(vals) else None
        if row["temp_c"] is None:          # hours without data yet (e.g. the future)
            continue
        rows.append(row)
    return rows


def _day(ts: int) -> str:
    return pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%Y-%m-%d")


def fetch_weather_range(lat: float, lon: float, start_ts: int, end_ts: int, run_log=None) -> list[dict]:
    """Hourly weather for [start_ts, end_ts] (UTC epoch seconds), any length."""
    out: list[dict] = []
    recent_cut = int(time.time()) - RECENT_DAYS * 86400
    s = int(start_ts)
    while s <= end_ts:
        e = min(s + CHUNK_DAYS * 86400, int(end_ts))
        url = RECENT_URL if s >= recent_cut else HISTORY_URL
        payload = get_json(url, {
            "latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY),
            "start_date": _day(s), "end_date": _day(e), "timezone": "GMT",
            "timeformat": "unixtime", "wind_speed_unit": "ms"}, SOURCE, run_log)
        if "hourly" not in payload:
            raise ApiError(SOURCE, f"unexpected response: {str(payload)[:200]}")
        out.extend(parse_hourly(payload, s, e))
        s = e + 3600
    return out