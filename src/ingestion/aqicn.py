"""AQICN (World Air Quality Index project) client.

The free API returns the *current* station reading near a coordinate.
Note: AQICN `iaqi` values are already AQI sub-indices (US EPA scale),
not concentrations. Hourly history is NOT available through the free
API, which is why the backfill uses OpenWeather pollutant history.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.ingestion.retry import ApiError, get_json
from src.utils.config import env
from src.utils.timeutil import to_ts

SOURCE = "aqicn"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None  # AQICN uses "-" when a station has no value


def parse_feed(payload: dict) -> dict[str, Any]:
    if payload.get("status") != "ok":
        raise ApiError(SOURCE, f"status={payload.get('status')} data={str(payload.get('data'))[:120]}")
    d = payload["data"]
    iaqi = d.get("iaqi", {}) or {}
    t = (d.get("time") or {}).get("iso")
    ts = to_ts(pd.Timestamp(t)) if t else to_ts(pd.Timestamp.now(tz="UTC"))
    return {
        "ts": ts,
        "aqi_aqicn": _num(d.get("aqi")),
        "aqicn_pm25_index": _num((iaqi.get("pm25") or {}).get("v")),
        "aqicn_pm10_index": _num((iaqi.get("pm10") or {}).get("v")),
        "aqicn_dominant": d.get("dominentpol"),
        "aqicn_station": (d.get("city") or {}).get("name"),
    }


def fetch_current(lat: float, lon: float, run_log=None) -> dict[str, Any]:
    token = env("AQICN_TOKEN", required=True)
    payload = get_json(f"https://api.waqi.info/feed/geo:{lat};{lon}/", {"token": token}, SOURCE, run_log)
    return parse_feed(payload)
