"""OpenWeather client.

Endpoints used
--------------
* One Call 3.0 current      /data/3.0/onecall                (needs "One Call by Call" plan)
* One Call 3.0 time machine /data/3.0/onecall/timemachine    (1 call = 1 historical hour)
* Current weather 2.5       /data/2.5/weather                (free fallback, no UV index)
* Air Pollution current     /data/2.5/air_pollution          (free)
* Air Pollution history     /data/2.5/air_pollution/history  (free, hourly, since late 2020)

All values are returned in metric units. Each parser returns a flat dict
(one observation) keyed by our column names.
"""
from __future__ import annotations

from typing import Any

from src.ingestion.retry import ApiError, get_json
from src.utils.config import env
from src.utils.timeutil import to_ts

BASE = "https://api.openweathermap.org"
SOURCE = "openweather"


def _key() -> str:
    return env("OPENWEATHER_API_KEY", required=True)  # type: ignore[return-value]


def _precip(block: dict) -> float:
    rain = block.get("rain", {}) or {}
    snow = block.get("snow", {}) or {}
    r = rain.get("1h", 0.0) if isinstance(rain, dict) else float(rain or 0)
    s = snow.get("1h", 0.0) if isinstance(snow, dict) else float(snow or 0)
    return float(r or 0) + float(s or 0)


def parse_onecall_point(p: dict) -> dict[str, Any]:
    """Parse one 'current' block or one time-machine 'data' item."""
    return {
        "ts": to_ts(_utc(p["dt"])),
        "temp_c": p.get("temp"),
        "humidity_pct": p.get("humidity"),
        "pressure_hpa": p.get("pressure"),
        "dew_point_c": p.get("dew_point"),
        "clouds_pct": p.get("clouds"),
        "uvi": p.get("uvi"),
        "wind_speed": p.get("wind_speed"),
        "wind_deg": p.get("wind_deg"),
        "precip_mm": _precip(p),
        "weather_main": (p.get("weather") or [{}])[0].get("main"),
        "source": "ow_onecall",
    }


def parse_weather25(p: dict) -> dict[str, Any]:
    main, wind = p.get("main", {}), p.get("wind", {})
    return {
        "ts": to_ts(_utc(p["dt"])),
        "temp_c": main.get("temp"),
        "humidity_pct": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "dew_point_c": None,
        "clouds_pct": (p.get("clouds") or {}).get("all"),
        "uvi": None,  # not provided by this endpoint -> imputed later
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "precip_mm": _precip(p),
        "weather_main": (p.get("weather") or [{}])[0].get("main"),
        "source": "ow_weather25",
    }


def parse_air_item(item: dict) -> dict[str, Any]:
    c = item.get("components", {})
    return {
        "ts": to_ts(_utc(item["dt"])),
        "pm2_5": c.get("pm2_5"),
        "pm10": c.get("pm10"),
        "o3": c.get("o3"),
        "no2": c.get("no2"),
        "so2": c.get("so2"),
        "co": c.get("co"),
        "ow_aqi_index": (item.get("main") or {}).get("aqi"),  # OpenWeather 1-5 scale (info only)
    }


def _utc(epoch: int):
    import pandas as pd

    return pd.Timestamp(int(epoch), unit="s", tz="UTC")


# ---------------------------------------------------------------- fetchers
def fetch_current_weather(lat: float, lon: float, run_log=None) -> dict[str, Any]:
    """One Call 3.0 current; falls back to the free 2.5 endpoint if One Call fails."""
    try:
        data = get_json(f"{BASE}/data/3.0/onecall",
                        {"lat": lat, "lon": lon, "appid": _key(), "units": "metric",
                         "exclude": "minutely,hourly,daily,alerts"}, SOURCE, run_log)
        return parse_onecall_point(data["current"])
    except (ApiError, KeyError) as exc:
        if run_log is not None:
            run_log.system("fallback", "openweather", "warning",
                           f"One Call failed ({exc}); using /data/2.5/weather")
        data = get_json(f"{BASE}/data/2.5/weather",
                        {"lat": lat, "lon": lon, "appid": _key(), "units": "metric"}, SOURCE, run_log)
        return parse_weather25(data)


def fetch_weather_at(lat: float, lon: float, ts: int, run_log=None) -> dict[str, Any]:
    """Historical weather for one hour (One Call 3.0 time machine). Costs 1 call."""
    data = get_json(f"{BASE}/data/3.0/onecall/timemachine",
                    {"lat": lat, "lon": lon, "dt": int(ts), "appid": _key(), "units": "metric"},
                    SOURCE, run_log)
    items = data.get("data") or []
    if not items:
        raise ApiError(SOURCE, f"empty time-machine response for dt={ts}")
    row = parse_onecall_point(items[0])
    row["ts"] = int(ts) - int(ts) % 3600  # align to the requested hour
    return row


def fetch_current_air(lat: float, lon: float, run_log=None) -> dict[str, Any]:
    data = get_json(f"{BASE}/data/2.5/air_pollution",
                    {"lat": lat, "lon": lon, "appid": _key()}, SOURCE, run_log)
    items = data.get("list") or []
    if not items:
        raise ApiError(SOURCE, "empty air pollution response")
    return parse_air_item(items[0])


def fetch_air_history(lat: float, lon: float, start_ts: int, end_ts: int, run_log=None) -> list[dict]:
    """Hourly pollutant history. We request in 30-day chunks to keep responses small."""
    out: list[dict] = []
    chunk = 30 * 24 * 3600
    s = int(start_ts)
    while s < end_ts:
        e = min(s + chunk, int(end_ts))
        data = get_json(f"{BASE}/data/2.5/air_pollution/history",
                        {"lat": lat, "lon": lon, "start": s, "end": e, "appid": _key()},
                        SOURCE, run_log)
        out.extend(parse_air_item(i) for i in data.get("list") or [])
        s = e
    return out
