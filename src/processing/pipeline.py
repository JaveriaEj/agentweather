"""Shared feature-pipeline logic used by BOTH the hourly job and the backfill.

Using one code path for history and live data guarantees the training
features are computed exactly like the serving features.
"""
from __future__ import annotations

import pandas as pd

from src.ingestion import aqicn, openmeteo, openweather
from src.ingestion.retry import ApiError
from src.processing.features import build_features, build_targets, target_columns
from src.processing.fuse import fuse_sources
from src.utils.config import horizon, settings
from src.utils.logger import get_logger
from src.utils.timeutil import floor_hour, to_ts, utcnow

log = get_logger(__name__)
HOUR = 3600
LIVE_WEATHER_HOURS = 6   # Open-Meteo: re-fetch the last 6 hours each run (also repairs gaps)


def weather_source() -> str:
    return settings().get("data_sources", {}).get("weather", "openmeteo")


def collect_live(run_log) -> dict[str, pd.DataFrame]:
    """Fetch the current observation from every source. A failing source is logged and skipped."""
    city = settings()["city"]
    out: dict[str, pd.DataFrame] = {}
    now_ts = to_ts(utcnow())
    sources = [("raw_air_quality", openweather.fetch_current_air), ("raw_aqicn", aqicn.fetch_current)]
    if weather_source() == "openmeteo":
        try:
            rows = openmeteo.fetch_weather_range(city["lat"], city["lon"],
                                                 now_ts - LIVE_WEATHER_HOURS * HOUR, now_ts, run_log)
            if not rows:
                raise ApiError(openmeteo.SOURCE, "no hourly rows returned")
            out["raw_weather"] = pd.DataFrame(rows)
            run_log.system("fetch", "raw_weather", "success", f"open-meteo {len(rows)} hours")
        except ApiError as exc:
            log.error("Open-Meteo failed (%s); falling back to OpenWeather current weather", exc)
            run_log.system("fallback", "raw_weather", "warning", f"open-meteo failed: {exc}")
            sources.insert(0, ("raw_weather", openweather.fetch_current_weather))
    else:
        sources.insert(0, ("raw_weather", openweather.fetch_current_weather))
    for name, fn in sources:
        try:
            row = fn(city["lat"], city["lon"], run_log=run_log)
            # Observations are stamped at the top of the current hour.
            row["ts"] = min(int(row["ts"]), now_ts) if row.get("ts") else now_ts
            row["ts"] = row["ts"] - row["ts"] % HOUR
            out[name] = pd.DataFrame([row])
            run_log.system("fetch", name, "success", f"fetched ts={row['ts']}")
        except (ApiError, RuntimeError, KeyError, TypeError) as exc:
            log.error("Source %s failed: %s", name, exc)
            run_log.system("fetch", name, "failed", str(exc))
    return out


def missing_hours(store, fg: str, start_ts: int, end_ts: int) -> list[int]:
    have = store.read(fg, start_ts, end_ts)
    have_ts = set(have["ts"].astype("int64")) if len(have) else set()
    return [t for t in range(start_ts, end_ts + 1, HOUR) if t not in have_ts]


def fill_weather_gaps(store, hours: list[int], run_log, max_calls: int,
                      batch_size: int = 200) -> int:
    """Fetch missing historical weather hours.

    Open-Meteo (default): one request per month of data, returns the number of hours filled.
    OpenWeather time machine: one call per hour (paid plan), returns #calls used.
    """
    city = settings()["city"]
    if not hours:
        return 0
    if weather_source() == "openmeteo":
        try:
            rows = openmeteo.fetch_weather_range(city["lat"], city["lon"], min(hours), max(hours), run_log)
        except ApiError as exc:
            log.warning("Open-Meteo history failed: %s", exc)
            return 0
        wanted = set(hours)
        rows = [r for r in rows if r["ts"] in wanted]
        if rows:
            store.insert("raw_weather", pd.DataFrame(rows))
        return len(rows)
    rows, calls = [], 0
    for ts in hours[:max_calls]:
        calls += 1
        try:
            rows.append(openweather.fetch_weather_at(city["lat"], city["lon"], ts, run_log=run_log))
        except ApiError as exc:
            log.warning("time machine failed for %s: %s", ts, exc)
            if exc.status in (401, 403):
                break
        if len(rows) >= batch_size:
            store.insert("raw_weather", pd.DataFrame(rows))
            rows = []
    if rows:
        store.insert("raw_weather", pd.DataFrame(rows))
    return calls


def fill_air_gaps(store, start_ts: int, end_ts: int, run_log) -> int:
    city = settings()["city"]
    if not missing_hours(store, "raw_air_quality", start_ts, end_ts):
        return 0
    try:
        rows = openweather.fetch_air_history(city["lat"], city["lon"], start_ts, end_ts + HOUR, run_log)
    except ApiError as exc:
        log.warning("air history failed: %s", exc)
        return 0
    if rows:
        store.insert("raw_air_quality", pd.DataFrame(rows))
    return len(rows)


def process_window(store, write_start_ts: int, write_end_ts: int, run_log) -> dict:
    """Read raw data, fuse, impute, engineer features and targets; upsert both.

    Reads extra history before `write_start_ts` so lags/rolling windows are complete.
    """
    pad = (settings()["features"]["min_history_hours"] + 8) * HOUR
    read_start = write_start_ts - pad
    weather = store.read("raw_weather", read_start, write_end_ts)
    air = store.read("raw_air_quality", read_start, write_end_ts)
    aq = store.read("raw_aqicn", read_start, write_end_ts)
    if weather.empty and air.empty:
        run_log.system("process", "fuse", "failed", "no raw data in window")
        return {"features": 0, "targets": 0}

    fused, quality = fuse_sources(weather, air, aq)
    feats = build_features(fused)
    targets = build_targets(fused)

    in_window = (feats["ts"] >= write_start_ts) & (feats["ts"] <= write_end_ts)
    feats_out = feats[in_window]
    # targets only where all 24 future values exist (newest 24h are forecast-only)
    t_ok = in_window & targets[target_columns()].notna().all(axis=1)
    targets_out = targets[t_ok]

    store.insert("hourly_features", feats_out)
    store.insert("training_targets", targets_out)
    info = {"features": int(len(feats_out)), "targets": int(len(targets_out)), **quality}
    run_log.system("process", "features", "success", "features/targets written", info)
    return info


def hourly_window() -> tuple[int, int]:
    end = to_ts(floor_hour(utcnow()))
    return end - (2 * horizon()) * HOUR, end