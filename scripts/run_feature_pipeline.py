"""SCRIPT 1 - hourly data collection + feature engineering.

Run:  python -m scripts.run_feature_pipeline
Schedule: GitHub Actions, every hour (.github/workflows/feature_pipeline.yml)

Steps: fetch live data (with retries/fallbacks) -> store raw -> repair any
gaps from missed runs (last 6h) -> fuse/impute/features/targets -> store
-> write pipeline logs. Exits with code 1 if no feature row was produced
for the current hour, so the agents workflow does not run on stale data.
"""
from __future__ import annotations

import sys

from src.monitoring.logs import RunLog
from src.processing.pipeline import (HOUR, collect_live, fill_air_gaps, fill_weather_gaps,
                                     hourly_window, missing_hours, process_window)
from src.store import get_store
from src.utils.logger import get_logger

log = get_logger("feature_pipeline")
GAP_REPAIR_HOURS = 6


def main() -> int:
    store = get_store()
    run_log = RunLog("feature_pipeline")
    status = "failed"
    try:
        live = collect_live(run_log)
        for fg, df in live.items():
            store.insert(fg, df)

        write_start, now_ts = hourly_window()
        gap_start = now_ts - GAP_REPAIR_HOURS * HOUR
        gaps = missing_hours(store, "raw_weather", gap_start, now_ts - HOUR)
        if gaps:
            used = fill_weather_gaps(store, gaps, run_log, max_calls=GAP_REPAIR_HOURS)
            run_log.system("gap_repair", "raw_weather", "success", f"{len(gaps)} gaps, {used} calls")
        fill_air_gaps(store, gap_start, now_ts, run_log)

        info = process_window(store, write_start, now_ts, run_log)
        latest = store.read("hourly_features", now_ts, now_ts)
        status = "success" if len(latest) else "failed"
        if "raw_weather" not in live or "raw_air_quality" not in live:
            status = "partial" if status == "success" else status
        run_log.system("run", "feature_pipeline", status, "finished", info)
        log.info("Feature pipeline %s: %s", status, info)
    except Exception as exc:  # noqa: BLE001
        log.exception("Feature pipeline crashed")
        run_log.system("run", "feature_pipeline", "failed", f"{type(exc).__name__}: {exc}")
    finally:
        run_log.flush(store)
    return 0 if status in ("success", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
