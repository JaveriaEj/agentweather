"""Historical backfill (resumable).

Run:  python -m scripts.backfill --days 180 --max-calls 900

* Air quality: OpenWeather Air Pollution history (free, few calls).
* Weather: One Call 3.0 time machine = 1 call per hour. With the free
  1,000 calls/day, 180 days (4,320 hours) needs ~5 daily runs. The script
  only fetches hours that are still missing, newest first, so you can run
  it every day (or via the backfill workflow) until it is complete.
* Then it rebuilds features + targets for the whole range with the SAME
  code as the hourly pipeline.

Limitation (documented): AQICN has no free hourly history API, so the
historical AQI is computed from OpenWeather pollutant concentrations.
"""
from __future__ import annotations

import argparse
import sys

from src.monitoring.logs import RunLog
from src.processing.pipeline import (HOUR, fill_air_gaps, fill_weather_gaps, missing_hours,
                                     process_window)
from src.store import get_store
from src.utils.logger import get_logger
from src.utils.timeutil import floor_hour, to_ts, utcnow

log = get_logger("backfill")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--max-calls", type=int, default=900,
                    help="max One Call time-machine calls in this run (free tier = 1000/day total)")
    ap.add_argument("--skip-weather", action="store_true")
    args = ap.parse_args(argv)

    store = get_store()
    run_log = RunLog("backfill")
    end = to_ts(floor_hour(utcnow())) - HOUR
    start = end - args.days * 24 * HOUR
    try:
        n_air = fill_air_gaps(store, start, end, run_log)
        log.info("Air-quality rows fetched: %d", n_air)

        gaps = missing_hours(store, "raw_weather", start, end)
        gaps.sort(reverse=True)  # newest first -> usable recent history sooner
        used = 0
        if gaps and not args.skip_weather:
            used = fill_weather_gaps(store, gaps, run_log, max_calls=args.max_calls)
        remaining = max(len(gaps) - used, 0)
        log.info("Weather hours missing: %d, fetched this run: %d, remaining: %d",
                 len(gaps), used, remaining)

        info = process_window(store, start, end, run_log)
        info.update({"weather_calls": used, "weather_remaining": remaining, "air_rows": n_air})
        status = "success" if remaining == 0 else "partial"
        run_log.system("run", "backfill", status, "backfill finished", info)
        log.info("Backfill %s: %s", status, info)
        if remaining:
            log.info("Run the backfill again tomorrow to fetch the remaining %d hours.", remaining)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.exception("Backfill failed")
        run_log.system("run", "backfill", "failed", str(exc))
        return 1
    finally:
        run_log.flush(store)


if __name__ == "__main__":
    sys.exit(main())
