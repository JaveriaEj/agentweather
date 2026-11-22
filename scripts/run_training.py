"""SCRIPT 2 - model training, evaluation and registration.

Run:  python -m scripts.run_training [--reason drift] [--families classical,deep_learning]
Schedule: daily + on demand when the Drift Agent dispatches train.yml.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.monitoring.logs import RunLog, new_id
from src.models.train import run_training
from src.store import get_store
from src.utils.logger import get_logger
from src.utils.timeutil import utcnow

log = get_logger("training")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="scheduled", help="scheduled | drift | manual")
    ap.add_argument("--drift-event-id", default="")
    ap.add_argument("--families", default="", help="comma list: statistical,classical,deep_learning")
    args = ap.parse_args(argv)
    families = set(args.families.split(",")) if args.families else None

    store = get_store()
    run_log = RunLog("training")
    code = 0
    try:
        summary = run_training(store, reason=args.reason, families=families, run_log=run_log)
        log.info("Training summary: %s", json.dumps(summary, default=str))
        outcome = summary["decision"]
    except Exception as exc:  # noqa: BLE001
        log.exception("Training failed")
        run_log.system("run", "training", "failed", f"{type(exc).__name__}: {exc}")
        outcome, code = f"failed: {exc}", 1
    if args.reason == "drift":
        # close the loop: record how the drift-triggered retraining ended
        store.insert("drift_log", pd.DataFrame([{
            "event_id": new_id(), "ts": int(utcnow().timestamp()), "metric": "all",
            "drift_type": "retraining", "score": float("nan"), "threshold": float("nan"),
            "drifted": 0, "action": "retraining_completed" if code == 0 else "retraining_failed",
            "retrain_status": outcome, "details": args.drift_event_id}]))
    run_log.flush(store)
    return code


if __name__ == "__main__":
    sys.exit(main())
