"""Builds the modelling dataset from the Feature Store and the time-based split.

Terminology
-----------
origin  = the hour t at which a forecast is made (we predict t+1..t+24).
frame   = contiguous hourly feature table (row position = hour).
Every model predicts from `frame` using ONLY rows at or before the origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.processing.features import feature_columns, target_columns
from src.utils.config import horizon, settings, target_vars

HOUR = 3600
RAIN_VAR = "rain_prob"


@dataclass
class ModelData:
    frame: pd.DataFrame          # contiguous hourly features (index = position)
    targets: pd.DataFrame        # aligned target columns (NaN where unknown)
    train_idx: np.ndarray        # origin positions used for training
    val_idx: np.ndarray          # origin positions used for validation
    feature_cols: list[str]

    def y(self, var: str, idx: np.ndarray) -> np.ndarray:
        if var == RAIN_VAR:
            thr = settings()["forecast"]["rain_threshold_mm"]
            raw = self.targets.loc[idx, [f"precip_mm_h{h:02d}" for h in range(1, horizon() + 1)]]
            return (raw.to_numpy() > thr).astype(float)
        return self.targets.loc[idx, [f"{var}_h{h:02d}" for h in range(1, horizon() + 1)]].to_numpy()


def contiguous(features: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a full hourly grid so row position == hour offset."""
    f = features.drop_duplicates("ts").sort_values("ts")
    grid = pd.DataFrame({"ts": np.arange(f["ts"].min(), f["ts"].max() + HOUR, HOUR, dtype="int64")})
    return grid.merge(f, on="ts", how="left").reset_index(drop=True)


def build_model_data(features: pd.DataFrame, targets: pd.DataFrame) -> ModelData:
    cfg = settings()["training"]
    frame = contiguous(features)
    tg = frame[["ts"]].merge(targets.drop(columns=["event_time"], errors="ignore"),
                             on="ts", how="left")
    tcols = target_columns()
    for c in tcols:
        if c not in tg:
            tg[c] = np.nan
    feat_cols = [c for c in feature_columns() if c in frame.columns]

    min_hist = 48
    ok = tg[tcols].notna().all(axis=1).to_numpy() & frame[target_vars()].notna().all(axis=1).to_numpy()
    ok[:min_hist] = False
    origins = np.flatnonzero(ok)
    if len(origins) < 24 * 21:
        raise ValueError(f"Only {len(origins)} usable training hours; need >= {24 * 21} (3 weeks). "
                         "Run the backfill longer.")
    n_val = int(len(origins) * cfg["validation_fraction"])
    val = origins[-n_val:]
    val_start_ts = frame.loc[val[0], "ts"]
    train = origins[frame.loc[origins, "ts"].to_numpy() <= val_start_ts - cfg["gap_hours"] * HOUR]
    return ModelData(frame=frame, targets=tg, train_idx=train, val_idx=val, feature_cols=feat_cols)


def load_model_data(store) -> ModelData:
    feats = store.read("hourly_features")
    targets = store.read("training_targets")
    if feats.empty or targets.empty:
        raise ValueError("No features/targets in the Feature Store. Run the backfill first.")
    return build_model_data(feats, targets)


def fourier_hour(ts: np.ndarray, k: int, tz_offset_h: float = 0.0) -> np.ndarray:
    hour = ((np.asarray(ts) // HOUR) + tz_offset_h) % 24
    cols = []
    for i in range(1, k + 1):
        cols += [np.sin(2 * np.pi * i * hour / 24), np.cos(2 * np.pi * i * hour / 24)]
    return np.column_stack(cols)
