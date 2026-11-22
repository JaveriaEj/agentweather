"""Common model interface + the Naive (24-hours-ago) baseline.

Every model answers: given the hourly `frame` and origin positions `idx`,
return an array of shape (len(idx), 24) for one variable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import horizon, settings

NON_NEGATIVE = {"aqi", "precip_mm", "humidity_pct", "uvi"}


class ForecastModel:
    family = "base"
    name = "base"

    def __init__(self, var: str):
        self.var = var

    def fit(self, data, train_idx: np.ndarray) -> "ForecastModel":
        return self

    def predict(self, frame: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def postprocess(self, pred: np.ndarray, frame: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        if self.var in NON_NEGATIVE:
            pred = np.clip(pred, 0, None)
        if self.var == "humidity_pct":
            pred = np.clip(pred, 0, 100)
        if self.var == "rain_prob":
            pred = np.clip(pred, 0, 1)
        if self.var == "uvi":  # physics: no UV when the sun is below the horizon
            from src.processing.fuse import solar_elevation_deg

            city = settings()["city"]
            H = pred.shape[1]
            fut = (frame["ts"].to_numpy()[np.asarray(idx)][:, None]
                   + 3600 * np.arange(1, H + 1)[None, :]).ravel()
            elev = solar_elevation_deg(pd.Series(fut), city["lat"], city["lon"]).to_numpy()
            pred = np.where(elev.reshape(pred.shape) <= 0, 0.0, pred)
        return pred


class NaiveBaseline(ForecastModel):
    """Prediction for hour t+h = observed value at t+h-24 (same hour yesterday)."""

    family = "baseline"
    name = "naive_24h"

    def predict(self, frame, idx):
        H = horizon()
        if self.var == "rain_prob":
            thr = settings()["forecast"]["rain_threshold_mm"]
            series = (frame["precip_mm"] > thr).astype(float).where(frame["precip_mm"].notna())
        else:
            series = frame[self.var]
        s = series.to_numpy(dtype=float)
        out = np.full((len(idx), H), np.nan)
        for r, i in enumerate(idx):
            lo = i - H + 1
            if lo >= 0:
                out[r] = s[lo:i + 1]
        # fall back to persistence (last value) if a lag is missing
        last = s[np.asarray(idx)]
        out = np.where(np.isnan(out), last[:, None], out)
        return out
