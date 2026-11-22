"""Evaluation metrics: RMSE, MAE, R² over all 24 horizons (+ per-horizon RMSE)."""
from __future__ import annotations

import numpy as np


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "rmse_h01": np.nan, "rmse_h24": np.nan}
    err = (yp - yt)[mask]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((yt[mask] - yt[mask].mean()) ** 2).sum())
    out = {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }
    if yt.ndim == 2:
        per_h = np.sqrt(np.nanmean((yp - yt) ** 2, axis=0))
        out["rmse_h01"], out["rmse_h24"] = float(per_h[0]), float(per_h[-1])
        out["rmse_by_horizon"] = [float(x) for x in per_h]
    return out
