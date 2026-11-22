"""Statistical model: SARIMAX with Fourier terms for the daily cycle.

Why Fourier terms instead of a seasonal order of 24? A seasonal SARIMA with
period 24 is very slow to fit on hourly data. sin/cos terms of the hour of
day capture the same daily pattern as exogenous regressors, quickly.

Forecasting at an origin: we keep the fitted parameters and re-run the
Kalman filter over the last `filter_window` hours up to the origin
(`.filter(params)` - no refit), then forecast 24 steps ahead.
"""
from __future__ import annotations

import warnings

import numpy as np

from src.models.base import ForecastModel
from src.models.dataset import fourier_hour
from src.utils.config import horizon, settings

FILTER_WINDOW = 336  # 14 days of state history per forecast


class SarimaxModel(ForecastModel):
    family = "statistical"
    name = "sarimax"

    def __init__(self, var: str):
        super().__init__(var)
        cfg = settings()["training"]["sarimax"]
        self.order = tuple(cfg["order"])
        self.k = int(cfg["fourier_k"])
        self.fit_window = int(cfg["fit_window_hours"])
        self.params = None

    def _model(self, endog, ts):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(endog, exog=fourier_hour(ts, self.k), order=self.order, trend="c",
                       enforce_stationarity=False, enforce_invertibility=False)

    def fit(self, data, train_idx):
        end = int(train_idx.max())
        start = max(0, end - self.fit_window + 1)
        y = data.frame[self.var].to_numpy(dtype=float)[start:end + 1]
        ts = data.frame["ts"].to_numpy()[start:end + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._model(y, ts).fit(disp=False, maxiter=200)
        self.params = np.asarray(res.params)
        return self

    def predict(self, frame, idx):
        H = horizon()
        y_all = frame[self.var].to_numpy(dtype=float)
        ts_all = frame["ts"].to_numpy()
        out = np.empty((len(idx), H))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for r, i in enumerate(idx):
                lo = max(0, i - FILTER_WINDOW + 1)
                res = self._model(y_all[lo:i + 1], ts_all[lo:i + 1]).filter(self.params)
                future_ts = ts_all[i] + 3600 * np.arange(1, H + 1)
                out[r] = res.forecast(H, exog=fourier_hour(future_ts, self.k))
        return self.postprocess(out, frame, idx)
