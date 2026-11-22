"""Classical ML: Ridge Regression and Random Forest (multi-output: 24 horizons at once).

Features = the engineered feature row at the origin hour.
Missing feature values are filled with TRAINING medians (fit on train only).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import ForecastModel
from src.utils.config import horizon, settings


class SklearnModel(ForecastModel):
    family = "classical"

    def __init__(self, var: str, kind: str = "ridge"):
        super().__init__(var)
        self.kind = kind
        self.name = kind
        self.pipe = None
        self.feature_cols: list[str] = []

    def _make(self):
        cfg = settings()["training"]
        if self.kind == "ridge":
            return Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler()),
                             ("model", Ridge(alpha=cfg["ridge_alpha"]))])
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("model", RandomForestRegressor(
                             n_estimators=cfg["rf_n_estimators"], max_depth=cfg["rf_max_depth"],
                             min_samples_leaf=3, max_features=0.5, n_jobs=-1,
                             random_state=cfg["random_seed"]))])

    def fit(self, data, train_idx):
        self.feature_cols = list(data.feature_cols)
        X = data.frame.loc[train_idx, self.feature_cols].to_numpy(dtype=float)
        y = data.y(self.var, train_idx)
        self.pipe = self._make().fit(X, y)
        return self

    def predict(self, frame, idx):
        X = frame.loc[idx, self.feature_cols].to_numpy(dtype=float)
        return self.postprocess(self.pipe.predict(X), frame, idx)


class RainProbabilityModel(ForecastModel):
    """P(precip > threshold) for each of the 24 horizons.

    One logistic regression per horizon. If a horizon has too few rainy
    hours in training (common in dry cities), it falls back to the
    climatological rain frequency for that horizon - documented, not hidden.
    """

    family = "classical"
    name = "logistic"

    def __init__(self, var: str = "rain_prob"):
        super().__init__(var)
        self.models: list = []
        self.feature_cols: list[str] = []

    def fit(self, data, train_idx):
        self.feature_cols = list(data.feature_cols)
        X = data.frame.loc[train_idx, self.feature_cols].to_numpy(dtype=float)
        Y = data.y("rain_prob", train_idx)
        self.models = []
        for h in range(horizon()):
            y = Y[:, h]
            if y.sum() >= 5 and (len(y) - y.sum()) >= 5:
                m = Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler()),
                              ("model", LogisticRegression(C=0.5, max_iter=1000))]).fit(X, y)
                self.models.append(m)
            else:
                self.models.append(float(y.mean()))
        return self

    @property
    def n_climatology(self) -> int:
        return sum(isinstance(m, float) for m in self.models)

    def predict(self, frame, idx):
        X = frame.loc[idx, self.feature_cols].to_numpy(dtype=float)
        cols = [np.full(len(idx), m) if isinstance(m, float) else m.predict_proba(X)[:, 1]
                for m in self.models]
        return self.postprocess(np.column_stack(cols), frame, idx)
