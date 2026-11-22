"""Model explainability.

Primary: SHAP (model-agnostic permutation explainer) on the classical
model of each variable, for the 1-hour-ahead and 24-hour-ahead outputs.
Fallback (only if the `shap` package is unavailable): sklearn permutation
importance. The method used is stored with every row so the dashboard
labels it honestly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def explain_sklearn(model, frame: pd.DataFrame, bg_idx, ex_idx, horizons=(0, 23),
                    top_k: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = model.feature_cols
    bg = frame.loc[rng.choice(bg_idx, size=min(20, len(bg_idx)), replace=False), cols].to_numpy(float)
    ex_sel = rng.choice(ex_idx, size=min(30, len(ex_idx)), replace=False)
    X = frame.loc[ex_sel, cols].to_numpy(float)
    bg = np.where(np.isnan(bg), np.nanmedian(bg, axis=0), bg)
    X = np.where(np.isnan(X), np.nanmedian(bg, axis=0), X)
    rows = []
    try:  # small batches: single-threaded prediction is much faster than spawning workers
        model.pipe.set_params(model__n_jobs=1)
    except ValueError:
        pass
    import os

    try:
        import shap  # noqa: F401
        have_shap = os.getenv("EXPLAIN_METHOD", "shap") != "permutation"
    except ImportError:
        have_shap = False

    perm_preds = None
    if not have_shap:
        # Permutation importance w.r.t. the model's own output: how much does the
        # prediction change when one feature is shuffled? (all horizons at once)
        base_all = model.pipe.predict(X)
        perm_preds = []
        for j in range(len(cols)):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            perm_preds.append(model.pipe.predict(Xp))

    for h in horizons:
        if have_shap:
            import shap

            def f(Z, h=h):
                return model.pipe.predict(Z)[:, h]

            explainer = shap.explainers.Permutation(f, shap.maskers.Independent(bg), seed=seed)
            sv = explainer(X, max_evals=2 * len(cols) + 1)
            imp = np.abs(sv.values).mean(axis=0)
            method = "shap_permutation"
        else:
            imp = np.array([np.sqrt(np.mean((p[:, h] - base_all[:, h]) ** 2)) for p in perm_preds])
            method = "permutation_importance"
        order = np.argsort(imp)[::-1][:top_k]
        for rank, j in enumerate(order, 1):
            rows.append({"feature": cols[j], "importance": float(imp[j]), "horizon": float(h + 1),
                         "importance_rank": float(rank), "method": method})
    try:
        model.pipe.set_params(model__n_jobs=-1)
    except ValueError:
        pass
    return pd.DataFrame(rows)
