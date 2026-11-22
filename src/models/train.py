"""Training orchestration: train all families, compare with the naive baseline,
select per variable, champion/challenger against production, register.

Replacement rule (per variable):
  1. a candidate must beat the naive baseline (validation RMSE) to be eligible;
  2. the best eligible candidate is compared with the CURRENT production
     model for that variable on the SAME validation hours;
  3. the production model is kept unless the candidate's RMSE is lower by
     more than `min_improvement_pct` (default 0.5%) - avoids churn from noise.
A new registry version is created only if at least one variable improved.
Models are trained on the training split only (never on validation hours),
so later comparisons on newer validation data stay fair.
"""
from __future__ import annotations

import importlib.util
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import NaiveBaseline
from src.models.bundle import ModelBundle
from src.models.classical import RainProbabilityModel, SklearnModel
from src.models.dataset import RAIN_VAR, ModelData, load_model_data
from src.models.evaluate import metrics
from src.models.explain import explain_sklearn
from src.monitoring.logs import RunLog, new_id
from src.utils.config import settings, target_vars
from src.utils.logger import get_logger
from src.utils.timeutil import utcnow

log = get_logger(__name__)


def _available(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def candidate_models(var: str, families: set[str] | None = None) -> list:
    if var == RAIN_VAR:
        cands = [NaiveBaseline(var), RainProbabilityModel(var)]
    else:
        cands = [NaiveBaseline(var), SklearnModel(var, "ridge"), SklearnModel(var, "random_forest")]
        if _available("statsmodels"):
            from src.models.sarimax import SarimaxModel

            cands.append(SarimaxModel(var))
        if _available("torch"):
            from src.models.lstm import LSTMModel

            cands.append(LSTMModel(var))
    if families:
        cands = [c for c in cands if c.family in families or c.family == "baseline"]
    return cands


def _component_train_end(prod, var: str) -> int | None:
    """Last training hour of the production model for `var` (None = unknown)."""
    if prod is None:
        return None
    comp = prod.meta.get("components", {}).get(var, {})
    if comp.get("train_end") is not None:
        return int(comp["train_end"])
    if comp.get("source") == "new":          # trained in that version's own run
        return int(prod.meta.get("train_range", [0, 0])[1])
    return None                              # carried over from an older version: unknown


def reference_stats(data: ModelData, bins: int = 10, sample: int = 500) -> dict:
    """Training-data distributions used by the Drift Agent (PSI + KS)."""
    out = {}
    rng = np.random.default_rng(0)
    rows = data.frame.loc[: int(data.train_idx.max())]
    for f in settings()["drift"]["monitored_features"]:
        if f not in rows:
            continue
        x = rows[f].dropna().to_numpy(float)
        if len(x) < 50:
            continue
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        counts, _ = np.histogram(x, bins=edges)
        out[f] = {"edges": edges.tolist(), "props": (counts / counts.sum()).tolist(),
                  "sample": rng.choice(x, size=min(sample, len(x)), replace=False).round(4).tolist(),
                  "mean": float(x.mean()), "std": float(x.std())}
    return out


def run_training(store, reason: str = "scheduled", families: set[str] | None = None,
                 run_log: RunLog | None = None) -> dict:
    cfg = settings()
    model_name = cfg["registry"]["model_name"]
    run_log = run_log or RunLog("training")
    run_id = run_log.run_id
    now_ts = int(utcnow().timestamp())

    data = load_model_data(store)
    eval_idx = data.val_idx[:: cfg["training"]["eval_origin_step"]]
    log.info("Training rows: %d | validation origins: %d (evaluated: %d)",
             len(data.train_idx), len(data.val_idx), len(eval_idx))

    # current production bundle (if any)
    prod = None
    latest = store.latest_model(model_name)
    if latest:
        try:
            prod = ModelBundle.load(latest[0], version=latest[1])
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load production bundle: %s", exc)

    # A production model may only be compared on hours it never trained on.
    # Each variable's model can come from an older version, so this is checked PER VARIABLE.
    val_start_ts = int(data.frame.loc[data.val_idx[0], "ts"])
    train_end_ts = int(data.frame.loc[data.train_idx[-1], "ts"])

    eval_rows, chosen, components, explain_models = [], {}, {}, {}
    improved_vars = []
    for var in target_vars() + [RAIN_VAR]:
        y_val = data.y(var, eval_idx)
        results = []
        for model in candidate_models(var, families):
            t0 = time.time()
            try:
                model.fit(data, data.train_idx)
                pred = model.predict(data.frame, eval_idx)
                m = metrics(y_val, pred)
                results.append((model, m))
                log.info("%-13s %-14s RMSE=%.3f MAE=%.3f R2=%.3f (%.1fs)", var, model.name,
                         m["rmse"], m["mae"], m["r2"], time.time() - t0)
            except Exception as exc:  # noqa: BLE001 - one failing family must not stop training
                log.exception("%s/%s failed", var, model.name)
                run_log.system("train", f"{var}/{model.name}", "failed", str(exc))
        base_rmse = next(m["rmse"] for mdl, m in results if mdl.family == "baseline")
        eligible = [(mdl, m) for mdl, m in results if mdl.family != "baseline" and m["rmse"] < base_rmse]
        best_mdl, best_m = min(eligible, key=lambda r: r[1]["rmse"]) if eligible else \
            next((mdl, m) for mdl, m in results if mdl.family == "baseline")

        # champion / challenger
        source, from_version = "new", None
        prod_end = _component_train_end(prod, var)
        if prod is not None and var in prod.models and (prod_end is None or prod_end >= val_start_ts):
            log.warning("%-13s production v%s was trained on (part of) the current validation period - "
                        "not a fair comparison, so it is not used as champion.", var, prod.version)
        if prod is not None and var in prod.models and prod_end is not None and prod_end < val_start_ts:
            try:
                p_m = metrics(y_val, prod.models[var].predict(data.frame, eval_idx))
            except Exception as exc:  # noqa: BLE001 - e.g. feature set changed
                log.warning("Production model for %s could not be evaluated: %s", var, exc)
                p_m = {"rmse": np.inf}
            log.info("%-13s production v%s RMSE=%.3f vs candidate %.3f", var, prod.version,
                     p_m["rmse"], best_m["rmse"])
            min_gain = cfg["training"].get("min_improvement_pct", 0.5) / 100.0
            if not best_m["rmse"] < p_m["rmse"] * (1 - min_gain):
                best_mdl, best_m = prod.models[var], p_m
                source, from_version = "carried_over", prod.version
        if source == "new":
            improved_vars.append(var)
        comp_train_end = train_end_ts if source == "new" else prod_end
        chosen[var] = best_mdl
        components[var] = {
            "family": best_mdl.family, "model": best_mdl.name, "source": source,
            "train_end": comp_train_end,
            "from_version": from_version, "val_rmse": best_m["rmse"], "val_mae": best_m.get("mae"),
            "val_r2": best_m.get("r2"), "baseline_rmse": base_rmse,
            "beats_baseline": bool(best_m["rmse"] < base_rmse),
            "rmse_by_horizon": best_m.get("rmse_by_horizon"),
            "note": ("climatology used for %d horizons (too few rainy hours)" % best_mdl.n_climatology)
            if isinstance(best_mdl, RainProbabilityModel) and best_mdl.n_climatology else "",
        }
        sk = [(mdl, m) for mdl, m in results if isinstance(mdl, SklearnModel)]
        if sk:
            explain_models[var] = min(sk, key=lambda r: r[1]["rmse"])[0]
        for mdl, m in results:
            eval_rows.append({
                "eval_id": f"{run_id}_{var}_{mdl.name}", "ts": now_ts, "run_id": run_id, "reason": reason,
                "target": var, "model": mdl.name, "family": mdl.family, "rmse": m["rmse"], "mae": m["mae"],
                "r2": m["r2"], "rmse_h01": m.get("rmse_h01"), "rmse_h24": m.get("rmse_h24"),
                "baseline_rmse": base_rmse, "beats_baseline": int(m["rmse"] < base_rmse),
                "selected": int(mdl is best_mdl), "n_val": float(len(eval_idx))})

    registered_version = None
    decision = "kept_production"
    if prod is None or improved_vars:
        meta = {
            "model_name": model_name, "trained_at": str(utcnow()), "reason": reason, "run_id": run_id,
            "train_range": [int(data.frame.loc[data.train_idx[0], "ts"]), int(data.frame.loc[data.train_idx[-1], "ts"])],
            "val_range": [int(data.frame.loc[data.val_idx[0], "ts"]), int(data.frame.loc[data.val_idx[-1], "ts"])],
            "feature_cols": data.feature_cols, "components": components,
            "improved_vars": improved_vars, "reference_stats": reference_stats(data),
        }
        bundle = ModelBundle(meta, chosen)
        with tempfile.TemporaryDirectory() as tmp:
            folder = bundle.save(Path(tmp) / "bundle")
            reg_metrics = {f"{v}_rmse": c["val_rmse"] for v, c in components.items()}
            reg_metrics["n_improved"] = len(improved_vars)
            desc = "; ".join(f"{v}:{c['model']}({c['source']})" for v, c in components.items())
            registered_version = store.register_model(model_name, folder, reg_metrics, desc)
        decision = "registered_new_version"
        log.info("Registered %s v%s (improved: %s)", model_name, registered_version, improved_vars)
    else:
        log.info("No variable improved - production v%s kept.", prod.version)

    for r in eval_rows:
        r["registered_version"] = float(registered_version) if registered_version else np.nan
        r["decision"] = decision
    store.insert("model_eval_history", pd.DataFrame(eval_rows))

    # explainability for the new version
    if registered_version:
        shap_rows = []
        for var, mdl in explain_models.items():
            try:
                df = explain_sklearn(mdl, data.frame, data.train_idx, data.val_idx)
                df["target"], df["explained_model"] = var, mdl.name
                df["production_model"] = components[var]["model"]
                shap_rows.append(df)
            except Exception as exc:  # noqa: BLE001
                log.warning("Explainability failed for %s: %s", var, exc)
        if shap_rows:
            sh = pd.concat(shap_rows, ignore_index=True)
            sh["model_version"] = float(registered_version)
            sh["ts"] = now_ts
            sh["row_id"] = [f"v{registered_version}_{i}" for i in range(len(sh))]
            store.insert("shap_importance", sh)

    summary = {"decision": decision, "registered_version": registered_version,
               "improved_vars": improved_vars, "reason": reason,
               "selected": {v: f"{c['model']} ({c['source']})" for v, c in components.items()}}
    run_log.system("run", "training", "success", decision, summary)
    return summary