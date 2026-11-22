import numpy as np

from src.models.base import NaiveBaseline
from src.models.bundle import ModelBundle
from src.models.dataset import load_model_data
from src.models.fuse_outputs import OUTPUT_COLUMNS
from src.models.train import run_training


def test_time_split_has_gap_and_no_overlap(filled_store):
    d = load_model_data(filled_store)
    gap = d.frame.loc[d.val_idx[0], "ts"] - d.frame.loc[d.train_idx[-1], "ts"]
    assert gap >= 24 * 3600
    assert d.train_idx.max() < d.val_idx.min()


def test_naive_baseline_uses_value_24h_ago(filled_store):
    d = load_model_data(filled_store)
    i = int(d.val_idx[5])
    pred = NaiveBaseline("temp_c").predict(d.frame, np.array([i]))[0]
    # forecast for t+h = observation at t+h-24
    assert pred[0] == d.frame.loc[i - 23, "temp_c"] and pred[23] == d.frame.loc[i, "temp_c"]


def test_training_registers_then_keeps_better_production(filled_store):
    s = filled_store
    first = run_training(s, families={"classical"})
    assert first["decision"] == "registered_new_version" and first["registered_version"] == 1
    second = run_training(s, families={"classical"})
    assert second["decision"] == "kept_production"            # identical candidates are not "better"
    assert len(s.list_models("weather_aqi_bundle")) == 1

    ev = s.read("model_eval_history")
    assert {"naive_24h", "ridge", "random_forest", "logistic"} <= set(ev["model"])
    sel = ev[(ev["selected"] == 1) & (ev["family"] != "baseline")]
    assert (sel["rmse"] < sel["baseline_rmse"]).all()        # only models that beat the baseline
    assert len(s.read("shap_importance")) > 0


def test_bundle_forecast_has_all_six_metric_groups(filled_store):
    run_training(filled_store, families={"classical"})
    path, v = filled_store.latest_model("weather_aqi_bundle")
    b = ModelBundle.load(path, version=v)
    d = load_model_data(filled_store)
    fc = b.forecast(d.frame, int(d.val_idx[-1]))
    assert len(fc) == 24 and set(OUTPUT_COLUMNS) <= set(fc.columns)
    assert fc[OUTPUT_COLUMNS].notna().all().all()
    assert (fc["rain_prob_pct"].between(0, 100)).all() and (fc["humidity_pct"].between(0, 100)).all()
    assert (fc["wind_dir_deg"].between(0, 360)).all() and (fc["uvi"] >= 0).all()
