import numpy as np
import pandas as pd

from src.processing.aqi_calc import compute_aqi_frame, sub_index
from src.processing.features import build_features, build_targets, complete_target_mask, feature_columns
from src.processing.fuse import fuse_sources, uv_to_wind, wind_to_uv
from src.processing.validate import validate_frame
from tests.synthetic import make_raw


def test_aqi_breakpoints():
    assert sub_index(0.0, "pm2_5") == 0
    assert sub_index(9.0, "pm2_5") == 50
    assert sub_index(35.4, "pm2_5") == 100
    assert sub_index(55.4, "pm2_5") == 150
    assert sub_index(154, "pm10") == 100
    assert sub_index(1000, "pm2_5") == 500
    assert np.isnan(sub_index(np.nan, "pm2_5"))


def test_overall_aqi_is_max_of_pollutants():
    df = pd.DataFrame({"pm2_5": [35.4], "pm10": [300.0]})
    assert compute_aqi_frame(df).iloc[0] == sub_index(300.0, "pm10")


def test_validation_sets_out_of_range_to_nan():
    df = pd.DataFrame({"ts": [0, 3600], "humidity_pct": [50, 140], "temp_c": [20, -99]})
    clean, invalid = validate_frame(df)
    assert clean["humidity_pct"].isna().sum() == 1 and invalid == {"humidity_pct": 1, "temp_c": 1}


def test_wind_roundtrip():
    spd, deg = pd.Series([5.0, 3.0, 7.0]), pd.Series([0.0, 90.0, 225.0])
    u, v = wind_to_uv(spd, deg)
    s2, d2 = uv_to_wind(u.to_numpy(), v.to_numpy())
    assert np.allclose(s2, spd) and np.allclose(d2, deg)


def test_fusion_one_row_per_hour_and_gap_imputation():
    w, a = make_raw(5)
    w = w.drop(index=[10, 11, 50, 51, 52, 53, 54])  # 2h gap (filled) and 5h gap (partly NaN)
    fused, rep = fuse_sources(w, a)
    assert fused["ts"].is_unique and (np.diff(fused["ts"]) == 3600).all()
    assert fused.loc[[10, 11], "temp_c"].notna().all()       # within ffill limit (3h)
    assert fused.loc[[53, 54], "temp_c"].isna().all()        # beyond the limit -> stays missing
    assert rep["imputed_cells"] > 0


def test_features_have_no_future_leakage():
    w, a = make_raw(6)
    fused, _ = fuse_sources(w, a)
    f1 = build_features(fused)
    t = 100
    fused2 = fused.copy()
    num = fused2.select_dtypes("number").columns.drop("ts")
    fused2.loc[t + 1:, num] = fused2.loc[t + 1:, num] * 3 + 50  # change ONLY the future
    f2 = build_features(fused2)
    cols = feature_columns()
    pd.testing.assert_frame_equal(f1.loc[:t, cols], f2.loc[:t, cols])


def test_targets_are_future_values_and_newest_24h_are_forecast_only():
    w, a = make_raw(4)
    fused, _ = fuse_sources(w, a)
    tg = build_targets(fused)
    assert tg.loc[10, "temp_c_h01"] == fused.loc[11, "temp_c"]
    assert tg.loc[10, "temp_c_h24"] == fused.loc[34, "temp_c"]
    mask = complete_target_mask(tg)
    assert not mask.iloc[-24:].any() and mask.iloc[:-24].all()
