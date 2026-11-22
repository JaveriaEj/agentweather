"""EDA computations shared by the EDA report script and the dashboard.

All findings are COMPUTED from the stored data - nothing is hardcoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import settings

EDA_VARS = ["aqi", "temp_c", "precip_mm", "humidity_pct", "uvi", "wind_speed", "pm2_5", "pressure_hpa"]
LABELS = {"aqi": "AQI", "temp_c": "Temperature (°C)", "precip_mm": "Precipitation (mm/h)",
          "humidity_pct": "Humidity (%)", "uvi": "UV Index", "wind_speed": "Wind speed (m/s)",
          "pm2_5": "PM2.5 (µg/m³)", "pressure_hpa": "Pressure (hPa)"}


def prepare(features: pd.DataFrame) -> pd.DataFrame:
    tz = settings()["city"]["timezone"]
    df = features.sort_values("ts").copy()
    df["time_local"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(tz)
    df["local_hour"] = df["time_local"].dt.hour
    df["local_date"] = df["time_local"].dt.date
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EDA_VARS if c in df]
    s = df[cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    s["missing_pct"] = df[cols].isna().mean() * 100
    s["skew"] = df[cols].skew()
    return s.round(3)


def outliers_iqr(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    rows = []
    for c in [c for c in EDA_VARS if c in df]:
        x = df[c].dropna()
        if x.empty:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n = int(((x < lo) | (x > hi)).sum())
        rows.append({"variable": c, "lower": lo, "upper": hi, "n_outliers": n,
                     "pct": 100 * n / len(x)})
    return pd.DataFrame(rows).round(3)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EDA_VARS if c in df]
    out = pd.DataFrame({"missing_after_imputation_pct": df[cols].isna().mean() * 100})
    if "n_imputed" in df:
        out.loc["(rows with any imputed value)", "missing_after_imputation_pct"] = np.nan
        out["note"] = ""
        out.loc["(rows with any imputed value)", "note"] = f"{(df['n_imputed'] > 0).mean() * 100:.2f}% of hours"
    return out


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EDA_VARS if c in df]
    return df.groupby("local_hour")[cols].mean()


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EDA_VARS if c in df]
    agg = {c: "mean" for c in cols}
    if "precip_mm" in agg:
        agg["precip_mm"] = "sum"
    return df.groupby("local_date").agg(agg)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in EDA_VARS if c in df]].corr().round(2)


def findings(df: pd.DataFrame) -> list[str]:
    out = []
    n_days = (df["ts"].max() - df["ts"].min()) / 86400
    out.append(f"Dataset covers {len(df):,} hourly records (~{n_days:.0f} days), "
               f"{df['time_local'].min():%Y-%m-%d} to {df['time_local'].max():%Y-%m-%d}.")
    prof = hourly_profile(df)
    if "aqi" in df and df["aqi"].notna().any():
        out.append(f"AQI averages {df['aqi'].mean():.0f}; it is worst around {int(prof['aqi'].idxmax()):02d}:00 "
                   f"and best around {int(prof['aqi'].idxmin()):02d}:00 local time. "
                   f"{(df['aqi'] > 100).mean() * 100:.1f}% of hours exceed AQI 100 (unhealthy for sensitive groups).")
    if "temp_c" in df:
        out.append(f"Temperature ranges {df['temp_c'].min():.1f}-{df['temp_c'].max():.1f}°C; the daily peak "
                   f"is typically around {int(prof['temp_c'].idxmax()):02d}:00, average daily swing "
                   f"{(df.groupby('local_date')['temp_c'].agg(lambda x: x.max() - x.min())).mean():.1f}°C.")
    if "precip_mm" in df:
        rainy = (df["precip_mm"] > settings()["forecast"]["rain_threshold_mm"]).mean() * 100
        out.append(f"Rain is recorded in {rainy:.1f}% of hours - "
                   + ("a strongly imbalanced target, so rain probability uses a classifier with "
                      "climatology fallback." if rainy < 10 else "enough rainy hours to model directly."))
    corr = correlations(df)
    if {"temp_c", "humidity_pct"} <= set(corr.columns):
        out.append(f"Temperature and humidity correlate at r={corr.loc['temp_c', 'humidity_pct']:.2f}.")
    if {"aqi", "wind_speed"} <= set(corr.columns):
        out.append(f"AQI vs wind speed correlation r={corr.loc['aqi', 'wind_speed']:.2f} "
                   "(negative = wind disperses pollution).")
    ol = outliers_iqr(df)
    if len(ol):
        top = ol.sort_values("pct", ascending=False).iloc[0]
        out.append(f"Most extreme values (3×IQR rule) are in {top['variable']} "
                   f"({int(top['n_outliers'])} hours, {top['pct']:.1f}%); they are kept (real events), not removed.")
    return out
