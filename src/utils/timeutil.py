"""Time helpers. Everything internal is UTC; `ts` = epoch seconds at the top of the hour."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def floor_hour(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").floor("h")


def to_ts(t) -> int:
    return int(floor_hour(t).timestamp())


def from_ts(ts) -> pd.Timestamp:
    return pd.Timestamp(int(ts), unit="s", tz="UTC")


def ts_series_to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("int64"), unit="s", utc=True)
