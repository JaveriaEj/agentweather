"""Hopsworks Feature Store + Model Registry implementation.

Login uses HOPSWORKS_API_KEY and HOPSWORKS_PROJECT (env vars / secrets).
Each feature group is created on first insert (version 1) with `ts` / ids
as primary key and `event_time` as event time. Inserts are upserts.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.store.base import BaseStore
from src.store.schemas import FEATURE_GROUPS, conform
from src.utils.config import env
from src.utils.logger import get_logger

log = get_logger(__name__)
FG_VERSION = 1


class HopsworksStore(BaseStore):
    name = "hopsworks"

    def __init__(self, wait_for_jobs: bool = False):
        import hopsworks  # imported lazily: heavy dependency

        self.project = hopsworks.login(
            api_key_value=env("HOPSWORKS_API_KEY", required=True),
            project=env("HOPSWORKS_PROJECT", required=True),
            host=env("HOPSWORKS_HOST", "c.app.hopsworks.ai"),
        )
        self.fs = self.project.get_feature_store()
        self._mr = None
        self.wait = wait_for_jobs
        self._fg_cache: dict = {}

    @property
    def mr(self):
        if self._mr is None:
            self._mr = self.project.get_model_registry()
        return self._mr

    # ------------------------------------------------------------ features
    def _get_fg(self, name: str, create_df: pd.DataFrame | None = None):
        if name in self._fg_cache:
            return self._fg_cache[name]
        meta = FEATURE_GROUPS[name]
        fg = None
        try:
            fg = self.fs.get_feature_group(name, version=FG_VERSION)
        except Exception:  # noqa: BLE001 - not found raises in some client versions
            fg = None
        if fg is None and create_df is not None:
            fg = self.fs.get_or_create_feature_group(
                name=name, version=FG_VERSION, primary_key=meta["pk"],
                event_time="event_time", description=meta["desc"], online_enabled=False)
        if fg is not None:
            self._fg_cache[name] = fg
        return fg

    @staticmethod
    def _align(fg, df: pd.DataFrame) -> pd.DataFrame:
        """Match the existing feature-group schema: add missing columns with
        empty defaults, drop unknown ones (logged), keep the stored order."""
        schema = {f.name: f.type.lower() for f in fg.features}
        extra = [c for c in df.columns if c not in schema]
        if extra:
            log.warning("%s: dropping columns not in schema %s (bump FG version to add them)", fg.name, extra)
        for name, typ in schema.items():
            if name not in df:
                df[name] = "" if typ == "string" else (0 if typ in ("bigint", "int") else float("nan"))
        return df[list(schema)]

    def insert(self, fg_name, df):
        if df is None or df.empty:
            return
        df = conform(df)
        fg = self._get_fg(fg_name, create_df=df)
        if getattr(fg, "features", None):
            df = self._align(fg, df)
        fg.insert(df, write_options={"wait_for_job": self.wait})
        log.info("Inserted %d rows into %s", len(df), fg_name)

    def read(self, fg_name, start_ts=None, end_ts=None):
        fg = self._get_fg(fg_name)
        if fg is None:
            return pd.DataFrame()
        try:
            q = fg.select_all()
            if start_ts is not None:
                q = q.filter(fg.ts >= int(start_ts))
            if end_ts is not None:
                q = q.filter(fg.ts <= int(end_ts))
            df = q.read()
        except Exception as exc:  # noqa: BLE001 - e.g. group exists but has no data yet
            log.warning("Read of %s failed: %s", fg_name, exc)
            return pd.DataFrame()
        if df.empty:
            return df
        pk = FEATURE_GROUPS[fg_name]["pk"]
        return df.drop_duplicates(subset=pk, keep="last").sort_values("ts").reset_index(drop=True)

    # ------------------------------------------------------------ models
    def register_model(self, model_name, local_dir, metrics, description) -> int:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        model = self.mr.python.create_model(name=model_name, metrics=numeric, description=description[:1000])
        model.save(str(local_dir))
        return int(model.version)

    def latest_model(self, model_name):
        try:
            models = self.mr.get_models(model_name)
        except Exception:  # noqa: BLE001
            return None
        if not models:
            return None
        newest = max(models, key=lambda m: m.version)
        return Path(newest.download()), int(newest.version)

    def list_models(self, model_name):
        try:
            models = self.mr.get_models(model_name)
        except Exception:  # noqa: BLE001
            return []
        return sorted(
            [{"version": int(m.version), "metrics": dict(m.training_metrics or {}),
              "description": m.description} for m in models],
            key=lambda m: m["version"])
