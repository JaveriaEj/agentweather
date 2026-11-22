"""File-based store with the same interface as Hopsworks.

Used ONLY for local development and the automated tests (no account
needed). Production runs (GitHub Actions + dashboard) use Hopsworks.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.store.base import BaseStore
from src.store.schemas import FEATURE_GROUPS, conform
from src.utils.config import ROOT


class LocalStore(BaseStore):
    name = "local"

    def __init__(self, root: Path | None = None):
        self.root = Path(root or ROOT / "data" / "local_store")
        (self.root / "fg").mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(parents=True, exist_ok=True)

    def _path(self, fg: str) -> Path:
        return self.root / "fg" / f"{fg}.pkl"

    def insert(self, fg_name: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        pk = FEATURE_GROUPS[fg_name]["pk"]
        new = conform(df)
        p = self._path(fg_name)
        if p.exists():
            old = pd.read_pickle(p)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=pk, keep="last")
        new.sort_values("ts").reset_index(drop=True).to_pickle(p)

    def read(self, fg_name, start_ts=None, end_ts=None) -> pd.DataFrame:
        p = self._path(fg_name)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_pickle(p)
        if start_ts is not None:
            df = df[df["ts"] >= int(start_ts)]
        if end_ts is not None:
            df = df[df["ts"] <= int(end_ts)]
        return df.reset_index(drop=True)

    def _model_root(self, name: str) -> Path:
        return self.root / "models" / name

    def register_model(self, model_name, local_dir, metrics, description) -> int:
        versions = [m["version"] for m in self.list_models(model_name)]
        v = max(versions, default=0) + 1
        dest = self._model_root(model_name) / f"v{v}"
        shutil.copytree(local_dir, dest)
        (dest / "_registry.json").write_text(json.dumps(
            {"version": v, "metrics": metrics, "description": description}, indent=2))
        return v

    def latest_model(self, model_name):
        models = self.list_models(model_name)
        if not models:
            return None
        v = max(m["version"] for m in models)
        return self._model_root(model_name) / f"v{v}", v

    def list_models(self, model_name):
        root = self._model_root(model_name)
        out = []
        if root.exists():
            for d in root.iterdir():
                meta = d / "_registry.json"
                if meta.exists():
                    out.append(json.loads(meta.read_text()))
        return sorted(out, key=lambda m: m["version"])
