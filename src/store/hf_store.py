"""Hugging Face Hub storage backend (free alternative to Hopsworks).

Feature Store  -> a PRIVATE dataset repo   (HF_DATASET_REPO, e.g. "your-name/agentweather-data")
                  fg/<feature_group>/<YYYY-MM>.parquet   (partitioned by month)
Model Registry -> a PRIVATE model repo     (HF_MODEL_REPO, e.g. "your-name/agentweather-models")
                  v<N>/bundle.json, v<N>/*.joblib, v<N>/_registry.json

Design
------
* Every Hub write is a git commit, so inserts are BUFFERED in memory and
  written in ONE commit per run by `commit()` (called automatically at the
  end of every pipeline via RunLog.flush).
* Optimistic concurrency: the commit is made against the commit we read
  (`parent_commit`). If another workflow committed in between, we re-read,
  merge again and retry - no rows are lost when jobs overlap.
* Monthly partitions keep each hourly commit small and reads fast.
Authentication: HF_TOKEN (a token with WRITE access).
"""
from __future__ import annotations

import io
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.store.base import BaseStore
from src.store.schemas import FEATURE_GROUPS, conform
from src.utils.config import env
from src.utils.logger import get_logger

log = get_logger(__name__)
FG_ROOT = "fg"


def normalize_repo_id(value: str) -> str:
    """Accept 'user/name', a full https://huggingface.co/... link, or values with stray spaces."""
    v = value.strip().strip('"').strip("'").strip()
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        if v.startswith(prefix):
            v = v[len(prefix):]
    for kind in ("datasets/", "models/"):
        if v.startswith(kind):
            v = v[len(kind):]
    return v.strip("/")


def _month(ts: int) -> str:
    return pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%Y-%m")


def _to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _from_path(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


class HuggingFaceStore(BaseStore):
    name = "huggingface"

    def __init__(self, api=None, hub=None, dataset_repo: str | None = None, model_repo: str | None = None,
                 token: str | None = None):
        if hub is None:
            import huggingface_hub as hub  # lazy: only needed for this backend
        self.hub = hub
        self.token = token or env("HF_TOKEN", required=True)
        self.api = api or hub.HfApi(token=self.token)
        self.dataset_repo = normalize_repo_id(dataset_repo or env("HF_DATASET_REPO", required=True))
        self.model_repo = normalize_repo_id(model_repo or env("HF_MODEL_REPO", required=True))
        log.info("Hugging Face repos: dataset=%s model=%s", self.dataset_repo, self.model_repo)
        for repo, kind in [(self.dataset_repo, "dataset"), (self.model_repo, "model")]:
            self.api.create_repo(repo_id=repo, repo_type=kind, private=True, exist_ok=True)
        self._pending: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
        self._cache: dict[str, pd.DataFrame] = {}
        self._files: list[str] | None = None
        self._head: str | None = None

    # ------------------------------------------------------------ helpers
    def _refresh_listing(self):
        info = self.api.repo_info(repo_id=self.dataset_repo, repo_type="dataset")
        self._head = info.sha
        self._files = self.api.list_repo_files(repo_id=self.dataset_repo, repo_type="dataset",
                                               revision=self._head)
        self._cache.clear()

    def _download(self, path: str, revision: str | None = None) -> pd.DataFrame:
        if revision is None and path in self._cache:
            return self._cache[path]
        local = self.hub.hf_hub_download(repo_id=self.dataset_repo, filename=path, repo_type="dataset",
                                         revision=revision or self._head, token=self.token)
        df = _from_path(local)
        if revision is None:
            self._cache[path] = df
        return df

    @staticmethod
    def _dedupe(df: pd.DataFrame, fg: str) -> pd.DataFrame:
        pk = FEATURE_GROUPS[fg]["pk"]
        return df.drop_duplicates(subset=pk, keep="last").sort_values("ts").reset_index(drop=True)

    # ------------------------------------------------------------ features
    def insert(self, fg_name, df):
        if df is None or df.empty:
            return
        df = conform(df)
        for month, part in df.groupby(df["ts"].map(_month)):
            self._pending[(fg_name, month)].append(part)

    def read(self, fg_name, start_ts=None, end_ts=None):
        if self._files is None:
            self._refresh_listing()
        lo = _month(start_ts) if start_ts is not None else "0000-00"
        hi = _month(end_ts) if end_ts is not None else "9999-99"
        prefix = f"{FG_ROOT}/{fg_name}/"
        frames = [self._download(f) for f in self._files
                  if f.startswith(prefix) and lo <= f[len(prefix):-8] <= hi]
        frames += [p for (fg, m), parts in self._pending.items() if fg == fg_name and lo <= m <= hi
                   for p in parts]
        if not frames:
            return pd.DataFrame()
        df = self._dedupe(pd.concat(frames, ignore_index=True), fg_name)
        if start_ts is not None:
            df = df[df["ts"] >= int(start_ts)]
        if end_ts is not None:
            df = df[df["ts"] <= int(end_ts)]
        return df.reset_index(drop=True)

    def commit(self, message: str = "pipeline update", max_attempts: int = 4) -> int:
        """Write all buffered rows in ONE commit. Returns the number of files written."""
        if not self._pending:
            return 0
        for attempt in range(1, max_attempts + 1):
            self._refresh_listing()
            ops = []
            for (fg, month), parts in self._pending.items():
                path = f"{FG_ROOT}/{fg}/{month}.parquet"
                frames = ([self._download(path)] if path in self._files else []) + parts
                merged = self._dedupe(pd.concat(frames, ignore_index=True), fg)
                ops.append(self.hub.CommitOperationAdd(path_in_repo=path, path_or_fileobj=_to_bytes(merged)))
            try:
                self.api.create_commit(repo_id=self.dataset_repo, repo_type="dataset", operations=ops,
                                       commit_message=message, parent_commit=self._head)
                n = len(ops)
                self._pending.clear()
                self._files = None  # force a fresh listing on the next read
                log.info("Committed %d file(s) to %s", n, self.dataset_repo)
                return n
            except Exception as exc:  # noqa: BLE001 - e.g. 412: repo changed since we read it
                log.warning("Commit attempt %d failed (%s); re-reading and retrying", attempt, exc)
                time.sleep(2 * attempt)
        raise RuntimeError(f"Could not commit to {self.dataset_repo} after {max_attempts} attempts")

    # ------------------------------------------------------------ models
    def list_models(self, model_name):
        files = self.api.list_repo_files(repo_id=self.model_repo, repo_type="model")
        out = []
        for f in files:
            if f.endswith("/_registry.json"):
                local = self.hub.hf_hub_download(repo_id=self.model_repo, filename=f, repo_type="model",
                                                 token=self.token)
                meta = json.loads(Path(local).read_text())
                if meta.get("model_name", model_name) == model_name:
                    out.append(meta)
        return sorted(out, key=lambda m: m["version"])

    def register_model(self, model_name, local_dir, metrics, description) -> int:
        v = max((m["version"] for m in self.list_models(model_name)), default=0) + 1
        (Path(local_dir) / "_registry.json").write_text(json.dumps(
            {"version": v, "model_name": model_name, "metrics": metrics, "description": description},
            indent=2, default=str))
        self.api.upload_folder(repo_id=self.model_repo, repo_type="model", folder_path=str(local_dir),
                               path_in_repo=f"v{v}", commit_message=f"{model_name} v{v}")
        return v

    def latest_model(self, model_name):
        models = self.list_models(model_name)
        if not models:
            return None
        v = models[-1]["version"]
        root = self.hub.snapshot_download(repo_id=self.model_repo, repo_type="model",
                                          allow_patterns=[f"v{v}/*"], token=self.token)
        return Path(root) / f"v{v}", v