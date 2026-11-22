"""Storage interface. Production = Hopsworks; local = pickle files (dev/tests only)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseStore(ABC):
    name = "base"

    @abstractmethod
    def insert(self, fg_name: str, df: pd.DataFrame) -> None:
        """Upsert rows (by primary key)."""

    @abstractmethod
    def read(self, fg_name: str, start_ts: int | None = None, end_ts: int | None = None) -> pd.DataFrame:
        """Read rows with start_ts <= ts <= end_ts. Empty frame if the group doesn't exist."""

    @abstractmethod
    def register_model(self, model_name: str, local_dir: Path, metrics: dict, description: str) -> int:
        """Upload a model folder; return the new version number."""

    @abstractmethod
    def latest_model(self, model_name: str) -> tuple[Path, int] | None:
        """Download the newest (= production) model version. None if none exists."""

    @abstractmethod
    def list_models(self, model_name: str) -> list[dict]:
        """All versions with their metrics."""

    def commit(self, message: str = "pipeline update") -> int:
        """Persist buffered writes. Hopsworks/local write immediately, so this is a no-op;
        the Hugging Face backend writes everything here in one commit."""
        return 0