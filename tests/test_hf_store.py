"""Hugging Face backend tested against an in-memory fake Hub (no network, no token)."""
import fnmatch
import io
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.store import hf_store
from src.store.hf_store import HuggingFaceStore


class FakeOp:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo, self.data = path_in_repo, path_or_fileobj


class FakeHub:
    """Mimics the parts of huggingface_hub the store uses."""

    CommitOperationAdd = FakeOp

    def __init__(self):
        self.repos: dict[str, dict] = {}
        self.tmp = Path(tempfile.mkdtemp())
        self.conflict_once = False

    # --- HfApi methods
    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.repos.setdefault(repo_id, {"files": {}, "sha": 0})

    def repo_info(self, repo_id, repo_type):
        return SimpleNamespace(sha=self.repos[repo_id]["sha"])

    def list_repo_files(self, repo_id, repo_type, revision=None):
        return list(self.repos[repo_id]["files"])

    def create_commit(self, repo_id, repo_type, operations, commit_message, parent_commit):
        repo = self.repos[repo_id]
        if self.conflict_once:          # simulate another workflow committing first
            self.conflict_once = False
            repo["files"]["fg/alerts/2000-01.parquet"] = pickle.dumps(pd.DataFrame(
                {"alert_id": ["other"], "ts": [946684800]}))
            repo["sha"] += 1
        if parent_commit != repo["sha"]:
            raise RuntimeError("412 Precondition Failed")
        for op in operations:
            repo["files"][op.path_in_repo] = op.data
        repo["sha"] += 1

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        for p in Path(folder_path).rglob("*"):
            if p.is_file():
                self.repos[repo_id]["files"][f"{path_in_repo}/{p.relative_to(folder_path).as_posix()}"] = p.read_bytes()
        self.repos[repo_id]["sha"] += 1

    # --- module-level functions
    def hf_hub_download(self, repo_id, filename, repo_type, token, revision=None):
        out = self.tmp / repo_id.replace("/", "_") / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.repos[repo_id]["files"][filename])
        return str(out)

    def snapshot_download(self, repo_id, repo_type, allow_patterns, token):
        root = self.tmp / f"snap_{repo_id.replace('/', '_')}"
        for f, data in self.repos[repo_id]["files"].items():
            if any(fnmatch.fnmatch(f, pat) for pat in allow_patterns):
                (root / f).parent.mkdir(parents=True, exist_ok=True)
                (root / f).write_bytes(data)
        return str(root)


@pytest.fixture
def hf(monkeypatch):
    # parquet needs pyarrow; the fake uses pickle so the test runs anywhere
    monkeypatch.setattr(hf_store, "_to_bytes", lambda df: pickle.dumps(df))
    monkeypatch.setattr(hf_store, "_from_path", lambda p: pickle.loads(Path(p).read_bytes()))
    hub = FakeHub()
    return hub, HuggingFaceStore(api=hub, hub=hub, dataset_repo="me/data", model_repo="me/models", token="x")


def test_buffered_writes_one_commit_and_monthly_partitions(hf):
    hub, s = hf
    df = pd.DataFrame({"ts": [1793491200, 1793494800, 1796083200], "temp_c": [20.0, 21.0, 22.0]})  # Nov + Dec 2026
    s.insert("hourly_features", df)
    assert hub.repos["me/data"]["files"] == {}               # nothing written before commit
    assert len(s.read("hourly_features")) == 3               # but readable from the buffer
    assert s.commit() == 2                                   # two monthly files, one commit
    assert hub.repos["me/data"]["sha"] == 1
    s.insert("hourly_features", pd.DataFrame({"ts": [1793491200], "temp_c": [25.0]}))  # upsert
    s.commit()
    out = s.read("hourly_features")
    assert len(out) == 3 and out.loc[out.ts == 1793491200, "temp_c"].item() == 25.0
    assert len(s.read("hourly_features", 1796000000, None)) == 1


def test_concurrent_commit_is_retried_without_losing_rows(hf):
    hub, s = hf
    s.insert("alerts", pd.DataFrame({"alert_id": ["mine"], "ts": [946684800]}))
    hub.conflict_once = True
    s.commit()
    ids = set(s.read("alerts")["alert_id"])
    assert ids == {"mine", "other"}


def test_model_registry_versions(hf, tmp_path):
    hub, s = hf
    for i in range(2):
        d = tmp_path / f"m{i}"
        d.mkdir()
        (d / "bundle.json").write_text("{}")
        assert s.register_model("weather_aqi_bundle", d, {"rmse": 1.0 - i}, "test") == i + 1
    path, v = s.latest_model("weather_aqi_bundle")
    assert v == 2 and (path / "bundle.json").exists()
    assert [m["version"] for m in s.list_models("weather_aqi_bundle")] == [1, 2]
