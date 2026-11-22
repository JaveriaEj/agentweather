import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.logs import RunLog  # noqa: E402
from src.processing.pipeline import process_window  # noqa: E402
from src.store import set_store  # noqa: E402
from src.store.local_store import LocalStore  # noqa: E402
from tests.synthetic import make_raw  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for k in ["GROQ_API_KEY", "GITHUB_TOKEN", "GH_DISPATCH_TOKEN", "GITHUB_REPOSITORY",
              "HOPSWORKS_API_KEY", "OPENWEATHER_API_KEY", "AQICN_TOKEN"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("STORE_BACKEND", "local")


@pytest.fixture
def store(tmp_path):
    s = LocalStore(tmp_path / "store")
    set_store(s)
    return s


@pytest.fixture
def filled_store(store):
    """Store with 45 days of SYNTHETIC raw data processed into features/targets."""
    w, a = make_raw(45)
    store.insert("raw_weather", w)
    store.insert("raw_air_quality", a)
    process_window(store, int(w.ts.min()), int(w.ts.max()), RunLog("test"))
    return store
