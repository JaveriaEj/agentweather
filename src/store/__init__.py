"""`get_store()` picks the storage backend.

STORE_BACKEND = huggingface | hopsworks | local.
If STORE_BACKEND is not set: Hugging Face when HF_TOKEN is set, else Hopsworks
when HOPSWORKS_API_KEY is set, else local files (development/tests only).
"""
from __future__ import annotations

from src.utils.config import env

_STORE = None


def get_store(force: str | None = None):
    global _STORE  # noqa: PLW0603
    backend = force or env("STORE_BACKEND") or (
        "huggingface" if env("HF_TOKEN") else "hopsworks" if env("HOPSWORKS_API_KEY") else "local")
    if _STORE is not None and _STORE.name == backend and force is None:
        return _STORE
    if backend == "huggingface":
        from src.store.hf_store import HuggingFaceStore

        _STORE = HuggingFaceStore()
    elif backend == "hopsworks":
        from src.store.hopsworks_store import HopsworksStore

        _STORE = HopsworksStore()
    else:
        from src.store.local_store import LocalStore

        _STORE = LocalStore()
    return _STORE


def set_store(store) -> None:
    """Used by tests to inject a store."""
    global _STORE  # noqa: PLW0603
    _STORE = store
