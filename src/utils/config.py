"""Loads YAML settings + environment variables (API keys) in one place."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

try:  # .env is optional (used locally; GitHub Actions uses Secrets)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def settings() -> dict:
    return load_yaml("settings.yaml")


def thresholds() -> dict:
    return load_yaml("thresholds.yaml")


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Read an environment variable. Also checks Streamlit secrets if available."""
    val = os.getenv(name)
    if val in (None, ""):
        try:  # Streamlit Community Cloud stores secrets in st.secrets
            import streamlit as st  # noqa: WPS433

            val = st.secrets.get(name)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            val = None
    if val in (None, ""):
        val = default
    if required and not val:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Add it to your .env file (local) or GitHub Secrets (CI)."
        )
    return val


def target_vars() -> list[str]:
    return list(settings()["forecast"]["target_vars"])


def horizon() -> int:
    return int(settings()["forecast"]["horizon_hours"])
