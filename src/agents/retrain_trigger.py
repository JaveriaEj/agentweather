"""Reliable retraining trigger + team notification via the GitHub REST API.

Inside GitHub Actions, GITHUB_TOKEN and GITHUB_REPOSITORY are provided
automatically. A `workflow_dispatch` created with GITHUB_TOKEN is one of
the few events that IS allowed to start another workflow run, so no
personal access token is needed (the workflow needs `actions: write`).
"""
from __future__ import annotations

import requests

from src.utils.config import env

API = "https://api.github.com"


def _headers():
    token = env("GH_DISPATCH_TOKEN") or env("GITHUB_TOKEN")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def dispatch_training(reason: str, drift_event_id: str, http=requests) -> tuple[bool, str]:
    headers, repo = _headers(), env("GITHUB_REPOSITORY")
    if not headers or not repo:
        return False, "GITHUB_TOKEN/GITHUB_REPOSITORY not set (local run) - run `python -m scripts.run_training --reason drift` manually"
    ref = env("GITHUB_REF_NAME", "main")
    try:
        r = http.post(f"{API}/repos/{repo}/actions/workflows/train.yml/dispatches", headers=headers,
                      json={"ref": ref, "inputs": {"reason": reason, "drift_event_id": drift_event_id}},
                      timeout=20)
        return (r.status_code == 204), f"HTTP {r.status_code} {r.text[:200]}"
    except requests.RequestException as exc:
        return False, str(exc)


def notify_team(title: str, body: str, http=requests) -> tuple[bool, str]:
    """Open a GitHub issue labelled 'drift' - repo watchers get notified."""
    headers, repo = _headers(), env("GITHUB_REPOSITORY")
    if not headers or not repo:
        return False, "no GitHub credentials (local run)"
    try:
        r = http.post(f"{API}/repos/{repo}/issues", headers=headers,
                      json={"title": title, "body": body, "labels": ["drift", "automated"]}, timeout=20)
        return (r.status_code == 201), f"HTTP {r.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)
