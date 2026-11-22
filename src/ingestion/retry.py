"""HTTP GET with retries, exponential backoff and failure logging.

Retried:   connection errors, timeouts, HTTP 429 (rate limit), HTTP 5xx.
Not retried: other 4xx (bad key, bad request) - retrying would not help.
"""
from __future__ import annotations

import re
import time
from typing import Any

import requests

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)
_SECRET_PARAMS = {"appid", "token", "api_key", "apikey"}


class ApiError(RuntimeError):
    def __init__(self, source: str, message: str, status: int | None = None):
        super().__init__(f"[{source}] {message}")
        self.source = source
        self.status = status


def _mask(url: str) -> str:
    return re.sub(r"(appid|token)=[^&]+", r"\1=***", url)


def get_json(
    url: str,
    params: dict[str, Any],
    source: str,
    run_log=None,
    max_retries: int | None = None,
    session: requests.Session | None = None,
    sleep=time.sleep,
) -> Any:
    cfg = settings()["api"]
    max_retries = cfg["max_retries"] if max_retries is None else max_retries
    http = session or requests
    safe_params = {k: ("***" if k in _SECRET_PARAMS else v) for k, v in params.items()}
    last_err = "unknown error"
    last_status = None
    attempt = 0
    for attempt in range(1, max_retries + 1):
        try:
            resp = http.get(url, params=params, timeout=cfg["timeout_s"])
            if resp.status_code == 200:
                return resp.json()
            last_status = resp.status_code
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = float(resp.headers.get("Retry-After", cfg["backoff_base_s"] ** attempt))
                log.warning("%s rate-limited/server error (attempt %d): %s", source, attempt, last_err)
                sleep(min(wait, 60))
                continue
            # Non-retryable client error (bad key, wrong subscription, ...)
            break
        except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            log.warning("%s request failed (attempt %d): %s", source, attempt, last_err)
            sleep(min(cfg["backoff_base_s"] ** attempt, 60))
    if run_log is not None:
        run_log.api_failure(source=source, endpoint=_mask(url), params=safe_params,
                            error=last_err, attempts=attempt)
    raise ApiError(source, last_err, status=last_status)
