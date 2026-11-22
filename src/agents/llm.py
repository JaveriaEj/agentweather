"""LLM client (Groq, OpenAI-compatible REST API) with full call logging.

Every call is recorded (agent, prompt, response, latency, status).
Failures are ALSO written to the separate failure log. The client never
raises: callers receive None and use their deterministic fallback.
"""
from __future__ import annotations

import time

import requests

from src.utils.config import env, settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self, run_log, session: requests.Session | None = None):
        cfg = settings()["llm"]
        self.cfg = cfg
        self.model = env("GROQ_MODEL", cfg["model"])
        self.api_key = env("GROQ_API_KEY")
        self.run_log = run_log
        self.http = session or requests

    def chat(self, agent: str, system: str, user: str, max_tokens: int = 700) -> str | None:
        prompt_log = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        if not self.api_key:
            self.run_log.llm_call(agent, self.model, prompt_log, "", 0.0, "failed", "GROQ_API_KEY not set")
            return None
        payload = {"model": self.model, "temperature": self.cfg["temperature"], "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if "gpt-oss" in self.model:  # reasoning model: keep thinking short, leave room for the answer
            payload["reasoning_effort"] = "low"
            payload["max_tokens"] = max_tokens + 1000
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        err = ""
        for attempt in range(1, self.cfg["max_retries"] + 2):
            t0 = time.time()
            try:
                r = self.http.post(self.cfg["base_url"], json=payload, headers=headers,
                                   timeout=self.cfg["timeout_s"])
                latency = time.time() - t0
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"].strip()
                    self.run_log.llm_call(agent, self.model, prompt_log, text, latency, "success")
                    return text
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                self.run_log.llm_call(agent, self.model, prompt_log, "", latency, "failed", err)
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
            except (requests.RequestException, KeyError, ValueError) as exc:
                err = f"{type(exc).__name__}: {exc}"
                self.run_log.llm_call(agent, self.model, prompt_log, "", time.time() - t0, "failed", err)
            time.sleep(2 ** attempt)
        log.warning("LLM unavailable for %s: %s -> using template fallback", agent, err)
        return None
