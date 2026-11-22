"""Agent 3 - Weather Report Agent.

Runs once a day (first run at/after the configured local hour) and
after every NEW alert (levels listed in thresholds.yaml `significant_levels`). Writes a news-style report with
the LLM; falls back to a deterministic template if the LLM fails.
All numbers given to the LLM come from the forecast - it is told not to
invent any.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.monitoring.logs import new_id
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.timeutil import utcnow

log = get_logger(__name__)
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def aqi_category(aqi: float) -> str:
    for lim, name in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for Sensitive Groups"),
                      (200, "Unhealthy"), (300, "Very Unhealthy")]:
        if aqi <= lim:
            return name
    return "Hazardous"


def compass(deg: float) -> str:
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def report_decision(state: dict) -> str:
    if not state.get("forecast"):
        return "none"
    if state.get("new_significant_alerts"):
        return "significant_alert"
    cfg = settings()
    tz = cfg["city"]["timezone"]
    now_local = utcnow().tz_convert(tz)
    last = state.get("ledger", {}).get("last_report_ts", 0)
    last_local_date = pd.Timestamp(last, unit="s", tz="UTC").tz_convert(tz).date() if last else None
    if now_local.hour >= cfg["report"]["daily_report_hour_local"] and last_local_date != now_local.date():
        return "daily"
    return "none"


def summarize(state: dict) -> dict:
    tz = settings()["city"]["timezone"]
    fc = pd.DataFrame(state["forecast"])
    t = pd.to_datetime(fc["target_ts"], unit="s", utc=True).dt.tz_convert(tz).dt.strftime("%H:%M")
    s = {
        "city": settings()["city"]["name"],
        "period": f"next 24 hours from {t.iloc[0]} local time",
        "temp_min_c": round(fc["temp_c"].min(), 1), "temp_min_at": t[fc["temp_c"].idxmin()],
        "temp_max_c": round(fc["temp_c"].max(), 1), "temp_max_at": t[fc["temp_c"].idxmax()],
        "total_precip_mm": round(fc["precip_mm"].sum(), 1),
        "max_rain_prob_pct": round(fc["rain_prob_pct"].max(), 0),
        "aqi_max": round(fc["aqi"].max(), 0), "aqi_max_category": aqi_category(fc["aqi"].max()),
        "aqi_mean": round(fc["aqi"].mean(), 0),
        "humidity_range_pct": [round(fc["humidity_pct"].min(), 0), round(fc["humidity_pct"].max(), 0)],
        "uv_max": round(fc["uvi"].max(), 1), "uv_max_at": t[fc["uvi"].idxmax()],
        "wind_max_ms": round(fc["wind_speed"].max(), 1),
        "wind_prevailing_dir": compass(float(np.degrees(np.arctan2(
            np.sin(np.radians(fc["wind_dir_deg"])).mean(), np.cos(np.radians(fc["wind_dir_deg"])).mean())))),
    }
    recent = pd.DataFrame(state.get("recent_obs") or [])
    changes = []
    if len(recent):
        if "temp_c" in recent and recent["temp_c"].notna().any():
            d = s["temp_max_c"] - recent["temp_c"].max()
            if abs(d) >= 2:
                changes.append(f"daytime high {'up' if d > 0 else 'down'} {abs(d):.1f}°C vs the last 24h")
        if "aqi" in recent and recent["aqi"].notna().any():
            d = s["aqi_mean"] - recent["aqi"].mean()
            if abs(d) >= 15:
                changes.append(f"average AQI {'worse' if d > 0 else 'better'} by {abs(d):.0f} points vs the last 24h")
    s["changes"] = changes
    s["alerts"] = [f"{a['label']} ({a['level']}) {a['time_window']}, peak {a['peak_value']}{a['unit']}"
                   for a in state.get("alerts", [])]
    return s


def template_report(s: dict) -> str:
    lines = [
        f"{s['city']} weather, {s['period']}.",
        f"Temperatures range from {s['temp_min_c']}°C (around {s['temp_min_at']}) to "
        f"{s['temp_max_c']}°C (around {s['temp_max_at']}). Humidity {s['humidity_range_pct'][0]:.0f}-"
        f"{s['humidity_range_pct'][1]:.0f}%.",
        (f"Rain: about {s['total_precip_mm']} mm in total, with up to {s['max_rain_prob_pct']:.0f}% chance."
         if s["total_precip_mm"] > 0.2 or s["max_rain_prob_pct"] >= 30 else "Little or no rain expected."),
        f"Air quality peaks near AQI {s['aqi_max']:.0f} ({s['aqi_max_category']}).",
        f"UV index up to {s['uv_max']} around {s['uv_max_at']}. Winds mostly from the "
        f"{s['wind_prevailing_dir']}, up to {s['wind_max_ms']} m/s.",
    ]
    if s["changes"]:
        lines.append("Notable changes: " + "; ".join(s["changes"]) + ".")
    lines.append("Active alerts: " + ("; ".join(s["alerts"]) if s["alerts"] else "none") + ".")
    return " ".join(lines)


def report_node(ctx, state: dict) -> dict:
    with ctx.run_log.timed_agent("weather_report_agent") as details:
        reason = report_decision(state)
        details["reason"] = reason
        s = summarize(state)
        system = ("You are a local TV weather presenter. Write a concise, friendly news-style weather "
                  "report (120-200 words). Use ONLY the numbers provided; do not invent values, "
                  "places or hazards. Mention every active alert. Plain text, no markdown headings.")
        text = ctx.llm.chat("weather_report_agent", system, json.dumps(s, default=str))
        used_llm = bool(text)
        if not text:
            text = template_report(s)
        report = {"report_id": new_id(), "ts": int(state["run_ts"]), "report_trigger": reason,
                  "report_text": text, "llm_used": int(used_llm), "facts": json.dumps(s, default=str),
                  "model_version": str(state.get("model_version"))}
        ctx.run_log.add_rows("weather_reports", [report])
        details.update(llm_used=used_llm, length=len(text))
        return {"report": report, "report_reason": reason}
