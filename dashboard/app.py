"""Streamlit dashboard - reads ONLY real data from the Feature Store / Model Registry.

Run locally:  streamlit run dashboard/app.py
Deploy:       Streamlit Community Cloud (main file: dashboard/app.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.eda import analysis as A  # noqa: E402
from src.utils.config import settings  # noqa: E402

st.set_page_config(page_title="Weather & AQI Forecast", page_icon="🌤️", layout="wide")
CFG = settings()
TZ = CFG["city"]["timezone"]
DAY = 86400


# ------------------------------------------------------------------ data access
@st.cache_resource(show_spinner="Connecting to the Feature Store…")
def store():
    from src.store import get_store

    return get_store()


@st.cache_data(ttl=600, show_spinner=False)
def read(fg: str, days: float | None = None) -> pd.DataFrame:
    now = int(pd.Timestamp.now(tz="UTC").timestamp())
    try:
        return store().read(fg, now - int(days * DAY) if days else None, None)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read `{fg}`: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def models() -> list[dict]:
    try:
        return store().list_models(CFG["registry"]["model_name"])
    except Exception:  # noqa: BLE001
        return []


def local(ts) -> pd.Series:
    return pd.to_datetime(pd.Series(ts).astype("int64"), unit="s", utc=True).dt.tz_convert(TZ)


def empty(msg: str):
    st.info(msg)


def aqi_cat(v):
    if v is None or pd.isna(v):
        return "no data"
    for lim, name in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy (SG)"), (200, "Unhealthy"),
                      (300, "Very Unhealthy")]:
        if v <= lim:
            return name
    return "Hazardous"


# ------------------------------------------------------------------ header
st.title(f"🌤️ AgentWeather — {CFG['city']['name']}")
st.caption("Your 24-hour weather & air-quality report, written by forecasting models and LangGraph agents. "
           "Every value is read live from the project's feature store; nothing is hardcoded.")

tabs = st.tabs(["📰 Weather Report", "24-h Forecast", "Agent Activity", "EDA", "Model Performance",
                "Explainability", "Drift & Failures", "System Logs"])

feats_recent = read("hourly_features", 3)
fc_all = read("forecasts", 2)
latest_fc = pd.DataFrame()
if len(fc_all):
    latest_fc = fc_all[fc_all["run_ts"] == fc_all["run_ts"].max()].sort_values("horizon")
LEVEL_ICON = {"advisory": "🟡", "warning": "🟠", "danger": "🔴", "emergency": "🟣"}

# ------------------------------------------------------------------ A + C. weather report (front page)
with tabs[0]:
    if feats_recent.empty:
        empty("No observations yet. Run the backfill and the hourly feature pipeline.")
    else:
        last = feats_recent.sort_values("ts").iloc[-1]
        upd = local([last["ts"]]).iloc[0]
        st.markdown(f"#### Now in {CFG['city']['name']} · updated {upd:%A %d %b, %H:%M}")
        c = st.columns(6)
        c[0].metric("Temperature", f"{last['temp_c']:.1f} °C")
        c[1].metric("Humidity", f"{last['humidity_pct']:.0f} %")
        c[2].metric("AQI (US EPA)", "n/a" if pd.isna(last["aqi"]) else f"{last['aqi']:.0f}",
                    aqi_cat(last["aqi"]), delta_color="off")
        aq = last.get("aqi_aqicn")
        c[3].metric("AQICN station AQI", "n/a" if pd.isna(aq) or aq == "" else f"{float(aq):.0f}")
        c[4].metric("Wind", f"{last['wind_speed']:.1f} m/s")
        c[5].metric("UV index", f"{last['uvi']:.1f}")

    # active alerts as banners (one per hazard, general-public wording)
    alerts_now = read("alerts", 1)
    if len(alerts_now):
        newest = alerts_now[alerts_now["ts"] == alerts_now["ts"].max()]
        pub = newest[newest["audience"] == "General public"]
        for _, a in (pub if len(pub) else newest.drop_duplicates("metric")).iterrows():
            st.warning(f"{LEVEL_ICON.get(a['level'], '⚠️')} **{a['label']}** ({a['level']}) · "
                       f"{a['time_window']} — {a['message']}")

    rep = read("weather_reports", 14)
    st.markdown("---")
    if rep.empty:
        empty("No report yet. The Weather Report Agent writes one daily and after every new alert.")
    else:
        rep = rep.sort_values("ts", ascending=False)
        r = rep.iloc[0]
        st.subheader(f"Weather report — {local([r['ts']]).iloc[0]:%A %d %b, %H:%M}")
        st.markdown(f"> {r['report_text']}")
        st.caption(f"Trigger: {r['report_trigger']} · "
                   f"{'written by the LLM' if int(r['llm_used']) else 'template fallback (LLM unavailable)'}")

    if len(latest_fc):
        f = latest_fc.assign(time=local(latest_fc["target_ts"]).values)
        st.markdown("#### Next 24 hours at a glance")
        glance = f.iloc[::3][["time", "temp_c", "aqi", "rain_prob_pct", "precip_mm", "humidity_pct", "uvi",
                              "wind_speed", "wind_dir_deg"]].copy()
        glance["time"] = glance["time"].dt.strftime("%a %H:%M")
        st.dataframe(glance.round(1).rename(columns={
            "temp_c": "Temp °C", "aqi": "AQI", "rain_prob_pct": "Rain %", "precip_mm": "Rain mm",
            "humidity_pct": "Humidity %", "uvi": "UV", "wind_speed": "Wind m/s", "wind_dir_deg": "Wind dir °"}),
            hide_index=True, use_container_width=True)
    if len(rep) > 1:
        with st.expander("Previous reports"):
            st.dataframe(rep.assign(time=local(rep["ts"]).values)[["time", "report_trigger", "llm_used",
                                                                  "report_text"]],
                         hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ B. forecast
with tabs[1]:
    if latest_fc.empty:
        empty("No forecast yet. The Forecast Agent runs hourly after the feature pipeline.")
    else:
        f = latest_fc.assign(time=local(latest_fc["target_ts"]).values)
        st.caption(f"Forecast made at {local([latest_fc['run_ts'].iloc[0]]).iloc[0]:%Y-%m-%d %H:%M} local · "
                   f"model version {latest_fc['model_version'].iloc[0]} · status {latest_fc['status'].iloc[0]}")
        st.caption(f"Models per variable: {latest_fc['models'].iloc[0]}")
        r1 = st.columns(2)
        r1[0].plotly_chart(px.line(f, x="time", y="aqi", markers=True, title="1 · AQI"), use_container_width=True)
        r1[1].plotly_chart(px.line(f, x="time", y="temp_c", markers=True, title="2 · Temperature (°C)"),
                           use_container_width=True)
        r2 = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=f["time"], y=f["precip_mm"], name="Precipitation (mm)")
        fig.add_scatter(x=f["time"], y=f["rain_prob_pct"], name="Rain probability (%)", yaxis="y2")
        fig.update_layout(title="3 · Precipitation & rain probability",
                          yaxis2=dict(overlaying="y", side="right", range=[0, 100]))
        r2[0].plotly_chart(fig, use_container_width=True)
        r2[1].plotly_chart(px.line(f, x="time", y="humidity_pct", markers=True, title="4 · Humidity (%)"),
                           use_container_width=True)
        r3 = st.columns(2)
        r3[0].plotly_chart(px.bar(f, x="time", y="uvi", title="5 · UV Index"), use_container_width=True)
        wind = go.Figure(go.Barpolar(r=f["wind_speed"], theta=f["wind_dir_deg"],
                                     text=f["time"].dt.strftime("%H:%M")))
        wind.update_layout(title="6 · Wind speed (m/s) by direction (from)")
        r3[1].plotly_chart(wind, use_container_width=True)
        st.dataframe(f[["time", "aqi", "temp_c", "precip_mm", "rain_prob_pct", "humidity_pct", "uvi",
                        "wind_speed", "wind_dir_deg"]].round(1), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ D. agents
with tabs[2]:
    runs = read("agent_runs", 3)
    alerts = read("alerts", 2)
    st.subheader("Recent alerts")
    if alerts.empty:
        st.success("No alerts in the last 48 hours.")
    else:
        a = alerts.sort_values("ts", ascending=False)
        st.dataframe(a.assign(issued=local(a["ts"]).values)[["issued", "level", "label", "audience", "time_window",
                                                             "peak_value", "rule", "llm_used", "message"]],
                     hide_index=True, use_container_width=True)
    st.subheader("Agent activity")
    if runs.empty:
        empty("No agent runs recorded yet.")
    else:
        runs = runs.assign(time=local(runs["ts"]).values)
        cols = st.columns(4)
        for col, agent in zip(cols, ["forecast_agent", "alert_agent", "weather_report_agent", "drift_agent"]):
            sub = runs[runs["agent"] == agent].sort_values("ts")
            if sub.empty:
                col.metric(agent, "never run")
            else:
                lastr = sub.iloc[-1]
                ok = (sub["status"] == "success").mean() * 100
                col.metric(agent.replace("_", " ").title(), lastr["status"], f"{ok:.0f}% success (3 d)",
                           delta_color="off")
                col.caption(f"last: {lastr['time']:%d %b %H:%M}")
        st.dataframe(runs.sort_values("ts", ascending=False)[["time", "agent", "status", "duration_s", "details"]],
                     hide_index=True, use_container_width=True)
    st.subheader("LLM calls")
    calls = read("llm_calls", 3)
    if calls.empty:
        empty("No LLM calls recorded yet.")
    else:
        c = st.columns(3)
        c[0].metric("LLM calls (3 d)", len(calls))
        c[1].metric("Success rate", f"{(calls['status'] == 'success').mean() * 100:.0f}%")
        c[2].metric("Median latency", f"{calls['latency_s'].median():.2f} s")

# ------------------------------------------------------------------ E. EDA
with tabs[3]:
    feats_all = read("hourly_features")
    if feats_all.empty:
        empty("No data for EDA yet.")
    else:
        df = A.prepare(feats_all)
        st.subheader("Key findings (computed from the data)")
        for line in A.findings(df):
            st.markdown(f"- {line}")
        v = st.selectbox("Variable", [x for x in A.EDA_VARS if x in df], format_func=lambda x: A.LABELS[x])
        c = st.columns(2)
        c[0].plotly_chart(px.histogram(df, x=v, nbins=50, title=f"Distribution — {A.LABELS[v]}"),
                          use_container_width=True)
        prof = A.hourly_profile(df).reset_index()
        c[1].plotly_chart(px.line(prof, x="local_hour", y=v, markers=True, title="Average by hour of day"),
                          use_container_width=True)
        daily = A.daily_series(df).reset_index()
        st.plotly_chart(px.line(daily, x="local_date", y=v, title="Daily trend"), use_container_width=True)
        c = st.columns(2)
        c[0].plotly_chart(px.imshow(A.correlations(df), text_auto=True, color_continuous_scale="RdBu_r",
                                    zmin=-1, zmax=1, title="Correlation matrix"), use_container_width=True)
        with c[1]:
            st.markdown("**Missing values**")
            st.dataframe(A.missing_report(df), use_container_width=True)
            st.markdown("**Outliers (3×IQR)**")
            st.dataframe(A.outliers_iqr(df), hide_index=True, use_container_width=True)
        st.markdown("**Summary statistics**")
        st.dataframe(A.summary_stats(df), use_container_width=True)

# ------------------------------------------------------------------ F. models
with tabs[4]:
    ev = read("model_eval_history")
    mlist = models()
    if mlist:
        best = mlist[-1]
        st.subheader(f"Production model: `{CFG['registry']['model_name']}` v{best['version']}")
        st.caption(best.get("description") or "")
        st.dataframe(pd.DataFrame([{"version": m["version"], **m.get("metrics", {})} for m in mlist]).round(3),
                     hide_index=True, use_container_width=True)
    if ev.empty:
        empty("No training runs recorded yet.")
    else:
        last_run = ev[ev["ts"] == ev["ts"].max()]
        st.caption(f"Latest training run: {local([last_run['ts'].iloc[0]]).iloc[0]:%Y-%m-%d %H:%M} · "
                   f"reason {last_run['reason'].iloc[0]} · decision {last_run['decision'].iloc[0]}")
        tgt = st.selectbox("Target", sorted(last_run["target"].unique()))
        sub = last_run[last_run["target"] == tgt].sort_values("rmse")
        st.plotly_chart(px.bar(sub, x="model", y="rmse", color="family",
                               title=f"Validation RMSE — {tgt} (lower is better; naive_24h = baseline)"),
                        use_container_width=True)
        st.dataframe(sub[["model", "family", "rmse", "mae", "r2", "rmse_h01", "rmse_h24", "baseline_rmse",
                          "beats_baseline", "selected"]].round(3), hide_index=True, use_container_width=True)
        st.markdown("**All targets — latest run**")
        piv = last_run.pivot_table(index="target", columns="model", values="rmse").round(3)
        st.dataframe(piv, use_container_width=True)
        hist = ev.assign(time=local(ev["ts"]).values)
        st.plotly_chart(px.line(hist[hist["selected"] == 1], x="time", y="rmse", color="target", markers=True,
                                title="Selected model RMSE over training runs"), use_container_width=True)

# ------------------------------------------------------------------ G. explainability
with tabs[5]:
    sh = read("shap_importance")
    if sh.empty:
        empty("No explanations yet (created after each new model version).")
    else:
        sh = sh[sh["model_version"] == sh["model_version"].max()]
        c = st.columns(2)
        tgt = c[0].selectbox("Target ", sorted(sh["target"].unique()))
        hz = c[1].radio("Horizon", sorted(sh["horizon"].unique()), format_func=lambda h: f"{int(h)} h ahead",
                        horizontal=True)
        s = sh[(sh["target"] == tgt) & (sh["horizon"] == hz)].sort_values("importance")
        meth = s["method"].iloc[0] if len(s) else ""
        st.caption(f"Method: **{meth}** on the `{s['explained_model'].iloc[0]}` model "
                   f"(production model for this target: `{s['production_model'].iloc[0]}`), "
                   f"model version {int(s['model_version'].iloc[0])}.")
        st.plotly_chart(px.bar(s, x="importance", y="feature", orientation="h",
                               title="Mean |SHAP value| — which features drive the forecast"),
                        use_container_width=True)

# ------------------------------------------------------------------ H. drift
with tabs[6]:
    dl = read("drift_log", 30)
    if dl.empty:
        empty("No drift checks yet.")
    else:
        dl = dl.assign(time=local(dl["ts"]).values)
        events = dl[(dl["drifted"] == 1) & (dl["drift_type"].isin(["data", "model"]))]
        c = st.columns(3)
        c[0].metric("Drift events (30 d)", len(events))
        # one "trigger" row per retraining request (the per-metric rows only say which checks drifted)
        trig = dl[(dl["drift_type"] == "trigger") & (dl["action"] == "retraining_triggered")]
        c[1].metric("Retraining triggered", len(trig))
        c[2].metric("Retraining completed", int((dl["action"] == "retraining_completed").sum()))
        for kind in ["data", "model"]:
            sub = dl[dl["drift_type"] == kind]
            if len(sub):
                fig = px.line(sub, x="time", y="score", color="metric", markers=True,
                              title=f"{kind.title()} drift score ({'PSI' if kind == 'data' else 'live RMSE / validation RMSE'})")
                fig.add_hline(y=float(sub["threshold"].iloc[0]), line_dash="dash", annotation_text="threshold")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Drift events & retraining history**")
        st.dataframe(dl[(dl["drifted"] == 1) | dl["drift_type"].isin(["retraining", "trigger"])]
                     .sort_values("ts", ascending=False)[["time", "drift_type", "metric", "score", "threshold",
                                                          "action", "retrain_status", "details"]],
                     hide_index=True, use_container_width=True)

    st.subheader("Failed agent (LLM) calls")
    fails = read("llm_failures", 14)
    if fails.empty:
        st.success("No failed LLM calls in the last 14 days.")
    else:
        fails = fails.sort_values("ts", ascending=False)
        st.dataframe(fails.assign(time=local(fails["ts"]).values)[["time", "agent", "model", "error",
                                                                   "latency_s"]],
                     hide_index=True, use_container_width=True)
        st.caption("Each failure fell back to a deterministic template, so alerts and reports were still produced.")

# ------------------------------------------------------------------ I. logs
with tabs[7]:
    which = st.radio("Log", ["Forecast Ledger", "Agent execution", "LLM calls", "Failed LLM calls",
                             "API failures", "Pipeline runs"], horizontal=True)
    fg = {"Forecast Ledger": ("forecast_ledger", 1), "Agent execution": ("agent_runs", 3),
          "LLM calls": ("llm_calls", 3), "Failed LLM calls": ("llm_failures", 14),
          "API failures": ("api_failures", 14), "Pipeline runs": ("pipeline_runs", 3)}[which]
    lg = read(*fg)
    if lg.empty:
        empty(f"No rows in `{fg[0]}` for the last {fg[1]} day(s).")
    else:
        lg = lg.sort_values("ts", ascending=False)
        lg.insert(0, "time", local(lg["ts"]).values)
        st.dataframe(lg.drop(columns=["event_time"], errors="ignore").head(2000), hide_index=True,
                     use_container_width=True)
        if which == "LLM calls" and "latency_s" in lg:
            st.caption(f"Success rate {(lg['status'] == 'success').mean() * 100:.0f}% · "
                       f"median latency {lg['latency_s'].median():.2f}s")