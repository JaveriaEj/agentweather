# AgentWeather — Agentic Multi-Metric Weather & AQI Forecasting System

*10Pearls SHINE Internship, Cohort 10*

An automated, serverless forecasting system for **one city** (default: Karachi; change it in
`config/settings.yaml`). It forecasts six metric groups **hourly for the next 24 hours**, using free or free-tier services:

| # | Metric group | Output variable(s) |
|---|---|---|
| 1 | AQI (US EPA scale) | `aqi` |
| 2 | Temperature | `temp_c` |
| 3 | Precipitation | `precip_mm`, `rain_prob_pct` |
| 4 | Humidity | `humidity_pct` |
| 5 | UV Index | `uvi` |
| 6 | Wind | `wind_speed` (m/s), `wind_dir_deg` (direction the wind comes from) |

### Brief → implementation map

| Brief requirement | Where |
|---|---|
| Hourly fetch → fuse → impute → features → Feature Store | `scripts/run_feature_pipeline.py`, `src/processing/` |
| Backfill with the same code; targets shifted ahead; newest 24 h forecast-only | `scripts/backfill.py`, `src/processing/features.py` |
| Daily training + on drift; ≥ 3 families + naive 24 h baseline; RMSE/MAE/R² per metric | `scripts/run_training.py`, `src/models/` |
| Model Registry; replace only if better on validation | `src/models/train.py` (champion/challenger) |
| 4 agents in one LangGraph graph, shared state, Forecast Ledger | `src/agents/graph.py`, `src/monitoring/ledger.py` |
| Alert rules for all six metrics; 3 audiences; LLM + fallback | `config/thresholds.yaml`, `src/agents/alert_agent.py` |
| Report daily and after a new alert; LLM + fallback | `src/agents/report_agent.py` |
| Data drift + model drift; alert team; extra training run | `src/agents/drift_agent.py`, `src/agents/retrain_trigger.py` |
| Forecast Ledger, agent call log, failure log, drift log in Feature Store | feature groups `forecast_ledger`, `llm_calls`, `llm_failures`, `drift_log` |
| GitHub Actions: ingestion hourly, training daily + on drift | `.github/workflows/` |
| EDA, SHAP explanations | `scripts/run_eda.py`, `src/models/explain.py` |
| Dashboard presented as a weather report (forecasts, report, agent activity/alerts, EDA, SHAP, drift, failed calls) | `dashboard/app.py` |
| Report | `docs/PROJECT_REPORT.md` |

---

## 1. Architecture

```
OpenWeather ─┐   AQICN ─┐
             ▼          ▼
GitHub Action (hourly) ── scripts/run_feature_pipeline.py
  fetch (retry/backoff/fallback) → validate → fuse 1 row/hour → impute → features/targets
             ▼
HOPSWORKS  Feature Store (raw, features, targets, forecasts, ledger, logs, drift)
           Model Registry (versioned "bundle": best model per variable)
     ▲            │                         │
     │            ▼                         ▼
GitHub Action (daily + on drift)     GitHub Action (after every ingestion)
scripts/run_training.py              scripts/run_agents.py  (LangGraph)
 baseline / SARIMAX / Ridge / RF /   Forecast → Alert → [Report] → Drift → [Retrain trigger] → Logs
 LSTM → compare → champion/challenger            │ Groq LLM (with template fallback)
             ▼                                   ▼
Streamlit Community Cloud dashboard (reads the Feature Store only, cached 10 min)
```

**Data flow:** every hour the feature pipeline stores raw observations, repairs gaps from missed
runs (last 6 h), and recomputes features/targets for the last 48 h (idempotent upserts by `ts`).
When it succeeds, the agents workflow starts automatically (`workflow_run`). Training runs daily and
whenever the Drift Agent dispatches it.

### LangGraph workflow (`src/agents/graph.py`)

```
START → load_context → forecast_agent → alert_agent
      → (new alert OR daily report due?) → weather_report_agent ─┐
      ────────────────────────────────────────────────────────────────────── → drift_agent
      → (drift over threshold?) → trigger_retraining → persist_logs → END
                                 └──────────────────→ persist_logs → END
```

* **Shared state within a run:** `ForecastState` (`src/agents/state.py`).
* **Shared state between runs:** the **Forecast Ledger** and logs in Hopsworks. `load_context`
  reads recent alerts (de-duplication), the last report time (daily schedule) and the last
  retraining trigger (12 h cooldown).

## 2. Repository structure

```
config/            settings.yaml (city, model, drift settings), thresholds.yaml (alert rules)
src/ingestion/     openweather.py, aqicn.py, retry.py (backoff, failure logging)
src/processing/    validate.py, fuse.py (fusion + imputation), aqi_calc.py, features.py, pipeline.py
src/store/         hopsworks_store.py (production), local_store.py (dev/tests), schemas.py
src/models/        dataset.py, base.py (naive baseline), sarimax.py, classical.py, lstm.py,
                   evaluate.py, train.py (selection + registry), bundle.py, explain.py (SHAP)
src/agents/        forecast_agent.py, alert_agent.py, report_agent.py, drift_agent.py,
                   retrain_trigger.py, llm.py, state.py, graph.py
src/monitoring/    logs.py (run logs), ledger.py (Forecast Ledger)
src/eda/           analysis.py (shared by the EDA report and the dashboard)
scripts/           run_feature_pipeline.py (Script 1), run_training.py (Script 2),
                   run_agents.py (Script 3), backfill.py, run_eda.py
dashboard/         app.py + requirements.txt (Streamlit Cloud)
.github/workflows/ feature_pipeline.yml, agents.yml, train.yml, backfill.yml, tests.yml
tests/             offline tests (synthetic data generator lives ONLY here)
docs/              PROJECT_REPORT.md
reports/eda/       generated EDA charts + findings
```

## 3. Accounts and API keys

| Service | Used for | How to get it |
|---|---|---|
| OpenWeather | live weather, weather history, air pollution | Sign up → subscribe to **One Call by Call** (card required, 1,000 calls/day free). **Set the daily limit to 1,000** in *Billing plans* so you are never charged. |
| AQICN | live station AQI | https://aqicn.org/data-platform/token |
| Hopsworks | Feature Store + Model Registry | https://app.hopsworks.ai → create a project → *Account settings → API keys* |
| Groq | LLM for alerts/reports | https://console.groq.com → API keys |
| GitHub | code + Actions | **public** repo (unlimited free Actions minutes) |
| Streamlit Community Cloud | dashboard | sign in with GitHub |

## 4. Local setup (step by step)

```bash
git clone https://github.com/<you>/weather-aqi-forecast.git
cd weather-aqi-forecast
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # then edit .env and paste your keys
pytest -q                            # offline tests, no keys needed
```

Run order the first time:

```bash
python -m scripts.backfill --days 180 --max-calls 900   # repeat daily until "remaining: 0"
python -m scripts.run_eda                               # EDA charts -> reports/eda/
python -m scripts.run_training                          # trains, compares, registers v1
python -m scripts.run_feature_pipeline                  # one hourly ingestion
python -m scripts.run_agents                            # forecast + agents
streamlit run dashboard/app.py                          # dashboard on http://localhost:8501
```

> Training needs at least 3 weeks of complete history (it stops with a clear message otherwise).
> 180 days of weather history = 4,320 time-machine calls ≈ 5 days at 900 calls/day.

`STORE_BACKEND=local` stores everything in `data/local_store/` instead of Hopsworks (for development
and the tests only).

## 5. GitHub Actions

1. Push the repo to GitHub (public).
2. *Settings → Secrets and variables → Actions → New repository secret*:
   `OPENWEATHER_API_KEY`, `AQICN_TOKEN`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, `GROQ_API_KEY`.
   Optional *variable* `GROQ_MODEL` to change the LLM.
3. *Actions* tab → enable workflows → run **Historical Backfill** manually (once per day until complete),
   then **Model Training** manually once.
4. From then on: **Feature Pipeline** hourly → **Forecast Agents** automatically after each success →
   **Model Training** daily at 02:17 PKT and on drift.

The Drift Agent starts retraining with the GitHub API (`workflow_dispatch` on `train.yml`) using the
built-in `GITHUB_TOKEN` (the agents workflow grants `actions: write`), and opens a GitHub issue labelled
`drift` to alert the team. Create the labels `drift` and `automated` once (Issues → Labels).

Known GitHub limits: scheduled runs can be delayed at busy times, and schedules are paused after 60 days
without repository activity — the gap-repair step re-fetches missed hours.

## 6. Dashboard deployment (Streamlit Community Cloud)

1. https://share.streamlit.io → *Create app* → choose the repo, branch `main`, main file `dashboard/app.py`.
2. *Advanced settings*: Python 3.11; *Secrets*: paste the contents of `.streamlit/secrets.toml.example`
   with your values.
3. Deploy. The app reads only from Hopsworks (cached 10 minutes). Free apps sleep after inactivity and wake on visit.

## 7. Design decisions (documentation)

### Data sources and their limits
* **Weather:** OpenWeather One Call 3.0 (current + time machine). Fallback for live data: free
  `/data/2.5/weather` (no UV → imputed).
* **Air quality:** OpenWeather Air Pollution (current + hourly history) gives pollutant
  concentrations; **AQICN** gives the live station AQI.
* **Limitation:** AQICN has no free hourly history API, so historical AQI is *computed* (US EPA
  breakpoints) from OpenWeather concentrations. Using the same method live keeps training and
  serving consistent.

### Source priority (fusion)
One row per UTC hour. AQI = computed US-EPA AQI from OpenWeather pollutants; if those are missing for an
hour, the AQICN value is used (`aqi_source` records which). AQICN's own value is always kept in
`aqi_aqicn` and shown on the dashboard.

### Missing-value imputation (leak-free)
1. Out-of-range values → missing (`validate.py`).
2. Forward-fill gaps ≤ 3 h (past values only).
3. UV at night (sun below horizon) = 0.
4. Longer gaps stay missing → those hours are excluded from training (`n_missing` column).

### Feature engineering
Calendar (hour/day/month, cyclic sin/cos, solar elevation), heat index, rain flag and, for 15 base
variables: lags (1, 2, 3, 6, 12, 24 h), rolling means (3, 6, 24 h), differences / rate of change (1, 3, 24 h).
Every feature at hour *t* uses data ≤ *t* (tested in `tests/test_processing.py`).

### Targets
`<var>_hXX` = value XX hours after *t*, XX = 1…24. The newest 24 hours have no complete target and are
used only for forecasting. Wind is modelled as u/v components (avoids the 359°→1° problem) and converted back.
Rain probability = classifier for "precip > 0.1 mm in hour t+h".

### Models
| Family | Model | Notes |
|---|---|---|
| Baseline | naive 24 h | forecast for t+h = observation at t+h−24 |
| Statistical | SARIMAX(1,0,1) + daily Fourier terms | parameters fitted once; Kalman filter re-run per origin |
| Classical ML | Ridge, Random Forest | 24 outputs at once; logistic regression per horizon for rain probability |
| Deep learning | LSTM (PyTorch) | 48 h input window → 24 outputs; early stopping on the end of the training period |

**Validation:** time-based split (last 20 % = validation, 24 h gap, no shuffling). Metrics: RMSE, MAE, R²
over all horizons (+ RMSE at 1 h and 24 h).

**Selection & registry:** per variable, a model must beat the naive baseline; the best eligible model is
compared with the current production model on the same validation hours and replaces it only if its RMSE
is lower by more than 0.5 %. A new registry version is created only if some variable improved. Every run's
metrics are stored in `model_eval_history`.

### Alerts
Deterministic rules in `config/thresholds.yaml` (US EPA AQI categories, WHO UV categories, AMS rain-rate
classes, Beaufort wind, NOAA heat index). The LLM only rewrites decided alerts for the general public,
sensitive groups and outdoor workers; its JSON output is validated, otherwise fixed templates are used.

### Drift
* Data drift: PSI (threshold 0.25) for 8 monitored features, last 7 days vs training data, plus KS p-value.
* Model drift: live RMSE of ledger forecasts vs observations ÷ validation RMSE (threshold 1.5).
* Weather is seasonal, so data drift will fire when the season changes — that is intended: it retrains on newer data.

### Logs (all in Hopsworks)
`forecast_ledger`, `agent_runs`, `llm_calls`, `llm_failures`, `api_failures`, `pipeline_runs`,
`drift_log`, `model_eval_history`, `alerts`, `weather_reports`, `shap_importance`.

## 8. Testing status

| Component | Status |
|---|---|
| Parsing, retries, validation, fusion, imputation, AQI, features, targets, leakage | ✅ automated tests (offline) |
| Baseline, Ridge, Random Forest, rain classifier, selection, registry logic (local store) | ✅ automated tests |
| Forecast/Alert/Report/Drift agents, ledger, cooldown, LLM fallback (sequential runner) | ✅ automated tests |
| SARIMAX, LSTM, LangGraph graph, SHAP | ⚠️ code complete; run `pytest` after `pip install -r requirements.txt` (the LangGraph test runs automatically when it is installed) |
| Real APIs, Hopsworks, Groq, GitHub dispatch, Streamlit Cloud | ⚠️ needs your keys — verify with the first-run checklist below |

### First-run checklist
1. `pytest -q` → all tests pass (including `test_langgraph_graph_runs`).
2. `python -m scripts.run_feature_pipeline` → log shows `Feature pipeline success`; Hopsworks UI shows `raw_weather`, `hourly_features`.
3. Backfill complete → `python -m scripts.run_training` logs the RMSE table for all 5 model types and `Registered weather_aqi_bundle v1`.
4. `python -m scripts.run_agents` → `Run summary` with `forecast_status: success`.
5. Dashboard shows real values in every tab.
