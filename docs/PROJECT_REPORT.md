# Project Report — AgentWeather: Agentic Multi-Metric Weather & AQI Forecasting System

*10Pearls SHINE Internship, Cohort 10*

| | |
|---|---|
| **Author** | Javeria Ejaz |
| **City** | Karachi, Pakistan (24.86° N, 67.00° E) |
| **Repository** | https://github.com/JaveriaEj/agentweather |
| **Live dashboard** | https://agentweather-njoqe99sukww4sdvkzhapf.streamlit.app |
| **Feature store / registry** | Hugging Face Hub: `JIAAAT/agentweather-data` (dataset), `JIAAAT/agentweather-models` (models), both private |
| **Data period** | ~Nov 2025 – Nov 2026 (one year of hourly history) + live hourly data |

> Numbers in sections 6, 8 and 11 come from runs of the system itself (training logs, the
> dashboard's EDA / Model Performance / Drift / System Logs tabs). Training runs daily, so the
> dashboard always shows the latest values.

---

## 1. Objective

Forecast six weather and air-quality metric groups for Karachi **hourly for the next 24 hours** —
AQI, temperature, precipitation (amount + probability of rain), humidity, UV index and wind
(speed + direction) — on a fully automated, free-tier, serverless stack, with a LangGraph agent
layer that writes alerts and a news-style weather report and monitors data and model drift.
Every agent decision is logged.

## 2. Architecture

```
Open-Meteo (weather) ─┐  OpenWeather (air pollution) ─┐  AQICN (station AQI) ─┐
                      ▼                               ▼                      ▼
GitHub Actions — hourly:  scripts/run_feature_pipeline.py
   fetch (retries, fallbacks) → validate → fuse (1 row/hour) → impute → features & targets
                      ▼
Hugging Face Hub (feature store + model registry)
   raw data, features, targets, forecasts, Forecast Ledger, alerts, reports, all logs
        ▲                     │                                  │
        │                     ▼                                  ▼
GitHub Actions — daily + on drift:          GitHub Actions — after every ingestion:
scripts/run_training.py                     scripts/run_agents.py (one LangGraph graph)
naive / SARIMAX / Ridge / RF / LSTM         Forecast → Alert → [Report] → Drift → [Retrain] → Logs
→ compare → champion/challenger → register            │ Groq LLM (gpt-oss-20b) + template fallback
                      ▼
Streamlit Community Cloud dashboard (reads only from the feature store)
```

The four agents are nodes of **one LangGraph graph**. Within a run they share a typed state;
**between runs** they share the **Forecast Ledger** and logs stored in the feature store
(`load_context` reads recent alerts, the last report time and the last retraining trigger at the
start of every run).

## 3. Data sources

| Source | Data | Used for | Notes |
|---|---|---|---|
| **Open-Meteo** (forecast + historical-forecast API) | temperature, humidity, dew point, pressure, cloud cover, precipitation, **UV index**, wind speed/direction | weather history (backfill) **and** live hourly weather | free, no key; same source for training and serving avoids train/serve skew |
| **OpenWeather Air Pollution API** | PM2.5, PM10, O₃, NO₂, SO₂, CO (µg/m³), hourly, with history | history + live; AQI target computed from it | free tier |
| **AQICN** | live station AQI | shown on the dashboard; fallback AQI source | free token; no free hourly history |
| OpenWeather current weather (2.5) | weather without UV | **fallback** if Open-Meteo is down | free tier |

**Deviation from the brief — OpenWeather weather history.** OpenWeather's historical weather
(One Call 3.0 "time machine") requires the paid *One Call by Call* subscription (with a payment
card); without it every call returned `HTTP 401`. The brief allows "any available sources", so
weather comes from Open-Meteo. OpenWeather and AQICN are still used for air quality.

**Limitation — AQICN history.** AQICN offers no free hourly history, so the historical AQI is
**computed with the US-EPA breakpoints** from OpenWeather pollutant concentrations
(overall AQI = max of the pollutant sub-indices; PM2.5 uses the 2024 breakpoints). Using the
same computation live keeps training and serving consistent; the AQICN value is kept separately
(`aqi_aqicn`) for display and as a fallback.

## 4. Preprocessing and imputation

* **Validation:** physically impossible values (e.g. humidity > 100 %, wind > 75 m/s) become missing.
* **Fusion:** all sources are aligned to the top of the hour (UTC) and joined into **one row per
  hour** on a continuous hourly grid. Source priority for AQI: computed from OpenWeather
  pollutants → AQICN station value if pollutants are missing (`aqi_source` records which).
* **Imputation (leak-free — past values only):**
  1. forward-fill gaps of **≤ 3 hours**;
  2. UV index at night (sun below the horizon) = 0;
  3. longer gaps stay missing; those hours are excluded from training (`n_missing` column).
* **Result (one year, 8,761 hours):** after imputation only **0.72 % of hours** contain any
  imputed value; PM2.5/AQI had the most gaps (~5 % of hours, from missing pollutant history),
  weather variables ~0.14 %.

## 5. Feature engineering

* **Calendar:** hour, day of week, month, day of year, cyclic sin/cos encodings, solar elevation.
* **Derived physics:** heat index (NOAA), rain flag, wind as **u/v components** (avoids the 359°→1° problem).
* **For 15 base variables:** lags (1, 2, 3, 6, 12, 24 h), rolling means (3, 6, 24 h),
  differences / rate of change (1, 3, 24 h) → ~210 features.
* **Leakage prevention:** every feature at hour *t* uses only data ≤ *t* (a unit test changes all
  future values and checks that past features are unchanged).
* **Targets:** each variable 1…24 hours ahead (`temp_c_h01 … temp_c_h24`); the newest 24 hours
  have no complete target and are used for forecasting only.
* **Rain probability** is not observed directly, so it is modelled as a classifier of
  "precipitation > 0.1 mm in hour t+h".

## 6. EDA findings

Computed from the stored data (dashboard → EDA tab; `scripts/run_eda.py` saves the charts in `reports/eda/`).

* **Seasonality / daily cycle:** temperature peaks around 13:00 local time with a daily swing of
  ~7 °C; UV follows the sun (0 at night); AQI is worst in the afternoon/evening and best in the
  early morning.
* **Correlations:** AQI ↔ PM2.5 r = 0.95 (PM2.5 drives Karachi's AQI); temperature ↔ pressure
  r = −0.64; wind ↔ pressure r = −0.56; temperature ↔ UV r = 0.53; AQI ↔ wind speed negative
  (wind disperses pollution).
* **Imbalanced rain:** rain occurs in only ~2–6 % of hours, so precipitation is a hard,
  zero-heavy target — the rain-probability classifier falls back to climatology for horizons
  with too few rainy hours.
* **Outliers (3×IQR):** precipitation 553 h (6.3 % — real rain events), PM2.5 274 h (3.3 %),
  AQI 120 h (1.4 %). They are real pollution/rain episodes and were **kept**, not removed.
* **Missing values:** see section 4.

## 7. Models

| Family | Model | Notes |
|---|---|---|
| Baseline | **Naive 24 h** | forecast for t+h = observation at t+h−24 |
| Statistical | **SARIMAX(1,0,1)** + daily Fourier terms | Fourier terms instead of a slow seasonal order of 24; parameters fitted once, Kalman filter re-run per forecast |
| Classical ML | **Ridge**, **Random Forest** | all 24 horizons predicted at once; median imputation fitted on training data |
| Deep learning | **LSTM** (PyTorch) | 48-hour input window → 24 outputs; early stopping on the end of the *training* period |
| Classification | **Logistic regression** per horizon | rain probability; climatology fallback |

**Validation:** time-based split — last 20 % of the history is validation, with a 24-hour gap;
no shuffling. Metrics: **RMSE, MAE, R²** over all 24 horizons (plus RMSE at +1 h and +24 h).

## 8. Model comparison

Training run on one year of data (6,214 training hours, 1,559 validation hours, 520 evaluated
forecast origins). Validation **RMSE** per target (lower is better); **bold** = selected model.

| Target | Naive 24 h | Ridge | Random Forest | SARIMAX | LSTM | Selected |
|---|---|---|---|---|---|---|
| AQI | 15.02 | 18.15 | 23.71 | **13.07** | 25.07 | SARIMAX |
| Temperature (°C) | **1.08** | 1.58 | 1.82 | 2.34 | 2.13 | naive (no model beat it) |
| Precipitation (mm) | 0.074 | 0.066 | 0.070 | 0.071 | **0.062** | LSTM |
| Humidity (%) | 11.90 | 11.29 | **10.83** | 19.63 | 12.80 | Random Forest |
| UV index | 0.48 | **0.39** | 0.43 | 0.65 | 0.41 | Ridge |
| Wind u (m/s) | 1.12 | **0.93** | 1.17 | 1.58 | 1.02 | Ridge |
| Wind v (m/s) | 0.96 | **0.85** | 0.91 | 1.89 | 0.87 | Ridge |
| Rain probability | 0.100 | — | — | — | — | Logistic (0.081) |

R² of the selected models: AQI 0.87, temperature 0.89, humidity 0.75, UV 0.97, wind u 0.56,
wind v 0.66. (Full RMSE/MAE/R² for every model and every run: dashboard → Model Performance.)

**Discussion**
* **No single family wins everywhere** — the bundle uses SARIMAX, LSTM, Random Forest, Ridge and
  the classifier, one per target.
* **Temperature:** the naive "same time yesterday" forecast is very strong for a coastal city
  with a regular daily cycle; no model beat it on validation, so the rule *"a model must beat the
  baseline"* keeps the baseline. This is reported honestly rather than hidden.
* **History length matters:** with only 30–90 days of history most models lost to the baseline
  (they had never seen the validation season). Extending to one year let Ridge, RF and LSTM beat
  the baseline for humidity, UV, wind and precipitation.
* **Precipitation** stays hard (negative R² for all models): rain is rare and bursty.

## 9. Model registry and replacement

Models are stored as a versioned **bundle** (best model per variable) in the Hugging Face model
repo (`v1/`, `v2/`, …). Rule (champion/challenger, per variable):
1. a candidate must beat the naive baseline;
2. it is compared with the current production model **on the same validation hours**;
3. it replaces production only if its RMSE is lower by more than 0.5 %.

A production model is only used as champion for a variable if it **never trained on that
variable's validation hours** (each variable records its own training end date). Every run's
metrics and decision (`registered_new_version` / `kept_production`) are stored in
`model_eval_history`. Training runs daily and whenever the Drift Agent requests it.

## 10. Explainability

SHAP (permutation explainer) is computed for the classical model of each target at +1 h and
+24 h after every new model version (dashboard → Explainability). Typical pattern: short-horizon
forecasts are dominated by the **current value and 1–3 h lags** of the same variable; for the
24-hour horizon the **24-hour lag, 24-hour rolling mean and calendar/solar features** become
most important — the model learns "same time tomorrow looks like same time today, adjusted by
the trend".

## 11. Agents and their performance

| Agent | Trigger | Deterministic part | LLM part | Fallback |
|---|---|---|---|---|
| **Forecast** | every hour after ingestion | loads latest features + best model, 24 h forecast for all metrics, saves it, writes the Forecast Ledger | — | marks `stale_data` if inputs are older than 2 h |
| **Alert** | every run | hazard detection with **fixed rules** (`config/thresholds.yaml`) for all six metrics (+ heat index) | wording for **general public, sensitive groups, outdoor workers** | fixed templates |
| **Weather Report** | daily and after every new alert | facts and summary statistics | news-style report | template report |
| **Drift** | every run | data drift (PSI + KS) and model drift (live RMSE ÷ validation RMSE) | — | — |

**Alert quality.** Thresholds follow published scales (US-EPA AQI categories, WHO UV index,
AMS rain-rate classes, Beaufort wind, NOAA heat index). The LLM can only rephrase alerts that the
rules already decided — it cannot add, remove or change levels or thresholds; its JSON answer is
validated, otherwise templates are used. Example produced by the system: *"Unhealthy for
Sensitive Groups (advisory): AQI expected to reach 149.4 (threshold ≥ 101) … Children, older
adults and people with heart or lung conditions should stay indoors and keep medication at
hand."* Alerts are de-duplicated for 12 hours using the ledger, so the same hazard does not
trigger a new report every hour.

**Drift detection.** The drift agent caught a **real problem** in the first cloud runs: two API
secrets were misconfigured, so no new air-quality data arrived and PM2.5/AQI inputs were stale.
Data-drift PSI rose far above the 0.25 threshold (up to ≈ 7) for 7 features, the agent logged the
drift events, **dispatched an extra training run and opened a GitHub issue**; the training run
registered a new model version. After the secrets were fixed the inputs returned to normal.
A 12-hour cooldown prevents retraining loops. Note: weather is seasonal, so PSI also rises when
the season changes — retraining on newer data is the intended response.

**LLM call failures.** Every LLM call (prompt, response, latency, status) is in `llm_calls`;
failures are also in `llm_failures`. The original model `llama-3.3-70b-versatile` was retired by
Groq, so every call failed with `HTTP 404 model_not_found`; **all alerts and reports were still
produced via the template fallback** — the system never stopped. After switching to
`openai/gpt-oss-20b` (with short reasoning effort so the answer fits the token limit) calls
succeed. Current counts and success rate: dashboard → Agent Activity / Drift & Failures.

## 12. Logging (all stored in the feature store)

| Log | Content |
|---|---|
| `forecast_ledger` | every forecast: run time, target time, metric, horizon, value, model name/version, status |
| `llm_calls` / `llm_failures` | agent, prompt, response, latency, status, error |
| `drift_log` | time, metric, drift type, score, threshold, action taken, retraining status |
| `agent_runs`, `pipeline_runs`, `api_failures` | every agent step, pipeline step and failed API call |
| `alerts`, `weather_reports`, `model_eval_history`, `shap_importance` | outputs and model history |

## 13. Issues encountered and fixes

| Issue | Fix |
|---|---|
| **Hopsworks:** account could not create a project ("Project limit reached"); the default host did not resolve | Storage layer is abstracted (`src/store/`); added a **Hugging Face Hub** backend (Parquet feature groups partitioned by month + versioned model repo). Hopsworks backend kept (`STORE_BACKEND=hopsworks`) |
| OpenWeather weather history needs a paid subscription (`HTTP 401`) | Weather from **Open-Meteo** (free, includes UV) for history *and* live data |
| AQICN has no free hourly history | AQI computed from OpenWeather pollutant history with US-EPA breakpoints |
| Rain probability is not observable | Classifier on "precip > 0.1 mm"; climatology fallback |
| Wind direction is circular | Predict u/v components, convert back |
| Every Hugging Face write is a commit; two workflows could overwrite each other | One buffered commit per run with optimistic concurrency (`parent_commit`) and automatic re-merge/retry |
| Re-training "improved" by floating-point noise, creating needless versions | Candidate must beat production by > 0.5 % |
| **Unfair comparison:** after extending history to one year, the old production model had trained on part of the new validation period and looked unrealistically good | Each variable stores its own training end date; a production model is only champion if it never saw the validation hours |
| Groq retired `llama-3.3-70b-versatile` (`404`) | Template fallbacks kept the system running; switched to `openai/gpt-oss-20b` (configurable via `GROQ_MODEL`) |
| `gpt-oss` "thinking" used up the token limit | `reasoning_effort = low` and a larger token budget |
| GitHub secrets copied with extra text / wrong token (`401`, invalid repo id) | Secret values are cleaned automatically (spaces, quotes, full URLs); clear error messages name the missing variable |
| Missing AQI showed as "nan · Hazardous" on the dashboard | Missing values shown as "n/a · no data" |
| Windows setup: PowerShell blocked the virtual environment; tests slow on a laptop | `Set-ExecutionPolicy -Scope Process Bypass`; smaller models inside the test suite only |

## 14. Testing

27 automated tests (pytest, run on every push by the *Tests* workflow) cover API parsing and
retries, validation, fusion, imputation, AQI breakpoints, **feature leakage**, targets, the naive
baseline, training/selection/registry rules, the unfair-champion case, all agents with the LLM
unavailable, PSI drift, model drift and cooldown, the LangGraph graph, and the Hugging Face store
(including concurrent commits) — all offline with synthetic data that lives only in `tests/`.

## 15. Limitations and future work

* Single city; one year of history (more years would help seasonal models).
* Open-Meteo weather and OpenWeather pollution are **model-based** data, not station measurements.
* Precipitation remains hard to forecast; a dedicated zero-inflated model could help.
* GitHub's hourly schedule can be delayed; the pipeline repairs gaps of up to 6 hours.
* If Hopsworks access is granted, switching back only needs `STORE_BACKEND=hopsworks` and its keys.
