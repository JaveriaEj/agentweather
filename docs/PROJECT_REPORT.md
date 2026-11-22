# Project Report — AgentWeather: Agentic Multi-Metric Weather & AQI Forecasting System

> **How to complete this report:** the method sections are already written. Every section marked
> **[FILL FROM SYSTEM]** must be completed with numbers from YOUR running system (dashboard tabs
> *Model Performance*, *Drift*, *System Logs*, and `reports/eda/findings.md`). Do not invent numbers.

**Author:** _your name_ · **City:** _city_ · **Repository:** _URL_ · **Live dashboard:** _URL_
**Data period:** _start – end_ · **Report date:** _date_

## 1. Objective
Forecast AQI, temperature, precipitation and rain probability, humidity, UV index, and wind speed and direction
hourly for the next 24 hours, fully automated on free-tier infrastructure, with an agentic layer for alerts,
reports and self-monitoring.

## 2. Architecture
See README §1. Components: GitHub Actions (hourly ingestion, agents after ingestion, daily training,
drift-triggered retraining), Hugging Face Hub (feature store as a private dataset repo, Model Registry as a private model repo; Hopsworks backend also implemented), LangGraph (four agents), Groq (LLM),
Streamlit Community Cloud (dashboard).

## 3. Data sources
| Source | Data | Frequency | Limitation |
|---|---|---|---|
| OpenWeather One Call 3.0 | weather (current + history) | hourly | 1,000 free calls/day; history costs 1 call per hour |
| OpenWeather Air Pollution | PM2.5, PM10, O3, NO2, SO2, CO | hourly, history available | model-based values, not station measurements |
| AQICN | station AQI | live only | no free hourly history API |

**[FILL FROM SYSTEM]** rows collected, % of hours from each source, API failures from the `api_failures` log.

## 4. Preprocessing and imputation
Validation ranges, one row per hour, source priority, forward-fill ≤ 3 h, night-time UV = 0, long gaps excluded
(README §7). **[FILL]** number of invalid values removed, % hours imputed, % hours excluded.

## 5. Feature engineering
Calendar, solar elevation, heat index, lags, rolling means, rate of change for 15 variables (~200 features).
Leakage prevention: features use data ≤ t only (unit-tested); targets are shifted 1–24 h; the newest 24 h are forecast-only;
time-based split with a 24 h gap.

## 6. EDA findings
**[FILL]** paste `reports/eda/findings.md` and the four charts (distributions, hourly patterns,
daily trends, correlations). Discuss seasonality, pollution peaks, outliers and rain imbalance.

## 7. Models
Naive 24 h baseline; SARIMAX + Fourier; Ridge; Random Forest; LSTM; logistic regression for rain probability.
Rationale and settings: README §7 and `config/settings.yaml`.

## 8. Model comparison
**[FILL]** For each target: table of RMSE / MAE / R² for every model and the baseline (Model Performance tab →
"All targets — latest run"), and which model was selected. Comment on targets where no model beats the
baseline (that variable is then served by the baseline and flagged).

## 9. Model registry and replacement
Champion/challenger rule (README §7). **[FILL]** list of registered versions, why each was registered, and runs
where production was kept.

## 10. Explainability
**[FILL]** top features for AQI and temperature at 1 h and 24 h (Explainability tab) and an interpretation.

## 11. Agents
| Agent | Trigger | Deterministic part | LLM part | Fallback |
|---|---|---|---|---|
| Forecast | hourly after ingestion | everything | — | stale-data flag if data is > 2 h old |
| Alert | every run | hazard rules (thresholds.yaml) | wording for 3 audiences | templates |
| Weather report | daily + after every new alert | facts and summary | news-style text | template report |
| Drift | every run | PSI and error ratio | — | — |

**[FILL] Agent performance:** run counts and success rate (Agents tab); alert counts by level; % of alerts and reports
written by the LLM vs fallback; examples of one LLM alert and one template alert; your assessment of alert quality.

## 12. Drift detection and retraining
**[FILL]** drift events (type, metric, score, threshold), retraining runs triggered and completed, outcome
(new version or production kept).

## 13. LLM and API failures
**[FILL]** counts from `llm_failures` and `api_failures` with typical errors (rate limits, timeouts) and how
the system behaved (retries, fallbacks).

## 14. Issues encountered and solutions
| Issue | Solution |
|---|---|
| AQICN has no free hourly history | Compute US-EPA AQI from OpenWeather pollutant history; AQICN used live and as fallback |
| Rain probability is not observable | Classify "precip > 0.1 mm"; climatology fallback for horizons with too few rainy hours |
| Wind direction is circular | Predict u/v components and convert back |
| History costs 1 API call per hour | Resumable, newest-first backfill with a per-run call budget |
| Re-training produced "better" models by floating-point noise, creating needless versions | Candidate must beat production by > 0.5 % RMSE |
| scikit-learn permutation importance rejected a plain adapter object (new estimator-tags API) | Own permutation-importance fallback; SHAP is the primary method |
| Hopsworks fixes schemas at first insert; some column names (`window`, `trigger`, `rank`) are SQL keywords | Consistent type casting (`conform`), schema alignment before insert, renamed columns |
| GitHub Actions runners keep no state | Forecast Ledger and logs persisted in the feature store and loaded at the start of each run |
| Hopsworks account could not create a project ("Project limit reached") | Pluggable storage layer; Hugging Face Hub backend (Parquet feature groups + versioned model repo) |
| Two workflows writing at the same time could overwrite each other | One buffered commit per run with optimistic concurrency (`parent_commit`) and automatic re-merge/retry |
| Retraining loops on persistent drift | 12-hour cooldown read from the drift log |
| **[FILL]** issues you hit during deployment | |

## 15. Limitations and future work
Single city; history length limited by the API budget; OpenWeather pollutant values are modelled rather than
measured at stations; GitHub cron delays; Hugging Face Hub storage/commit quotas.
