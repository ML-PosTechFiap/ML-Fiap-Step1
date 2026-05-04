# Deploy Architecture

## Chosen Architecture: Real-Time Inference (REST API)

### Decision

We chose **real-time (synchronous) REST inference** over batch scoring for the following reasons:

| Factor | Real-Time (chosen) | Batch |
|--------|--------------------|-------|
| Latency requirement | < 200 ms per prediction | Hours acceptable |
| Use case trigger | Agent opens customer record | Nightly ETL |
| Integration | CRM system calls API on open | Scheduled job |
| Infra complexity | Medium (FastAPI + Docker) | Low (cron + scripts) |
| Cost | Constant (API always on) | Lower (compute only when needed) |

The retention team's primary workflow is triggered when an agent opens a customer record in the CRM. A batch model would produce stale scores (up to 24 h old) and require the CRM to join on a scoring table. A synchronous API returns a fresh prediction in milliseconds, enabling the agent to make a real-time decision.

---

## Infrastructure Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Docker Compose                            │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐ │
│  │  trainer    │───▶│   mlflow    │    │       api            │ │
│  │  (one-shot) │    │  :5000      │    │  FastAPI :8000       │ │
│  └─────────────┘    └─────────────┘    │  /predict            │ │
│         │                 ▲            │  /health             │ │
│         │ writes pkl      │ register   │  /metrics (Prom.)    │ │
│         ▼                 │            └──────────┬───────────┘ │
│  ┌─────────────────────────────────┐              │ scrape      │
│  │   models/tuned_model.pkl        │              ▼             │
│  └─────────────────────────────────┘   ┌──────────────────────┐ │
│         │ file-watcher hot-reload       │   prometheus :9090   │ │
│         └──────────────────────────────│                      │ │
│                                        └──────────┬───────────┘ │
│                                                   │ datasource  │
│                                        ┌──────────▼───────────┐ │
│                                        │    grafana :3000      │ │
│                                        │  dashboards + alerts  │ │
│                                        └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

External: CRM / Retention Tool ──HTTP POST /predict──▶ api:8000
```

---

## Service Responsibilities

| Service | Role | Port |
|---------|------|------|
| `mlflow` | Experiment tracking + model registry | 5000 |
| `trainer` | Full 10-step pipeline; exits after training | — |
| `api` | FastAPI inference; hot-reloads model on file change | 8000 |
| `prometheus` | Metrics scraping every 15 s | 9090 |
| `grafana` | Dashboards + alert rules | 3000 |

---

## Data Flow

1. **Training path** (offline, triggered manually or by CI):
   ```
   CSV → trainer → EDA → Feature Engineering → Model training
                       → MLflow (params, metrics, artifacts)
                       → models/tuned_model.pkl
   ```

2. **Inference path** (real-time, per request):
   ```
   HTTP POST /predict
     → Pydantic validation
     → apply_feature_engineering
     → ShowWithPandas.prepare_for_ml
     → reindex to expected 25 features
     → model.predict + predict_proba
     → ChurnOutput (prediction, probability, version)
   ```

3. **Model hot-reload** (zero-downtime):
   ```
   Async file watcher (5 s polling)
     → detects mtime change on tuned_model.pkl
     → joblib.load into app.state.model
     → no restart needed
   ```

---

## SLOs

| SLO | Target |
|-----|--------|
| Availability | ≥ 99.5% (measured over 30 days) |
| p95 latency (`/predict`) | < 200 ms |
| p99 latency (`/predict`) | < 500 ms |
| Error rate (5xx) | < 0.5% of requests |
| Model freshness | Retrained within 7 days of drift alert |
