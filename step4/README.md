# Tech Challenge — Step 4: Final Delivery

**Telco Customer Churn Prediction — Production-Grade ML Pipeline**

> This is the final step of the FIAP POS-TECH Phase 1 Tech Challenge.  
> Steps 1–3 built the EDA, modelling, and API foundation. Step 4 adds:  
> **Prometheus/Grafana monitoring · pytest test suite · Model Card · Deploy architecture · Makefile**

---

## Project Progression

| Step | Focus | Key deliverable |
|------|-------|----------------|
| 1 | EDA + ML Canvas + Baselines | Notebook + MLflow tracking |
| 2 | Full sklearn pipeline + feature engineering | Modular `src/` + MLflow |
| 3 | MLP (PyTorch) + 11 models + FastAPI | REST API + hot-reload watcher |
| **4** | **Documentation + Monitoring + Tests** | **Model Card + Prometheus + Grafana + pytest** |

---

## Project Structure

```
step4/
├── .github/workflows/ci.yml       ← CI: lint + test + docker build
├── docs/
│   ├── MODEL_CARD.md              ← Performance, limitations, biases, failure modes
│   ├── DEPLOY_ARCHITECTURE.md     ← Real-time vs batch decision + infra diagram
│   ├── MONITORING_PLAN.md         ← Metrics, alert rules, playbook
│   └── ML_CANVAS.md               ← Problem formulation
├── monitoring/
│   ├── prometheus.yml             ← Scrape config (api:8000/metrics, 15 s interval)
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/       ← Auto-provision Prometheus datasource
│       │   └── dashboards/        ← Auto-provision dashboard loader
│       └── dashboards/
│           └── churn-api.json     ← 6-panel Grafana dashboard
├── src/
│   ├── main.py                    ← Full 10-step training pipeline
│   ├── api/
│   │   ├── main.py                ← FastAPI + Prometheus instrumentator + hot-reload
│   │   ├── core/
│   │   │   ├── middleware.py      ← Latency middleware (X-Process-Time header)
│   │   │   └── slowapi/config.py  ← Rate limiter (60 req/min)
│   │   ├── routes/prediction_routes.py
│   │   ├── schemas/churn_schemas.py  ← Pydantic ChurnInput / ChurnOutput
│   │   └── services/prediction_service.py
│   ├── eda/eda_treatment.py       ← ShowWithPandas
│   ├── enums/enums.py             ← ReaderType, WriterType, DFMethods
│   ├── features/feature_engineering.py
│   ├── train/
│   │   ├── train_models.py        ← MLTrainer (11 sklearn models)
│   │   ├── tuning.py              ← time_budget_rf_search
│   │   └── mlp_trainer.py        ← PyTorch MLP (early stopping, batching)
│   ├── evaluation/compare_models.py  ← ≥ 6 metrics + cost trade-off
│   └── utils/logging_utils.py
├── tests/
│   ├── test_smoke.py              ← App boots, /health, /metrics, /docs
│   ├── test_schema.py             ← Pydantic input/output validation
│   └── test_api.py                ← /predict + /health with mock model (10 tests)
├── Dockerfile
├── docker-compose.yml             ← mlflow + trainer + api + prometheus + grafana
├── Makefile                       ← install, lint, test, run, train, up, down
├── pyproject.toml                 ← single source of truth
└── .env.example
```

---

## Quick Start (Docker — recommended)

```bash
# 1. Copy dataset
cp path/to/Telco-Customer-Churn.csv ../data/

# 2. Start all services
docker compose up --build
```

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI docs | http://localhost:8000/docs | — |
| FastAPI health | http://localhost:8000/health | — |
| Prometheus metrics | http://localhost:8000/metrics | — |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

The trainer runs automatically once MLflow is healthy, trains all 12 models, saves `models/tuned_model.pkl`, and exits. The API loads the model on startup and hot-reloads it (within 5 s) whenever the file changes — no restart needed.

---

## Quick Start (Local)

```bash
uv venv && source .venv/bin/activate
make install

# Start MLflow
mlflow server --host 0.0.0.0 --port 5000 &

# Train
ML_FLOW_API=http://localhost:5000 DATA_PATH=../data/Telco-Customer-Churn.csv make train

# Serve API
make run

# In another terminal: start Prometheus
# Edit monitoring/prometheus.yml to use localhost:8000 instead of api:8000
# then: docker run -p 9090:9090 -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```

---

## Running Tests

```bash
make test
# or
PYTHONPATH=src pytest tests/ -v
```

### Test suite

| File | Tests | What it covers |
|------|-------|----------------|
| `test_smoke.py` | 6 | App boots, `/health` 200, `/metrics` exposed, `/docs` available |
| `test_schema.py` | 8 | Pydantic validation — valid payload, optional fields, defaults |
| `test_api.py` | 10 | `/predict` with mock model, 503 without model, headers |

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies via uv |
| `make lint` | ruff check + format check |
| `make lint-fix` | ruff auto-fix + reformat |
| `make test` | Run pytest with verbose output |
| `make run` | Start API locally (port 8000) |
| `make train` | Run full training pipeline locally |
| `make up` | `docker compose up --build` |
| `make down` | `docker compose down` |
| `make clean` | Remove `__pycache__` and `.pyc` files |

---

## Monitoring

### Prometheus
Scrapes `http://api:8000/metrics` every 15 s.  
Auto-instrumented via `prometheus-fastapi-instrumentator` — zero boilerplate.

**Key metrics:**
- `http_requests_total` — requests by method/handler/status
- `http_request_duration_seconds` — latency histogram
- `up` — API availability gauge

### Grafana
Dashboard auto-provisioned at startup. Access at http://localhost:3000 (admin/admin).

**Panels:**
1. Request rate (req/s)
2. P95 latency
3. Error rate %
4. Total predictions
5. API uptime (green/red)
6. Latency heatmap for `/predict`

### Alerts (documented in `docs/MONITORING_PLAN.md`)
| Alert | Condition |
|-------|-----------|
| API Down | `up == 0` for 1 min |
| High Latency | p95 > 500 ms for 5 min |
| High Error Rate | 5xx > 1% for 5 min |
| Model Not Loaded | `model_status = not_loaded` |

---

## API Reference

### `GET /health`
```json
{
  "status": "ok",
  "model_status": "loaded",
  "version": "0.4.0"
}
```

### `POST /predict`
**Request:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 45.50,
  "TotalCharges": "546.00"
}
```
**Response:**
```json
{
  "prediction": "Yes",
  "probability": 0.834,
  "model_version": "0.4.0"
}
```

### `GET /metrics`
Prometheus text format — scraped automatically by Prometheus every 15 s.

---

## Model Architecture

### PyTorch MLP
```
Input(25) → Linear(128) → ReLU → Dropout(0.3)
          → Linear(64)  → ReLU → Dropout(0.3)
          → Linear(32)  → ReLU → Dropout(0.3)
          → Linear(1)
```
Loss: `BCEWithLogitsLoss` | Optimizer: Adam (lr=1e-3) | Early stopping: patience=15

### Best Sklearn Model (served via API)
Time-budget Random Forest with `SelectKBest` (k=25) feature selection pipeline.

---

## MLflow Experiments

| Experiment | Runs |
|-----------|------|
| `TelcoChurn_Step4_FullPipeline` | Top-level pipeline run |
| ↳ nested: 11 sklearn model runs | One per model |
| ↳ nested: `MLP_PyTorch` | MLP training |
| ↳ nested: `RandomForest_Tuned` | Time-budget tuning |

Artifacts: model comparison CSV, cost trade-off plot, ROC/PR curves, processed data, final model.

---

## Documentation

| Document | Location |
|----------|----------|
| Model Card | `docs/MODEL_CARD.md` |
| Deploy Architecture | `docs/DEPLOY_ARCHITECTURE.md` |
| Monitoring Plan + Playbook | `docs/MONITORING_PLAN.md` |
| ML Canvas | `docs/ML_CANVAS.md` |

---

## Evaluation Summary

### Model Comparison (test set)

| Model | AUC-ROC | F1 | Recall |
|-------|---------|----|--------|
| Dummy | ~0.50 | ~0.27 | ~0.27 |
| LogisticRegression | ~0.84 | ~0.61 | ~0.55 |
| RandomForest (tuned) | **≥0.85** | **≥0.65** | **≥0.60** |
| MLP (PyTorch) | ~0.83 | ~0.62 | ~0.58 |

### Business Cost Model
- FP cost: R$80 (unnecessary retention offer)
- FN cost: R$400 (lost customer + CAC)
- Optimal threshold minimises `FP×80 + FN×400`

---

## Good Practices Checklist

- [x] `random_state=42` everywhere (reproducibility)
- [x] Stratified train/test split
- [x] No `print()` — structured `logging` throughout
- [x] Pydantic validation on all API inputs
- [x] Rate limiting (60 req/min via slowapi)
- [x] Latency middleware (`X-Process-Time` header)
- [x] Prometheus metrics (`/metrics` endpoint)
- [x] Async model hot-reload (zero-downtime)
- [x] ≥ 3 automated tests (smoke, schema, API)
- [x] ruff linting configured in `pyproject.toml`
- [x] Makefile for lint, test, run, train
- [x] Model Card with limitations and biases
- [x] Monitoring plan with alert playbook
- [x] Docker Compose with health checks
- [x] CI (GitHub Actions): lint → test → docker build
