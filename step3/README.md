# Tech Challenge — Step 3: MLP (PyTorch) + Full Model Comparison + Production API

Completes the progression from Step 1 (notebooks) → Step 2 (basic pipeline) → **Step 3 (full production)**.

## What's new in Step 3

| Feature | Step 2 | Step 3 |
|---------|--------|--------|
| Preprocessing | `apply_feature_engineering` | Full `ShowWithPandas` EDA class + graphics |
| Models | 5 (Dummy, LR, RF, XGB, LGBM) | **11 sklearn + PyTorch MLP** |
| MLP | — | ReLU, BCEWithLogitsLoss, Adam, early stopping, mini-batch |
| Metrics | AUC, F1, Recall | **≥6 metrics**: AUC-ROC, PR-AUC, F1, Recall, Precision, Accuracy + business cost |
| Cost analysis | — | FP/FN trade-off curve + optimal threshold per model |
| ROC/PR plots | — | Side-by-side curves for all models |
| RF tuning | — | Time-budget random search (5 h) |
| API hot-reload | No | **Async file watcher — zero-downtime** |
| MLflow | Basic | All experiments + evaluation artifacts registered |

---

## Project structure

```
step3/
├── .github/workflows/ci.yml
├── docs/ML_CANVAS.md
├── src/
│   ├── main.py                        ← full 10-step pipeline
│   ├── api/
│   │   ├── main.py                    ← FastAPI + async hot-reload watcher
│   │   ├── core/middleware.py         ← LatencyMiddleware
│   │   ├── core/slowapi/config.py     ← rate limiter
│   │   ├── routes/prediction_routes.py
│   │   ├── services/prediction_service.py
│   │   └── schemas/churn_schemas.py
│   ├── eda/eda_treatment.py           ← ShowWithPandas (EDA + graphics)
│   ├── enums/enums.py                 ← ReaderType, WriterType, DFMethods
│   ├── features/feature_engineering.py
│   ├── train/
│   │   ├── train_models.py            ← MLTrainer (11 models)
│   │   ├── tuning.py                  ← time_budget_rf_search
│   │   └── mlp_trainer.py             ← PyTorch MLP (NEW)
│   ├── evaluation/
│   │   └── compare_models.py          ← ≥4 metrics + cost trade-off (NEW)
│   └── utils/logging_utils.py
├── Dockerfile
├── docker-compose.yml                 ← mlflow + trainer + api
└── pyproject.toml                     ← includes torch (CPU)
```

---

## MLP Architecture

```
Input(25) → Linear(128) → ReLU → Dropout(0.3)
          → Linear(64)  → ReLU → Dropout(0.3)
          → Linear(32)  → ReLU → Dropout(0.3)
          → Linear(1)
```

- **Loss**: `BCEWithLogitsLoss`
- **Optimizer**: Adam (lr=1e-3)
- **Batching**: `DataLoader` with batch_size=64
- **Early stopping**: patience=15 on validation loss

---

## Business cost model

From ML Canvas (CAC=R$400, retention=R$80):

| Outcome | Cost |
|---------|------|
| TP — churn caught, retained | −R$80 (retention) + R$400 saved = **+R$320** |
| FP — false alarm, wrong offer | **−R$80** |
| FN — missed churn, customer leaves | **−R$400** |
| TN — correct no-churn | R$0 |

`evaluate/compare_models.py` sweeps thresholds to find the operating point that minimises `FP×80 + FN×400`.

---

## Quick start

```bash
# 1. Add dataset
cp path/to/Telco-Customer-Churn.csv src/data/entry/

# 2. Start everything
docker compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

The trainer runs automatically after MLflow is healthy, trains all models, and saves `tuned_model.pkl`. The API loads it on startup and hot-reloads whenever the file changes (zero-downtime).

---

## Run locally

```bash
pip install uv
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r pyproject.toml --no-sources

# MLflow
mlflow server --host 0.0.0.0 --port 5000 &

# Train
ML_FLOW_API=http://localhost:5000 PYTHONPATH=src python src/main.py

# Serve
PYTHONPATH=src uvicorn api.main:app --reload --port 8000
```

---

## MLflow experiments

| Experiment | Runs |
|-----------|------|
| `TelcoChurn_Step3_FullPipeline` | Top-level pipeline run |
| ↳ nested: 11 sklearn model runs | One per model |
| ↳ nested: `MLP_PyTorch` | MLP training run |
| ↳ nested: `RandomForest_Tuned` | Time-budget tuning |
| `rf_time_budget_tuning` | RF search iterations |

Artifacts logged: model comparison CSV, cost trade-off plot, ROC/PR curves, processed data, final model.
