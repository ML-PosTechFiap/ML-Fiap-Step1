# Tech Challenge — Step 2: Clean Training Pipeline

Evolves from Step 1 (notebooks / messy code) into a clean, structured pipeline.

## What's in Step 2

| Feature | Step 1 | Step 2 | Step 3 |
|---------|--------|--------|--------|
| Structure | Notebooks + messy scripts | Clean modular package | Clean modular package |
| Preprocessing | Raw / mixed | `apply_feature_engineering` + null handling + encoding | Full `ShowWithPandas` + graphics |
| Models | Unstructured | **5: Dummy, LR, RF, XGB, LGBM** | 11 sklearn + PyTorch MLP |
| Metrics | Mixed | **AUC, F1, Recall** | ≥6 metrics + business cost |
| MLflow | No | **Basic: params + metrics** | Full artifacts + curves |
| API | No | **No** | FastAPI + async hot-reload |
| Docker | No | MLflow + trainer | MLflow + trainer + API |

---

## Project structure

```
step2/
├── .github/workflows/ci.yml
├── docs/ML_CANVAS.md
├── src/
│   ├── main.py                      ← 6-step pipeline
│   ├── eda/eda_treatment.py         ← ShowWithPandas (EDA + basic preprocessing)
│   ├── enums/enums.py               ← ReaderType, WriterType, DFMethods
│   ├── features/feature_engineering.py
│   ├── train/train_models.py        ← MLTrainer (5 models + AUC/F1/Recall)
│   └── utils/logging_utils.py
├── Dockerfile
├── docker-compose.yml               ← mlflow + trainer
└── pyproject.toml
```

---

## Quick start

```bash
# 1. Add dataset
cp path/to/Telco-Customer-Churn.csv src/data/entry/

# 2. Run
docker compose up --build
```

MLflow UI → http://localhost:5000

---

## Run locally

```bash
pip install uv
uv venv && source .venv/bin/activate
uv pip install -r pyproject.toml --no-sources

mlflow server --host 0.0.0.0 --port 5000 &
ML_FLOW_API=http://localhost:5000 PYTHONPATH=src python src/main.py
```

---

## Models & metrics

5 models trained and compared by **AUC-ROC**: Dummy, Logistic Regression, Random Forest, XGBoost, LightGBM.

Per-model MLflow metrics: `cv_auc`, `test_auc`, `test_f1`, `test_recall`, `test_accuracy`.

Best model saved to `src/models/tuned_model.pkl`.
