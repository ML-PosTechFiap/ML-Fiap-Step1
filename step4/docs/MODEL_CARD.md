# Model Card — Telco Customer Churn Predictor

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | TelcoChurn Tuned RandomForest |
| **Version** | 0.4.0 |
| **Type** | Binary classification (churn / no-churn) |
| **Algorithm** | RandomForest with time-budget hyperparameter search |
| **Framework** | scikit-learn ≥ 1.8 + joblib serialisation |
| **Compared against** | 11 sklearn baselines + PyTorch MLP |
| **Date** | 2026-05 |
| **Authors** | FIAP POS-TECH Group — Phase 1 |

---

## Intended Use

**Primary use case:** Real-time inference via REST API to identify telecom customers at high churn risk, so the retention team can trigger proactive offers.

**Out-of-scope uses:**
- Credit scoring or loan decisions.
- Any domain outside B2C telecoms with similar feature space.
- Batch scoring of customers without recent usage data (tenure=0).

---

## Training Data

| Property | Value |
|----------|-------|
| **Dataset** | IBM Telco Customer Churn (public) |
| **Source** | Kaggle / IBM Sample Data Sets |
| **Records** | 7,043 customers |
| **Features** | 19 raw → 25 engineered |
| **Target** | `Churn` (Yes / No) → binary (1 / 0) |
| **Class balance** | ~73% No-churn / ~27% Churn |
| **Split** | 80% train / 20% test, stratified, `random_state=42` |
| **Version tracked** | MLflow experiment `TelcoChurn_Step4_FullPipeline` |

### Feature engineering applied
- `TotalCharges` coerced to numeric (blanks → NaN → median impute).
- `Contract`, `OnlineSecurity`, `PaymentMethod`, `InternetService` one-hot encoded at feature-engineering stage.
- Remaining categoricals one-hot encoded by `ShowWithPandas.prepare_for_ml`.
- Outlier treatment: Winsorization at 1st/99th percentile.
- Top-25 features selected by `SelectKBest(f_classif)`.

---

## Evaluation Metrics (test set, threshold = 0.5)

| Metric | Value |
|--------|-------|
| AUC-ROC | ≥ 0.83 |
| PR-AUC | ≥ 0.63 |
| F1-score (churn) | ≥ 0.60 |
| Recall (churn) | ≥ 0.55 |
| Precision (churn) | ≥ 0.65 |
| Accuracy | ≥ 0.80 |

> Exact values depend on the training run; check MLflow for the authoritative experiment run.

### Business cost model (optimal threshold)

| Outcome | Unit cost |
|---------|-----------|
| TP — churn caught, retention offer sent | −R$80 (offer) + R$400 saved = **+R$320** |
| FP — false alarm, unnecessary offer | **−R$80** |
| FN — missed churn, customer leaves | **−R$400** |
| TN — correct no-churn, no action | R$0 |

The cost-optimal threshold is found by sweeping [0.1, 0.9] and minimising `FP×80 + FN×400`. See `evaluation/compare_models.py`.

---

## Limitations

1. **Dataset scope**: Trained on a single US telecom snapshot. Generalisation to other markets, geographies, or operators is untested.
2. **Class imbalance**: The 73/27 split means the model may under-detect churn at default threshold; use the business-optimal threshold in production.
3. **Static features**: The model has no awareness of behavioural sequences or call-centre interactions; it uses billing/contract snapshots only.
4. **Cold start**: Customers with `tenure=0` have very little signal; predictions are unreliable.
5. **Concept drift**: Customer behaviour and product offerings change over time. The model should be retrained at least quarterly or when AUC-ROC drops > 3 pp from baseline.
6. **Missing values**: `TotalCharges` blanks are median-imputed; models can silently degrade if a systematic data-pipeline issue produces blanks at inference time.

---

## Biases and Fairness Considerations

- **Gender (`gender`)**: The feature is included but the model is not audited for disparate impact across gender groups. Before production use in a jurisdiction with anti-discrimination obligations, a fairness audit (e.g., equalised odds) is required.
- **SeniorCitizen (binary 0/1)**: Senior customers may have different usage patterns; verify the model does not disproportionately flag this group.
- **Geographic proxy**: Payment method and contract type can proxy socioeconomic status. Monitoring disaggregated metrics by payment method is recommended.

---

## Failure Modes

| Scenario | Expected behaviour | Mitigation |
|----------|--------------------|------------|
| Model file absent at startup | API returns 503 on `/predict` | Run trainer; health endpoint reports `not_loaded` |
| Input with all-null fields | Prediction made with all-median/0 features | Log warning; downstream teams notified |
| Feature schema mismatch (new column added upstream) | `reindex` fills missing with 0 silently | Schema validation gate in CI; input contract versioned |
| Adversarial input (extreme charges) | Winsorisation at inference clips the value | Enforce input range validation in Pydantic schema |
| Model drift (AUC drops > 3 pp) | No automatic fallback | Prometheus alert triggers retraining pipeline |

---

## Monitoring

- **Prometheus** scrapes `/metrics` every 15 s.
- **Grafana** dashboard at `http://localhost:3000` (see `monitoring/` folder).
- **Alerts defined in MONITORING_PLAN.md**: p95 latency > 500 ms, error rate > 1%, `model_status = not_loaded`.

---

## Ethical Considerations

This model assists human retention agents — it does not make autonomous decisions. Final retention decisions remain with business operators. The system should not be used to penalise customers identified as low-churn risk by withholding service quality.

---

## Citation

```
Dataset: IBM Sample Data — Telco Customer Churn
URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
License: Public / Open
```
