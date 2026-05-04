from fastapi import APIRouter, Request

router = APIRouter(prefix="/drift", tags=["drift"])


@router.get("/status")
async def drift_status(request: Request):
    store = getattr(request.app.state, "drift_store", None)
    service = getattr(request.app.state, "drift_service", None)
    report = getattr(request.app.state, "last_drift_report", None)
    last_check = getattr(request.app.state, "last_drift_check", None)

    n_samples = store.count() if store else 0

    if service is None or not service.is_ready():
        return {
            "status": "disabled",
            "reason": "Reference stats not found. Run the trainer first to generate reference_stats.json.",
            "production_samples": n_samples,
        }

    if report is None:
        return {
            "status": "pending",
            "reason": f"Not enough samples yet ({n_samples}/50 minimum).",
            "production_samples": n_samples,
        }

    return {
        "status": "ok",
        "drift_detected": report.drift_detected,
        "retrain_recommended": report.retrain_recommended,
        "drifted_features": report.drifted_features,
        "feature_scores": report.feature_scores,
        "feature_pvalues": {k: v for k, v in report.feature_pvalues.items() if v == v},  # drop NaN
        "production_churn_rate": round(report.production_churn_rate, 4),
        "reference_churn_rate": round(report.reference_churn_rate, 4),
        "prediction_drift": report.prediction_drift,
        "production_samples": report.n_samples,
        "last_check": last_check,
    }
