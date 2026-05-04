import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.core.drift_store import DriftStore
from api.core.metrics import (
    churn_rate_gauge,
    drift_checks_counter,
    drift_detected_gauge,
    feature_drift_score_gauge,
    features_drifted_gauge,
    production_samples_gauge,
)
from api.core.middleware import LatencyMiddleware
from api.core.slowapi import limiter
from api.routes.drift_routes import router as drift_router
from api.routes.prediction_routes import router as prediction_router
from api.services.drift_service import DriftService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_models_dir = os.getenv("MODELS_PATH", str(Path(__file__).parents[2] / "models"))
MODEL_PATH = Path(_models_dir) / "tuned_model.pkl"
REFERENCE_STATS_PATH = Path(_models_dir) / "reference_stats.json"

DRIFT_CHECK_INTERVAL = int(os.getenv("DRIFT_CHECK_INTERVAL", "300"))  # seconds


async def watch_model_file(app: FastAPI):
    """Hot-reloads model whenever the file is updated on disk (zero-downtime)."""
    last_mtime = os.path.getmtime(MODEL_PATH) if MODEL_PATH.exists() else 0
    while True:
        await asyncio.sleep(5)
        try:
            if MODEL_PATH.exists():
                current_mtime = os.path.getmtime(MODEL_PATH)
                if current_mtime > last_mtime:
                    logger.info("[Watcher] New model detected — hot-reloading...")
                    app.state.model = joblib.load(MODEL_PATH)
                    last_mtime = current_mtime
                    logger.info("[Watcher] Model reloaded successfully.")
                    # Reload reference stats in case they were regenerated alongside the model
                    app.state.drift_service.reload()
        except Exception as e:
            logger.error(f"[Watcher] Reload failed: {e}")


async def run_drift_check(app: FastAPI):
    """Periodically computes drift metrics and updates Prometheus gauges."""
    while True:
        await asyncio.sleep(DRIFT_CHECK_INTERVAL)
        try:
            store: DriftStore = app.state.drift_store
            service: DriftService = app.state.drift_service

            production_samples_gauge.set(store.count())

            production_df = store.get_dataframe()
            report = service.compute_drift(production_df)

            if report is not None:
                drift_checks_counter.inc()
                drift_detected_gauge.set(1 if report.drift_detected else 0)
                features_drifted_gauge.set(len(report.drifted_features))
                churn_rate_gauge.set(report.production_churn_rate)
                for feat, score in report.feature_scores.items():
                    feature_drift_score_gauge.labels(feature=feat).set(score)

                app.state.last_drift_report = report
                app.state.last_drift_check = datetime.now(timezone.utc).isoformat()

                if report.drift_detected:
                    logger.warning(
                        "[Drift] DRIFT DETECTED — features: %s | churn rate: %.2f%% (ref %.2f%%) | retrain recommended: %s",
                        report.drifted_features,
                        report.production_churn_rate * 100,
                        report.reference_churn_rate * 100,
                        report.retrain_recommended,
                    )
                else:
                    logger.info(
                        "[Drift] No drift detected — %d samples | churn rate: %.2f%%",
                        report.n_samples,
                        report.production_churn_rate * 100,
                    )
        except Exception as e:
            logger.error(f"[Drift] Check failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        app.state.model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Run the trainer first.")
        app.state.model = None

    app.state.drift_store = DriftStore(maxlen=1000)
    app.state.drift_service = DriftService(REFERENCE_STATS_PATH)
    app.state.last_drift_report = None
    app.state.last_drift_check = None

    watcher_task = asyncio.create_task(watch_model_file(app))
    drift_task = asyncio.create_task(run_drift_check(app))
    yield
    watcher_task.cancel()
    drift_task.cancel()
    logger.info("API shutdown")


app = FastAPI(
    title="ML-Churn-Prediction",
    description=(
        "Step 4: Churn prediction API — full monitoring stack "
        "(Prometheus metrics, Grafana dashboards, data drift detection) + Model Card."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

# Prometheus — auto-instrument all routes and expose /metrics
Instrumentator().instrument(app).expose(app)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(LatencyMiddleware)
app.include_router(prediction_router, tags=["prediction"])
app.include_router(drift_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
