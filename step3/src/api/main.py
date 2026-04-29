import asyncio
import joblib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.core.slowapi import limiter
from api.core.middleware import LatencyMiddleware
from api.routes.prediction_routes import router as prediction_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parents[3]
MODEL_PATH = ROOT_PATH / "models" / "tuned_model.pkl"


async def watch_model_file(app: FastAPI):
    """Async file watcher — hot-reloads model whenever the file is updated on disk."""
    last_mtime = os.path.getmtime(MODEL_PATH) if MODEL_PATH.exists() else 0
    while True:
        await asyncio.sleep(5)
        try:
            if MODEL_PATH.exists():
                current_mtime = os.path.getmtime(MODEL_PATH)
                if current_mtime > last_mtime:
                    logger.info("[Watcher] New model detected — hot-reloading (zero-downtime)...")
                    app.state.model = joblib.load(MODEL_PATH)
                    last_mtime = current_mtime
                    logger.info("[Watcher] Model reloaded successfully.")
        except Exception as e:
            logger.error(f"[Watcher] Reload failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        app.state.model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Run the trainer first.")
        app.state.model = None

    watcher_task = asyncio.create_task(watch_model_file(app))
    yield
    watcher_task.cancel()
    logger.info("API shutdown")


app = FastAPI(
    title="ML-Churn-Prediction",
    description="Step 3: Churn prediction API — best model from full pipeline (sklearn + MLP).",
    version="0.3.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(LatencyMiddleware)
app.include_router(prediction_router, tags=["prediction"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
