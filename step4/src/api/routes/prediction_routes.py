from fastapi import APIRouter, HTTPException, Request, status

from api.core.drift_store import DriftStore
from api.core.slowapi import limiter
from api.schemas.churn_schemas import ChurnInput, ChurnOutput
from api.services.prediction_service import PredictionService

router = APIRouter()


@router.get("/health")
@limiter.limit("20/minute")
async def health(request: Request):
    model_status = "loaded" if request.app.state.model is not None else "not_loaded"
    return {"status": "ok", "model_status": model_status, "version": request.app.version}


@router.post("/predict", response_model=ChurnOutput, status_code=status.HTTP_200_OK)
@limiter.limit("60/hour")
async def predict_churn(request: Request, input_data: ChurnInput):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run the trainer first")

    service = PredictionService(request.app.state.model)
    try:
        raw = input_data.model_dump()
        result = service.predict(raw)

        store: DriftStore = getattr(request.app.state, "drift_store", None)
        if store is not None:
            store.log(raw, result["prediction"], result["probability"])

        return ChurnOutput(
            prediction=result["prediction"],
            probability=result["probability"],
            model_version=request.app.version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
