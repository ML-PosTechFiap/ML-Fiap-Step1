from fastapi import APIRouter, Request, HTTPException, status

from api.core.slowapi import limiter
from api.schemas.churn_schemas import ChurnInput, ChurnOutput
from api.services.prediction_service import PredictionService

router = APIRouter()


@router.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    model_status = "loaded" if request.app.state.model is not None else "not_loaded"
    return {"status": "ok", "model_status": model_status, "version": request.app.version}


@router.post("/predict", response_model=ChurnOutput, status_code=status.HTTP_200_OK)
@limiter.limit("60/minute")
async def predict_churn(request: Request, input_data: ChurnInput):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run the trainer first")

    service = PredictionService(request.app.state.model)
    try:
        result = service.predict(input_data.model_dump())
        return ChurnOutput(
            prediction=result["prediction"],
            probability=result["probability"],
            model_version=request.app.version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
