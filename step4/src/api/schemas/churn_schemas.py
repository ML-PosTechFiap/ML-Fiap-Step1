from typing import Optional

from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    gender: Optional[str] = Field(None, description="Female or Male")
    SeniorCitizen: Optional[int] = Field(None, description="0 or 1")
    Partner: Optional[str] = Field(None, description="Yes or No")
    Dependents: Optional[str] = Field(None, description="Yes or No")
    tenure: Optional[int] = Field(None, description="Months with company (0–72)")
    PhoneService: Optional[str] = Field(None, description="Yes or No")
    MultipleLines: Optional[str] = Field(None)
    InternetService: Optional[str] = Field(None, description="DSL, Fiber optic, or No")
    OnlineSecurity: Optional[str] = Field(None)
    OnlineBackup: Optional[str] = Field(None)
    DeviceProtection: Optional[str] = Field(None)
    TechSupport: Optional[str] = Field(None)
    StreamingTV: Optional[str] = Field(None)
    StreamingMovies: Optional[str] = Field(None)
    Contract: Optional[str] = Field(None, description="Month-to-month, One year, or Two year")
    PaperlessBilling: Optional[str] = Field(None, description="Yes or No")
    PaymentMethod: Optional[str] = Field(None)
    MonthlyCharges: Optional[float] = Field(None)
    TotalCharges: Optional[str] = Field(None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
            }
        }
    }


class ChurnOutput(BaseModel):
    prediction: str = Field(..., description="Yes or No")
    probability: float = Field(..., description="Churn probability (0.0–1.0)")
    model_version: str = Field("0.4.0", description="Model version")
