import logging
from typing import Any

import pandas as pd

from eda import ShowWithPandas
from features import apply_feature_engineering

logger = logging.getLogger(__name__)

EXPECTED_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "OnlineSecurity_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "InternetService_DSL", "InternetService_Fiber optic",
    "Partner_Yes", "Dependents_Yes",
    "OnlineBackup_No internet service", "DeviceProtection_No internet service",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingMovies_No internet service",
    "PaperlessBilling_Yes",
]


class PredictionService:
    def __init__(self, model: Any):
        self.model = model
        self.eda_app = ShowWithPandas()

    def predict(self, input_data: dict) -> dict:
        if self.model is None:
            raise ValueError("Model is not loaded")

        df = pd.DataFrame([input_data])
        df_fe = apply_feature_engineering(df)
        df_ml = self.eda_app.prepare_for_ml(
            df_fe,
            normalize=False,
            handle_outliers_method="winsorize",
            handle_nulls_method="median",
            encoding="onehot",
        )
        df_final = df_ml.reindex(columns=EXPECTED_FEATURES, fill_value=0)

        prediction_idx = self.model.predict(df_final)[0]
        probabilities = self.model.predict_proba(df_final)[0]

        return {
            "prediction": "Yes" if int(prediction_idx) == 1 else "No",
            "probability": float(probabilities[1]),
        }
