import pandas as pd
from utils.logging_utils import logger


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_engineered = df.copy()

    if "Churn" in df_engineered.columns or "customerID" in df_engineered.columns:
        logger.info("Detected Churn dataset — applying specific feature engineering")

        if "TotalCharges" in df_engineered.columns:
            df_engineered["TotalCharges"] = pd.to_numeric(
                df_engineered["TotalCharges"], errors="coerce"
            )

        cols_to_encode = ["Contract", "OnlineSecurity", "PaymentMethod", "InternetService"]
        for col in cols_to_encode:
            if col in df_engineered.columns:
                dummies = pd.get_dummies(df_engineered[col], prefix=col)
                df_engineered = pd.concat([df_engineered, dummies], axis=1)
                df_engineered.drop(col, axis=1, inplace=True)

        if "InternetService_No" in df_engineered.columns:
            df_engineered.drop("InternetService_No", axis=1, inplace=True)

        if "Churn" in df_engineered.columns:
            df_engineered["Churn"] = df_engineered["Churn"].map({"Yes": 1, "No": 0})

        if "customerID" in df_engineered.columns:
            df_engineered.drop("customerID", axis=1, inplace=True)

    return df_engineered
