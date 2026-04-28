"""
Step 2 — Training Pipeline

Steps:
  1. Load data
  2. Show dataframe analysis (EDA)
  3. Feature engineering + basic preprocessing
  4. Save processed data
  5. Train 5 models (Dummy, LR, RF, XGB, LGBM)
  6. Save best model
"""

import os
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from eda import ShowWithPandas
from enums import DFMethods, WriterType
from features import apply_feature_engineering
from train import MLTrainer
from utils.logging_utils import logger

warnings.filterwarnings("ignore")

ROOT_PATH = Path(__file__).parents[2]
_data_override = os.getenv("DATA_PATH")
DATA_PATH = (
    Path(_data_override)
    if _data_override
    else ROOT_PATH / "data" / "Telco-Customer-Churn.csv"
)
NAME_PROCESS = "TelcoChurn"


def load_data() -> tuple[pd.DataFrame, str]:
    logger.info("--- STEP 1: LOADING DATA ---")
    app = ShowWithPandas()
    extension = DATA_PATH.suffix.lower().lstrip(".")
    df = app._load_content(DATA_PATH, extension)
    logger.info(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {DATA_PATH}")
    return df, extension


def show_dataframe(df: pd.DataFrame, extension: str):
    logger.info("--- STEP 2: DATAFRAME ANALYSIS ---")
    ShowWithPandas().show_content(df, extension, category=DFMethods.ALL)


def apply_eda(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- STEP 3: FEATURE ENGINEERING + PREPROCESSING ---")
    app = ShowWithPandas()
    df_fe = apply_feature_engineering(df)
    df_ml = app.handle_nulls(df_fe, strategy="median")
    df_ml = app.encode_categorical(df_ml, method="onehot")
    df_ml = app.final_nan_cleanup(df_ml)
    logger.info(f"Processed shape: {df_ml.shape}")
    return df_ml


def save_processed_data(df: pd.DataFrame, extension: str) -> Path:
    logger.info("--- STEP 4: SAVING PROCESSED DATA ---")
    output_path = ROOT_PATH / "data" / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"processed_data.{extension}"
    WriterType(extension.lower()).write(df, file_path)
    mlflow.log_artifact(str(file_path), artifact_path="data")
    logger.info(f"Saved to {file_path}")
    return file_path


def train_models(df_ml: pd.DataFrame):
    logger.info("--- STEP 5: TRAINING MODELS ---")

    target = "Churn" if "Churn" in df_ml.columns else df_ml.columns[-1]
    X = df_ml.drop(columns=[target])
    y = df_ml[target]

    assert df_ml.isna().sum().sum() == 0, "Dataset still contains NaN"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlflow.log_param("num_samples", len(df_ml))
    mlflow.log_param("num_features", X_train.shape[1])

    trainer = MLTrainer()
    k = min(25, X_train.shape[1])
    X_train_sel, X_test_sel, selector = trainer.feature_selection(
        X_train, X_test, y_train, k=k
    )
    mlflow.log_param("k_features_selected", k)
    logger.info(f"Feature selection: {X_train_sel.shape[1]} features")

    best_model, results = trainer.train_and_evaluate(
        X_train_sel, X_test_sel, y_train, y_test
    )

    models_dir = ROOT_PATH / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "tuned_model.pkl"
    trainer.save_model(best_model, str(model_file))
    mlflow.log_artifact(str(model_file), artifact_path="model")
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model",
        registered_model_name=f"{NAME_PROCESS}_Best_Model",
    )

    selected_features = X_train.columns[selector.get_support()]
    trainer.show_feature_importance(best_model, selected_features)

    return results


if __name__ == "__main__":
    load_dotenv()

    mlflow_uri = os.getenv("ML_FLOW_API", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(f"{NAME_PROCESS}_Step2_Pipeline")

    logger.info("=== STEP 2: PIPELINE START ===")

    with mlflow.start_run(run_name="Step2_Pipeline"):
        df_raw, extension = load_data()
        show_dataframe(df_raw, extension)
        df_ml = apply_eda(df_raw)
        save_processed_data(df_ml, extension)
        train_models(df_ml)

    logger.info("=== PIPELINE COMPLETE ===")
