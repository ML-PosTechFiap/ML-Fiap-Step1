"""
Step 3 — Full Training Pipeline

Steps:
  1. Load data
  2. Show dataframe analysis (EDA)
  3. Feature engineering + prepare for ML (EDA treatment)
  4. Save processed data
  5. Train all sklearn models (11 models)
  6. Train PyTorch MLP
  7. Compare all models (≥ 4 metrics + business cost)
  8. Plot ROC/PR curves + cost trade-off analysis
  9. Tune best sklearn model with time-budget random search
  10. Save final model
"""

import os
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from eda import ShowWithPandas
from enums import DFMethods, WriterType
from features import apply_feature_engineering
from train import MLTrainer, time_budget_rf_search, MLPTrainer
from evaluation import compare_all_models, analyze_cost_tradeoff
from evaluation.compare_models import plot_roc_pr_curves
from utils.logging_utils import logger

warnings.filterwarnings("ignore")

ROOT_PATH = Path(__file__).parents[2]
_data_override = os.getenv("DATA_PATH")
DATA_PATH = Path(_data_override) if _data_override else ROOT_PATH / "data" / "Telco-Customer-Churn.csv"
GRAPHS_PATH = ROOT_PATH / "graphs"
NAME_PROCESS = "TelcoChurn"


def load_data() -> tuple[pd.DataFrame, str]:
    logger.info("--- STEP 1: LOADING DATA ---")
    app = ShowWithPandas()
    extension = DATA_PATH.suffix.lower().lstrip(".")
    df = app._load_content(DATA_PATH, extension)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {DATA_PATH}")
    return df, extension


def show_dataframe(df: pd.DataFrame, extension: str):
    logger.info("--- STEP 2: DATAFRAME ANALYSIS ---")
    ShowWithPandas().show_content(df, extension, category=DFMethods.ALL)


def apply_eda(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("--- STEP 3: EDA + FEATURE ENGINEERING ---")
    app = ShowWithPandas()

    plots_path = app.generate_graphics(df, save_dir="graphs")
    mlflow.log_artifacts(str(plots_path), artifact_path="plots_raw")

    df_fe = apply_feature_engineering(df)

    df_ml = app.prepare_for_ml(
        df_fe,
        normalize=False,
        handle_outliers_method="winsorize",
        handle_nulls_method="median",
        encoding="onehot",
    )
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


def train_all_models(df_ml: pd.DataFrame):
    logger.info("--- STEP 5-6: TRAINING (SKLEARN + MLP) ---")

    target = "Churn" if "Churn" in df_ml.columns else df_ml.columns[-1]
    X = df_ml.drop(columns=[target])
    y = df_ml[target]

    assert df_ml.isna().sum().sum() == 0, "Dataset still contains NaN"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlflow.log_param("num_samples", len(df_ml))
    mlflow.log_param("num_features_raw", X_train.shape[1])

    trainer = MLTrainer()
    k = min(25, X_train.shape[1])
    X_train_sel, X_test_sel, selector = trainer.feature_selection(X_train, X_test, y_train, k=k)
    mlflow.log_param("k_features_selected", k)
    logger.info(f"Feature selection: {X_train_sel.shape[1]} features")

    # --- sklearn models ---
    logger.info("Training 11 sklearn models...")
    _, sklearn_results = trainer.train_and_evaluate(X_train_sel, X_test_sel, y_train, y_test)

    # --- PyTorch MLP ---
    logger.info("Training PyTorch MLP...")
    mlp_trainer = MLPTrainer(
        input_dim=X_train_sel.shape[1],
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        lr=1e-3,
        batch_size=64,
        max_epochs=200,
        patience=15,
    )
    mlp_metrics = mlp_trainer.train_and_log(X_train_sel, X_test_sel, y_train, y_test)
    sklearn_results["MLP (PyTorch)"] = {"model": mlp_trainer, "accuracy": mlp_metrics["accuracy"]}

    # --- comparison ---
    logger.info("--- STEP 7: MODEL COMPARISON ---")
    model_results = {}
    for name, data in sklearn_results.items():
        model = data["model"]
        if name == "MLP (PyTorch)":
            y_pred = mlp_trainer.predict(X_test_sel)
            y_proba = mlp_trainer.predict_proba(X_test_sel)[:, 1]
        else:
            y_pred = model.predict(X_test_sel)
            y_proba = model.predict_proba(X_test_sel)[:, 1] if hasattr(model, "predict_proba") else None
        model_results[name] = {"y_true": y_test.values, "y_pred": y_pred, "y_proba": y_proba}

    eval_dir = ROOT_PATH / "graphs"
    comparison_df = compare_all_models(model_results, save_dir=eval_dir)

    # --- cost trade-off ---
    logger.info("--- STEP 8: COST TRADE-OFF ANALYSIS ---")
    proba_models = {n: d["y_proba"] for n, d in model_results.items() if d["y_proba"] is not None}
    analyze_cost_tradeoff(proba_models, y_test.values, save_dir=eval_dir)
    plot_roc_pr_curves(proba_models, y_test.values, save_dir=eval_dir)

    # --- time-budget tuning ---
    logger.info("--- STEP 9: TIME-BUDGET RF TUNING ---")
    tuned_model, tuning_metrics = time_budget_rf_search(
        X_train_sel, y_train, X_test_sel, y_test,
        time_limit_minutes=60,
        acc_target=0.90,
    )
    mlflow.log_params({k: str(v) for k, v in tuned_model.named_steps["model"].get_params().items()})
    mlflow.log_metrics({f"tuned_{k}": v for k, v in tuning_metrics.items()})

    # --- save best model ---
    logger.info("--- STEP 10: SAVING FINAL MODEL ---")
    models_dir = ROOT_PATH / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "tuned_model.pkl"
    trainer.save_model(tuned_model, str(model_file))
    mlflow.log_artifact(str(model_file), artifact_path="model")
    mlflow.sklearn.log_model(
        tuned_model,
        artifact_path="final_tuned_model",
        registered_model_name=f"{NAME_PROCESS}_Tuned_RF_Model",
    )

    selected_features = X_train.columns[selector.get_support()]
    trainer.show_feature_importance(tuned_model.named_steps["model"], selected_features)

    return comparison_df, tuned_model


if __name__ == "__main__":
    load_dotenv()

    mlflow_uri = os.getenv("ML_FLOW_API", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(f"{NAME_PROCESS}_Step3_FullPipeline")

    logger.info("=== STEP 3: FULL PIPELINE START ===")

    with mlflow.start_run(run_name="Step3_Full_Pipeline"):
        df_raw, extension = load_data()
        show_dataframe(df_raw, extension)
        df_ml = apply_eda(df_raw)
        save_processed_data(df_ml, extension)
        comparison_df, final_model = train_all_models(df_ml)

    logger.info("=== PIPELINE COMPLETE ===")
    logger.info(f"\n{comparison_df.to_string()}")
