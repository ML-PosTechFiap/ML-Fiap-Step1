"""
Model comparison and business cost analysis.

Metrics (≥ 4):  Accuracy, AUC-ROC, PR-AUC, F1, Recall, Precision
Cost model (from ML Canvas):
  - FN: customer churns undetected → loss of R$400 (CAC)
  - FP: wrong retention attempt    → wasted R$80  (retention cost)
  - TP: churn caught               → spend R$80, save R$400 → net +R$320
  - TN: no cost
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.logging_utils import logger

FP_COST = 80    # R$ wasted retention attempt
FN_COST = 400   # R$ lost churned customer (CAC)
TP_NET_GAIN = FN_COST - FP_COST  # R$320 net gain per correctly caught churn


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["business_cost_avoided"] = float(tp * TP_NET_GAIN - fp * FP_COST - fn * FN_COST)
    metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"] = int(tp), int(fp), int(fn), int(tn)
    return metrics


def compare_all_models(model_results: dict, save_dir: Path | None = None) -> pd.DataFrame:
    """
    model_results: {
        "Model Name": {
            "y_pred": array,
            "y_proba": array | None,
            "y_true": array,
        }
    }
    Returns a DataFrame sorted by roc_auc descending.
    Logs comparison table and plots to MLflow.
    """
    rows = []
    for name, data in model_results.items():
        m = compute_metrics(data["y_true"], data["y_pred"], data.get("y_proba"))
        m["model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")
    priority_cols = ["roc_auc", "pr_auc", "recall", "f1", "precision", "accuracy", "business_cost_avoided"]
    df = df[[c for c in priority_cols if c in df.columns] + [c for c in df.columns if c not in priority_cols]]

    if "roc_auc" in df.columns:
        df = df.sort_values("roc_auc", ascending=False)

    logger.info("\n=== MODEL COMPARISON ===\n%s", df.to_string())

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "model_comparison.csv"
        df.to_csv(csv_path)
        mlflow.log_artifact(str(csv_path), artifact_path="evaluation")
        logger.info(f"Comparison saved to {csv_path}")

    return df


def analyze_cost_tradeoff(
    models_with_proba: dict,
    y_true,
    save_dir: Path | None = None,
    n_thresholds: int = 100,
) -> dict:
    """
    Sweep probability thresholds for each model and find the one minimising
    total business cost (FP×80 + FN×400).

    models_with_proba: {"Model Name": y_proba_array}
    Returns dict: {"Model Name": {"optimal_threshold", "min_cost", "tp", "fp", "fn"}}
    """
    results = {}
    thresholds = np.linspace(0, 1, n_thresholds)

    fig, axes = plt.subplots(1, len(models_with_proba), figsize=(6 * len(models_with_proba), 5))
    if len(models_with_proba) == 1:
        axes = [axes]

    for ax, (name, y_proba) in zip(axes, models_with_proba.items()):
        costs = []
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()
            total_cost = fp * FP_COST + fn * FN_COST
            costs.append((t, total_cost, tp, fp, fn, tn))

        costs_arr = np.array(costs)
        best_idx = np.argmin(costs_arr[:, 1])
        opt_t, min_cost = costs_arr[best_idx, 0], costs_arr[best_idx, 1]
        tp_opt, fp_opt, fn_opt = int(costs_arr[best_idx, 2]), int(costs_arr[best_idx, 3]), int(costs_arr[best_idx, 4])

        results[name] = {
            "optimal_threshold": float(opt_t),
            "min_business_cost_R$": float(min_cost),
            "tp": tp_opt,
            "fp": fp_opt,
            "fn": fn_opt,
            "cost_avoided_vs_no_model": float(fn_opt * FN_COST - min_cost),
        }

        logger.info(
            f"{name} | optimal threshold={opt_t:.2f} | cost=R${min_cost:.0f} | "
            f"TP={tp_opt} FP={fp_opt} FN={fn_opt}"
        )

        ax.plot(costs_arr[:, 0], costs_arr[:, 1], label="Total cost")
        ax.axvline(opt_t, color="red", linestyle="--", label=f"Opt t={opt_t:.2f}")
        ax.set_title(f"{name}\nOpt threshold = {opt_t:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Business cost (R$)")
        ax.legend()

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / "cost_tradeoff.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(str(fig_path), artifact_path="evaluation")
        logger.info(f"Cost trade-off plot saved to {fig_path}")

    plt.close(fig)

    cost_df = pd.DataFrame(results).T
    logger.info("\n=== COST TRADE-OFF SUMMARY ===\n%s", cost_df.to_string())
    if save_dir:
        cost_csv = save_dir / "cost_tradeoff.csv"
        cost_df.to_csv(cost_csv)
        mlflow.log_artifact(str(cost_csv), artifact_path="evaluation")

    return results


def plot_roc_pr_curves(models_with_proba: dict, y_true, save_dir: Path | None = None):
    """Plots ROC and PR curves for all models side by side."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    for name, y_proba in models_with_proba.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        ax_pr.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.legend(loc="lower right")

    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / "roc_pr_curves.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(str(fig_path), artifact_path="evaluation")

    plt.close(fig)
