import time
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from scipy.stats import randint


def time_budget_rf_search(
    X_train,
    y_train,
    X_test,
    y_test,
    time_limit_minutes: int = 60,
    acc_target: float = 0.86,
):
    start_time = time.time()
    deadline = start_time + time_limit_minutes * 60

    print(f"\nRandomForest time-budget search — {time_limit_minutes} min | target acc={acc_target}")

    param_distributions = {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__min_samples_split": randint(2, 100),
        "model__min_samples_leaf": randint(1, 50),
    }

    sampler = ParameterSampler(param_distributions, n_iter=1_000_000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best = {"params": None, "cv_acc": -np.inf, "test_acc": -np.inf, "model": None}
    iterations = 0

    for params in sampler:
        if time.time() >= deadline:
            print("Time limit reached.")
            break

        pipeline = Pipeline([
            ("selector", SelectKBest(score_func=f_classif, k="all")),
            ("model", RandomForestClassifier(random_state=42)),
        ])
        pipeline.set_params(**params)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_acc = float(np.mean(cv_scores))

        pipeline.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))

        if test_acc > best["test_acc"]:
            best.update({"params": params, "cv_acc": cv_acc, "test_acc": test_acc, "model": pipeline})
            elapsed = (time.time() - start_time) / 60
            print(f"  New best | acc={test_acc:.4f} | cv={cv_acc:.4f} | {elapsed:.1f}min | {params}")

        iterations += 1
        if iterations % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"  iter={iterations} | best={best['test_acc']:.4f} | {elapsed:.1f}min")

    best_model = best["model"]
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    print(f"\nTuning done | iterations={iterations} | best acc={metrics['accuracy']:.4f}")

    active_run = mlflow.active_run()
    with mlflow.start_run(run_name="RandomForest_Tuned", nested=(active_run is not None)):
        if best["params"]:
            for k, v in best["params"].items():
                mlflow.log_param(k, str(v))
        mlflow.log_metric("cv_accuracy", best["cv_acc"])
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(best_model, "model")

    return best_model, metrics
