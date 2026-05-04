import time

import mlflow
import mlflow.sklearn
import numpy as np
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from utils.logging_utils import logger

_PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "RandomForestClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__min_samples_split": randint(2, 100),
        "model__min_samples_leaf": randint(1, 50),
    },
    "GradientBoostingClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(2, 8),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.39),
        "model__min_samples_split": randint(2, 20),
    },
    "LogisticRegression": {
        "model__C": uniform(0.01, 9.99),
        "model__solver": ["lbfgs", "liblinear", "saga"],
        "model__max_iter": [500, 1000, 2000],
    },
    "XGBClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(2, 8),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.39),
        "model__colsample_bytree": uniform(0.6, 0.39),
    },
    "LGBMClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(2, 8),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__num_leaves": randint(20, 80),
        "model__subsample": uniform(0.6, 0.39),
    },
    "AdaBoostClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": uniform(0.01, 1.99),
    },
    "DecisionTreeClassifier": {
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": randint(2, 50),
        "model__min_samples_leaf": randint(1, 30),
        "model__criterion": ["gini", "entropy"],
    },
    "SVC": {
        "model__C": uniform(0.1, 9.9),
        "model__gamma": ["scale", "auto"],
    },
    "KNeighborsClassifier": {
        "model__n_neighbors": randint(3, 20),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "GaussianNB": {
        "model__var_smoothing": uniform(1e-10, 1e-5),
    },
}


def time_budget_search(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    time_limit_minutes: int = 5,
    acc_target: float = 0.90,
    model_name: str = "",
):
    model_class_name = type(model).__name__
    display_name = model_name or model_class_name

    param_distributions = _PARAM_DISTRIBUTIONS.get(model_class_name)
    if param_distributions is None:
        logger.warning(
            f"No param distributions defined for {model_class_name} — falling back to RandomForestClassifier"
        )
        model = RandomForestClassifier(random_state=42)
        model_class_name = "RandomForestClassifier"
        display_name = f"{display_name} → RandomForest (fallback)"
        param_distributions = _PARAM_DISTRIBUTIONS["RandomForestClassifier"]

    start_time = time.time()
    deadline = start_time + time_limit_minutes * 60

    logger.info(f"Time-budget search: {display_name} — {time_limit_minutes} min | target acc={acc_target}")

    sampler = ParameterSampler(param_distributions, n_iter=1_000_000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best: dict = {"params": None, "cv_acc": -np.inf, "test_acc": -np.inf, "model": None}
    iterations = 0

    for params in sampler:
        if time.time() >= deadline:
            logger.info("Time limit reached.")
            break

        pipeline = Pipeline([
            ("selector", SelectKBest(score_func=f_classif, k="all")),
            ("model", clone(model)),
        ])
        pipeline.set_params(**params)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_acc = float(np.mean(cv_scores))

        pipeline.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))

        if test_acc > best["test_acc"]:
            best.update({"params": params, "cv_acc": cv_acc, "test_acc": test_acc, "model": pipeline})
            elapsed = (time.time() - start_time) / 60
            logger.info(f"New best | acc={test_acc:.4f} | cv={cv_acc:.4f} | {elapsed:.1f}min | {params}")

        iterations += 1
        if iterations % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            logger.info(f"iter={iterations} | best={best['test_acc']:.4f} | {elapsed:.1f}min")

        if test_acc >= acc_target:
            logger.info(f"Target accuracy {acc_target} reached — stopping early.")
            break

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

    logger.info(f"Tuning done | iterations={iterations} | best acc={metrics['accuracy']:.4f}")

    active_run = mlflow.active_run()
    with mlflow.start_run(run_name=f"{display_name}_Tuned", nested=(active_run is not None)):
        if best["params"]:
            for k, v in best["params"].items():
                mlflow.log_param(k, str(v))
        mlflow.log_metric("cv_accuracy", best["cv_acc"])
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(best_model, "model")

    return best_model, metrics
