import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils.logging_utils import logger


class MLTrainer:

    def handle_missing(self, X_train, X_test):
        imputer = SimpleImputer(strategy="median")
        return imputer.fit_transform(X_train), imputer.transform(X_test)

    def feature_selection(self, X_train, X_test, y_train, k=25):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        return X_train_sel, X_test_sel, selector

    def get_models(self):
        return {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        }

    def cross_validation(self, model, X_train, y_train):
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        return float(scores.mean())

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        models = self.get_models()
        results = {}
        best_model, best_score, best_name = None, 0.0, None

        for name, model in models.items():
            logger.info(f"Training {name}...")
            with mlflow.start_run(run_name=name, nested=True):
                cv_auc = self.cross_validation(model, X_train, y_train)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = float(accuracy_score(y_test, preds))
                f1 = float(f1_score(y_test, preds, zero_division=0))
                rec = float(recall_score(y_test, preds, zero_division=0))

                auc = 0.0
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)[:, 1]
                    auc = float(roc_auc_score(y_test, proba))

                logger.info(
                    f"{name} | AUC={auc:.4f} | F1={f1:.4f} | Recall={rec:.4f} | Acc={acc:.4f}"
                )
                logger.info(classification_report(y_test, preds))

                mlflow.log_param("model_type", name)
                mlflow.log_metric("cv_auc", cv_auc)
                mlflow.log_metric("test_auc", auc)
                mlflow.log_metric("test_f1", f1)
                mlflow.log_metric("test_recall", rec)
                mlflow.log_metric("test_accuracy", acc)
                mlflow.sklearn.log_model(model, name)

                results[name] = {
                    "model": model,
                    "auc": auc,
                    "f1": f1,
                    "recall": rec,
                    "accuracy": acc,
                }

                if auc > best_score:
                    best_score, best_model, best_name = auc, model, name

        logger.info(f"\nBest model: {best_name} (AUC={best_score:.4f})")
        return best_model, results

    def show_feature_importance(self, model, feature_names):
        if hasattr(model, "feature_importances_"):
            df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            logger.info("\nTop feature importances:\n%s", df.head(15).to_string())

    def save_model(self, model, filename="trained_model.pkl"):
        joblib.dump(model, filename)
        logger.info(f"Model saved: {filename}")
