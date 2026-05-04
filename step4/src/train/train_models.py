import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils.logging_utils import logger

_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


class MLTrainer:

    def split_features_target(self, df: pd.DataFrame, target: str):
        return df.drop(columns=[target]), df[target]

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
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Gaussian Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        }

    def cross_validation(self, model, X_train, y_train) -> float:
        scores = cross_val_score(model, X_train, y_train, cv=_CV, scoring="roc_auc")
        return float(scores.mean())

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        models = self.get_models()
        results = {}
        best_model, best_score, best_name = None, 0, None

        for name, model in models.items():
            logger.info(f"Training {name}")
            with mlflow.start_run(run_name=name, nested=True):
                cv_score = self.cross_validation(model, X_train, y_train)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                logger.info(f"{name} | cv_auc={cv_score:.4f} | test_acc={acc:.4f}")
                logger.info(f"\n{classification_report(y_test, preds)}")

                mlflow.log_param("model_type", name)
                mlflow.log_metric("cv_roc_auc", cv_score)
                mlflow.log_metric("test_accuracy", acc)
                mlflow.sklearn.log_model(model, name)

                results[name] = {"model": model, "accuracy": acc}

                if acc > best_score:
                    best_score, best_model, best_name = acc, model, name

        logger.info(f"Best model: {best_name} (accuracy={best_score:.4f})")
        return best_model, best_name, results

    def hyperparameter_tuning(self, model, X_train, y_train):
        param_grids = {
            RandomForestClassifier: {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
            DecisionTreeClassifier: {
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10],
            },
            GradientBoostingClassifier: {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
            },
            SVC: {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
            LogisticRegression: {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
            KNeighborsClassifier: {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
            },
            XGBClassifier: {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7],
            },
            LGBMClassifier: {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [31, 50],
            },
            AdaBoostClassifier: {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0],
            },
            GaussianNB: {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        }

        param_grid = param_grids.get(type(model))
        if param_grid is None:
            logger.warning(f"No tuning grid for {type(model).__name__}")
            return model

        grid = GridSearchCV(model, param_grid, cv=_CV, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        logger.info(f"Best params: {grid.best_params_}  CV AUC: {grid.best_score_:.4f}")
        return grid.best_estimator_

    def show_feature_importance(self, model, feature_names):
        if hasattr(model, "feature_importances_"):
            df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            logger.info(f"Top feature importances:\n{df.head(15).to_string()}")

    def save_model(self, model, filename="trained_model.pkl"):
        joblib.dump(model, filename)
        logger.info(f"Model saved: {filename}")
