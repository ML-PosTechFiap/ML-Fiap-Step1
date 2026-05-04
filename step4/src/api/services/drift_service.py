import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "Partner",
    "Dependents",
    "TechSupport",
]

# PSI > 0.2 → significant drift; KS p-value < 0.05 → drift detected
PSI_THRESHOLD = 0.2
KS_ALPHA = 0.05
PREDICTION_DRIFT_THRESHOLD = 0.10  # 10 percentage-point shift in churn rate
MIN_SAMPLES = 50


@dataclass
class DriftReport:
    n_samples: int
    drift_detected: bool
    drifted_features: list[str]
    feature_scores: dict[str, float]
    feature_pvalues: dict[str, float]
    production_churn_rate: float
    reference_churn_rate: float
    prediction_drift: bool
    retrain_recommended: bool


class DriftService:
    def __init__(self, reference_path: Path):
        self._ref_path = reference_path
        self._ref_stats: Optional[dict] = None
        self._load_reference()

    def is_ready(self) -> bool:
        return self._ref_stats is not None

    def _load_reference(self) -> None:
        if not self._ref_path.exists():
            logger.warning("[Drift] Reference stats not found at %s — drift detection disabled.", self._ref_path)
            return
        with open(self._ref_path) as f:
            self._ref_stats = json.load(f)
        logger.info("[Drift] Reference stats loaded from %s", self._ref_path)

    def reload(self) -> None:
        self._load_reference()

    def compute_drift(self, production_df: pd.DataFrame) -> Optional[DriftReport]:
        if not self.is_ready():
            return None
        if len(production_df) < MIN_SAMPLES:
            logger.info("[Drift] Skipping — only %d/%d samples.", len(production_df), MIN_SAMPLES)
            return None

        drifted: list[str] = []
        scores: dict[str, float] = {}
        pvalues: dict[str, float] = {}

        for feat in NUMERIC_FEATURES:
            ref_sample = np.array(self._ref_stats["numeric"].get(feat, {}).get("sample", []))
            if len(ref_sample) == 0 or feat not in production_df.columns:
                continue
            prod_vals = production_df[feat].dropna().astype(float).values
            if len(prod_vals) < 10:
                continue
            ks_stat, p_val = stats.ks_2samp(ref_sample, prod_vals)
            scores[feat] = float(ks_stat)
            pvalues[feat] = float(p_val)
            if p_val < KS_ALPHA:
                drifted.append(feat)

        for feat in CATEGORICAL_FEATURES:
            ref_dist = self._ref_stats["categorical"].get(feat, {})
            if not ref_dist or feat not in production_df.columns:
                continue
            prod_counts = production_df[feat].fillna("Unknown").value_counts(normalize=True)
            psi = self._compute_psi(ref_dist, prod_counts)
            scores[feat] = float(psi)
            pvalues[feat] = float("nan")
            if psi > PSI_THRESHOLD:
                drifted.append(feat)

        prod_churn_rate = 0.0
        if "prediction" in production_df.columns and len(production_df) > 0:
            prod_churn_rate = float((production_df["prediction"] == "Yes").mean())

        ref_churn_rate = float(self._ref_stats.get("churn_rate", 0.0))
        prediction_drift = abs(prod_churn_rate - ref_churn_rate) > PREDICTION_DRIFT_THRESHOLD
        drift_detected = bool(drifted) or prediction_drift

        return DriftReport(
            n_samples=len(production_df),
            drift_detected=drift_detected,
            drifted_features=drifted,
            feature_scores=scores,
            feature_pvalues=pvalues,
            production_churn_rate=prod_churn_rate,
            reference_churn_rate=ref_churn_rate,
            prediction_drift=prediction_drift,
            retrain_recommended=drift_detected,
        )

    @staticmethod
    def _compute_psi(expected: dict, actual: pd.Series) -> float:
        psi = 0.0
        for cat, exp_pct in expected.items():
            act_pct = float(actual.get(cat, 0.0))
            exp_pct = max(float(exp_pct), 1e-6)
            act_pct = max(act_pct, 1e-6)
            psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        return psi
