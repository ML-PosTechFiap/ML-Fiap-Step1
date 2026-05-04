from prometheus_client import Counter, Gauge

drift_detected_gauge = Gauge(
    "churn_model_drift_detected",
    "1 if any feature or prediction drift is detected, 0 otherwise",
)
features_drifted_gauge = Gauge(
    "churn_model_features_drifted_total",
    "Number of features with detected drift",
)
feature_drift_score_gauge = Gauge(
    "churn_model_feature_drift_score",
    "Drift score per feature (KS statistic for numeric, PSI for categorical)",
    ["feature"],
)
production_samples_gauge = Gauge(
    "churn_model_production_samples_total",
    "Number of predictions currently in the drift monitoring buffer",
)
churn_rate_gauge = Gauge(
    "churn_model_churn_rate",
    "Rolling churn rate from production predictions (0.0–1.0)",
)
drift_checks_counter = Counter(
    "churn_model_drift_checks_total",
    "Total number of drift checks performed",
)
