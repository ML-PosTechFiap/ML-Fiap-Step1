import threading
from collections import deque

import pandas as pd

DRIFT_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "Partner",
    "Dependents",
    "TechSupport",
]


class DriftStore:
    """Thread-safe ring buffer that logs raw prediction inputs for drift analysis."""

    def __init__(self, maxlen: int = 1000):
        self._buffer: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def log(self, raw_input: dict, prediction: str, probability: float) -> None:
        record = {k: raw_input.get(k) for k in DRIFT_FEATURES}
        record["prediction"] = prediction
        record["probability"] = probability
        with self._lock:
            self._buffer.append(record)

    def get_dataframe(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self._buffer))

    def count(self) -> int:
        with self._lock:
            return len(self._buffer)
