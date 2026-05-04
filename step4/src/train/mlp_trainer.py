"""
PyTorch MLP for binary churn classification.

Architecture:
  Input → Linear → ReLU → Dropout → ... → Linear(1) → BCEWithLogitsLoss

Training:
  Mini-batch gradient descent via DataLoader
  Early stopping on validation loss (patience configurable)
  Adam optimizer
"""

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from utils.logging_utils import logger


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class MLPTrainer:
    """
    Sklearn-compatible wrapper around a PyTorch MLP.
    Supports fit(), predict(), predict_proba() and MLflow logging.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
        val_fraction: float = 0.15,
        device: str | None = None,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_: _MLP | None = None
        self.threshold: float = 0.5

    def _to_tensor(self, X, y=None):
        X_t = torch.tensor(np.array(X, dtype=np.float32), device=self.device)
        if y is not None:
            y_t = torch.tensor(np.array(y, dtype=np.float32), device=self.device)
            return X_t, y_t
        return X_t

    def fit(self, X_train, y_train):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=self.val_fraction,
            random_state=42,
            stratify=y_train,
        )

        X_tr_t, y_tr_t = self._to_tensor(X_tr, y_tr)
        X_val_t, y_val_t = self._to_tensor(X_val, y_val)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model_ = _MLP(self.input_dim, self.hidden_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        logger.info(
            f"MLP training: device={self.device} | layers={self.hidden_layers} | "
            f"dropout={self.dropout} | lr={self.lr} | batch={self.batch_size}"
        )

        for epoch in range(1, self.max_epochs + 1):
            # --- train ---
            self.model_.train()
            train_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(X_tr_t)

            # --- validate ---
            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_val_t), y_val_t).item()

            if epoch % 20 == 0:
                logger.info(
                    f"  epoch {epoch:>4} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
                )

            # --- early stopping ---
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        logger.info(f"MLP training complete | best_val_loss={best_val_loss:.4f}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first")
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(self._to_tensor(X))
            proba = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def train_and_log(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Fit MLP, evaluate on test set, log everything to MLflow.
        Returns dict of metrics.
        """
        from sklearn.metrics import average_precision_score

        with mlflow.start_run(run_name="MLP_PyTorch", nested=True):
            mlflow.log_params({
                "model_type": "MLP_PyTorch",
                "hidden_layers": str(self.hidden_layers),
                "dropout": self.dropout,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "activation": "ReLU",
                "loss_fn": "BCEWithLogitsLoss",
                "optimizer": "Adam",
                "device": str(self.device),
            })

            self.fit(X_train, y_train)

            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "pr_auc": float(average_precision_score(y_test, y_proba)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            }

            mlflow.log_metrics(metrics)
            logger.info(f"MLP metrics: {metrics}")

        return metrics
