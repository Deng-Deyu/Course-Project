"""
models.py
---------
Model definitions for SolarCast (光伏出力预测系统).

Models implemented:
  1. LightGBMForecaster  — gradient boosted trees baseline.
  2. LSTMForecaster      — PyTorch LSTM deep learning model.
  3. SequenceDataset     — PyTorch Dataset for sliding-window sequences.
"""

import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. LightGBM Forecaster (wrapper for clean API)
# ===========================================================================

class LightGBMForecaster:
    """
    Thin wrapper around lightgbm.LGBMRegressor with sensible defaults
    for a 15-minute interval solar power forecasting task.

    Hyper-parameters are chosen based on grid search results reported in
    recent solar forecasting literature (e.g., Wan et al., 2023).
    """

    DEFAULT_PARAMS = {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(self, params: dict = None):
        import lightgbm as lgb  # imported here to keep module importable without lgb

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb

        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ] if eval_set else []

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        logger.info("LightGBM training complete. Best iteration: %d", self.model.best_iteration_)
        return self

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(X)
        # Clamp negative predictions to zero (AC power cannot be negative)
        return np.clip(preds, 0, None)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def booster_(self):
        return self.model.booster_


# ===========================================================================
# 2. PyTorch Dataset for sliding-window sequences
# ===========================================================================

class SequenceDataset(Dataset):
    """
    Constructs overlapping fixed-length sequences for LSTM training.

    Args:
        X  : numpy array of shape (N, n_features) — scaled feature matrix.
        y  : numpy array of shape (N,)             — target (AC_POWER).
        seq_len : number of time steps per input sequence (default 24 = 6 h).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]        # (seq_len, n_features)
        y_target = self.y[idx + self.seq_len]           # scalar
        return x_seq, y_target


# ===========================================================================
# 3. LSTM Model
# ===========================================================================

class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM with dropout regularisation for solar power forecasting.

    Architecture:
      Input  -> LSTM (n_layers) -> Dropout -> FC -> ReLU -> FC -> Output

    The final FC layer outputs a single scalar (AC_POWER prediction).
    ReLU before the output ensures non-negative predictions.

    Args:
        input_size  : number of features per time step.
        hidden_size : number of LSTM hidden units (default 128).
        n_layers    : number of stacked LSTM layers (default 2).
        dropout     : dropout probability between LSTM layers (default 0.2).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            out: (batch_size,) — predicted AC_POWER values
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]    # take last time step output
        out = self.head(last_hidden)        # (batch, 1)
        # Clamp to non-negative
        out = torch.relu(out.squeeze(1))    # (batch,)
        return out


# ===========================================================================
# Helper: count model parameters
# ===========================================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
