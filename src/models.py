"""
models.py
---------
Model definitions for SolarCast (光伏出力预测系统).

Models implemented:
  1. LightGBMForecaster  — gradient boosted trees baseline.
  2. LSTMForecaster      — PyTorch LSTM deep learning model.
  3. Seq2SeqLSTM         — encoder-decoder for multi-step prediction.
  4. MCDropoutLSTM       — LSTM with Monte Carlo Dropout for uncertainty.
  5. SequenceDataset     — PyTorch Dataset for sliding-window sequences.
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
# 3. Multi-Step Dataset for Seq2Seq
# ===========================================================================

class MultiStepDataset(Dataset):
    """
    Sliding-window dataset for multi-step forecasting.

    Args:
        X        : numpy array of shape (N, n_features)
        y        : numpy array of shape (N,)
        seq_len  : input sequence length
        horizon  : number of future steps to predict (default 4 = 1 hour)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24, horizon: int = 4):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]                  # (seq_len, n_features)
        y_target = self.y[idx + self.seq_len: idx + self.seq_len + self.horizon]  # (horizon,)
        return x_seq, y_target


# ===========================================================================
# 4. LSTM Model (single-step forecasting)
# ===========================================================================

class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM with dropout regularisation for solar power forecasting.

    Architecture:
      Input  -> LSTM (n_layers) -> Dropout -> FC -> FC -> Output

    The final FC layer outputs a single scalar (AC_POWER prediction).
    ReLU is NOT applied at output — target values may be negative after
    StandardScaler normalization. Clipping to non-negative is done after
    inverse_transform during inference.

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

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            out: (batch_size,) — predicted AC_POWER values (in scaled space)
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]    # take last time step output
        out = self.dropout(last_hidden)
        out = torch.relu(self.fc1(out))     # (batch, 64)
        out = self.fc2(out)                 # (batch, 1) — NO ReLU: allow negative in scaled space
        return out.squeeze(1)               # (batch,)


# ===========================================================================
# 5. Seq2Seq LSTM for Multi-Step Forecasting
# ===========================================================================

class Seq2SeqLSTM(nn.Module):
    """
    Encoder-Decoder LSTM for multi-step forecasting.

    Encoder: processes the input sequence into a context vector.
    Decoder: autoregressively generates predictions for the forecast horizon,
             using the last step's prediction as input for the next step.

    Args:
        input_size  : number of features per time step.
        hidden_size : number of LSTM hidden units.
        horizon     : number of future steps to predict.
        n_layers    : number of stacked LSTM layers.
        dropout     : dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        horizon: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.horizon = horizon

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
            teacher_forcing_ratio: probability of using true values during training
        Returns:
            out: (batch_size, horizon) — predicted values
        """
        batch_size = x.size(0)

        # Encode input sequence
        _, (hidden, cell) = self.encoder(x)

        # Decoder: start from last input step features, predict step by step
        decoder_input = x[:, -1:, :]   # (batch, 1, input_size)
        outputs = []

        for t in range(self.horizon):
            dec_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(dec_out[:, -1, :])   # (batch, 1)
            outputs.append(pred)

            # Prepare next decoder input: use last known features updated with new prediction
            # For simplicity, repeat the last input step features and replace lag features
            next_input = decoder_input[:, -1:, :].clone()
            outputs_tensor = torch.cat(outputs, dim=1)   # for teacher forcing
            if t > 0:
                # Update lag features with previous predictions
                for lag_k in range(1, 5):
                    lag_col = lag_k - 1 + 8   # index of ac_lag_{k} in feature vector
                    if lag_k <= t + 1:
                        next_input[:, 0, lag_col] = outputs[-lag_k].squeeze(-1) if lag_k <= t else pred.squeeze(-1)

            decoder_input = next_input

        out = torch.cat(outputs, dim=1)   # (batch, horizon)
        return out


# ===========================================================================
# 6. LSTM with MC Dropout for Uncertainty Estimation
# ===========================================================================

class MCDropoutLSTM(nn.Module):
    """
    LSTM with dropout enabled at inference time for Monte Carlo uncertainty estimation.

    Architecture: same as LSTMForecaster but dropout is always enabled.
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

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dropout is always active (train + eval) for MC sampling.
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)               # NO ReLU
        return out.squeeze(1)

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout prediction with uncertainty.

        Args:
            x: input tensor (n_samples_batch, seq_len, features)
            n_samples: number of MC iterations
        Returns:
            mean: mean prediction
            std: uncertainty (standard deviation)
        """
        self.train()  # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x).cpu().numpy()
                predictions.append(pred)
        predictions = np.stack(predictions, axis=0)   # (n_samples, batch)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std


# ===========================================================================
# Helper: count model parameters
# ===========================================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
