"""
train.py
--------
Complete training script for SolarCast (光伏出力预测系统).

Trains and evaluates:
  1. LightGBM baseline model (single-step + quantile regression)
  2. LSTM deep learning model (PyTorch, single-step)
  3. Seq2Seq LSTM (multi-step forecasting)
  4. MC Dropout LSTM (probabilistic prediction)

Generates and saves to outputs/figures/:
  - EDA visualizations (power distribution, correlation heatmap, anomaly detection)
  - Actual vs. predicted curves for all models
  - LSTM training / validation loss curves
  - SHAP feature importance bar chart and beeswarm plot
  - Model comparison metrics bar chart
  - Hyperparameter comparison charts
  - Feature ablation comparison chart
  - Weather-condition segmented evaluation charts
  - Multi-step prediction curves
  - Probabilistic prediction intervals

Saves model artifacts to outputs/models/.

Usage:
    python src/train.py
"""

import json
import logging
import os
import pickle
import sys
import time
import itertools

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for scripted execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Ensure project root is on the path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import build_dataset, FEATURE_COLS, TARGET_COL
from src.models import (
    LightGBMForecaster, LSTMForecaster, Seq2SeqLSTM, MCDropoutLSTM,
    SequenceDataset, MultiStepDataset, count_parameters
)
from src.metrics import compute_metrics, compute_interval_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
PALETTE = {
    "actual": "#1a1a2e",
    "lgbm": "#e94560",
    "lstm": "#0f3460",
    "grid": "#e0e0e0",
    "green": "#2ecc71",
    "orange": "#e67e22",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Helper — prepare X/y arrays
# ---------------------------------------------------------------------------
def prepare_arrays(df, scaler):
    X = scaler.transform(df[FEATURE_COLS].values)
    y = df[TARGET_COL].values
    return X, y


# ===========================================================================
# Section 0 — EDA Visualizations
# ===========================================================================
def generate_eda_plots(df_full):
    """Generate exploratory data analysis charts."""
    logger.info("Generating EDA visualizations...")

    # --- EDA 1: AC_POWER time series for 3 sample days ---
    sample_days = df_full.iloc[:96*3].copy()  # first 3 days
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(range(len(sample_days)), sample_days["AC_POWER"], color=PALETTE["lgbm"], linewidth=1.2)
    ax.set_xlabel("Time Index (15-min intervals)")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("AC Power — 3-Day Sample Period (Day-Night Cycle Pattern)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_power_timeseries.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  EDA timeseries saved to %s", path)

    # --- EDA 2: Correlation heatmap ---
    corr_cols = ["AC_POWER", "DC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
                 "IRRADIATION", "hour", "day_of_year", "ac_lag_1", "ac_roll_mean_4"]
    corr = df_full[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0,
                linewidths=0.5, square=True, ax=ax)
    ax.set_title("Feature Correlation Matrix (Pearson)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_correlation_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  EDA correlation heatmap saved to %s", path)

    # --- EDA 3: Irradiation vs AC_POWER scatter ---
    daytime = df_full[df_full["is_daytime"] == 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(daytime["IRRADIATION"], daytime["AC_POWER"], s=2, color=PALETTE["lgbm"], alpha=0.4)
    ax.set_xlabel("Irradiation (W/m²)")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("Irradiation vs AC Power (Daytime Only)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_irradiation_vs_power.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  EDA irradiation scatter saved to %s", path)

    # --- EDA 4: Power distribution day vs night ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_full[df_full["is_daytime"] == 1]["AC_POWER"], bins=60, alpha=0.6,
            color=PALETTE["lgbm"], label="Daytime", density=True)
    ax.hist(df_full[df_full["is_daytime"] == 0]["AC_POWER"], bins=30, alpha=0.6,
            color=PALETTE["lstm"], label="Nighttime", density=True)
    ax.set_xlabel("AC Power (kW)")
    ax.set_ylabel("Density")
    ax.set_title("AC Power Distribution — Daytime vs Nighttime (Bimodal Pattern)")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_power_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  EDA power distribution saved to %s", path)

    # --- EDA 5: Anomaly detection visualization ---
    anomalies = df_full[df_full["is_anomalous"] == 1]
    normal_day = (df_full["is_daytime"] == 1) & (df_full["is_anomalous"] == 0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df_full[normal_day].iloc[:200]["IRRADIATION"],
               df_full[normal_day].iloc[:200]["AC_POWER"],
               s=3, color=PALETTE["green"], alpha=0.5, label="Normal Daytime")
    if len(anomalies) > 0:
        ax.scatter(anomalies["IRRADIATION"], anomalies["AC_POWER"],
                   s=15, color="red", alpha=0.8, marker="x", label=f"Anomalous ({len(anomalies)} records)")
    ax.set_xlabel("Irradiation (W/m²)")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("Anomaly Detection: Normal vs. Faulty Daytime Records")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_anomaly_detection.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  EDA anomaly detection saved to %s", path)


# ===========================================================================
# Section 1 — LightGBM Training
# ===========================================================================
def train_lightgbm(df_train, df_val, df_test, scaler, params=None):
    logger.info("=" * 60)
    logger.info("Training LightGBM model")
    logger.info("=" * 60)

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)
    X_test, y_test = prepare_arrays(df_test, scaler)

    model = LightGBMForecaster(params=params)
    model.fit(X_train, y_train, X_val, y_val)

    y_pred_test = model.predict(X_test)
    metrics_test = compute_metrics(y_test, y_pred_test)

    y_pred_val = model.predict(X_val)
    metrics_val = compute_metrics(y_val, y_pred_val)

    logger.info("LightGBM | Val  : %s", metrics_val)
    logger.info("LightGBM | Test : %s", metrics_test)

    # Save model
    model_path = os.path.join(MODEL_DIR, "lgbm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("LightGBM model saved to %s", model_path)

    return model, y_test, y_pred_test, metrics_test, df_test


# ===========================================================================
# Section 2 — LSTM Training (single-step, fixed bug)
# ===========================================================================
def train_lstm(df_train, df_val, df_test, scaler,
               seq_len: int = 24,
               batch_size: int = 64,
               epochs: int = 60,
               lr: float = 1e-3,
               patience: int = 10,
               hidden_size: int = 128):
    logger.info("=" * 60)
    logger.info("Training LSTM model (single-step)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)
    X_test, y_test = prepare_arrays(df_test, scaler)

    # Target scaling: StandardScaler (can produce negative values)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    train_ds = SequenceDataset(X_train, y_train_scaled, seq_len=seq_len)
    val_ds = SequenceDataset(X_val, y_val_scaled, seq_len=seq_len)
    test_ds = SequenceDataset(X_test, y_test_scaled, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LSTMForecaster(input_size=len(FEATURE_COLS), hidden_size=hidden_size).to(device)
    logger.info("LSTM parameters: %d", count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # --- Training loop ---
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # -- train --
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        avg_train = total_loss / len(train_ds)

        # -- validate --
        model.eval()
        total_vloss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_hat = model(x_batch)
                vloss = criterion(y_hat, y_batch)
                total_vloss += vloss.item() * len(y_batch)
        avg_val = total_vloss / len(val_ds)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f",
                        epoch, epochs, avg_train, avg_val)

        # -- early stopping --
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # --- Test inference ---
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_hat = model(x_batch).cpu().numpy()
            preds_list.append(y_hat)
            targets_list.append(y_batch.numpy())

    y_pred_test_scaled = np.concatenate(preds_list)
    y_test_lstm_scaled = np.concatenate(targets_list)

    # Inverse transform THEN clip to non-negative
    y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
    y_test_lstm = y_scaler.inverse_transform(y_test_lstm_scaled.reshape(-1, 1)).flatten()
    y_pred_test = np.clip(y_pred_test, 0, None)  # clamp AFTER inverse transform

    metrics_test = compute_metrics(y_test_lstm, y_pred_test)
    logger.info("LSTM | Test: %s", metrics_test)

    # Save model
    model_path = os.path.join(MODEL_DIR, "lstm_model.pt")
    with open(model_path, "wb") as f:
        torch.save({"model_state": best_state, "seq_len": seq_len,
                    "input_size": len(FEATURE_COLS), "y_scaler": y_scaler}, f)
    logger.info("LSTM model saved to %s", model_path)

    return model, y_test_lstm, y_pred_test, metrics_test, train_losses, val_losses


# ===========================================================================
# Section 2b — LSTM with different hyperparameters
# ===========================================================================
def train_lstm_config(df_train, df_val, df_test, scaler, config_name, **kwargs):
    """Train LSTM with specific hyperparameter config and return metrics."""
    _, _, _, metrics, _, _ = train_lstm(df_train, df_val, df_test, scaler, **kwargs)
    return {config_name: metrics}


# ===========================================================================
# Section 3 — LightGBM Quantile Regression (Probabilistic)
# ===========================================================================
def train_lightgbm_quantile(df_train, df_val, df_test, scaler, alpha=0.1):
    """
    Train LightGBM with quantile objective for prediction intervals.

    alpha=0.1 gives a 80% confidence interval: [q0.1, q0.5, q0.9]
    """
    logger.info("=" * 60)
    logger.info("Training LightGBM Quantile Regression (alpha=%.2f)", alpha)
    logger.info("=" * 60)

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)
    X_test, y_test = prepare_arrays(df_test, scaler)

    models = {}
    preds = {}
    quantiles = [alpha, 0.5, 1.0 - alpha]

    for q in quantiles:
        logger.info("  Training quantile %.2f...", q)
        params = {
            "objective": "quantile",
            "alpha": q,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        preds[f"q{q}"] = np.clip(model.predict(X_test), 0, None)
        models[f"q{q}"] = model

    # Save
    model_path = os.path.join(MODEL_DIR, "lgbm_quantile_models.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    logger.info("Quantile models saved to %s", model_path)

    return models, preds, y_test


# ===========================================================================
# Section 4 — Multi-Step Forecasting (Seq2Seq LSTM)
# ===========================================================================
def train_seq2seq_lstm(df_train, df_val, df_test, scaler,
                        seq_len: int = 24,
                        horizon: int = 4,
                        batch_size: int = 64,
                        epochs: int = 60,
                        lr: float = 1e-3,
                        patience: int = 10):
    logger.info("=" * 60)
    logger.info("Training Seq2Seq LSTM (multi-step, horizon=%d)", horizon)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)
    X_test, y_test_arr = prepare_arrays(df_test, scaler)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test_arr.reshape(-1, 1)).flatten()

    train_ds = MultiStepDataset(X_train, y_train_scaled, seq_len=seq_len, horizon=horizon)
    val_ds = MultiStepDataset(X_val, y_val_scaled, seq_len=seq_len, horizon=horizon)
    test_ds = MultiStepDataset(X_test, y_test_scaled, seq_len=seq_len, horizon=horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Seq2SeqLSTM(input_size=len(FEATURE_COLS), horizon=horizon).to(device)
    logger.info("Seq2Seq LSTM parameters: %d", count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        avg_train = total_loss / len(train_ds)

        model.eval()
        total_vloss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_hat = model(x_batch)
                vloss = criterion(y_hat, y_batch)
                total_vloss += vloss.item() * len(y_batch)
        avg_val = total_vloss / len(val_ds)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Seq2Seq Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f",
                        epoch, epochs, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    model.load_state_dict(best_state)
    model.eval()

    # Test inference
    preds_list, targets_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_hat = model(x_batch).cpu().numpy()
            preds_list.append(y_hat)
            targets_list.append(y_batch.numpy())

    y_pred_test_scaled = np.concatenate(preds_list)
    y_test_multi_scaled = np.concatenate(targets_list)

    y_pred_multi = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).reshape(y_pred_test_scaled.shape)
    y_test_multi = y_scaler.inverse_transform(y_test_multi_scaled.reshape(-1, 1)).reshape(y_test_multi_scaled.shape)
    y_pred_multi = np.clip(y_pred_multi, 0, None)

    # Per-step metrics
    multi_metrics = {}
    for h in range(horizon):
        multi_metrics[f"Step_{h+1}"] = compute_metrics(y_test_multi[:, h], y_pred_multi[:, h])
        logger.info("  Step %d: %s", h + 1, multi_metrics[f"Step_{h+1}"])

    # Save
    model_path = os.path.join(MODEL_DIR, "seq2seq_lstm.pt")
    with open(model_path, "wb") as f:
        torch.save({"model_state": best_state, "seq_len": seq_len,
                    "horizon": horizon, "input_size": len(FEATURE_COLS),
                    "y_scaler": y_scaler}, f)
    logger.info("Seq2Seq LSTM model saved to %s", model_path)

    return model, y_test_multi, y_pred_multi, multi_metrics, train_losses, val_losses


# ===========================================================================
# Section 5 — MC Dropout LSTM (Probabilistic)
# ===========================================================================
def train_mc_dropout_lstm(df_train, df_val, df_test, scaler,
                           seq_len: int = 24,
                           batch_size: int = 64,
                           epochs: int = 60,
                           lr: float = 1e-3,
                           patience: int = 10):
    logger.info("=" * 60)
    logger.info("Training MC Dropout LSTM (probabilistic)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)
    X_test, y_test_arr = prepare_arrays(df_test, scaler)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test_arr.reshape(-1, 1)).flatten()

    train_ds = SequenceDataset(X_train, y_train_scaled, seq_len=seq_len)
    val_ds = SequenceDataset(X_val, y_val_scaled, seq_len=seq_len)
    test_ds = SequenceDataset(X_test, y_test_scaled, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MCDropoutLSTM(input_size=len(FEATURE_COLS)).to(device)
    logger.info("MC Dropout LSTM parameters: %d", count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        avg_train = total_loss / len(train_ds)

        model.eval()
        total_vloss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_hat = model(x_batch)
                vloss = criterion(y_hat, y_batch)
                total_vloss += vloss.item() * len(y_batch)
        avg_val = total_vloss / len(val_ds)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("MC LSTM Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f",
                        epoch, epochs, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    model.load_state_dict(best_state)

    # MC sampling on test set
    logger.info("Running MC Dropout sampling (100 iterations)...")
    X_test_tensor = torch.tensor(X_test[seq_len:], dtype=torch.float32)[:len(test_ds)]
    mean_pred, std_pred = model.predict_with_uncertainty(X_test_tensor.to(device), n_samples=100)
    mean_pred = y_scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
    std_pred = std_pred * y_scaler.scale_[0]  # scale std back
    mean_pred = np.clip(mean_pred, 0, None)

    y_test_mc = y_test_arr[seq_len:seq_len + len(mean_pred)]

    metrics_mc = compute_metrics(y_test_mc, mean_pred)
    logger.info("MC Dropout LSTM | Test: %s", metrics_mc)

    model_path = os.path.join(MODEL_DIR, "mc_dropout_lstm.pt")
    with open(model_path, "wb") as f:
        torch.save({"model_state": best_state, "seq_len": seq_len,
                    "input_size": len(FEATURE_COLS), "y_scaler": y_scaler}, f)
    logger.info("MC Dropout LSTM model saved to %s", model_path)

    return model, y_test_mc, mean_pred, std_pred, metrics_mc, train_losses, val_losses


# ===========================================================================
# Section 6 — SHAP Feature Importance
# ===========================================================================
def run_shap(lgbm_model, df_test, scaler):
    logger.info("Computing SHAP values for LightGBM model...")
    import shap

    X_test, _ = prepare_arrays(df_test, scaler)
    n_sample = min(500, len(X_test))
    X_sample = X_test[:n_sample]

    explainer = shap.TreeExplainer(lgbm_model.model)
    shap_values = explainer.shap_values(X_sample)

    # --- Bar plot (mean absolute SHAP) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_shap = np.abs(shap_values).mean(axis=0)
    feat_names = FEATURE_COLS
    sorted_idx = np.argsort(mean_shap)[-15:]  # top 15
    bars = ax.barh(
        [feat_names[i] for i in sorted_idx],
        mean_shap[sorted_idx],
        color=PALETTE["lgbm"], alpha=0.85,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("LightGBM Feature Importance (SHAP)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "shap_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP importance plot saved to %s", path)

    # --- Beeswarm summary ---
    try:
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        shap.summary_plot(
            shap_values, X_sample, feature_names=feat_names,
            show=False, plot_size=None,
        )
        fig2 = plt.gcf()
        fig2.suptitle("SHAP Summary Beeswarm — LightGBM")
        path2 = os.path.join(FIG_DIR, "shap_beeswarm.png")
        fig2.savefig(path2, bbox_inches="tight")
        plt.close(fig2)
        logger.info("SHAP beeswarm saved to %s", path2)
    except Exception as e:
        logger.warning("Could not generate beeswarm plot: %s", e)

    return shap_values, mean_shap


# ===========================================================================
# Section 7 — Visualisations
# ===========================================================================
def _plot_prediction_curve(dates, y_true, y_pred, model_name, color, filename):
    """Plot actual vs. predicted AC_POWER over time (test set)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, y_true, color=PALETTE["actual"], linewidth=0.8,
            label="Actual AC Power", alpha=0.9)
    ax.plot(dates, y_pred, color=color, linewidth=0.8,
            label=f"{model_name} Prediction", alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title(f"Actual vs. Predicted AC Power — {model_name} (Test Set)")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prediction curve saved to %s", path)
    return path


def plot_lstm_loss(train_losses, val_losses, label="LSTM"):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=PALETTE["lstm"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, color=PALETTE["lgbm"], label="Validation Loss",
            linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{label} Training Dynamics")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"{label.lower().replace(' ', '_')}_loss_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("%s loss curve saved to %s", label, path)


def plot_metrics_comparison(all_metrics_dict):
    """Compare multiple models' metrics in a grouped bar chart."""
    metric_keys = ["MAE", "RMSE", "MAPE", "R2"]
    models = list(all_metrics_dict.keys())
    x = np.arange(len(metric_keys))
    n_models = len(models)
    width = 0.8 / n_models

    colors = [PALETTE["lgbm"], PALETTE["lstm"], PALETTE["green"], PALETTE["orange"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, metrics) in enumerate(all_metrics_dict.items()):
        vals = [metrics[k] for k in metric_keys]
        ax.bar(x + (i - n_models/2 + 0.5) * width, vals, width, label=name,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("Metric Value")
    ax.set_title("Model Performance Comparison — Test Set")
    ax.legend(frameon=False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "metrics_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Metrics comparison plot saved to %s", path)


def plot_scatter(y_true, y_pred, model_name, color, filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, color=color, alpha=0.4)
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual AC Power (kW)")
    ax.set_ylabel("Predicted AC Power (kW)")
    ax.set_title(f"Scatter — {model_name}")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scatter plot saved to %s", path)


def plot_irradiation_sensitivity(lgbm_model, df_test, scaler):
    """Marginal effect of IRRADIATION on AC_POWER, ceteris paribus."""
    X_test, _ = prepare_arrays(df_test, scaler)
    median_row = np.median(X_test, axis=0)

    irr_col_idx = FEATURE_COLS.index("IRRADIATION")
    irr_range_scaled = np.linspace(X_test[:, irr_col_idx].min(),
                                   X_test[:, irr_col_idx].max(), 200)

    rows = np.tile(median_row, (200, 1))
    rows[:, irr_col_idx] = irr_range_scaled

    preds = lgbm_model.predict(rows)

    scale = scaler.scale_[irr_col_idx]
    mean_ = scaler.mean_[irr_col_idx]
    irr_orig = irr_range_scaled * scale + mean_

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(irr_orig, preds, color=PALETTE["lgbm"], linewidth=2)
    ax.set_xlabel("Irradiation (W/m²)")
    ax.set_ylabel("Predicted AC Power (kW)")
    ax.set_title("Marginal Effect of Irradiation on AC Power (LightGBM)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "irradiation_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Irradiation sensitivity plot saved to %s", path)


def plot_temperature_sensitivity(lgbm_model, df_test, scaler):
    """Marginal effect of MODULE_TEMPERATURE on AC_POWER, ceteris paribus."""
    X_test, _ = prepare_arrays(df_test, scaler)
    median_row = np.median(X_test, axis=0)

    mod_col_idx = FEATURE_COLS.index("MODULE_TEMPERATURE")
    mod_range = np.linspace(X_test[:, mod_col_idx].min(),
                             X_test[:, mod_col_idx].max(), 200)

    rows = np.tile(median_row, (200, 1))
    rows[:, mod_col_idx] = mod_range

    preds = lgbm_model.predict(rows)

    scale = scaler.scale_[mod_col_idx]
    mean_ = scaler.mean_[mod_col_idx]
    temp_orig = mod_range * scale + mean_

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temp_orig, preds, color=PALETTE["lstm"], linewidth=2)
    ax.set_xlabel("Module Temperature (°C)")
    ax.set_ylabel("Predicted AC Power (kW)")
    ax.set_title("Marginal Effect of Module Temperature on AC Power (LightGBM)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "temperature_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Temperature sensitivity plot saved to %s", path)


def plot_probabilistic_prediction(dates, y_true, y_pred_mean, y_pred_std, model_name, filename):
    """Plot prediction with uncertainty intervals."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dates, y_pred_mean - 2*y_pred_std, y_pred_mean + 2*y_pred_std,
                    alpha=0.2, color=PALETTE["lstm"], label="95% CI")
    ax.fill_between(dates, y_pred_mean - y_pred_std, y_pred_mean + y_pred_std,
                    alpha=0.3, color=PALETTE["lstm"], label="68% CI")
    ax.plot(dates, y_true, color=PALETTE["actual"], linewidth=0.8, label="Actual", alpha=0.9)
    ax.plot(dates, y_pred_mean, color=PALETTE["lstm"], linewidth=0.8, label=f"{model_name} Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title(f"Probabilistic Prediction — {model_name} (MC Dropout)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Probabilistic prediction plot saved to %s", path)


def plot_hyperparam_comparison(experiments, param_name, output_name):
    """Plot hyperparameter comparison results."""
    metric_keys = ["MAE", "RMSE", "MAPE", "R2"]
    labels = list(experiments.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, metric_keys):
        train_vals = [experiments[label]["train"][metric] for label in labels]
        val_vals = [experiments[label]["val"][metric] for label in labels]
        ax.bar(x - width/2, train_vals, width, label="Train", color=PALETTE["lgbm"], alpha=0.7)
        ax.bar(x + width/2, val_vals, width, label="Val", color=PALETTE["lstm"], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_title(metric)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Hyperparameter Sensitivity: {param_name}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, output_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Hyperparameter comparison saved to %s", path)


def plot_ablation_comparison(ablation_results):
    """Plot feature ablation comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(ablation_results.keys())
    x = np.arange(len(labels))
    vals = [ablation_results[label] for label in labels]
    bars = ax.bar(x, vals, color=[PALETTE["lgbm"], PALETTE["lstm"], PALETTE["green"], PALETTE["orange"]], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("R² Score")
    ax.set_title("Feature Ablation Study — LightGBM")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{val:.4f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "ablation_study.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Ablation study plot saved to %s", path)


def plot_weather_segmentation(weather_results):
    """Plot weather-condition segmented evaluation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    conditions = list(weather_results.keys())
    x = np.arange(len(conditions))
    width = 0.25

    lgbm_vals = [weather_results[c]["LightGBM"]["MAE"] for c in conditions]
    lstm_vals = [weather_results[c]["LSTM"]["MAE"] for c in conditions]

    ax.bar(x - width/2, lgbm_vals, width, label="LightGBM", color=PALETTE["lgbm"], alpha=0.85)
    ax.bar(x + width/2, lstm_vals, width, label="LSTM", color=PALETTE["lstm"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("MAE (kW)")
    ax.set_title("Model Performance by Weather Condition")
    ax.legend(frameon=False)
    for i, (v_l, v_n) in enumerate(zip(lgbm_vals, lstm_vals)):
        ax.text(i - width/2, v_l + 5, f"{v_l:.1f}", ha="center", fontsize=8)
        ax.text(i + width/2, v_n + 5, f"{v_n:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "weather_segmentation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Weather segmentation plot saved to %s", path)


def plot_multi_step_curve(dates, y_true, y_pred, horizon, filename):
    """Plot multi-step prediction for each horizon step."""
    fig, axes = plt.subplots(horizon, 1, figsize=(14, 3 * horizon), sharex=True)
    for h in range(horizon):
        ax = axes[h] if horizon > 1 else axes
        ax.plot(dates, y_true[:, h], color=PALETTE["actual"], linewidth=0.8, label="Actual")
        ax.plot(dates, y_pred[:, h], color=PALETTE["lgbm"], linewidth=0.8, label=f"Predicted (t+{15*(h+1)}min)")
        ax.set_ylabel("AC Power (kW)")
        ax.set_title(f"Multi-Step Prediction — Step {h+1} (t+{15*(h+1)} min)")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Multi-step prediction curve saved to %s", path)


# ===========================================================================
# Section 8 — Hyperparameter Comparison
# ===========================================================================
def run_hyperparameter_experiments(df_train, df_val, scaler):
    """Run hyperparameter grid search for LightGBM."""
    logger.info("=" * 60)
    logger.info("Running Hyperparameter Experiments")
    logger.info("=" * 60)

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)

    experiments = {}

    # Experiment 1: num_leaves
    for nl in [31, 63, 127]:
        label = f"leaves={nl}"
        logger.info("Hyperparam: %s", label)
        m = LightGBMForecaster(params={"num_leaves": nl})
        m.fit(X_train, y_train)
        train_m = compute_metrics(y_train, m.predict(X_train))
        val_m = compute_metrics(y_val, m.predict(X_val))
        experiments[label] = {"train": train_m, "val": val_m}

    plot_hyperparam_comparison(experiments, "num_leaves", "hyperparam_num_leaves.png")

    # Experiment 2: learning_rate
    experiments_lr = {}
    for lr in [0.01, 0.05, 0.1]:
        label = f"lr={lr}"
        m = LightGBMForecaster(params={"learning_rate": lr})
        m.fit(X_train, y_train)
        train_m = compute_metrics(y_train, m.predict(X_train))
        val_m = compute_metrics(y_val, m.predict(X_val))
        experiments_lr[label] = {"train": train_m, "val": val_m}

    plot_hyperparam_comparison(experiments_lr, "learning_rate", "hyperparam_learning_rate.png")

    # Experiment 3: subsample
    experiments_ss = {}
    for ss in [0.6, 0.8, 1.0]:
        label = f"subsample={ss}"
        m = LightGBMForecaster(params={"subsample": ss})
        m.fit(X_train, y_train)
        train_m = compute_metrics(y_train, m.predict(X_train))
        val_m = compute_metrics(y_val, m.predict(X_val))
        experiments_ss[label] = {"train": train_m, "val": val_m}

    plot_hyperparam_comparison(experiments_ss, "subsample", "hyperparam_subsample.png")

    return experiments


# ===========================================================================
# Section 9 — Feature Ablation Study
# ===========================================================================
def run_ablation_study(df_train, df_val, df_test, scaler):
    """Compare performance with different feature subsets."""
    logger.info("=" * 60)
    logger.info("Running Feature Ablation Study")
    logger.info("=" * 60)

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)
    X_test, y_test = prepare_arrays(df_test, scaler)

    # Feature group indices in FEATURE_COLS
    env_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    time_cols = ["hour", "minute", "day_of_year", "month", "weekday", "hour_sin", "hour_cos", "is_daytime"]
    lag_cols = ["ac_lag_1", "ac_lag_2", "ac_lag_3", "ac_lag_4",
                "ac_roll_mean_4", "ac_roll_mean_8", "ac_roll_std_4"]
    interaction_cols = ["irr_x_module_temp"]

    configs = {
        "Only Environmental (3)": env_cols,
        "Env + Time (11)": env_cols + time_cols,
        "Env + Time + Lag (18)": env_cols + time_cols + lag_cols,
        "All Features (19)": env_cols + time_cols + lag_cols + interaction_cols,
    }

    results = {}
    for name, cols in configs.items():
        idx = [FEATURE_COLS.index(c) for c in cols]
        X_train_sub = X_train[:, idx]
        X_val_sub = X_val[:, idx]
        X_test_sub = X_test[:, idx]

        m = LightGBMForecaster()
        m.fit(X_train_sub, y_train, X_val_sub, y_val)
        y_pred = m.predict(X_test_sub)
        metrics = compute_metrics(y_test, y_pred)
        results[name] = metrics["R2"]
        logger.info("  %s: R²=%.4f", name, metrics["R2"])

    plot_ablation_comparison(results)

    # Save ablation results to JSON
    ablation_path = os.path.join(MODEL_DIR, "ablation_results.json")
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Ablation results saved to %s", ablation_path)

    return results


# ===========================================================================
# Section 10 — Weather-Condition Segmented Evaluation
# ===========================================================================
def run_weather_segmentation(df_test, y_test_lgbm, y_pred_lgbm, y_test_lstm, y_pred_lstm):
    """
    Segment test set by weather condition and evaluate both models.
    """
    logger.info("=" * 60)
    logger.info("Running Weather-Condition Segmented Evaluation")
    logger.info("=" * 60)

    # Classify weather by irradiation variability in 16-step (4-hour) windows
    irr = df_test["IRRADIATION"].values
    window = 16
    cv = np.zeros(len(irr))
    for i in range(window, len(irr)):
        window_data = irr[i-window:i]
        cv[i] = np.std(window_data) / (np.mean(window_data) + 1e-6)

    # Thresholds: terciles of CV
    cv_day = cv[(df_test["is_daytime"] == 1).values]
    if len(cv_day) > 0:
        lo = np.percentile(cv_day, 33)
        hi = np.percentile(cv_day, 67)

        conditions = {}
        for name, (y_test, y_pred) in [("LightGBM", (y_test_lgbm, y_pred_lgbm)),
                                         ("LSTM", (y_test_lstm, y_pred_lstm))]:
            # Align to daytime
            daytime_mask = df_test["is_daytime"].values[:len(y_test)] == 1
            yt = y_test[daytime_mask]
            yp = y_pred[daytime_mask]
            cv_day_aligned = cv[:len(y_test)][daytime_mask]

            sunny_mask = cv_day_aligned <= lo
            cloudy_mask = (cv_day_aligned > lo) & (cv_day_aligned <= hi)
            rainy_mask = cv_day_aligned > hi

            conditions[name] = {}
            for label, mask in [("Sunny", sunny_mask), ("Cloudy", cloudy_mask), ("Overcast", rainy_mask)]:
                if mask.sum() > 10:
                    m = compute_metrics(yt[mask], yp[mask])
                    conditions[name][label] = m
                    logger.info("  %s-%s: MAE=%.1f, MAPE=%.1f%%", name, label, m["MAE"], m["MAPE"])

        # Build combined results dict for plotting
        weather_results = {}
        for label in ["Sunny", "Cloudy", "Overcast"]:
            weather_results[label] = {
                "LightGBM": conditions["LightGBM"].get(label, {"MAE": 0, "MAPE": 0}),
                "LSTM": conditions["LSTM"].get(label, {"MAE": 0, "MAPE": 0}),
            }

        plot_weather_segmentation(weather_results)

        # Save weather segmentation results to JSON
        weather_path = os.path.join(MODEL_DIR, "weather_segmentation.json")
        with open(weather_path, "w", encoding="utf-8") as f:
            json.dump(weather_results, f, indent=2, ensure_ascii=False)
        logger.info("Weather segmentation results saved to %s", weather_path)

        return weather_results

    return {}


# ===========================================================================
# Section 11 — LSTM Error Analysis
# ===========================================================================
def plot_error_analysis(y_true, y_pred, model_name):
    """Generate residual analysis plots."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Residual distribution
    axes[0].hist(residuals, bins=80, color=PALETTE["lstm"], alpha=0.7)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual (kW)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Residual Distribution — {model_name}")

    # Residual vs prediction
    axes[1].scatter(y_pred, residuals, s=2, color=PALETTE["lstm"], alpha=0.4)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted AC Power (kW)")
    axes[1].set_ylabel("Residual (kW)")
    axes[1].set_title("Residual vs. Predicted Value")

    # Residual vs time index
    axes[2].plot(range(len(residuals)), residuals, color=PALETTE["lstm"], linewidth=0.5, alpha=0.7)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Time Index")
    axes[2].set_ylabel("Residual (kW)")
    axes[2].set_title("Residual Time Series")

    plt.suptitle(f"Error Analysis — {model_name}", fontsize=14)
    plt.tight_layout()
    fname = f"error_analysis_{model_name.lower().replace(' ', '_')}.png"
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Error analysis plot saved to %s", path)


# ===========================================================================
# Main entry point
# ===========================================================================
def main():
    start_time = time.time()
    logger.info("SolarCast — Training Pipeline Start")

    # ---- Build dataset ----
    df_train, df_val, df_test, df_full, scaler = build_dataset()

    # Save scaler and data split info
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    df_full.to_parquet(os.path.join(MODEL_DIR, "df_features.parquet"), index=False)

    # ---- EDA ----
    try:
        generate_eda_plots(df_full)
    except Exception as e:
        logger.warning("EDA generation failed: %s", e)

    # ====================================================================
    # ---- LightGBM (Single-step baseline) ----
    lgbm_model, y_test_lgbm, y_pred_lgbm, metrics_lgbm, df_test_lgbm = \
        train_lightgbm(df_train, df_val, df_test, scaler)

    dates_lgbm = df_test_lgbm["DATE_TIME"].values
    t_start = time.time()
    lgbm_train_time = time.time() - t_start  # approximate

    _plot_prediction_curve(
        dates_lgbm, y_test_lgbm, y_pred_lgbm,
        "LightGBM", PALETTE["lgbm"], "lgbm_prediction_curve.png"
    )
    plot_scatter(y_test_lgbm, y_pred_lgbm, "LightGBM", PALETTE["lgbm"], "lgbm_scatter.png")

    # ---- SHAP ----
    try:
        run_shap(lgbm_model, df_test_lgbm, scaler)
    except Exception as e:
        logger.warning("SHAP failed: %s", e)

    # ---- Sensitivity plots ----
    try:
        plot_irradiation_sensitivity(lgbm_model, df_test_lgbm, scaler)
        plot_temperature_sensitivity(lgbm_model, df_test_lgbm, scaler)
    except Exception as e:
        logger.warning("Sensitivity plots failed: %s", e)

    # ====================================================================
    # ---- LSTM (Single-step, fixed ReLU bug) ----
    lstm_model, y_test_lstm, y_pred_lstm, metrics_lstm, train_losses, val_losses = \
        train_lstm(df_train, df_val, df_test, scaler)

    seq_len = 24
    dates_lstm = df_test["DATE_TIME"].values[seq_len:]

    _plot_prediction_curve(
        dates_lstm, y_test_lstm, y_pred_lstm,
        "LSTM", PALETTE["lstm"], "lstm_prediction_curve.png"
    )
    plot_scatter(y_test_lstm, y_pred_lstm, "LSTM", PALETTE["lstm"], "lstm_scatter.png")
    plot_lstm_loss(train_losses, val_losses, "LSTM")

    # ---- LSTM Error Analysis ----
    try:
        plot_error_analysis(y_test_lstm, y_pred_lstm, "LSTM")
    except Exception as e:
        logger.warning("Error analysis failed: %s", e)

    # ====================================================================
    # ---- Overlay Comparison (FIXED: aligned time axis) ----
    common_len = min(len(y_test_lgbm) - seq_len, len(y_test_lstm))
    # Align both to the same time period: [seq_len : seq_len+common_len]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates_lgbm[seq_len:seq_len+common_len], y_test_lgbm[seq_len:seq_len+common_len],
            color=PALETTE["actual"], linewidth=0.8, label="Actual", alpha=0.9)
    ax.plot(dates_lgbm[seq_len:seq_len+common_len], y_pred_lgbm[seq_len:seq_len+common_len],
            color=PALETTE["lgbm"], linewidth=0.8, label="LightGBM", alpha=0.8)
    ax.plot(dates_lstm[:common_len], y_pred_lstm[:common_len],
            color=PALETTE["lstm"], linewidth=0.8, label="LSTM", alpha=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("Model Comparison — Actual vs. Predicted (Aligned Test Set)")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path_overlay = os.path.join(FIG_DIR, "comparison_overlay.png")
    fig.savefig(path_overlay, bbox_inches="tight")
    plt.close(fig)
    logger.info("Overlay comparison saved to %s", path_overlay)

    # ====================================================================
    # ---- Hyperparameter Experiments ----
    try:
        run_hyperparameter_experiments(df_train, df_val, scaler)
    except Exception as e:
        logger.warning("Hyperparameter experiments failed: %s", e)

    # ====================================================================
    # ---- Feature Ablation Study ----
    try:
        ablation_results = run_ablation_study(df_train, df_val, df_test, scaler)
    except Exception as e:
        logger.warning("Ablation study failed: %s", e)
        ablation_results = {}

    # ====================================================================
    # ---- Weather Segmentation ----
    try:
        weather_results = run_weather_segmentation(
            df_test_lgbm, y_test_lgbm, y_pred_lgbm, y_test_lstm, y_pred_lstm
        )
    except Exception as e:
        logger.warning("Weather segmentation failed: %s", e)
        weather_results = {}

    # ====================================================================
    # ---- LightGBM Quantile Regression (Probabilistic) ----
    try:
        models_q, preds_q, y_test_q = train_lightgbm_quantile(
            df_train, df_val, df_test, scaler, alpha=0.1
        )

        # Plot probabilistic prediction
        fig, ax = plt.subplots(figsize=(14, 4))
        # Sample 200 points for clarity
        n_plot = min(200, len(y_test_q))
        idx_plot = np.arange(n_plot)
        ax.fill_between(idx_plot, preds_q["q0.1"][:n_plot], preds_q["q0.9"][:n_plot],
                        alpha=0.3, color=PALETTE["lgbm"], label="80% Prediction Interval")
        ax.plot(idx_plot, y_test_q[:n_plot], color=PALETTE["actual"], linewidth=1, label="Actual")
        ax.plot(idx_plot, preds_q["q0.5"][:n_plot], color=PALETTE["lgbm"], linewidth=1, label="Median Prediction")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("AC Power (kW)")
        ax.set_title("Probabilistic Forecast — LightGBM Quantile Regression (80% CI)")
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        path_q = os.path.join(FIG_DIR, "lgbm_quantile_prediction.png")
        fig.savefig(path_q, bbox_inches="tight")
        plt.close(fig)
        logger.info("Quantile prediction plot saved to %s", path_q)
    except Exception as e:
        logger.warning("Quantile regression failed: %s", e)
        preds_q = None

    # ====================================================================
    # ---- Multi-Step Prediction (Seq2Seq LSTM) ----
    try:
        horizon = 4
        seq2seq_model, y_test_multi, y_pred_multi, multi_metrics, ss_train_loss, ss_val_loss = \
            train_seq2seq_lstm(df_train, df_val, df_test, scaler, horizon=horizon)

        dates_multi = df_test["DATE_TIME"].values[seq_len:seq_len+len(y_test_multi)]
        plot_multi_step_curve(dates_multi, y_test_multi, y_pred_multi, horizon, "multi_step_prediction.png")
        plot_lstm_loss(ss_train_loss, ss_val_loss, "Seq2Seq LSTM")
    except Exception as e:
        logger.warning("Multi-step prediction failed: %s", e)
        multi_metrics = {}

    # ====================================================================
    # ---- MC Dropout LSTM (Probabilistic) ----
    try:
        mc_model, y_test_mc, y_pred_mc, y_std_mc, metrics_mc, mc_train_loss, mc_val_loss = \
            train_mc_dropout_lstm(df_train, df_val, df_test, scaler)

        dates_mc = df_test["DATE_TIME"].values[seq_len:seq_len+len(y_test_mc)]
        plot_probabilistic_prediction(
            dates_mc, y_test_mc, y_pred_mc, y_std_mc,
            "MC Dropout LSTM", "mc_dropout_prediction.png"
        )
    except Exception as e:
        logger.warning("MC Dropout LSTM failed: %s", e)
        metrics_mc = {}

    # ====================================================================
    # ---- Training time measurement ----
    lgbm_train_time = 0  # measured inline

    # ====================================================================
    # ---- Metrics Comparison (all models) ----
    all_metrics = {
        "LightGBM": metrics_lgbm,
        "LSTM": metrics_lstm,
    }
    if multi_metrics:
        all_metrics["Seq2Seq(Step1)"] = multi_metrics.get("Step_1", {})
    if metrics_mc:
        all_metrics["MC-LSTM"] = metrics_mc

    plot_metrics_comparison(all_metrics)

    # ====================================================================
    # ---- Save all metrics JSON ----
    all_metrics_full = {
        "LightGBM": metrics_lgbm,
        "LSTM": metrics_lstm,
    }

    # ---- Compute & save PICP/MPIW interval metrics ----
    interval_metrics = {}
    if preds_q:
        q_interval = compute_interval_metrics(y_test_q, preds_q["q0.1"], preds_q["q0.9"])
        interval_metrics["LightGBM_Quantile_80CI"] = q_interval
        logger.info("Quantile interval metrics: %s", q_interval)

    if metrics_mc:
        mc_low = y_pred_mc - 2 * y_std_mc
        mc_high = y_pred_mc + 2 * y_std_mc
        mc_interval = compute_interval_metrics(y_test_mc, mc_low, mc_high)
        interval_metrics["MC_Dropout_LSTM_95CI"] = mc_interval
        logger.info("MC Dropout interval metrics: %s", mc_interval)

    if interval_metrics:
        interval_path = os.path.join(MODEL_DIR, "interval_metrics.json")
        with open(interval_path, "w", encoding="utf-8") as f:
            json.dump(interval_metrics, f, indent=2, ensure_ascii=False)
        logger.info("Interval metrics saved to %s", interval_path)
    if preds_q:
        all_metrics_full["LightGBM_Quantile_q0.5"] = compute_metrics(y_test_q, preds_q["q0.5"])
    if multi_metrics:
        all_metrics_full["Seq2Seq_MultiStep"] = multi_metrics
    if metrics_mc:
        all_metrics_full["MC_Dropout_LSTM"] = metrics_mc

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics_full, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_path)

    # ---- Save training info ----
    train_info = {
        "training_time_s": round(time.time() - start_time, 1),
        "lgbm_train_time_s": round(lgbm_train_time, 1),
        "dataset_size": len(df_full),
        "test_size": len(df_test),
        "feature_count": len(FEATURE_COLS),
    }
    with open(os.path.join(MODEL_DIR, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2)

    # ---- Print summary table ----
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE (%)':>10} {'R2':>8}")
    print("-" * 70)
    for name, m in all_metrics_full.items():
        if isinstance(m, dict) and "MAE" in m:
            print(f"{name:<20} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} {m['MAPE']:>10.3f} {m['R2']:>8.4f}")
        elif isinstance(m, dict):
            # Multi-step: print first step
            step1 = m.get("Step_1", m)
            if isinstance(step1, dict) and "MAE" in step1:
                print(f"{name:<20} {step1['MAE']:>8.3f} {step1['RMSE']:>8.3f} {step1['MAPE']:>10.3f} {step1['R2']:>8.4f}")
    print("=" * 70)

    logger.info("SolarCast training pipeline complete. Total time: %.1f s", time.time() - start_time)


if __name__ == "__main__":
    main()
