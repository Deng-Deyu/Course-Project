"""
train.py
--------
Complete training script for SolarCast (光伏出力预测系统).

Trains and evaluates:
  1. LightGBM baseline model.
  2. LSTM deep learning model (PyTorch).

Generates and saves to outputs/figures/:
  - actual vs. predicted curves for both models.
  - LSTM training / validation loss curves.
  - SHAP feature importance bar chart and beeswarm plot.
  - Model comparison metrics bar chart.

Saves model artifacts to outputs/models/.

Usage:
    python src/train.py
"""

import json
import logging
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for scripted execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on the path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import build_dataset, FEATURE_COLS, TARGET_COL
from src.models import LightGBMForecaster, LSTMForecaster, SequenceDataset, count_parameters
from src.metrics import compute_metrics

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
# Section 1 — LightGBM Training
# ===========================================================================
def train_lightgbm(df_train, df_val, df_test, scaler):
    logger.info("=" * 60)
    logger.info("Training LightGBM model")
    logger.info("=" * 60)

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)
    X_test, y_test = prepare_arrays(df_test, scaler)

    model = LightGBMForecaster()
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
# Section 2 — LSTM Training
# ===========================================================================
def train_lstm(df_train, df_val, df_test, scaler,
               seq_len: int = 24,
               batch_size: int = 64,
               epochs: int = 60,
               lr: float = 1e-3,
               patience: int = 10):
    logger.info("=" * 60)
    logger.info("Training LSTM model")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    X_train, y_train = prepare_arrays(df_train, scaler)
    X_val, y_val = prepare_arrays(df_val, scaler)
    X_test, y_test = prepare_arrays(df_test, scaler)

    from sklearn.preprocessing import StandardScaler
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

    model = LSTMForecaster(input_size=len(FEATURE_COLS)).to(device)
    logger.info("LSTM parameters: %d", count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False)
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

    y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
    y_test_lstm = y_scaler.inverse_transform(y_test_lstm_scaled.reshape(-1, 1)).flatten()
    y_pred_test = np.clip(y_pred_test, 0, None)  # clamp to non-negative

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
# Section 3 — SHAP Feature Importance
# ===========================================================================
def run_shap(lgbm_model, df_test, scaler):
    logger.info("Computing SHAP values for LightGBM model...")
    import shap

    X_test, _ = prepare_arrays(df_test, scaler)
    # Use a subsample to keep computation fast
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
# Section 4 — Visualisations
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


def plot_lstm_loss(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=PALETTE["lstm"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, color=PALETTE["lgbm"], label="Validation Loss",
            linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Training Dynamics")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "lstm_loss_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("LSTM loss curve saved to %s", path)


def plot_metrics_comparison(metrics_lgbm, metrics_lstm):
    metrics_keys = ["MAE", "RMSE", "MAPE", "R2"]
    x = np.arange(len(metrics_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    vals_lgbm = [metrics_lgbm[k] for k in metrics_keys]
    vals_lstm = [metrics_lstm[k] for k in metrics_keys]

    ax.bar(x - width / 2, vals_lgbm, width, label="LightGBM", color=PALETTE["lgbm"], alpha=0.85)
    ax.bar(x + width / 2, vals_lstm, width, label="LSTM", color=PALETTE["lstm"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_keys)
    ax.set_ylabel("Metric Value")
    ax.set_title("Model Performance Comparison — Test Set")
    ax.legend(frameon=False)

    for i, (v_l, v_n) in enumerate(zip(vals_lgbm, vals_lstm)):
        ax.text(i - width / 2, v_l + 0.01 * max(vals_lgbm + vals_lstm),
                f"{v_l:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, v_n + 0.01 * max(vals_lgbm + vals_lstm),
                f"{v_n:.3f}", ha="center", va="bottom", fontsize=8)

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
    """
    Marginal effect of IRRADIATION on AC_POWER prediction, holding
    all other features at their median value (ceteris paribus).
    """
    X_test, _ = prepare_arrays(df_test, scaler)
    median_row = np.median(X_test, axis=0)

    irr_col_idx = FEATURE_COLS.index("IRRADIATION")
    irr_range_scaled = np.linspace(X_test[:, irr_col_idx].min(),
                                   X_test[:, irr_col_idx].max(), 200)

    rows = np.tile(median_row, (200, 1))
    rows[:, irr_col_idx] = irr_range_scaled

    preds = lgbm_model.predict(rows)

    # Recover original irradiation values via inverse transform
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
    """
    Marginal effect of MODULE_TEMPERATURE on AC_POWER prediction,
    ceteris paribus (holding all others at median).
    """
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


# ===========================================================================
# Main entry point
# ===========================================================================
def main():
    logger.info("SolarCast — Training Pipeline Start")

    # ---- Build dataset ----
    df_train, df_val, df_test, df_full, scaler = build_dataset()

    # Save scaler and data split info
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    df_full.to_parquet(os.path.join(MODEL_DIR, "df_features.parquet"), index=False)

    # ---- LightGBM ----
    lgbm_model, y_test_lgbm, y_pred_lgbm, metrics_lgbm, df_test_lgbm = \
        train_lightgbm(df_train, df_val, df_test, scaler)

    dates_lgbm = df_test_lgbm["DATE_TIME"].values

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

    # ---- LSTM ----
    lstm_model, y_test_lstm, y_pred_lstm, metrics_lstm, train_losses, val_losses = \
        train_lstm(df_train, df_val, df_test, scaler)

    # LSTM test set is seq_len-shifted — align dates
    seq_len = 24
    dates_lstm = df_test["DATE_TIME"].values[seq_len:]

    _plot_prediction_curve(
        dates_lstm, y_test_lstm, y_pred_lstm,
        "LSTM", PALETTE["lstm"], "lstm_prediction_curve.png"
    )
    plot_scatter(y_test_lstm, y_pred_lstm, "LSTM", PALETTE["lstm"], "lstm_scatter.png")
    plot_lstm_loss(train_losses, val_losses)

    # ---- Comparison ----
    plot_metrics_comparison(metrics_lgbm, metrics_lstm)

    # Overlay comparison plot (same axis)
    fig, ax = plt.subplots(figsize=(14, 4))
    # Use LGBM dates (longer, since no seq_len offset)
    common_len = min(len(y_test_lgbm), len(y_test_lstm))
    ax.plot(dates_lgbm[:common_len], y_test_lgbm[:common_len],
            color=PALETTE["actual"], linewidth=0.8, label="Actual", alpha=0.9)
    ax.plot(dates_lgbm[:common_len], y_pred_lgbm[:common_len],
            color=PALETTE["lgbm"], linewidth=0.8, label="LightGBM", alpha=0.8)
    ax.plot(dates_lstm[:common_len], y_pred_lstm[:common_len],
            color=PALETTE["lstm"], linewidth=0.8, label="LSTM", alpha=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("AC Power (kW)")
    ax.set_title("Model Comparison — Actual vs. Predicted (Test Set)")
    ax.legend(frameon=False)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path_overlay = os.path.join(FIG_DIR, "comparison_overlay.png")
    fig.savefig(path_overlay, bbox_inches="tight")
    plt.close(fig)
    logger.info("Overlay comparison saved to %s", path_overlay)

    # ---- Save metrics JSON ----
    all_metrics = {"LightGBM": metrics_lgbm, "LSTM": metrics_lstm}
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_path)

    # ---- Print summary table ----
    print("\n" + "=" * 56)
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE (%)':>10} {'R2':>8}")
    print("-" * 56)
    for name, m in all_metrics.items():
        print(f"{name:<12} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} {m['MAPE']:>10.3f} {m['R2']:>8.4f}")
    print("=" * 56)

    logger.info("SolarCast training pipeline complete.")


if __name__ == "__main__":
    main()
