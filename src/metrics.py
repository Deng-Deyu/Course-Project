"""
metrics.py
----------
Shared evaluation metrics for SolarCast.

Functions:
  - compute_metrics(y_true, y_pred) -> dict
      Returns MAE, RMSE, MAPE (excludes zero-power records), R².
  - compute_interval_metrics(y_true, y_pred_low, y_pred_high) -> dict
      Returns prediction interval coverage and width.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE, and R² for regression evaluation.

    MAPE is computed only on non-zero actual values to avoid
    division-by-zero artefacts from nocturnal zero-power records.

    Args:
        y_true : 1-D array of ground-truth AC_POWER values.
        y_pred : 1-D array of predicted AC_POWER values.

    Returns:
        dict with keys: MAE, RMSE, MAPE, R2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE on daytime (non-zero) records only
    mask = y_true > 1.0           # exclude near-zero night-time records
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4), "R2": round(r2, 6)}


def compute_interval_metrics(y_true: np.ndarray,
                              y_pred_low: np.ndarray,
                              y_pred_high: np.ndarray,
                              alpha: float = 0.8) -> dict:
    """
    Evaluate probabilistic prediction intervals.

    Args:
        y_true      : ground-truth values.
        y_pred_low  : lower bound of prediction interval.
        y_pred_high : upper bound of prediction interval.
        alpha       : target confidence level (e.g. 0.8 for 80% CI).

    Returns:
        dict with keys: coverage (actual fraction in interval),
                        avg_width (average interval width),
                        PICP (prediction interval coverage probability),
                        MPIW (mean prediction interval width).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_low = np.asarray(y_pred_low, dtype=float)
    y_pred_high = np.asarray(y_pred_high, dtype=float)

    in_interval = (y_true >= y_pred_low) & (y_true <= y_pred_high)
    coverage = in_interval.mean()

    widths = y_pred_high - y_pred_low
    avg_width = np.mean(widths)

    # Daytime-only
    mask = y_true > 1.0
    if mask.sum() > 0:
        coverage_day = in_interval[mask].mean()
        avg_width_day = np.mean(widths[mask])
    else:
        coverage_day = float("nan")
        avg_width_day = float("nan")

    return {
        "PICP": round(coverage, 4),           # overall coverage
        "PICP_day": round(coverage_day, 4),   # daytime coverage
        "MPIW": round(avg_width, 2),          # mean width (kW)
        "MPIW_day": round(avg_width_day, 2),  # daytime mean width (kW)
    }
