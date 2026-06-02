"""
metrics.py
----------
Shared evaluation metrics for SolarCast.

Functions:
  - compute_metrics(y_true, y_pred) -> dict
      Returns MAE, RMSE, MAPE (excludes zero-power records), R².
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
