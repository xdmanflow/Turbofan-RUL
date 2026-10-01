"""Evaluation metrics: RMSE and the NASA PHM08 asymmetric scoring function."""
import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """NASA PHM08 scoring function (lower is better).

    d = predicted - true. Late predictions (d > 0) are penalized more
    heavily than early ones (d < 0), reflecting the safety cost of
    overestimating the remaining life.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(y_true, y_pred, critical_threshold: int = 50) -> dict:
    """RMSE, NASA score, and RMSE in the critical zone (true RUL < threshold)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true < critical_threshold
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "rmse_critical": rmse(y_true[mask], y_pred[mask]) if mask.any() else float("nan"),
    }
