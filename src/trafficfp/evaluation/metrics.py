"""Metryki jakości klasyfikacji."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """Zwraca 95-procentowy przedział ufności Wilsona dla proporcji."""
    if trials <= 0:
        raise ValueError("Liczba prób musi być dodatnia")
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def summarize(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> dict:
    """Liczy metryki zbiorcze, metryki per klasa i macierz pomyłek."""
    labels = list(labels)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    low, high = wilson_interval(int((y_true == y_pred).sum()), y_true.size)
    return {
        "n_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci95": [low, high],
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
