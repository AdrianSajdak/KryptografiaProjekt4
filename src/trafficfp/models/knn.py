"""Potok klasyfikacji k-NN ze standaryzacją cech."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trafficfp.config import KnnConfig


def make_knn(n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform") -> Pipeline:
    """Buduje potok, w którym skaler uczy się wyłącznie na danych treningowych foldu.

    k-NN zależy od skali cech, a rozmiary pakietów i logarytmy odstępów mają różne zakresy.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)),
        ]
    )


def parameter_grid(config: KnnConfig) -> dict[str, list]:
    """Zamienia konfigurację na siatkę parametrów dla GridSearchCV."""
    return {
        "knn__n_neighbors": list(config.neighbors),
        "knn__metric": list(config.metrics),
        "knn__weights": list(config.weights),
    }
