"""Modele referencyjne do porównania z k-NN: las losowy i SVM z jądrem RBF."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_random_forest(n_estimators: int = 300, max_depth: int | None = None, min_samples_leaf: int = 1, seed: int = 42) -> Pipeline:
    """Las losowy nie zależy od skali cech, więc potok nie zawiera standaryzacji."""
    return Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        ]
    )


def make_svm(c: float = 10.0, gamma: float | str = "scale") -> Pipeline:
    """SVM z jądrem RBF, podobnie jak k-NN, wymaga standaryzacji cech."""
    return Pipeline([("scaler", StandardScaler()), ("model", SVC(C=c, gamma=gamma, kernel="rbf"))])


def baseline_grids(config) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Zwraca modele referencyjne i ich siatki parametrów z sekcji `baselines` konfiguracji."""
    return {
        "random_forest": (
            make_random_forest(n_estimators=config.rf_n_estimators, seed=config.seed),
            {"model__max_depth": list(config.rf_max_depth), "model__min_samples_leaf": list(config.rf_min_samples_leaf)},
        ),
        "svm": (
            make_svm(),
            {"model__C": list(config.svm_c), "model__gamma": list(config.svm_gamma)},
        ),
    }
