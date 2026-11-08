"""Strojenie k-NN i ocena na zbiorach testowych.

Konfiguracja wybierana jest wyłącznie na podstawie walidacji krzyżowej na zbiorze
treningowym. Zbiór testowy służy tylko do końcowej oceny wybranych konfiguracji.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from trafficfp.config import Config
from trafficfp.data.dataset import FlowStore
from trafficfp.data.labels import APPS, CATEGORIES
from trafficfp.evaluation.metrics import summarize
from trafficfp.evaluation.splits import blocked_folds, chronological_holdout, random_holdout
from trafficfp.features.build import prefix_matrix, volume_matrix
from trafficfp.models.controls import chance_control, majority_control, permute_labels
from trafficfp.models.knn import make_knn, parameter_grid

PRETRAINING = "pretraining"
SCRIPT_RETRAINING = "Retraining(script-triggered)"
HUMAN_RETRAINING = "Retraining(human-triggered)"

TARGETS = {"category": CATEGORIES, "app": APPS}
_SCORERS = ("f1_macro", "accuracy", "balanced_accuracy")


@dataclass(frozen=True)
class Candidate:
    """Pełny opis konfiguracji ataku."""

    feature_set: str
    n_packets: int
    n_neighbors: int
    metric: str
    weights: str

    @classmethod
    def from_row(cls, row: pd.Series) -> "Candidate":
        return cls(
            feature_set=str(row["feature_set"]),
            n_packets=int(row["n_packets"]),
            n_neighbors=int(row["n_neighbors"]),
            metric=str(row["metric"]),
            weights=str(row["weights"]),
        )

    def model(self):
        return make_knn(self.n_neighbors, self.metric, self.weights)


class FeatureCache:
    """Przechowuje policzone macierze cech dla wszystkich przepływów magazynu."""

    def __init__(self, store: FlowStore) -> None:
        self._store = store
        self._matrices: dict[tuple[str, int], np.ndarray] = {}

    def matrix(self, feature_set: str, n_packets: int) -> np.ndarray:
        key = (feature_set, n_packets)
        if key not in self._matrices:
            self._matrices[key] = prefix_matrix(
                self._store, range(len(self._store)), feature_set, n_packets
            )
        return self._matrices[key]


def search_configurations(
    cache: FeatureCache, train: np.ndarray, labels: np.ndarray, apps: np.ndarray, config: Config
) -> pd.DataFrame:
    """Przeszukuje siatkę zestawów cech, długości prefiksu i parametrów k-NN."""
    selection = config.knn.selection_scoring
    folds = blocked_folds(apps[train], config.split.cv_folds)
    frames = []
    for feature_set in config.features.feature_sets:
        for n_packets in config.features.packet_counts:
            search = GridSearchCV(
                make_knn(),
                parameter_grid(config.knn),
                scoring=list(_SCORERS),
                refit=False,
                cv=folds,
                n_jobs=-1,
            )
            search.fit(cache.matrix(feature_set, n_packets)[train], labels[train])
            results = pd.DataFrame(search.cv_results_)
            frames.append(
                pd.DataFrame(
                    {
                        "feature_set": feature_set,
                        "n_packets": n_packets,
                        "n_neighbors": results["param_knn__n_neighbors"].astype(int),
                        "metric": results["param_knn__metric"].astype(str),
                        "weights": results["param_knn__weights"].astype(str),
                        **{f"cv_{name}": results[f"mean_test_{name}"] for name in _SCORERS},
                        **{f"cv_{name}_std": results[f"std_test_{name}"] for name in _SCORERS},
                    }
                )
            )
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(
        [f"cv_{selection}", "n_packets"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _fit_predict(model, features: np.ndarray, labels: np.ndarray, fit: np.ndarray, targets: dict):
    model.fit(features[fit], labels[fit])
    return {name: model.predict(features[indices]) for name, indices in targets.items()}


def evaluate_target(store: FlowStore, cache: FeatureCache, target: str, config: Config) -> tuple[pd.DataFrame, dict]:
    """Stroi k-NN dla wybranego poziomu etykiet i ocenia wynik na zbiorach testowych."""
    meta = store.meta
    label_names = TARGETS[target]
    labels = meta[target].to_numpy(dtype=str)
    apps = meta["app"].to_numpy(dtype=str)
    pretraining = np.flatnonzero(meta["partition"] == PRETRAINING)

    local_train, local_test = chronological_holdout(meta.iloc[pretraining], config.split.test_fraction)
    train, test = pretraining[local_train], pretraining[local_test]

    table = search_configurations(cache, train, labels, apps, config)
    best = Candidate.from_row(table.iloc[0])
    features = cache.matrix(best.feature_set, best.n_packets)

    evaluations = {
        "test_chronological": test,
        "test_script_retraining": np.flatnonzero(meta["partition"] == SCRIPT_RETRAINING),
        "test_human_retraining": np.flatnonzero(meta["partition"] == HUMAN_RETRAINING),
    }
    predictions = _fit_predict(best.model(), features, labels, train, evaluations)
    report: dict = {
        "target": target,
        "split": {"train": int(train.size), "test": int(test.size), "test_fraction": config.split.test_fraction},
        "best_configuration": best.__dict__,
        "best_cv": {f"cv_{name}": float(table.iloc[0][f"cv_{name}"]) for name in _SCORERS},
        "accuracy_threshold": config.accuracy_threshold,
        "evaluations": {
            name: summarize(labels[evaluations[name]], predicted, label_names)
            for name, predicted in predictions.items()
        },
    }

    local_random_train, local_random_test = random_holdout(
        meta.iloc[pretraining], config.split.test_fraction, config.seed
    )
    random_train, random_test = pretraining[local_random_train], pretraining[local_random_test]
    random_predictions = _fit_predict(best.model(), features, labels, random_train, {"random": random_test})
    report["evaluations"]["test_random"] = summarize(labels[random_test], random_predictions["random"], label_names)

    report["controls"] = _evaluate_controls(store, best, features, labels, train, evaluations, label_names, config)
    report["feature_set_comparison"] = _compare_feature_sets(table, cache, labels, train, test, label_names)
    return table, report


def _evaluate_controls(store, best, features, labels, train, evaluations, label_names, config) -> dict:
    """Ocenia grupę kontrolną na tych samych zbiorach testowych co główny model."""
    volume = volume_matrix(store, np.arange(len(store)))
    placeholder = np.zeros((len(store), 1))
    scrambled = labels.copy()
    scrambled[train] = permute_labels(labels[train], config.seed)

    controls = {
        "majority": (majority_control(), placeholder, labels),
        "chance": (chance_control(config.seed), placeholder, labels),
        "permuted_labels_knn": (best.model(), features, scrambled),
        "whole_flow_volume_knn": (best.model(), volume, labels),
    }
    results = {}
    for name, (model, matrix, fit_labels) in controls.items():
        predictions = _fit_predict(model, matrix, fit_labels, train, evaluations)
        results[name] = {
            split: summarize(labels[evaluations[split]], predicted, label_names)
            for split, predicted in predictions.items()
        }
    return results


def _compare_feature_sets(table, cache, labels, train, test, label_names) -> dict:
    """Ocenia najlepszą konfigurację każdego zestawu cech, wybraną w walidacji krzyżowej."""
    comparison = {}
    for feature_set, group in table.groupby("feature_set", sort=False):
        candidate = Candidate.from_row(group.iloc[0])
        features = cache.matrix(candidate.feature_set, candidate.n_packets)
        predicted = _fit_predict(candidate.model(), features, labels, train, {"test": test})["test"]
        summary = summarize(labels[test], predicted, label_names)
        comparison[feature_set] = {
            "configuration": candidate.__dict__,
            "cv_f1_macro": float(group.iloc[0]["cv_f1_macro"]),
            "test_accuracy": summary["accuracy"],
            "test_f1_macro": summary["f1_macro"],
        }
    return comparison
