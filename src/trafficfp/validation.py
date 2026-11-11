"""Walidacja metody na ruchu TLS po TCP (CESNET-TLS22).

Procedura jest ta sama co dla zbioru UC Davis: strojenie k-NN w blokowej walidacji
krzyżowej, wybór według makro-F1, ocena na zbiorze testowym i grupa kontrolna.
Podział jest chronologiczny według tygodni: uczymy się na pierwszym tygodniu
pomiaru, a testujemy na drugim.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from trafficfp.config import Config
from trafficfp.data.dataset import FlowStore
from trafficfp.data.labels import CATEGORIES
from trafficfp.evaluation.metrics import summarize
from trafficfp.experiment import (
    Candidate,
    FeatureCache,
    _compare_feature_sets,
    _evaluate_controls,
    _fit_predict,
    search_configurations,
)


def grid_config(config: Config) -> Config:
    """Siatka strojenia dla PPI: najwyżej 30 pakietów i krótsza lista wartości k."""
    return replace(
        config,
        features=replace(config.features, packet_counts=config.cesnet.packet_counts),
        knn=replace(config.knn, neighbors=config.cesnet.neighbors),
    )


def weekly_split(meta: pd.DataFrame, train_week: str, test_week: str) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca przepływy obu tygodni uporządkowane w czasie."""
    order = np.argsort(meta["start_time"].to_numpy(), kind="stable")
    weeks = meta["partition"].to_numpy()[order]
    return order[weeks == train_week], order[weeks == test_week]


def evaluate_cesnet(store: FlowStore, config: Config) -> tuple[pd.DataFrame, dict]:
    meta = store.meta
    labels = meta["category"].to_numpy(dtype=str)
    apps = meta["app"].to_numpy(dtype=str)
    train, test = weekly_split(meta, config.cesnet.train_week, config.cesnet.test_week)
    cache = FeatureCache(store)
    tuned = grid_config(config)
    # Foldy stratyfikujemy według kategorii, bo część usług ma zbyt mało przepływów.
    table = search_configurations(cache, train, labels, labels, tuned)
    best = Candidate.from_row(table.iloc[0])
    features = cache.matrix(best.feature_set, best.n_packets)
    evaluations = {"test_week": test}
    predicted = _fit_predict(best.model(), features, labels, train, evaluations)["test_week"]
    per_app = {
        app: {"category": str(labels[test][apps[test] == app][0]), "n_samples": int((apps[test] == app).sum()), "accuracy": float((predicted[apps[test] == app] == labels[test][apps[test] == app]).mean())}
        for app in np.unique(apps[test])
    }
    report = {
        "dataset": "CESNET-TLS22",
        "split": {"train": int(train.size), "test": int(test.size), "train_week": config.cesnet.train_week, "test_week": config.cesnet.test_week},
        "best_configuration": best.__dict__,
        "best_cv": {"cv_f1_macro": float(table.iloc[0]["cv_f1_macro"]), "cv_accuracy": float(table.iloc[0]["cv_accuracy"])},
        "accuracy_threshold": config.accuracy_threshold,
        "evaluations": {"test_week": summarize(labels[test], predicted, CATEGORIES)},
        "per_app": per_app,
        "controls": _evaluate_controls(store, best, features, labels, train, evaluations, CATEGORIES, config),
        "feature_set_comparison": _compare_feature_sets(table, cache, labels, train, test, CATEGORIES),
    }
    return table, report
