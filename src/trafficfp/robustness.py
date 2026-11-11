"""Testy odporności k-NN na padding w dwóch scenariuszach przeciwnika.

Przeciwnik naiwny uczy się na ruchu bez obrony, a klasyfikuje ruch z wypełnieniem.
Przeciwnik adaptacyjny zna obronę (zasada Kerckhoffsa), więc stroi i uczy k-NN na
ruchu z wypełnieniem tą samą procedurą co model główny. Obie wersje są oceniane na
tych samych zbiorach testowych co model główny, po nałożeniu na nie obrony.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from trafficfp.config import Config
from trafficfp.data.dataset import FlowStore
from trafficfp.defenses.padding import Defense, apply_defense, bandwidth_overhead
from trafficfp.evaluation.metrics import summarize
from trafficfp.evaluation.splits import chronological_holdout
from trafficfp.experiment import (
    HUMAN_RETRAINING,
    PRETRAINING,
    SCRIPT_RETRAINING,
    TARGETS,
    Candidate,
    FeatureCache,
    search_configurations,
)

SPLITS = ("test_chronological", "test_script_retraining", "test_human_retraining")
_KEPT = ("n_samples", "accuracy", "accuracy_ci95", "balanced_accuracy", "f1_macro")


def evaluation_sets(meta: pd.DataFrame, test_fraction: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Odtwarza zbiór uczący i zbiory testowe modelu głównego."""
    pretraining = np.flatnonzero(meta["partition"] == PRETRAINING)
    local_train, local_test = chronological_holdout(meta.iloc[pretraining], test_fraction)
    sets = {
        "test_chronological": pretraining[local_test],
        "test_script_retraining": np.flatnonzero(meta["partition"] == SCRIPT_RETRAINING),
        "test_human_retraining": np.flatnonzero(meta["partition"] == HUMAN_RETRAINING),
    }
    return pretraining[local_train], sets


def _evaluate(candidate: Candidate, fit_features, test_features, labels, train, sets, label_names) -> dict:
    model = candidate.model().fit(fit_features[train], labels[train])
    results = {}
    for split in SPLITS:
        summary = summarize(labels[sets[split]], model.predict(test_features[sets[split]]), label_names)
        results[split] = {key: summary[key] for key in _KEPT}
    return results


def naive_results(clean: FeatureCache, padded: FeatureCache, candidates: dict[str, Candidate], labels, train, sets, label_names) -> dict:
    """Model uczony na ruchu bez obrony klasyfikuje ruch z wypełnieniem."""
    return {
        feature_set: {
            "configuration": candidate.__dict__,
            **_evaluate(
                candidate,
                clean.matrix(candidate.feature_set, candidate.n_packets),
                padded.matrix(candidate.feature_set, candidate.n_packets),
                labels,
                train,
                sets,
                label_names,
            ),
        }
        for feature_set, candidate in candidates.items()
    }


def adaptive_results(padded: FeatureCache, labels, apps, train, sets, label_names, config: Config) -> dict:
    """Przeciwnik stroi k-NN na ruchu z wypełnieniem i ocenia każdy zestaw cech."""
    table = search_configurations(padded, train, labels, apps, config)
    by_feature_set = {}
    for feature_set, group in table.groupby("feature_set", sort=False):
        candidate = Candidate.from_row(group.iloc[0])
        features = padded.matrix(candidate.feature_set, candidate.n_packets)
        by_feature_set[feature_set] = {
            "configuration": candidate.__dict__,
            "cv_f1_macro": float(group.iloc[0]["cv_f1_macro"]),
            **_evaluate(candidate, features, features, labels, train, sets, label_names),
        }
    best = Candidate.from_row(table.iloc[0])
    return {
        "best_configuration": best.__dict__,
        "best_cv_f1_macro": float(table.iloc[0]["cv_f1_macro"]),
        "by_feature_set": by_feature_set,
    }


def evaluate_padding(
    store: FlowStore,
    defenses: Sequence[Defense],
    clean_report: dict,
    target: str,
    config: Config,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Ocenia obu przeciwników dla każdej obrony i zapisuje narzut przepustowości."""
    meta = store.meta
    label_names = TARGETS[target]
    labels = meta[target].to_numpy(dtype=str)
    apps = meta["app"].to_numpy(dtype=str)
    train, sets = evaluation_sets(meta, config.split.test_fraction)
    candidates = {
        feature_set: Candidate(**item["configuration"])
        for feature_set, item in clean_report["feature_set_comparison"].items()
    }
    clean = FeatureCache(store)
    result = {
        "target": target,
        "max_size": config.padding.max_size,
        "seed": config.seed,
        "main_feature_set": clean_report["best_configuration"]["feature_set"],
        "baseline": naive_results(clean, clean, candidates, labels, train, sets, label_names),
        "defenses": [],
    }
    for defense in defenses:
        if progress:
            progress(f"obrona: {defense.label}")
        padded_store = apply_defense(store, defense, config.padding.max_size, config.seed)
        padded = FeatureCache(padded_store)
        result["defenses"].append(
            {
                "key": defense.key,
                "kind": defense.kind,
                "parameter": defense.parameter,
                "label": defense.label,
                "overhead": bandwidth_overhead(store.sizes, padded_store.sizes),
                "naive": naive_results(clean, padded, candidates, labels, train, sets, label_names),
                "adaptive": adaptive_results(padded, labels, apps, train, sets, label_names, config),
            }
        )
    return result


def flatten(result: dict) -> pd.DataFrame:
    """Zamienia wyniki na tabelę: obrona, przeciwnik, zestaw cech, trafność."""
    rows = []

    def add(defense, label, overhead, adversary, feature_set, item):
        row = {"defense": defense, "label": label, "overhead": overhead, "adversary": adversary, "feature_set": feature_set}
        for split in SPLITS:
            row[f"{split}_accuracy"] = item[split]["accuracy"]
            row[f"{split}_f1_macro"] = item[split]["f1_macro"]
        rows.append(row)

    for feature_set, item in result["baseline"].items():
        add("none", "brak obrony", 0.0, "brak obrony", feature_set, item)
    for defense in result["defenses"]:
        for feature_set, item in defense["naive"].items():
            add(defense["key"], defense["label"], defense["overhead"], "naiwny", feature_set, item)
        for feature_set, item in defense["adaptive"]["by_feature_set"].items():
            add(defense["key"], defense["label"], defense["overhead"], "adaptacyjny", feature_set, item)
    return pd.DataFrame(rows)
