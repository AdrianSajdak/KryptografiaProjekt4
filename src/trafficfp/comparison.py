"""Porównanie k-NN z modelami referencyjnymi i istotność grup cech.

Modele referencyjne dostają te same cechy, ten sam podział danych i te same foldy
walidacji krzyżowej co k-NN. Istotność cech liczymy na ostatnim foldzie walidacyjnym,
więc zbiory testowe nie wpływają na żadną decyzję.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from trafficfp.config import Config
from trafficfp.data.dataset import FlowStore
from trafficfp.evaluation.metrics import summarize
from trafficfp.evaluation.splits import blocked_folds
from trafficfp.experiment import TARGETS, Candidate, FeatureCache
from trafficfp.features.build import feature_names
from trafficfp.models.baselines import baseline_grids
from trafficfp.robustness import SPLITS, evaluation_sets
from trafficfp.shift import feature_group

_KEPT = ("n_samples", "accuracy", "accuracy_ci95", "balanced_accuracy", "f1_macro")
TOP_FEATURES = 10


def _plain(value):
    """Zamienia typy NumPy na typy zapisywalne w JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def macro_f1(model, features: np.ndarray, labels: np.ndarray, label_names) -> float:
    return float(f1_score(labels, model.predict(features), labels=list(label_names), average="macro", zero_division=0))


def group_permutation_importance(model, features, labels, groups, label_names, repeats: int, seed: int) -> dict:
    """Mierzy spadek makro-F1 po wspólnym przetasowaniu wszystkich cech jednej grupy.

    Tasowanie całej grupy naraz omija problem cech skorelowanych, które przy tasowaniu
    pojedynczo zastępowałyby się nawzajem.
    """
    rng = np.random.default_rng(seed)
    baseline = macro_f1(model, features, labels, label_names)
    result = {}
    for group in dict.fromkeys(groups):
        columns = np.flatnonzero(groups == group)
        drops = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[:, columns] = features[rng.permutation(len(features))][:, columns]
            drops.append(baseline - macro_f1(model, shuffled, labels, label_names))
        result[str(group)] = {"mean": float(np.mean(drops)), "std": float(np.std(drops)), "n_features": int(columns.size)}
    return {"baseline_f1_macro": baseline, "groups": result}


def _evaluate(model, features, labels, train, sets, label_names) -> dict:
    started = time.perf_counter()
    model.fit(features[train], labels[train])
    fit_seconds = time.perf_counter() - started
    results = {"fit_seconds": fit_seconds}
    for split in SPLITS:
        started = time.perf_counter()
        predicted = model.predict(features[sets[split]])
        if split == "test_chronological":
            results["predict_seconds"] = time.perf_counter() - started
        summary = summarize(labels[sets[split]], predicted, label_names)
        results[split] = {key: summary[key] for key in _KEPT}
    return results


def compare_models(store: FlowStore, clean_report: dict, target: str, config: Config) -> dict:
    """Stroi modele referencyjne, ocenia je na zbiorach testowych i liczy istotność cech."""
    meta = store.meta
    label_names = TARGETS[target]
    best = Candidate(**clean_report["best_configuration"])
    features = FeatureCache(store).matrix(best.feature_set, best.n_packets)
    names = feature_names(best.feature_set, best.n_packets)
    groups = np.array([feature_group(name) for name in names])
    labels = meta[target].to_numpy(dtype=str)
    apps = meta["app"].to_numpy(dtype=str)
    train, sets = evaluation_sets(meta, config.split.test_fraction)
    folds = blocked_folds(apps[train], config.split.cv_folds)

    models = {"knn": {"parameters": best.__dict__, "cv_f1_macro": clean_report["best_cv"]["cv_f1_macro"], **_evaluate(best.model(), features, labels, train, sets, label_names)}}
    fitted = {}
    for name, (estimator, grid) in baseline_grids(config.baselines).items():
        search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=folds, n_jobs=1 if name == "random_forest" else -1, refit=False)
        search.fit(features[train], labels[train])
        model = clone(estimator).set_params(**search.best_params_)
        models[name] = {
            "parameters": {key.removeprefix("model__"): _plain(value) for key, value in search.best_params_.items()},
            "cv_f1_macro": float(search.best_score_),
            **_evaluate(model, features, labels, train, sets, label_names),
        }
        fitted[name] = model

    fit, check = folds[-1]
    validation = {"knn": best.model(), "random_forest": clone(fitted["random_forest"])}
    importance = {}
    for name, model in validation.items():
        model.fit(features[train[fit]], labels[train[fit]])
        importance[name] = group_permutation_importance(
            model, features[train[check]], labels[train[check]], groups, label_names, repeats=10, seed=config.seed
        )
    single = permutation_importance(
        validation["knn"], features[train[check]], labels[train[check]], scoring="f1_macro", n_repeats=5, random_state=config.seed
    )
    order = np.argsort(single.importances_mean)[::-1][:TOP_FEATURES]
    forest = fitted["random_forest"].named_steps["model"]
    forest_order = np.argsort(forest.feature_importances_)[::-1][:TOP_FEATURES]
    return {
        "target": target,
        "feature_set": best.feature_set,
        "n_packets": best.n_packets,
        "models": models,
        "group_importance": importance,
        "knn_top_features": [
            {"feature": names[i], "group": str(groups[i]), "mean": float(single.importances_mean[i]), "std": float(single.importances_std[i])}
            for i in order
        ],
        "forest_top_features": [
            {"feature": names[i], "group": str(groups[i]), "importance": float(forest.feature_importances_[i])} for i in forest_order
        ],
        "forest_group_importance": {
            str(group): float(forest.feature_importances_[groups == group].sum()) for group in dict.fromkeys(groups)
        },
    }
