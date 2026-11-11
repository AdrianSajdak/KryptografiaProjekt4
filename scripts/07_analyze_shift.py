"""Analiza przesunięcia danych między ruchem skryptowym a ruchem ludzi (4 kategorie)."""

import json

import numpy as np

from trafficfp.config import load_config
from trafficfp.data.dataset import FlowStore
from trafficfp.evaluation.metrics import summarize
from trafficfp.experiment import TARGETS, Candidate, FeatureCache
from trafficfp.features.build import feature_names
from trafficfp.robustness import evaluation_sets
from trafficfp.shift import (
    class_conditional_shift,
    domain_classifier_accuracy,
    matched_sample,
    nearest_distances,
    temporal_blocks,
)

TARGET = "category"
TRAIN_FRACTION = 0.5
BLOCKS = 5
DECILES = 10
#: Część najstarszych przepływów każdej aplikacji użyta do uczenia w analizie pokrycia.
COVERAGE_FRACTIONS = (0.5, 0.6, 0.7, 0.8)
DOMAIN_REPEATS = 20
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def quantiles(values: np.ndarray) -> dict:
    return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES}


def main() -> None:
    config = load_config()
    store = FlowStore.load(config.paths.processed)
    meta = store.meta
    clean = json.loads((config.paths.tables / f"results_{TARGET}.json").read_text(encoding="utf-8"))
    best = Candidate(**clean["best_configuration"])
    features = FeatureCache(store).matrix(best.feature_set, best.n_packets)
    names = feature_names(best.feature_set, best.n_packets)
    labels = meta[TARGET].to_numpy(dtype=str)
    apps = meta["app"].to_numpy(dtype=str)
    train, sets = evaluation_sets(meta, config.split.test_fraction)
    test, human = sets["test_chronological"], sets["test_human_retraining"]
    rng = np.random.default_rng(config.seed)

    shift_test = class_conditional_shift(features, labels, train, test, names)
    shift_human = class_conditional_shift(features, labels, train, human, names)
    merged = shift_human.merge(shift_test[["feature", "ks", "smd"]], on="feature", suffixes=("_human", "_test"))

    model = best.model().fit(features[train], labels[train])
    distances = {split: nearest_distances(model, features[indices]) for split, indices in sets.items()}
    predicted_human = model.predict(features[human])
    edges = np.quantile(distances["test_human_retraining"], [0, 1 / 3, 2 / 3, 1])
    bins = np.clip(np.searchsorted(edges, distances["test_human_retraining"], side="right") - 1, 0, 2)
    by_distance = [
        {
            "tercile": index + 1,
            "distance_from": float(edges[index]),
            "distance_to": float(edges[index + 1]),
            "n_samples": int((bins == index).sum()),
            "accuracy": float((predicted_human[bins == index] == labels[human][bins == index]).mean()),
        }
        for index in range(3)
    ]

    pretraining = np.flatnonzero(meta["partition"] == "pretraining")
    start = meta["start_time"].to_numpy(dtype=float)
    coverage = []
    for fraction in COVERAGE_FRACTIONS:
        local_train, local_blocks = temporal_blocks(meta.iloc[pretraining], fraction, BLOCKS)
        early = pretraining[local_train]
        later = np.concatenate([pretraining[block] for block in local_blocks])
        predicted = best.model().fit(features[early], labels[early]).predict(features[later])
        summary = summarize(labels[later], predicted, TARGETS[TARGET])
        coverage.append(
            {
                "train_fraction": fraction,
                "n_samples": summary["n_samples"],
                "accuracy": summary["accuracy"],
                "accuracy_ci95": summary["accuracy_ci95"],
                "per_app": {app: float((predicted[apps[later] == app] == labels[later][apps[later] == app]).mean()) for app in np.unique(apps)},
            }
        )

    local_train, local_blocks = temporal_blocks(meta.iloc[pretraining], TRAIN_FRACTION, BLOCKS)
    early, blocks = pretraining[local_train], [pretraining[block] for block in local_blocks]
    decay_model = best.model().fit(features[early], labels[early])
    train_end = {app: start[early][apps[early] == app].max() for app in np.unique(apps)}
    decay = []
    for index, block in enumerate(blocks):
        gaps = np.array([start[i] - train_end[apps[i]] for i in block]) / 3600
        predicted = decay_model.predict(features[block])
        decay.append(
            {
                "block": index + 1,
                "n_samples": int(block.size),
                "accuracy": float((predicted == labels[block]).mean()),
                "per_app": {app: float((predicted[apps[block] == app] == labels[block][apps[block] == app]).mean()) for app in np.unique(apps)},
                "median_hours_after_training": float(np.median(gaps)),
            }
        )

    behaviour = {}
    for app in np.unique(apps):
        group = meta.iloc[pretraining][apps[pretraining] == app].sort_values("start_time", kind="stable")
        parts = [group.iloc[part] for part in np.array_split(np.arange(len(group)), DECILES)]
        behaviour[app] = {
            "out_packet_share_median": [float((part["out_packets"] / part["n_packets"]).median()) for part in parts],
            "duration_median": [float(part["duration"].median()) for part in parts],
            "first_start": [float(part["start_time"].min()) for part in parts],
        }

    pool = np.concatenate([train, test])
    domain = [
        domain_classifier_accuracy(best.model, features, human, matched_sample(apps, human, pool, rng), config.split.cv_folds, config.seed + repeat)
        for repeat in range(DOMAIN_REPEATS)
    ]
    domain_control = [
        domain_classifier_accuracy(
            best.model,
            features,
            matched_sample(apps, human, test, rng),
            matched_sample(apps, human, train, rng),
            config.split.cv_folds,
            config.seed + repeat,
        )
        for repeat in range(DOMAIN_REPEATS)
    ]

    result = {
        "target": TARGET,
        "configuration": best.__dict__,
        "feature_shift": {
            "mean_ks_test": float(shift_test["ks"].mean()),
            "mean_ks_human": float(shift_human["ks"].mean()),
            "mean_ks_by_group": merged.groupby("group")[["ks_test", "ks_human"]].mean().to_dict("index"),
            "top_human": merged.head(10).to_dict("records"),
        },
        "nearest_distance": {split: quantiles(values) for split, values in distances.items()},
        "human_accuracy_by_distance": by_distance,
        "class_coverage": coverage,
        "temporal_blocks": {"train_fraction": TRAIN_FRACTION, "blocks": decay},
        "behaviour_by_decile": behaviour,
        "domain_classifier": {
            "repeats": DOMAIN_REPEATS,
            "human_vs_script_mean": float(np.mean(domain)),
            "human_vs_script_std": float(np.std(domain)),
            "test_vs_train_mean": float(np.mean(domain_control)),
            "test_vs_train_std": float(np.std(domain_control)),
        },
    }
    tables = config.paths.tables
    merged.to_csv(tables / f"shift_features_{TARGET}.csv", index=False, float_format="%.5f", encoding="utf-8")
    (tables / f"shift_{TARGET}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for item in coverage:
        print(f"uczenie na {item['train_fraction']:.0%}: {item['accuracy']:.4f}, Google Drive {item['per_app']['Google Drive']:.3f}")
    print("odległość do sąsiada (mediana):", {split: round(values["p50"], 2) for split, values in result["nearest_distance"].items()})
    print("trafność ruchu ludzi wg tercyli odległości:", [round(item["accuracy"], 3) for item in by_distance])
    print("klasyfikator domeny:", round(result["domain_classifier"]["human_vs_script_mean"], 3), round(result["domain_classifier"]["test_vs_train_mean"], 3))
    print("średnie KS test / człowiek:", round(result["feature_shift"]["mean_ks_test"], 3), round(result["feature_shift"]["mean_ks_human"], 3))


if __name__ == "__main__":
    main()
