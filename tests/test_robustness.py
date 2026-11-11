from dataclasses import replace

import numpy as np
import pandas as pd

from trafficfp.config import load_config
from trafficfp.data.dataset import FlowStore
from trafficfp.defenses.padding import Defense
from trafficfp.robustness import evaluate_padding, evaluation_sets, flatten

PARTITIONS = ["pretraining"] * 40 + ["Retraining(script-triggered)"] * 6 + ["Retraining(human-triggered)"] * 6


def make_store(seed=0, packets=8):
    rng = np.random.default_rng(seed)
    apps = np.array(["Youtube", "Google Drive"] * (len(PARTITIONS) // 2))
    times, sizes, directions = [], [], []
    for app in apps:
        gap = 0.01 if app == "Youtube" else 0.05
        times.append(np.cumsum(np.full(packets, gap) + rng.uniform(0, 0.002, packets)) - gap)
        base = 1200 if app == "Youtube" else 300
        sizes.append(np.clip(base + rng.integers(-50, 50, packets), 60, 1400))
        directions.append(np.where(np.arange(packets) % 3 == 0, 1, -1))
    meta = pd.DataFrame(
        {
            "partition": PARTITIONS,
            "app": apps,
            "category": np.where(apps == "Youtube", "wideo", "pliki"),
            "start_time": np.arange(len(apps), dtype=float),
            "n_packets": packets,
            "total_bytes": [int(s.sum()) for s in sizes],
            "duration": [float(t[-1]) for t in times],
        }
    )
    return FlowStore(
        meta=meta,
        offsets=np.arange(len(apps) + 1) * packets,
        times=np.concatenate(times),
        sizes=np.concatenate(sizes).astype(np.uint16),
        directions=np.concatenate(directions).astype(np.int8),
    )


def small_config():
    config = load_config()
    return replace(
        config,
        features=replace(config.features, packet_counts=(5,), feature_sets=("timing", "stats")),
        knn=replace(config.knn, neighbors=(1, 3)),
        split=replace(config.split, cv_folds=2),
    )


def clean_report():
    configuration = {"n_packets": 5, "n_neighbors": 1, "metric": "manhattan", "weights": "uniform"}
    return {
        "best_configuration": {"feature_set": "stats", **configuration},
        "feature_set_comparison": {name: {"configuration": {"feature_set": name, **configuration}} for name in ("timing", "stats")},
    }


def test_evaluation_sets_put_latest_flows_in_test():
    store = make_store()
    train, sets = evaluation_sets(store.meta, 0.2)
    start = store.meta["start_time"].to_numpy()

    assert len(train) == 32 and len(sets["test_chronological"]) == 8
    for app in ("Youtube", "Google Drive"):
        app_train = train[store.meta["app"].to_numpy()[train] == app]
        app_test = sets["test_chronological"][store.meta["app"].to_numpy()[sets["test_chronological"]] == app]
        assert start[app_train].max() < start[app_test].min()
    assert len(sets["test_human_retraining"]) == 6


def test_padding_does_not_change_timing_only_model():
    store = make_store()
    result = evaluate_padding(store, [Defense("constant")], clean_report(), "category", small_config())
    defense = result["defenses"][0]

    assert defense["overhead"] > 0
    for split in ("test_chronological", "test_human_retraining"):
        assert defense["naive"]["timing"][split]["accuracy"] == result["baseline"]["timing"][split]["accuracy"]
    assert set(defense["adaptive"]["by_feature_set"]) == {"timing", "stats"}


def test_flatten_lists_every_adversary_and_feature_set():
    store = make_store(1)
    result = evaluate_padding(store, [Defense("block", 512)], clean_report(), "category", small_config())
    table = flatten(result)

    assert sorted(table["adversary"].unique()) == ["adaptacyjny", "brak obrony", "naiwny"]
    assert len(table) == 2 + 2 + 2
    assert table.loc[table["adversary"] == "brak obrony", "overhead"].eq(0).all()
