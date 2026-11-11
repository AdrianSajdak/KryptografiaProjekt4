import numpy as np
import pandas as pd
import pytest

from trafficfp.models.knn import make_knn
from trafficfp.shift import (
    class_conditional_shift,
    domain_classifier_accuracy,
    feature_group,
    matched_sample,
    nearest_distances,
    temporal_blocks,
)


def test_feature_groups():
    assert feature_group("size_in_mean") == "rozmiary"
    assert feature_group("gap_all_p50") == "odstępy czasu"
    assert feature_group("hist_out_100") == "histogramy"
    assert feature_group("out_byte_share") == "ogólne"


def test_identical_distributions_have_no_shift():
    features = np.tile(np.arange(10, dtype=float), 2).reshape(-1, 1)
    labels = np.array(["a"] * 20)
    shift = class_conditional_shift(features, labels, np.arange(10), np.arange(10, 20), ["f"])

    assert shift.loc[0, "ks"] == 0.0
    assert shift.loc[0, "smd"] == 0.0


def test_shift_is_measured_within_classes():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    features = np.concatenate([base, base + 3.0]).reshape(-1, 1)
    labels = np.array(["a"] * 100 + ["b"] * 100 + ["a"] * 100 + ["b"] * 100)
    shift = class_conditional_shift(features, labels, np.arange(200), np.arange(200, 400), ["f"])

    assert shift.loc[0, "ks"] > 0.8
    assert shift.loc[0, "smd"] == pytest.approx(3.0, rel=0.15)


def test_nearest_distance_is_zero_for_training_points():
    features = np.array([[0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [6.0, 4.0]])
    model = make_knn(1).fit(features, np.array(["a", "a", "b", "b"]))

    np.testing.assert_allclose(nearest_distances(model, features), 0.0)


def test_temporal_blocks_keep_order_within_application():
    meta = pd.DataFrame({"app": ["x"] * 10 + ["y"] * 10, "start_time": list(range(10)) + list(range(100, 110))})
    train, blocks = temporal_blocks(meta, 0.5, 5)

    assert sorted(train.tolist()) == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    assert [sorted(block.tolist()) for block in blocks] == [[5, 15], [6, 16], [7, 17], [8, 18], [9, 19]]


def test_domain_classifier_separates_distinct_groups():
    rng = np.random.default_rng(1)
    features = np.vstack([rng.normal(0, 1, (40, 3)), rng.normal(6, 1, (40, 3))])
    accuracy = domain_classifier_accuracy(lambda: make_knn(3), features, np.arange(40), np.arange(40, 80), 5, 0)

    assert accuracy > 0.95


def test_domain_classifier_requires_equal_groups():
    with pytest.raises(ValueError):
        domain_classifier_accuracy(make_knn, np.zeros((5, 1)), np.arange(2), np.arange(2, 5), 2, 0)


def test_matched_sample_copies_application_counts():
    apps = np.array(["x", "x", "y", "x", "x", "y", "y", "y"])
    chosen = matched_sample(apps, np.array([0, 1, 2]), np.arange(3, 8), np.random.default_rng(0))

    assert sorted(apps[chosen].tolist()) == ["x", "x", "y"]
