import numpy as np
import pandas as pd
import pytest

from trafficfp.evaluation.metrics import summarize, wilson_interval
from trafficfp.evaluation.splits import blocked_folds, chronological_holdout, random_holdout
from trafficfp.models.controls import chance_control, majority_control, permute_labels
from trafficfp.models.knn import make_knn


def make_meta():
    apps = ["a"] * 10 + ["b"] * 20
    starts = list(range(10, 0, -1)) + list(range(100, 120))
    return pd.DataFrame({"app": apps, "start_time": np.asarray(starts, dtype=float)})


def test_chronological_holdout_tests_only_later_flows():
    meta = make_meta()
    train, test = chronological_holdout(meta, 0.2)

    for app in ("a", "b"):
        train_times = meta.iloc[train].query("app == @app")["start_time"]
        test_times = meta.iloc[test].query("app == @app")["start_time"]
        assert train_times.max() < test_times.min()


def test_chronological_holdout_has_no_overlap_and_keeps_all_flows():
    meta = make_meta()
    train, test = chronological_holdout(meta, 0.2)

    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(meta)


def test_chronological_holdout_respects_fraction_per_application():
    meta = make_meta()
    _, test = chronological_holdout(meta, 0.2)

    assert meta.iloc[test]["app"].value_counts().to_dict() == {"a": 2, "b": 4}


def test_chronological_holdout_orders_train_by_time_within_application():
    meta = make_meta()
    train, _ = chronological_holdout(meta, 0.2)

    selected = meta.iloc[train]
    for _, group in selected.groupby("app"):
        assert group["start_time"].is_monotonic_increasing


def test_random_holdout_is_stratified_and_repeatable():
    meta = make_meta()
    first = random_holdout(meta, 0.2, seed=1)
    second = random_holdout(meta, 0.2, seed=1)

    np.testing.assert_array_equal(first[1], second[1])
    assert meta.iloc[first[1]]["app"].value_counts().to_dict() == {"a": 2, "b": 4}


def test_blocked_folds_keep_each_application_contiguous():
    apps = np.array(["a"] * 10 + ["b"] * 20)
    for _, validation in blocked_folds(apps, 2):
        for app in ("a", "b"):
            positions = validation[apps[validation] == app]
            assert np.all(np.diff(positions) == 1)


def test_wilson_interval_contains_proportion():
    low, high = wilson_interval(90, 100)

    assert low < 0.9 < high
    assert 0.0 <= low and high <= 1.0


def test_wilson_interval_rejects_zero_trials():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_summarize_reports_confusion_matrix_in_label_order():
    result = summarize(np.array(["a", "a", "b"]), np.array(["a", "b", "b"]), ["b", "a"])

    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]


def test_knn_pipeline_separates_toy_classes():
    X = np.vstack([np.zeros((10, 2)), np.full((10, 2), 10.0)])
    y = np.array(["x"] * 10 + ["y"] * 10)

    model = make_knn(n_neighbors=3).fit(X, y)

    assert model.score(X, y) == 1.0


def test_scaler_is_fitted_on_training_data_only():
    train = np.array([[0.0], [2.0]])
    y = np.array(["x", "y"])

    model = make_knn(n_neighbors=1).fit(train, y)
    model.predict(np.array([[1000.0]]))

    assert model.named_steps["scaler"].mean_[0] == pytest.approx(1.0)


def test_majority_control_predicts_most_frequent_class():
    X, y = np.zeros((3, 1)), np.array(["a", "b", "b"])

    assert majority_control().fit(X, y).predict(X).tolist() == ["b", "b", "b"]


def test_chance_control_is_repeatable():
    X, y = np.zeros((50, 1)), np.array(["a", "b"] * 25)

    first = chance_control(3).fit(X, y).predict(X)
    second = chance_control(3).fit(X, y).predict(X)

    np.testing.assert_array_equal(first, second)


def test_permute_labels_preserves_class_counts():
    labels = np.array(["a"] * 5 + ["b"] * 5)
    shuffled = permute_labels(labels, seed=1)

    assert sorted(shuffled.tolist()) == sorted(labels.tolist())
    assert not np.array_equal(shuffled, labels)
