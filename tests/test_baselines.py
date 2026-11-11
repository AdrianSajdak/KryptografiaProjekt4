import numpy as np

from trafficfp.comparison import group_permutation_importance
from trafficfp.config import load_config
from trafficfp.models.baselines import baseline_grids, make_random_forest, make_svm
from trafficfp.models.knn import make_knn


def separable_data(seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["a", "b"], 60)
    informative = np.where(labels == "a", 0.0, 5.0)[:, None] + rng.normal(0, 0.5, (120, 2))
    noise = rng.normal(0, 1, (120, 3))
    return np.hstack([informative, noise]), labels


def test_baseline_models_learn_separable_classes():
    features, labels = separable_data()
    for model in (make_random_forest(n_estimators=20), make_svm()):
        assert (model.fit(features, labels).predict(features) == labels).mean() > 0.95


def test_svm_pipeline_standardizes_features():
    assert make_svm().steps[0][0] == "scaler"


def test_baseline_grids_follow_configuration():
    grids = baseline_grids(load_config().baselines)

    assert set(grids) == {"random_forest", "svm"}
    assert grids["svm"][1]["model__gamma"] == ["scale", 0.01]
    assert None in grids["random_forest"][1]["model__max_depth"]


def test_group_importance_finds_informative_group():
    features, labels = separable_data(1)
    groups = np.array(["sygnał", "sygnał", "szum", "szum", "szum"])
    model = make_knn(3).fit(features, labels)
    result = group_permutation_importance(model, features, labels, groups, ["a", "b"], repeats=5, seed=0)

    assert result["groups"]["sygnał"]["mean"] > 0.3
    assert abs(result["groups"]["szum"]["mean"]) < 0.1
