import numpy as np
import pandas as pd
import pytest

from trafficfp.data.dataset import FlowStore
from trafficfp.features.build import FEATURE_SETS, feature_names, prefix_matrix, volume_matrix
from trafficfp.features.sequence import log_interarrival_ms, signed_size_vector, timing_vector
from trafficfp.features.statistical import burst_lengths, statistical_names, statistical_vector

TIMES = np.array([0.0, 0.001, 0.003, 0.004])
SIZES = np.array([100, 200, 300, 400])
DIRECTIONS = np.array([1, 1, -1, 1], dtype=np.int8)


def make_store():
    return FlowStore(
        meta=pd.DataFrame(
            {"n_packets": [4, 2], "total_bytes": [1000, 300], "duration": [0.004, 1.0]}
        ),
        offsets=np.array([0, 4, 6]),
        times=np.concatenate([TIMES, [0.0, 1.0]]),
        sizes=np.concatenate([SIZES, [100, 200]]).astype(np.uint16),
        directions=np.concatenate([DIRECTIONS, [1, -1]]).astype(np.int8),
    )


def test_signed_sizes_carry_direction():
    np.testing.assert_array_equal(
        signed_size_vector(SIZES, DIRECTIONS, 4), [100, 200, -300, 400]
    )


def test_short_sequences_are_zero_padded():
    np.testing.assert_array_equal(
        signed_size_vector(SIZES[:2], DIRECTIONS[:2], 4), [100, 200, 0, 0]
    )


def test_timing_vector_has_one_value_less_than_packets():
    vector = timing_vector(TIMES, 4)

    assert vector.size == 3
    np.testing.assert_allclose(vector, np.log1p([1.0, 2.0, 1.0]))


def test_negative_gaps_are_clipped_to_zero():
    assert log_interarrival_ms(np.array([1.0, 0.5])).tolist() == [0.0]


def test_burst_lengths_count_runs_of_equal_direction():
    assert burst_lengths(DIRECTIONS).tolist() == [2, 1, 1]


def test_burst_lengths_of_empty_flow():
    assert burst_lengths(np.array([], dtype=np.int8)).size == 0


def test_statistical_vector_matches_declared_names():
    vector = statistical_vector(TIMES, SIZES, DIRECTIONS)

    assert vector.size == len(statistical_names())
    assert np.isfinite(vector).all()


def test_statistics_handle_flow_in_one_direction_only():
    vector = statistical_vector(TIMES, SIZES, np.ones(4, dtype=np.int8))

    assert np.isfinite(vector).all()
    assert vector[0] == 1.0


def test_statistics_reflect_direction_shares():
    vector = statistical_vector(TIMES, SIZES, DIRECTIONS)

    assert vector[0] == pytest.approx(0.75)
    assert vector[1] == pytest.approx(700 / 1000)


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_prefix_matrix_shape_matches_names(feature_set):
    matrix = prefix_matrix(make_store(), [0, 1], feature_set, 3)

    assert matrix.shape == (2, len(feature_names(feature_set, 3)))


def test_prefix_matrix_ignores_packets_after_the_prefix():
    store = make_store()
    first = prefix_matrix(store, [0], "sizes", 2)

    np.testing.assert_array_equal(first, [[100, 200]])


def test_unknown_feature_set_is_rejected():
    with pytest.raises(ValueError):
        prefix_matrix(make_store(), [0], "brak", 3)


def test_volume_matrix_uses_whole_flow():
    matrix = volume_matrix(make_store(), [0])

    np.testing.assert_allclose(matrix, [[np.log1p(4), np.log1p(1000), np.log1p(0.004)]])
