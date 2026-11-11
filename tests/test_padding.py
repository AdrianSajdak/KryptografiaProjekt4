import numpy as np
import pandas as pd
import pytest

from trafficfp.config import load_config
from trafficfp.data.dataset import FlowStore
from trafficfp.defenses.padding import (
    Defense,
    apply_defense,
    bandwidth_overhead,
    defenses_from_config,
    pad_constant,
    pad_random,
    pad_to_block,
)

SIZES = np.array([54, 100, 128, 129, 700, 1486])
MAX_SIZE = 1486


def make_store():
    return FlowStore(
        meta=pd.DataFrame({"n_packets": [4, 2], "total_bytes": [982, 1615], "min_size": [54, 129], "max_size": [700, 1486]}),
        offsets=np.array([0, 4, 6]),
        times=np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.5]),
        sizes=np.array([54, 100, 128, 700, 129, 1486], dtype=np.uint16),
        directions=np.array([1, -1, 1, -1, 1, -1], dtype=np.int8),
    )


def test_constant_padding_hides_every_size():
    assert pad_constant(SIZES, MAX_SIZE).tolist() == [MAX_SIZE] * SIZES.size


def test_block_padding_rounds_up_to_multiple():
    assert pad_to_block(SIZES, 128, MAX_SIZE).tolist() == [128, 128, 128, 256, 768, 1486]


def test_block_padding_never_exceeds_maximum_size():
    assert pad_to_block(np.array([1400]), 512, MAX_SIZE).tolist() == [MAX_SIZE]


def test_random_padding_stays_within_bounds():
    padded = pad_random(SIZES, 100, MAX_SIZE, np.random.default_rng(0))

    assert np.all(padded >= SIZES)
    assert np.all(padded - SIZES <= 100)
    assert np.all(padded <= MAX_SIZE)


def test_random_padding_is_reproducible_with_seed():
    first = Defense("random", 200).apply(SIZES, MAX_SIZE, seed=7)
    second = Defense("random", 200).apply(SIZES, MAX_SIZE, seed=7)

    np.testing.assert_array_equal(first, second)


def test_padding_rejects_packets_above_maximum():
    with pytest.raises(ValueError):
        pad_constant(np.array([1500]), MAX_SIZE)


def test_defense_requires_parameter():
    with pytest.raises(ValueError):
        Defense("block")


def test_defense_labels_and_keys():
    assert Defense("constant").key == "constant"
    assert Defense("block", 256).key == "block_256"
    assert Defense("random", 50).label == "losowe 0–50 B"


def test_bandwidth_overhead():
    assert bandwidth_overhead(np.array([100, 100]), np.array([150, 150])) == pytest.approx(0.5)


def test_apply_defense_changes_only_sizes_and_byte_counts():
    store = make_store()
    padded = apply_defense(store, Defense("block", 128), MAX_SIZE, seed=0)

    np.testing.assert_array_equal(padded.times, store.times)
    np.testing.assert_array_equal(padded.directions, store.directions)
    assert padded.sizes.tolist() == [128, 128, 128, 768, 256, 1486]
    assert padded.meta["total_bytes"].tolist() == [1152, 1742]
    assert padded.meta["max_size"].tolist() == [768, 1486]
    assert store.sizes.tolist() == [54, 100, 128, 700, 129, 1486]


def test_defenses_from_config_cover_all_variants():
    config = load_config().padding
    defenses = defenses_from_config(config)

    assert len(defenses) == 1 + len(config.block_sizes) + len(config.random_max_extra)
    assert len({defense.key for defense in defenses}) == len(defenses)
