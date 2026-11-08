import numpy as np
import pandas as pd
import pytest

from trafficfp.data.dataset import FlowStore, repair_packet_order
from trafficfp.data.loader import Flow
from trafficfp.data.validate import find_duplicates, quality_report


def make_flow(times):
    n = len(times)
    return Flow(
        flow_id="p/x",
        partition="p",
        app="Youtube",
        start_time=0.0,
        times=np.asarray(times, dtype=float),
        sizes=np.arange(100, 100 + n),
        directions=np.ones(n, dtype=np.int8),
    )


def make_store():
    meta = pd.DataFrame({"flow_id": ["a", "b"], "n_packets": [3, 2]})
    return FlowStore(
        meta=meta,
        offsets=np.array([0, 3, 5]),
        times=np.array([0.0, 1.0, 2.0, 0.0, 0.5]),
        sizes=np.array([10, 20, 30, 40, 50], dtype=np.uint16),
        directions=np.array([1, -1, 1, 1, -1], dtype=np.int8),
    )


def test_ordered_flow_stays_unchanged():
    flow = make_flow([0.0, 0.1, 0.2])
    times, sizes, _, disorders = repair_packet_order(flow)

    assert disorders == 0
    np.testing.assert_array_equal(times, flow.times)
    np.testing.assert_array_equal(sizes, flow.sizes)


def test_disordered_packets_are_sorted_with_their_sizes():
    flow = make_flow([0.0, 0.3, 0.2])
    times, sizes, _, disorders = repair_packet_order(flow)

    assert disorders == 1
    np.testing.assert_allclose(times, [0.0, 0.2, 0.3])
    np.testing.assert_array_equal(sizes, [100, 102, 101])


def test_store_returns_packets_of_requested_flow():
    times, sizes, directions = make_store().packets(1)

    np.testing.assert_allclose(times, [0.0, 0.5])
    np.testing.assert_array_equal(sizes, [40, 50])
    np.testing.assert_array_equal(directions, [1, -1])


def test_store_applies_packet_limit():
    _, sizes, _ = make_store().packets(0, limit=2)

    np.testing.assert_array_equal(sizes, [10, 20])


def test_store_round_trip_keeps_content(tmp_path):
    store = make_store()
    store.save(tmp_path)

    loaded = FlowStore.load(tmp_path, mmap=False)

    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded.sizes, store.sizes)


def test_duplicates_are_detected_by_start_packets_and_bytes():
    meta = pd.DataFrame(
        {"start_time": [1.0, 1.0, 2.0], "n_packets": [5, 5, 5], "total_bytes": [9, 9, 9]}
    )

    assert len(find_duplicates(meta)) == 2


def test_quality_report_rejects_incomplete_dataset():
    meta = pd.DataFrame({"partition": ["pretraining"], "app": ["Youtube"]})

    with pytest.raises(ValueError):
        quality_report(meta)
