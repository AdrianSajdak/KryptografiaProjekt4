import zipfile

import numpy as np
import pytest

from trafficfp.data.labels import APPS, CATEGORIES, category_of
from trafficfp.data.loader import (
    INBOUND,
    OUTBOUND,
    iter_archive_flows,
    parse_flow_bytes,
    parse_member_path,
)

SAMPLE = (
    b"100.000000000\t0\t1412\t1\n"
    b"100.000250000\t0.00025\t365\t1\n"
    b"100.047000000\t0.047\t1412\t0\n"
)


def test_parse_flow_bytes_returns_relative_times_and_directions():
    start, times, sizes, directions = parse_flow_bytes(SAMPLE)

    assert start == pytest.approx(100.0)
    np.testing.assert_allclose(times, [0.0, 0.00025, 0.047])
    np.testing.assert_array_equal(sizes, [1412, 365, 1412])
    np.testing.assert_array_equal(directions, [OUTBOUND, OUTBOUND, INBOUND])


def test_parse_flow_bytes_rejects_empty_input():
    with pytest.raises(Exception):
        parse_flow_bytes(b"")


def test_parse_member_path_splits_partition_app_and_name():
    assert parse_member_path("pretraining/Google Doc/GoogleDoc-7.txt") == (
        "pretraining",
        "Google Doc",
        "GoogleDoc-7",
    )


def test_parse_member_path_rejects_unexpected_layout():
    with pytest.raises(ValueError):
        parse_member_path("pretraining/plik.txt")


def test_iter_archive_flows_reads_flows_without_extracting(tmp_path):
    archive = tmp_path / "pretraining.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("pretraining/", "")
        bundle.writestr("pretraining/Youtube/", "")
        bundle.writestr("pretraining/Youtube/Youtube-1.txt", SAMPLE)

    flows = list(iter_archive_flows(archive))

    assert len(flows) == 1
    assert flows[0].flow_id == "pretraining/Youtube-1"
    assert flows[0].app == "Youtube"
    assert flows[0].n_packets == 3


def test_every_application_maps_to_a_known_category():
    assert {category_of(app) for app in APPS} == set(CATEGORIES)


def test_unknown_application_is_rejected():
    with pytest.raises(ValueError):
        category_of("Netflix")
