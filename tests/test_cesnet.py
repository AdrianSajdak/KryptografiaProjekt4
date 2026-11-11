import gzip
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from trafficfp.data.cesnet import APP_TO_CATEGORY, build_cesnet_store, day_members, parse_ppi, sample_day
from trafficfp.data.labels import CATEGORIES
from trafficfp.validation import weekly_split

HEADER = "ID,BYTES,BYTES_REV,PACKETS,PACKETS_REV,DURATION,PPI,PPI_LEN,PPI_DURATION,PPI_ROUNDTRIPS,APP,CATEGORY"


def row(index, app, length):
    gaps = [0] + [5] * (length - 1)
    directions = [1, -1] * (length // 2) + [1] * (length % 2)
    sizes = [100 + i for i in range(length)]
    ppi = f'"[{gaps}, {directions}, {sizes}]"'
    return f"{index},1000,5000,10,20,1.5,{ppi},{length},0.1,2,{app},X"


def make_archive(rows):
    text = "\n".join([HEADER, *rows]) + "\n"
    payload = gzip.compress(text.encode("utf-8"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("W-2021-41/1_Mon/flows-20211011.csv.gz", payload)
        archive.writestr("W-2021-40/1_Mon/flows-20211004.csv.gz", payload)
        archive.writestr("W-2021-40/1_Mon/stats-20211004.json", "{}")
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def test_every_selected_service_maps_to_project_category():
    assert set(APP_TO_CATEGORY.values()) == set(CATEGORIES)


def test_parse_ppi_converts_gaps_to_relative_times():
    times, sizes, directions = parse_ppi("[[0, 8, 0, 90], [1, -1, -1, 1], [517, 1428, 991, 93]]")

    np.testing.assert_allclose(times, [0.0, 0.008, 0.008, 0.098])
    assert sizes.tolist() == [517, 1428, 991, 93]
    assert directions.tolist() == [1, -1, -1, 1]


def test_day_members_are_sorted_by_date():
    archive = make_archive([row(0, "youtube", 12)])

    assert day_members(archive) == [
        ("W-2021-40", "W-2021-40/1_Mon/flows-20211004.csv.gz"),
        ("W-2021-41", "W-2021-41/1_Mon/flows-20211011.csv.gz"),
    ]


def test_sample_day_filters_services_and_short_flows():
    rows = [row(0, "youtube", 12), row(1, "youtube", 5), row(2, "spotify", 30), row(3, "github", 20), row(4, "dropbox", 10)]
    archive = make_archive(rows)
    day = sample_day(archive, "W-2021-40/1_Mon/flows-20211004.csv.gz", 5, 10, np.random.default_rng(0), chunksize=2)

    assert sorted(day["ID"].tolist()) == [0, 2, 4]
    assert dict(zip(day["APP"], day["category"])) == {"youtube": "wideo", "spotify": "audio", "dropbox": "pliki"}


def test_sample_day_limits_flows_per_category():
    archive = make_archive([row(i, "bing", 15) for i in range(20)])
    day = sample_day(archive, "W-2021-40/1_Mon/flows-20211004.csv.gz", 4, 10, np.random.default_rng(0))

    assert len(day) == 4
    assert day["ID"].is_monotonic_increasing


def test_build_store_keeps_packets_and_whole_flow_volume():
    archive = make_archive([row(0, "youtube", 12), row(1, "spotify", 30)])
    day = sample_day(archive, "W-2021-40/1_Mon/flows-20211004.csv.gz", 5, 10, np.random.default_rng(0))
    day["week"], day["date"], day["day_index"] = "W-2021-40", "20211004", 0
    store = build_cesnet_store(day)

    assert store.meta["app"].tolist() == ["spotify", "youtube"]
    assert store.offsets.tolist() == [0, 30, 42]
    assert store.meta["n_packets"].tolist() == [30, 30]
    assert store.meta["total_bytes"].tolist() == [6000, 6000]
    _, sizes, directions = store.packets(0, 3)
    assert sizes.tolist() == [100, 101, 102]
    assert directions.tolist() == [1, -1, 1]


def test_weekly_split_orders_flows_in_time():
    meta = pd.DataFrame({"partition": ["b", "a", "a", "b"], "start_time": [4.0, 2.0, 1.0, 3.0]})
    train, test = weekly_split(meta, "a", "b")

    assert train.tolist() == [2, 1]
    assert test.tolist() == [3, 0]


def test_parse_ppi_rejects_malformed_field():
    with pytest.raises(ValueError):
        parse_ppi("[[0, 1], [1, -1]]")
