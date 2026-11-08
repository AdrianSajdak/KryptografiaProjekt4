"""Budowanie macierzy cech dla wybranych przepływów."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trafficfp.data.dataset import FlowStore
from trafficfp.features.sequence import (
    signed_size_vector,
    size_feature_names,
    timing_feature_names,
    timing_vector,
)
from trafficfp.features.statistical import statistical_names, statistical_vector

FEATURE_SETS = ("sizes", "timing", "sizes+timing", "stats", "all")

#: Wykorzystuje cały przepływ, więc służy wyłącznie jako grupa kontrolna.
VOLUME_NAMES = ["log_packets", "log_bytes_total", "log_duration"]


def _vector_for(feature_set: str, times, sizes, directions, n_packets: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    if feature_set in ("sizes", "sizes+timing", "all"):
        parts.append(signed_size_vector(sizes, directions, n_packets))
    if feature_set in ("timing", "sizes+timing", "all"):
        parts.append(timing_vector(times, n_packets))
    if feature_set in ("stats", "all"):
        parts.append(statistical_vector(times, sizes, directions))
    return np.concatenate(parts)


def feature_names(feature_set: str, n_packets: int) -> list[str]:
    """Zwraca nazwy kolumn macierzy cech."""
    names: list[str] = []
    if feature_set in ("sizes", "sizes+timing", "all"):
        names += size_feature_names(n_packets)
    if feature_set in ("timing", "sizes+timing", "all"):
        names += timing_feature_names(n_packets)
    if feature_set in ("stats", "all"):
        names += statistical_names()
    return names


def prefix_matrix(
    store: FlowStore, indices: Sequence[int], feature_set: str, n_packets: int
) -> np.ndarray:
    """Liczy cechy wyłącznie z pierwszych `n_packets` pakietów każdego przepływu."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Nieznany zestaw cech: {feature_set!r}")
    rows = [
        _vector_for(feature_set, *store.packets(int(index), n_packets), n_packets)
        for index in indices
    ]
    return np.vstack(rows)


def volume_matrix(store: FlowStore, indices: Sequence[int]) -> np.ndarray:
    """Zwraca liczbę pakietów, bajtów i czas trwania całego przepływu."""
    selected = store.meta.iloc[np.asarray(indices)]
    return np.column_stack(
        [
            np.log1p(selected["n_packets"].to_numpy(dtype=np.float64)),
            np.log1p(selected["total_bytes"].to_numpy(dtype=np.float64)),
            np.log1p(selected["duration"].to_numpy(dtype=np.float64)),
        ]
    )
