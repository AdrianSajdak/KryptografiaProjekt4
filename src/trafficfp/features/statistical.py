"""Cechy statystyczne obserwowanego fragmentu przepływu.

Zestaw nawiązuje do cech k-fingerprinting (Hayes i Danezis, 2016): liczności
i udziały kierunków, statystyki rozmiarów i odstępów, serie (bursts), histogram
rozmiarów oraz położenie pakietów wychodzących w sekwencji.
"""

from __future__ import annotations

import numpy as np

from trafficfp.features.sequence import log_interarrival_ms

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SIZE_BIN_EDGES = np.arange(0, 1501, 100)

_SUMMARY_NAMES = ["mean", "std", "min", "max", *[f"p{int(q * 100)}" for q in QUANTILES]]
_GROUPS = ("all", "out", "in")


def _summary(values: np.ndarray) -> list[float]:
    """Zwraca statystyki rozkładu albo same zera, gdy brak próbek."""
    if values.size == 0:
        return [0.0] * len(_SUMMARY_NAMES)
    return [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        *np.quantile(values, QUANTILES).tolist(),
    ]


def burst_lengths(directions: np.ndarray) -> np.ndarray:
    """Zwraca długości kolejnych serii pakietów o tym samym kierunku."""
    directions = np.asarray(directions)
    if directions.size == 0:
        return np.empty(0, dtype=np.int64)
    changes = np.flatnonzero(np.diff(directions)) + 1
    edges = np.concatenate(([0], changes, [directions.size]))
    return np.diff(edges)


def _size_histogram(sizes: np.ndarray) -> list[float]:
    if sizes.size == 0:
        return [0.0] * (SIZE_BIN_EDGES.size - 1)
    counts, _ = np.histogram(np.minimum(sizes, SIZE_BIN_EDGES[-1] - 1), bins=SIZE_BIN_EDGES)
    return (counts / sizes.size).tolist()


def statistical_vector(times: np.ndarray, sizes: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Liczy wektor cech o stałej długości i kolejności zgodnej z `statistical_names`."""
    times = np.asarray(times, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    directions = np.asarray(directions)
    outbound = directions > 0
    masks = {"all": np.ones(sizes.size, dtype=bool), "out": outbound, "in": ~outbound}

    total_bytes = float(sizes.sum())
    out_bytes = float(sizes[outbound].sum())
    span = float(times[-1] - times[0]) if times.size else 0.0
    positions = np.flatnonzero(outbound) / max(sizes.size - 1, 1)
    bursts = burst_lengths(directions)

    vector: list[float] = [
        float(outbound.mean()) if sizes.size else 0.0,
        out_bytes / total_bytes if total_bytes else 0.0,
        np.log1p(total_bytes),
        np.log1p(span * 1000.0),
        float(positions.mean()) if positions.size else 0.0,
        float(positions.std()) if positions.size else 0.0,
        float(bursts.size) / max(sizes.size, 1),
        float(bursts.mean()) if bursts.size else 0.0,
        float(bursts.std()) if bursts.size else 0.0,
        float(bursts.max()) if bursts.size else 0.0,
    ]
    for group in _GROUPS:
        vector.extend(_summary(sizes[masks[group]]))
    for group in _GROUPS:
        vector.extend(_summary(log_interarrival_ms(times[masks[group]])))
    vector.extend(_size_histogram(sizes[outbound]))
    vector.extend(_size_histogram(sizes[~outbound]))
    return np.asarray(vector, dtype=np.float64)


def statistical_names() -> list[str]:
    """Zwraca nazwy cech w kolejności wektora."""
    names = [
        "out_packet_share",
        "out_byte_share",
        "log_bytes",
        "log_span_ms",
        "out_position_mean",
        "out_position_std",
        "burst_rate",
        "burst_len_mean",
        "burst_len_std",
        "burst_len_max",
    ]
    for prefix in ("size", "gap"):
        for group in _GROUPS:
            names.extend(f"{prefix}_{group}_{stat}" for stat in _SUMMARY_NAMES)
    for group in ("out", "in"):
        names.extend(
            f"hist_{group}_{SIZE_BIN_EDGES[i]}" for i in range(SIZE_BIN_EDGES.size - 1)
        )
    return names
