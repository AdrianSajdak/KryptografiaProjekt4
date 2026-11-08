"""Cechy sekwencyjne: kolejne rozmiary pakietów i odstępy czasu między nimi."""

from __future__ import annotations

import numpy as np


def log_interarrival_ms(times: np.ndarray) -> np.ndarray:
    """Zamienia znaczniki czasu na log(1 + odstęp w milisekundach).

    Skala logarytmiczna spłaszcza rozrzut odstępów obejmujący kilka rzędów wielkości.
    """
    gaps = np.maximum(np.diff(np.asarray(times, dtype=np.float64)), 0.0)
    return np.log1p(gaps * 1000.0)


def _fit_length(values: np.ndarray, length: int) -> np.ndarray:
    """Przycina wektor albo dopełnia go zerami do podanej długości."""
    result = np.zeros(length, dtype=np.float64)
    used = min(length, values.size)
    result[:used] = values[:used]
    return result


def signed_size_vector(sizes: np.ndarray, directions: np.ndarray, n_packets: int) -> np.ndarray:
    """Zwraca rozmiary pierwszych pakietów ze znakiem kierunku."""
    signed = np.asarray(sizes, dtype=np.float64) * np.asarray(directions, dtype=np.float64)
    return _fit_length(signed, n_packets)


def timing_vector(times: np.ndarray, n_packets: int) -> np.ndarray:
    """Zwraca n_packets - 1 odstępów czasu między pierwszymi pakietami."""
    return _fit_length(log_interarrival_ms(times), n_packets - 1)


def size_feature_names(n_packets: int) -> list[str]:
    return [f"size_{index}" for index in range(1, n_packets + 1)]


def timing_feature_names(n_packets: int) -> list[str]:
    return [f"gap_{index}" for index in range(2, n_packets + 1)]
