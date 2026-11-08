"""Podział danych odporny na wyciek informacji między zbiorami.

Każda aplikacja była zbierana w osobnym, ciągłym okresie, a kolejne przepływy
powstawały w odstępach kilkudziesięciu sekund. Podział losowy umieszczałby sąsiednie
w czasie przepływy po obu stronach podziału, dlatego głównym wariantem jest
podział chronologiczny w obrębie każdej aplikacji.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def chronological_holdout(
    meta: pd.DataFrame, test_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Oddziela na test najpóźniej zebrane przepływy każdej aplikacji.

    Zwraca pozycje wierszy `meta`, posortowane chronologicznie w obrębie aplikacji.
    """
    train_parts, test_parts = [], []
    positions = pd.Series(np.arange(len(meta)), index=meta.index)
    for _, group in meta.sort_values("start_time", kind="stable").groupby("app", sort=True):
        n_test = math.ceil(len(group) * test_fraction)
        ordered = positions.loc[group.index].to_numpy()
        train_parts.append(ordered[:-n_test])
        test_parts.append(ordered[-n_test:])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def random_holdout(
    meta: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Losowy podział stratyfikowany, używany jako punkt odniesienia dla literatury."""
    positions = np.arange(len(meta))
    train, test = train_test_split(
        positions, test_size=test_fraction, stratify=meta["app"], random_state=seed
    )
    return np.sort(train), np.sort(test)


def blocked_folds(apps: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tworzy foldy walidacji krzyżowej ciągłe w czasie w obrębie aplikacji.

    Wejście musi być uporządkowane chronologicznie w obrębie aplikacji. Bez losowania
    StratifiedKFold przypisuje kolejne próbki klasy do kolejnych foldów.
    """
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=False)
    return list(splitter.split(np.zeros(len(apps)), apps))
