"""Analiza przesunięcia danych między ruchem skryptowym a ruchem ludzi.

Partycja ludzka służy tu wyłącznie do opisu różnic. Żaden wynik tej analizy nie
wpływa na wybór konfiguracji modelu, więc partycja pozostaje niezależnym testem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

GROUPS = {"size_": "rozmiary", "gap_": "odstępy czasu", "hist_": "histogramy"}


def feature_group(name: str) -> str:
    """Przypisuje cechę statystyczną do jednej z czterech grup."""
    for prefix, group in GROUPS.items():
        if name.startswith(prefix):
            return group
    return "ogólne"


def class_conditional_shift(
    features: np.ndarray, labels: np.ndarray, reference: np.ndarray, other: np.ndarray, names: list[str]
) -> pd.DataFrame:
    """Mierzy przesunięcie każdej cechy osobno w każdej klasie i uśrednia wynik.

    Porównanie w obrębie klas oddziela zmianę rozkładu cech od zmiany proporcji klas.
    `ks` to statystyka Kołmogorowa–Smirnowa (0: te same rozkłady, 1: rozłączne),
    a `smd` to bezwzględna różnica średnich w odchyleniach standardowych zbioru odniesienia.
    """
    classes = sorted(set(labels[reference]) & set(labels[other]))
    rows = []
    for column, name in enumerate(names):
        ks, smd = [], []
        for label in classes:
            base = features[reference[labels[reference] == label], column]
            shifted = features[other[labels[other] == label], column]
            ks.append(ks_2samp(base, shifted).statistic)
            spread = base.std()
            smd.append(abs(shifted.mean() - base.mean()) / spread if spread > 0 else 0.0)
        rows.append({"feature": name, "group": feature_group(name), "ks": float(np.mean(ks)), "smd": float(np.mean(smd))})
    return pd.DataFrame(rows).sort_values("ks", ascending=False, kind="stable").reset_index(drop=True)


def nearest_distances(model, features: np.ndarray) -> np.ndarray:
    """Zwraca odległość każdego przepływu do najbliższego przepływu uczącego modelu."""
    scaled = model.named_steps["scaler"].transform(features)
    distances, _ = model.named_steps["knn"].kneighbors(scaled, n_neighbors=1)
    return distances[:, 0]


def temporal_blocks(meta: pd.DataFrame, train_fraction: float, n_blocks: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Dzieli przepływy każdej aplikacji na najstarszą część uczącą i kolejne bloki w czasie.

    Zwraca pozycje wierszy `meta`. Blok o numerze 0 leży najbliżej danych uczących.
    """
    train_parts: list[np.ndarray] = []
    block_parts: list[list[np.ndarray]] = [[] for _ in range(n_blocks)]
    positions = pd.Series(np.arange(len(meta)), index=meta.index)
    for _, group in meta.sort_values("start_time", kind="stable").groupby("app", sort=True):
        ordered = positions.loc[group.index].to_numpy()
        cut = int(len(ordered) * train_fraction)
        train_parts.append(ordered[:cut])
        for index, block in enumerate(np.array_split(ordered[cut:], n_blocks)):
            block_parts[index].append(block)
    return np.concatenate(train_parts), [np.concatenate(parts) for parts in block_parts]


def domain_classifier_accuracy(
    make_model, features: np.ndarray, first: np.ndarray, second: np.ndarray, n_folds: int, seed: int
) -> float:
    """Sprawdza, jak dobrze klasyfikator odróżnia dwie grupy przepływów.

    Wynik bliski 50% oznacza brak widocznej różnicy, a bliski 100% silne przesunięcie danych.
    Obie grupy muszą mieć tę samą liczność.
    """
    if first.size != second.size:
        raise ValueError("Grupy muszą mieć tę samą liczność")
    samples = np.concatenate([first, second])
    domain = np.concatenate([np.ones(first.size, dtype=int), np.zeros(second.size, dtype=int)])
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    correct = 0
    for fit, check in splitter.split(samples, domain):
        model = make_model().fit(features[samples[fit]], domain[fit])
        correct += int((model.predict(features[samples[check]]) == domain[check]).sum())
    return correct / samples.size


def matched_sample(apps: np.ndarray, target: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Losuje z `pool` tyle przepływów każdej aplikacji, ile ma ich grupa `target`."""
    chosen = []
    for app in np.unique(apps[target]):
        candidates = pool[apps[pool] == app]
        chosen.append(rng.choice(candidates, size=int((apps[target] == app).sum()), replace=False))
    return np.concatenate(chosen)
