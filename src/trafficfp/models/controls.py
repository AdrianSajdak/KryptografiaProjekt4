"""Klasyfikatory i przekształcenia grupy kontrolnej."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier


def majority_control() -> DummyClassifier:
    """Zawsze wskazuje najliczniejszą klasę ze zbioru treningowego."""
    return DummyClassifier(strategy="most_frequent")


def chance_control(seed: int) -> DummyClassifier:
    """Losuje etykietę zgodnie z rozkładem klas ze zbioru treningowego."""
    return DummyClassifier(strategy="stratified", random_state=seed)


def permute_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    """Tasuje etykiety, zrywając zależność między cechami i klasą."""
    return np.random.default_rng(seed).permutation(labels)
