"""Symulacja obron typu padding przez przekształcenie rozmiarów pakietów.

Obrona dopisuje bajty wypełnienia, więc pakiet po obronie nigdy nie jest mniejszy
od pierwotnego i nie przekracza największego dopuszczalnego rozmiaru, który pełni
rolę MTU. Czasy i kierunki pakietów pozostają bez zmian. To uproszczenie: prawdziwa
obrona wydłuża transmisję, więc mogłaby też zmienić odstępy czasu.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from trafficfp.data.dataset import FlowStore

KINDS = ("constant", "block", "random")


def _check_sizes(sizes: np.ndarray, max_size: int) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size and int(sizes.max()) > max_size:
        raise ValueError(f"Pakiet większy niż dopuszczalny rozmiar {max_size} B")
    return sizes


def pad_constant(sizes: np.ndarray, max_size: int) -> np.ndarray:
    """Dopełnia każdy pakiet do największego dopuszczalnego rozmiaru."""
    sizes = _check_sizes(sizes, max_size)
    return np.full_like(sizes, max_size)


def pad_to_block(sizes: np.ndarray, block: int, max_size: int) -> np.ndarray:
    """Dopełnia pakiet do najbliższej wielokrotności bloku, nie więcej niż `max_size`."""
    if block <= 0:
        raise ValueError("Rozmiar bloku musi być dodatni")
    sizes = _check_sizes(sizes, max_size)
    rounded = -(-sizes // block) * block
    return np.minimum(rounded, max_size)


def pad_random(sizes: np.ndarray, max_extra: int, max_size: int, rng: np.random.Generator) -> np.ndarray:
    """Dopisuje do każdego pakietu losową liczbę bajtów z przedziału [0, max_extra]."""
    if max_extra < 0:
        raise ValueError("Liczba dopisanych bajtów nie może być ujemna")
    sizes = _check_sizes(sizes, max_size)
    extra = rng.integers(0, max_extra + 1, size=sizes.size)
    return np.minimum(sizes + extra, max_size)


@dataclass(frozen=True)
class Defense:
    """Jedna obrona z parametrem: rozmiarem bloku albo górną granicą wypełnienia."""

    kind: str
    parameter: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Nieznana obrona: {self.kind!r}")
        if self.kind != "constant" and self.parameter is None:
            raise ValueError("Ta obrona wymaga parametru")

    @property
    def key(self) -> str:
        return self.kind if self.parameter is None else f"{self.kind}_{self.parameter}"

    @property
    def label(self) -> str:
        if self.kind == "constant":
            return "stały rozmiar"
        if self.kind == "block":
            return f"blok {self.parameter} B"
        return f"losowe 0–{self.parameter} B"

    def apply(self, sizes: np.ndarray, max_size: int, seed: int) -> np.ndarray:
        if self.kind == "constant":
            return pad_constant(sizes, max_size)
        if self.kind == "block":
            return pad_to_block(sizes, self.parameter, max_size)
        return pad_random(sizes, self.parameter, max_size, np.random.default_rng(seed))


def defenses_from_config(config) -> list[Defense]:
    """Buduje listę obron z sekcji `padding` konfiguracji."""
    return [
        Defense("constant"),
        *(Defense("block", int(block)) for block in config.block_sizes),
        *(Defense("random", int(extra)) for extra in config.random_max_extra),
    ]


def bandwidth_overhead(original: np.ndarray, padded: np.ndarray) -> float:
    """Zwraca względny przyrost liczby przesłanych bajtów."""
    total = float(np.asarray(original, dtype=np.int64).sum())
    if total == 0:
        raise ValueError("Brak bajtów do porównania")
    return float(np.asarray(padded, dtype=np.int64).sum()) / total - 1.0


def apply_defense(store: FlowStore, defense: Defense, max_size: int, seed: int) -> FlowStore:
    """Zwraca kopię magazynu, w której rozmiary pakietów przeszły przez obronę."""
    padded = defense.apply(store.sizes, max_size, seed)
    starts = np.asarray(store.offsets[:-1], dtype=np.int64)
    meta = store.meta.copy()
    meta["total_bytes"] = np.add.reduceat(padded, starts)
    meta["min_size"] = np.minimum.reduceat(padded, starts)
    meta["max_size"] = np.maximum.reduceat(padded, starts)
    return replace(store, meta=meta, sizes=padded.astype(np.uint16))
