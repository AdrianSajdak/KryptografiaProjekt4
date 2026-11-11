"""Wczytywanie konfiguracji eksperymentów z pliku YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed: Path
    tables: Path
    figures: Path


@dataclass(frozen=True)
class FeatureConfig:
    packet_counts: tuple[int, ...]
    feature_sets: tuple[str, ...]


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float
    cv_folds: int


@dataclass(frozen=True)
class KnnConfig:
    neighbors: tuple[int, ...]
    metrics: tuple[str, ...]
    weights: tuple[str, ...]
    selection_scoring: str


@dataclass(frozen=True)
class PaddingConfig:
    max_size: int
    block_sizes: tuple[int, ...]
    random_max_extra: tuple[int, ...]


@dataclass(frozen=True)
class BaselineConfig:
    seed: int
    rf_n_estimators: int
    rf_max_depth: tuple[int | None, ...]
    rf_min_samples_leaf: tuple[int, ...]
    svm_c: tuple[float, ...]
    svm_gamma: tuple[float | str, ...]


@dataclass(frozen=True)
class CesnetConfig:
    raw: Path
    processed: Path
    per_category_per_day: int
    min_packets: int
    train_week: str
    test_week: str
    packet_counts: tuple[int, ...]
    neighbors: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    features: FeatureConfig
    split: SplitConfig
    knn: KnnConfig
    accuracy_threshold: float
    padding: PaddingConfig
    baselines: BaselineConfig
    cesnet: CesnetConfig


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    """Wczytuje konfigurację i zamienia ścieżki na bezwzględne."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    paths = {key: PROJECT_ROOT / value for key, value in raw["paths"].items()}
    knn = raw["knn"]
    return Config(
        seed=int(raw["seed"]),
        paths=Paths(**paths),
        features=FeatureConfig(
            packet_counts=tuple(raw["features"]["packet_counts"]),
            feature_sets=tuple(raw["features"]["feature_sets"]),
        ),
        split=SplitConfig(
            test_fraction=float(raw["split"]["test_fraction"]),
            cv_folds=int(raw["split"]["cv_folds"]),
        ),
        knn=KnnConfig(
            neighbors=tuple(knn["neighbors"]),
            metrics=tuple(knn["metrics"]),
            weights=tuple(knn["weights"]),
            selection_scoring=knn["selection_scoring"],
        ),
        accuracy_threshold=float(raw["success"]["accuracy_threshold"]),
        padding=PaddingConfig(
            max_size=int(raw["padding"]["max_size"]),
            block_sizes=tuple(raw["padding"]["block_sizes"]),
            random_max_extra=tuple(raw["padding"]["random_max_extra"]),
        ),
        baselines=BaselineConfig(
            seed=int(raw["seed"]),
            rf_n_estimators=int(raw["baselines"]["random_forest"]["n_estimators"]),
            rf_max_depth=tuple(raw["baselines"]["random_forest"]["max_depth"]),
            rf_min_samples_leaf=tuple(raw["baselines"]["random_forest"]["min_samples_leaf"]),
            svm_c=tuple(float(value) for value in raw["baselines"]["svm"]["C"]),
            svm_gamma=tuple(raw["baselines"]["svm"]["gamma"]),
        ),
        cesnet=CesnetConfig(
            raw=PROJECT_ROOT / raw["cesnet"]["raw"],
            processed=PROJECT_ROOT / raw["cesnet"]["processed"],
            per_category_per_day=int(raw["cesnet"]["per_category_per_day"]),
            min_packets=int(raw["cesnet"]["min_packets"]),
            train_week=str(raw["cesnet"]["train_week"]),
            test_week=str(raw["cesnet"]["test_week"]),
            packet_counts=tuple(raw["cesnet"]["packet_counts"]),
            neighbors=tuple(raw["cesnet"]["neighbors"]),
        ),
    )
