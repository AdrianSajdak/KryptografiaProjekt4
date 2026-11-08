"""Zwarty magazyn przepływów i budowanie go z archiwów źródłowych."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from trafficfp.data.labels import category_of
from trafficfp.data.loader import Flow, iter_archive_flows

_META_FILE = "flows_meta.parquet"
_ARRAYS = {"offsets": "offsets.npy", "times": "times.npy", "sizes": "sizes.npy", "directions": "directions.npy"}


@dataclass(frozen=True)
class FlowStore:
    """Pakiety wszystkich przepływów w jednej tablicy z indeksem `offsets`."""

    meta: pd.DataFrame
    offsets: np.ndarray
    times: np.ndarray
    sizes: np.ndarray
    directions: np.ndarray

    def __len__(self) -> int:
        return len(self.meta)

    def packets(self, index: int, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zwraca czasy, rozmiary i kierunki pierwszych `limit` pakietów przepływu."""
        begin = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        if limit is not None:
            end = min(end, begin + limit)
        return self.times[begin:end], self.sizes[begin:end], self.directions[begin:end]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.meta.to_parquet(directory / _META_FILE, index=False)
        for attribute, filename in _ARRAYS.items():
            np.save(directory / filename, getattr(self, attribute), allow_pickle=False)

    @classmethod
    def load(cls, directory: Path, mmap: bool = True) -> "FlowStore":
        mode = "r" if mmap else None
        arrays = {
            attribute: np.load(directory / filename, mmap_mode=mode, allow_pickle=False)
            for attribute, filename in _ARRAYS.items()
        }
        return cls(meta=pd.read_parquet(directory / _META_FILE), **arrays)


def repair_packet_order(flow: Flow) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Porządkuje pakiety według czasu i zwraca liczbę par w złej kolejności."""
    disorders = int((np.diff(flow.times) < 0).sum())
    if disorders == 0:
        return flow.times, flow.sizes, flow.directions, 0
    order = np.argsort(flow.times, kind="stable")
    times = flow.times[order]
    return times - times[0], flow.sizes[order], flow.directions[order], disorders


def build_flow_store(archives: Iterable[Path]) -> FlowStore:
    """Scala archiwa w magazyn i uzupełnia metadane przepływów."""
    records: list[dict] = []
    times: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    directions: list[np.ndarray] = []

    for archive in archives:
        for flow in tqdm(iter_archive_flows(archive), desc=archive.stem, unit="flow"):
            flow_times, flow_sizes, flow_directions, disorders = repair_packet_order(flow)
            records.append(
                {
                    "flow_id": flow.flow_id,
                    "partition": flow.partition,
                    "app": flow.app,
                    "category": category_of(flow.app),
                    "start_time": flow.start_time,
                    "n_packets": flow.n_packets,
                    "total_bytes": int(flow_sizes.sum()),
                    "duration": float(flow_times[-1]),
                    "out_packets": int((flow_directions > 0).sum()),
                    "first_direction": int(flow.directions[0]),
                    "min_size": int(flow_sizes.min()),
                    "max_size": int(flow_sizes.max()),
                    "disordered_pairs": disorders,
                }
            )
            times.append(flow_times.astype(np.float64))
            sizes.append(flow_sizes.astype(np.uint16))
            directions.append(flow_directions.astype(np.int8))

    meta = pd.DataFrame.from_records(records)
    offsets = np.concatenate(([0], np.cumsum(meta["n_packets"].to_numpy(dtype=np.int64))))
    return FlowStore(
        meta=meta,
        offsets=offsets,
        times=np.concatenate(times),
        sizes=np.concatenate(sizes),
        directions=np.concatenate(directions),
    )
