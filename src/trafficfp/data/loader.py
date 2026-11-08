"""Parser plików przepływów zbioru UC Davis QUIC.

Każdy plik .txt opisuje jeden przepływ, a każdy wiersz jeden pakiet:
    znacznik_czasu <TAB> czas_względny <TAB> rozmiar <TAB> kierunek

Parser korzysta wyłącznie z metadanych dostępnych biernemu podsłuchującemu.
Archiwa czytane są bezpośrednio z ZIP, bez rozpakowywania na dysk.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

#: Kierunek w pliku źródłowym: 1 = klient -> serwer, 0 = serwer -> klient.
RAW_DIRECTION_OUTBOUND = 1

#: Kierunek po normalizacji.
OUTBOUND = 1
INBOUND = -1

_COLUMNS = ["timestamp", "relative", "size", "direction"]
_DTYPES = {"timestamp": "float64", "relative": "float64", "size": "int32", "direction": "int8"}


@dataclass(frozen=True)
class Flow:
    """Przepływ w postaci trzech równoległych tablic o długości liczby pakietów."""

    flow_id: str
    partition: str
    app: str
    start_time: float
    times: np.ndarray
    sizes: np.ndarray
    directions: np.ndarray

    @property
    def n_packets(self) -> int:
        return int(self.sizes.size)


def parse_flow_bytes(raw: bytes) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Zamienia zawartość pliku przepływu na tablice (start, czasy, rozmiary, kierunki).

    Czasy wyliczane są z bezwzględnego znacznika, ponieważ kolumna czasu
    względnego ma w źródle tylko sześć cyfr znaczących.
    """
    frame = pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        header=None,
        names=_COLUMNS,
        dtype=_DTYPES,
        engine="c",
    )
    if frame.empty:
        raise ValueError("Plik przepływu nie zawiera pakietów")

    absolute = frame["timestamp"].to_numpy(dtype=np.float64)
    start = float(absolute[0])
    directions = np.where(
        frame["direction"].to_numpy() == RAW_DIRECTION_OUTBOUND, OUTBOUND, INBOUND
    ).astype(np.int8)
    return start, absolute - start, frame["size"].to_numpy(dtype=np.int32), directions


def parse_member_path(member: str) -> tuple[str, str, str]:
    """Rozkłada ścieżkę `partycja/aplikacja/nazwa.txt` na trzy składniki."""
    parts = PurePosixPath(member).parts
    if len(parts) != 3:
        raise ValueError(f"Nieoczekiwana ścieżka w archiwum: {member!r}")
    partition, app, filename = parts
    return partition, app, PurePosixPath(filename).stem


def iter_archive_flows(archive: Path) -> Iterator[Flow]:
    """Iteruje po przepływach archiwum ZIP w ustalonej kolejności nazw."""
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(
            name for name in bundle.namelist() if name.lower().endswith(".txt")
        )
        for member in members:
            partition, app, stem = parse_member_path(member)
            start, times, sizes, directions = parse_flow_bytes(bundle.read(member))
            yield Flow(
                flow_id=f"{partition}/{stem}",
                partition=partition,
                app=app,
                start_time=start,
                times=times,
                sizes=sizes,
                directions=directions,
            )
