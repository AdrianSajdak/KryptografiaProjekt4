"""Walidacyjny zbiór CESNET-TLS22: ruch TLS po TCP z sieci CESNET2 (4–17.10.2021).

Archiwum zawiera 14 dziennych plików CSV skompresowanych gzipem. Każdy wiersz to jeden
przepływ, a kolumna PPI zapisuje do 30 początkowych pakietów z danymi w postaci
[[odstępy w ms], [kierunki], [rozmiary]]. Plików nie rozpakowujemy: czytamy je
strumieniowo bezpośrednio z archiwum ZIP.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from trafficfp.data.dataset import FlowStore

ARCHIVE_NAME = "cesnet-tls22.zip"
ARCHIVE_URL = "https://zenodo.org/records/10610895/files/cesnet-tls22.zip?download=1"
ARCHIVE_MD5 = "5b15a7543ef71c3e64efde5d5cf08e08"
ARCHIVE_SIZE = 9_482_192_513

#: Usługi CESNET-TLS22 odpowiadające czterem kategoriom ruchu z instrukcji projektu.
#: Kategorie samego zbioru są szersze (np. „Streaming media” obejmuje też CDN-y czcionek),
#: dlatego wybieramy pojedyncze usługi. apple-itunes pomijamy, bo obejmuje też sklep z aplikacjami.
APP_TO_CATEGORY = {
    "youtube": "wideo",
    "netflix": "wideo",
    "twitch": "wideo",
    "vimeo": "wideo",
    "o2tv": "wideo",
    "amazon-prime": "wideo",
    "spotify": "audio",
    "soundcloud": "audio",
    "google-www": "www",
    "bing": "www",
    "duckduckgo": "www",
    "seznam-search": "www",
    "google-drive": "pliki",
    "dropbox": "pliki",
    "microsoft-onedrive": "pliki",
    "owncloud": "pliki",
    "ulozto": "pliki",
    "uschovna": "pliki",
    "cesnet-filesender": "pliki",
}
COLUMNS = ["ID", "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "DURATION", "PPI", "PPI_LEN", "APP"]
_DAY_FILE = re.compile(r"(W-\d{4}-\d{2})/(\d)_\w+/flows-(\d{8})\.csv\.gz$")


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download_archive(directory: Path) -> Path:
    """Pobiera archiwum z Zenodo, chyba że poprawna kopia już istnieje, i sprawdza MD5."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ARCHIVE_NAME
    if not (target.exists() and target.stat().st_size == ARCHIVE_SIZE):
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(ARCHIVE_URL) as response, open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 22)
        partial.replace(target)
    if md5_of(target) != ARCHIVE_MD5:
        raise ValueError(f"Niezgodna suma MD5 archiwum {target}")
    return target


def day_members(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Zwraca dzienne pliki przepływów jako pary (tydzień, data RRRRMMDD) w kolejności dat."""
    days = []
    for name in archive.namelist():
        match = _DAY_FILE.search(name)
        if match:
            days.append((match.group(3), match.group(1), name))
    return [(week, name) for _, week, name in sorted(days)]


def parse_ppi(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zamienia pole PPI na czasy względne [s], rozmiary [B] i kierunki (+1 od klienta)."""
    gaps, directions, sizes = json.loads(text)
    times = np.cumsum(np.asarray(gaps, dtype=np.float64)) / 1000.0
    if times.size:
        times -= times[0]
    return times, np.asarray(sizes, dtype=np.int64), np.asarray(directions, dtype=np.int64)


def sample_day(
    archive: zipfile.ZipFile,
    member: str,
    per_category: int,
    min_packets: int,
    rng: np.random.Generator,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Losuje z jednego dnia po `per_category` przepływów każdej kategorii.

    Bierzemy tylko wybrane usługi i przepływy mające co najmniej `min_packets` pakietów w PPI.
    """
    parts = []
    with archive.open(member) as raw, gzip.open(raw, "rt", encoding="utf-8") as text:
        for chunk in pd.read_csv(text, usecols=COLUMNS, chunksize=chunksize):
            chunk = chunk[chunk["APP"].isin(APP_TO_CATEGORY.keys()) & (chunk["PPI_LEN"] >= min_packets)]
            if len(chunk):
                parts.append(chunk)
    day = pd.concat(parts, ignore_index=True)
    day["category"] = day["APP"].map(APP_TO_CATEGORY)
    chosen = []
    for _, group in day.groupby("category", sort=True):
        take = min(per_category, len(group))
        chosen.append(group.iloc[np.sort(rng.choice(len(group), size=take, replace=False))])
    return pd.concat(chosen, ignore_index=True)


def build_cesnet_store(samples: pd.DataFrame) -> FlowStore:
    """Buduje magazyn przepływów z wylosowanych wierszy.

    `start_time` służy tylko do porządkowania przepływów: zbiór nie podaje czasu
    rozpoczęcia, więc łączymy numer dnia z identyfikatorem wiersza w pliku dnia.
    """
    times, sizes, directions, lengths = [], [], [], []
    for text in samples["PPI"]:
        flow_times, flow_sizes, flow_directions = parse_ppi(text)
        times.append(flow_times)
        sizes.append(flow_sizes)
        directions.append(flow_directions)
        lengths.append(flow_times.size)
    meta = pd.DataFrame(
        {
            "flow_id": samples["date"].astype(str) + "-" + samples["ID"].astype(str),
            "partition": samples["week"].to_numpy(),
            "app": samples["APP"].to_numpy(),
            "category": samples["category"].to_numpy(),
            "date": samples["date"].to_numpy(),
            "start_time": samples["day_index"].to_numpy(dtype=np.float64) * 1e9 + samples["ID"].to_numpy(dtype=np.float64),
            "ppi_len": np.asarray(lengths, dtype=np.int64),
            "n_packets": (samples["PACKETS"] + samples["PACKETS_REV"]).to_numpy(dtype=np.int64),
            "total_bytes": (samples["BYTES"] + samples["BYTES_REV"]).to_numpy(dtype=np.int64),
            "duration": samples["DURATION"].to_numpy(dtype=np.float64),
        }
    )
    offsets = np.concatenate(([0], np.cumsum(meta["ppi_len"].to_numpy())))
    return FlowStore(
        meta=meta,
        offsets=offsets,
        times=np.concatenate(times),
        sizes=np.concatenate(sizes).astype(np.uint16),
        directions=np.concatenate(directions).astype(np.int8),
    )
