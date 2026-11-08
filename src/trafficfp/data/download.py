"""Pobieranie zbioru UC Davis QUIC i kontrola integralności archiwów.

Link z instrukcji projektu (github.com/ucdavis/quic-data) jest nieaktywny.
Dane pobierane są z folderu Google Drive wskazanego w repozytorium autorów
zbioru: github.com/shrezaei/Semi-supervised-Learning-QUIC-
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GOOGLE_DRIVE_FOLDER_ID = "1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE"

#: Sumy SHA-256 archiwów zarejestrowane przy pierwszym pobraniu.
ARCHIVE_SHA256 = {
    "pretraining.zip": "0c87e7fb41e30144607e878822fe710e91e280f5750d85f5be1c468fda57f88e",
    "Retraining(script-triggered).zip": "ca032ebbb51f9ed1f4b1422c0641790cdd809fe98e5b32aa333a571155128a61",
    "Retraining(human-triggered).zip": "752ef08a4d6ad5ee75396bcdb73371cce6286317ebcbabfd0bea5b29f4f33262",
}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Oblicza sumę SHA-256 pliku bez wczytywania go w całości do pamięci."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives(directory: Path) -> list[Path]:
    """Sprawdza obecność i sumy kontrolne archiwów, zwraca ich ścieżki."""
    verified = []
    for name, expected in ARCHIVE_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Brak archiwum: {path}")
        actual = sha256_of(path)
        if actual != expected:
            raise ValueError(f"Niezgodna suma SHA-256 pliku {name}: {actual}")
        verified.append(path)
    return verified


def download_dataset(directory: Path) -> list[Path]:
    """Pobiera brakujące archiwa z Google Drive i weryfikuje ich integralność."""
    import gdown

    directory.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        id=GOOGLE_DRIVE_FOLDER_ID,
        output=str(directory),
        quiet=False,
        resume=True,
        retries=3,
    )
    return verify_archives(directory)
