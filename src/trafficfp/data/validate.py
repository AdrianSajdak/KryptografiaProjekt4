"""Kontrola jakości danych po zbudowaniu magazynu przepływów."""

from __future__ import annotations

import pandas as pd

#: Liczności publikowane dla tego zbioru (m.in. dokumentacja tcbench).
EXPECTED_FLOWS = {
    "pretraining": {"Google Doc": 1221, "Google Drive": 1634, "Google Music": 592, "Google Search": 1915, "Youtube": 1077},
    "Retraining(script-triggered)": {"Google Doc": 30, "Google Drive": 30, "Google Music": 30, "Google Search": 30, "Youtube": 30},
    "Retraining(human-triggered)": {"Google Doc": 15, "Google Drive": 18, "Google Music": 15, "Google Search": 15, "Youtube": 20},
}

#: Największy rozmiar ramki dopuszczony w kontroli poprawności.
MAX_PLAUSIBLE_SIZE = 1514


def check_expected_counts(meta: pd.DataFrame) -> None:
    """Zgłasza błąd, gdy liczba przepływów różni się od opublikowanej."""
    counts = meta.groupby(["partition", "app"]).size()
    for partition, apps in EXPECTED_FLOWS.items():
        for app, expected in apps.items():
            actual = int(counts.get((partition, app), 0))
            if actual != expected:
                raise ValueError(f"{partition}/{app}: oczekiwano {expected}, jest {actual}")


def find_duplicates(meta: pd.DataFrame) -> pd.DataFrame:
    """Zwraca przepływy o identycznym starcie, liczbie pakietów i liczbie bajtów."""
    keys = ["start_time", "n_packets", "total_bytes"]
    return meta[meta.duplicated(subset=keys, keep=False)].sort_values(keys)


def quality_report(meta: pd.DataFrame) -> dict:
    """Podsumowuje jakość danych w postaci gotowej do zapisu w JSON."""
    check_expected_counts(meta)
    duplicates = find_duplicates(meta)
    return {
        "flows_total": int(len(meta)),
        "packets_total": int(meta["n_packets"].sum()),
        "flows_by_partition": {key: int(value) for key, value in meta["partition"].value_counts().items()},
        "min_packets_per_flow": int(meta["n_packets"].min()),
        "size_min": int(meta["min_size"].min()),
        "size_max": int(meta["max_size"].max()),
        "sizes_within_limit": bool(meta["max_size"].max() <= MAX_PLAUSIBLE_SIZE),
        "first_packet_outbound_fraction": float((meta["first_direction"] == 1).mean()),
        "flows_with_disordered_packets": int((meta["disordered_pairs"] > 0).sum()),
        "disordered_packet_pairs": int(meta["disordered_pairs"].sum()),
        "duplicate_flows": int(len(duplicates)),
    }
