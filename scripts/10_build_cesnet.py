"""Losuje zrównoważoną próbkę przepływów CESNET-TLS22 i zapisuje ją jako magazyn przepływów."""

import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from trafficfp.config import load_config
from trafficfp.data.cesnet import ARCHIVE_NAME, build_cesnet_store, day_members, sample_day


def main() -> None:
    config = load_config()
    settings = config.cesnet
    rng = np.random.default_rng(config.seed)
    samples = []
    with zipfile.ZipFile(settings.raw / ARCHIVE_NAME) as archive:
        for day_index, (week, member) in enumerate(tqdm(day_members(archive), desc="dni", unit="dzień")):
            day = sample_day(archive, member, settings.per_category_per_day, settings.min_packets, rng)
            day["week"] = week
            day["date"] = member.rsplit("-", 1)[-1][:8]
            day["day_index"] = day_index
            samples.append(day)
    store = build_cesnet_store(pd.concat(samples, ignore_index=True))
    store.save(settings.processed)

    meta = store.meta
    counts = meta.groupby(["category", "app", "partition"]).size().unstack("partition", fill_value=0).reset_index()
    counts.to_csv(config.paths.tables / "cesnet_counts.csv", index=False, encoding="utf-8")
    print(meta.groupby(["partition", "category"]).size().unstack())
    print("przepływów:", len(meta), "pakietów w PPI:", int(meta["ppi_len"].sum()))


if __name__ == "__main__":
    main()
