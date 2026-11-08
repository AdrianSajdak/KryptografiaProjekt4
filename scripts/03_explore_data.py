"""Etap 1: analiza eksploracyjna zbioru i wykresy do sprawozdania."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trafficfp.config import load_config
from trafficfp.data.dataset import FlowStore
from trafficfp.visualization import FIGURE_WIDTH, INK_SECONDARY, SERIES, apply_style, save_figure

PRETRAINING = "pretraining"
HUMAN = "Retraining(human-triggered)"
SCRIPT = "Retraining(script-triggered)"
PARTITION_LABELS = {PRETRAINING: "pretraining (skrypt)", SCRIPT: "retraining (skrypt)", HUMAN: "retraining (człowiek)"}
SIZE_BINS = np.arange(0, 1551, 50)
SIZE_SAMPLE_PACKETS = 150


def save_class_counts(meta: pd.DataFrame, directory) -> None:
    table = meta.pivot_table(index=["category", "app"], columns="partition", values="flow_id", aggfunc="count")
    table[[PRETRAINING, SCRIPT, HUMAN]].to_csv(directory / "eda_class_counts.csv", encoding="utf-8")


def save_flow_summary(meta: pd.DataFrame, directory) -> None:
    summary = (
        meta.groupby(["partition", "app"])
        .agg(
            flows=("flow_id", "count"),
            packets_median=("n_packets", "median"),
            bytes_median=("total_bytes", "median"),
            duration_median=("duration", "median"),
            duration_p05=("duration", lambda s: s.quantile(0.05)),
            duration_p95=("duration", lambda s: s.quantile(0.95)),
        )
        .round(2)
    )
    summary.to_csv(directory / "eda_flow_summary.csv", encoding="utf-8")


def save_prefix_times(store: FlowStore, packet_counts, directory) -> None:
    """Zapisuje medianę czasu potrzebnego na zaobserwowanie N pierwszych pakietów."""
    meta = store.meta
    rows = []
    for count in packet_counts:
        ends = store.offsets[:-1] + count - 1
        times = pd.Series(np.asarray(store.times[ends]), index=meta.index)
        selected = meta["partition"] == PRETRAINING
        for app, value in times[selected].groupby(meta.loc[selected, "app"]).median().items():
            rows.append({"n_packets": count, "app": app, "median_seconds": round(float(value), 3)})
    table = pd.DataFrame(rows).pivot(index="n_packets", columns="app", values="median_seconds")
    table.to_csv(directory / "eda_prefix_time.csv", encoding="utf-8")


def plot_durations(meta: pd.DataFrame, path) -> None:
    order = (
        meta[meta["partition"] == PRETRAINING].groupby("app")["duration"].median().sort_values().index.tolist()
    )
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 3.0))
    offsets = {PRETRAINING: 0.18, HUMAN: -0.18}
    for color, (partition, offset) in zip(SERIES, offsets.items()):
        groups = [meta[(meta["partition"] == partition) & (meta["app"] == app)]["duration"] for app in order]
        positions = np.arange(len(order)) + offset
        box = axis.boxplot(
            groups, positions=positions, orientation="horizontal", widths=0.26, whis=(5, 95),
            showfliers=False, patch_artist=True, manage_ticks=False,
        )
        for patch in box["boxes"]:
            patch.set(facecolor=color, edgecolor=color, alpha=0.45, linewidth=1)
        for element in ("whiskers", "caps", "medians"):
            for line in box[element]:
                line.set(color=color, linewidth=1.2)
        axis.plot([], [], color=color, linewidth=5, alpha=0.6, label=PARTITION_LABELS[partition])
    axis.set_xscale("log")
    axis.set_yticks(range(len(order)), order)
    axis.set_xlabel("Czas trwania przepływu [s], skala logarytmiczna")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="lower right")
    save_figure(figure, path)


def plot_capture_timeline(meta: pd.DataFrame, path) -> None:
    order = sorted(meta["app"].unique())
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 2.6))
    dates = pd.to_datetime(meta["start_time"], unit="s")
    offsets = {PRETRAINING: 0.2, SCRIPT: 0.0, HUMAN: -0.2}
    for color, (partition, offset) in zip(SERIES, offsets.items()):
        mask = meta["partition"] == partition
        positions = meta.loc[mask, "app"].map(order.index) + offset
        axis.scatter(dates[mask], positions, s=6, color=color, label=PARTITION_LABELS[partition], linewidths=0)
    axis.set_yticks(range(len(order)), order)
    axis.set_xlabel("Data zebrania przepływu (2018)")
    axis.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d.%m"))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3)
    save_figure(figure, path)


def plot_packet_sizes(store: FlowStore, path) -> None:
    """Rysuje rozkład rozmiarów pierwszych pakietów osobno dla obu kierunków."""
    meta = store.meta
    apps = sorted(meta["app"].unique())
    figure, axes = plt.subplots(len(apps), 1, figsize=(FIGURE_WIDTH, 5.4), sharex=True, sharey=True)
    selected = meta.index[meta["partition"] == PRETRAINING]
    for axis, app in zip(axes, apps):
        sizes, directions = [], []
        for index in selected[meta.loc[selected, "app"] == app]:
            _, flow_sizes, flow_directions = store.packets(int(index), SIZE_SAMPLE_PACKETS)
            sizes.append(np.asarray(flow_sizes))
            directions.append(np.asarray(flow_directions))
        size_array, direction_array = np.concatenate(sizes), np.concatenate(directions)
        for color, sign, label in ((SERIES[0], 1, "klient → serwer"), (SERIES[1], -1, "serwer → klient")):
            values = size_array[direction_array == sign]
            weights = np.full(values.size, 100.0 / size_array.size)
            axis.hist(values, bins=SIZE_BINS, weights=weights, color=color, alpha=0.75, label=label, rwidth=0.86)
        axis.set_title(app, fontsize=9, pad=2)
        axis.grid(axis="x", visible=False)
    axes[0].legend(loc="upper center", ncol=2)
    axes[2].set_ylabel("Udział pakietów [%]", color=INK_SECONDARY)
    axes[-1].set_xlabel(f"Rozmiar pakietu [B], pierwsze {SIZE_SAMPLE_PACKETS} pakietów przepływu")
    figure.tight_layout(h_pad=0.4)
    save_figure(figure, path)


def main() -> None:
    config = load_config()
    apply_style()
    store = FlowStore.load(config.paths.processed)
    config.paths.tables.mkdir(parents=True, exist_ok=True)

    save_class_counts(store.meta, config.paths.tables)
    save_flow_summary(store.meta, config.paths.tables)
    save_prefix_times(store, config.features.packet_counts, config.paths.tables)

    plot_durations(store.meta, config.paths.figures / "eda_flow_duration.png")
    plot_capture_timeline(store.meta, config.paths.figures / "eda_capture_timeline.png")
    plot_packet_sizes(store, config.paths.figures / "eda_packet_sizes.png")


if __name__ == "__main__":
    main()
