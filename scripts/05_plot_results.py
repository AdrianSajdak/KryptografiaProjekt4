"""Etap 3: wykresy wyników k-NN do sprawozdania."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trafficfp.config import load_config
from trafficfp.visualization import (
    FIGURE_WIDTH,
    INK,
    INK_SECONDARY,
    MUTED,
    SERIES,
    SURFACE,
    apply_style,
    save_figure,
)

FEATURE_LABELS = {
    "sizes": "rozmiary",
    "timing": "odstępy czasu",
    "sizes+timing": "rozmiary + odstępy",
    "stats": "statystyki",
    "all": "wszystkie",
}
CONTROL_LABELS = {
    "whole_flow_volume_knn": "k-NN: objętość całego przepływu",
    "majority": "klasa większościowa",
    "permuted_labels_knn": "k-NN: przetasowane etykiety",
    "chance": "losowanie wg rozkładu klas",
}
TEST_LABELS = {
    "test_chronological": "test chronologiczny\n(główny)",
    "test_script_retraining": "retraining\n(skrypt)",
    "test_human_retraining": "retraining\n(człowiek)",
}


def percent(value: float) -> str:
    """Formatuje procent z przecinkiem dziesiętnym."""
    return f"{value:.1f}".replace(".", ",")


def load_results(config, target):
    path = config.paths.tables / f"results_{target}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def plot_cv_by_packets(table: pd.DataFrame, path) -> None:
    best = table.groupby(["feature_set", "n_packets"])["cv_f1_macro"].max().reset_index()
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 3.2))
    for color, (feature_set, label) in zip(SERIES, FEATURE_LABELS.items()):
        series = best[best["feature_set"] == feature_set].sort_values("n_packets")
        axis.plot(
            series["n_packets"], series["cv_f1_macro"] * 100, color=color, linewidth=2,
            marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, label=label,
        )
    packets = sorted(best["n_packets"].unique())
    axis.set_xticks(packets)
    axis.set_xlabel("Liczba początkowych pakietów widzianych przez przeciwnika (N)")
    axis.set_ylabel("Macro-F1 w walidacji krzyżowej [%]")
    axis.legend(loc="lower right", ncol=2)
    save_figure(figure, path)


def plot_confusion(report: dict, axis, title: str) -> None:
    summary = report["evaluations"]["test_chronological"]
    labels = summary["labels"]
    matrix = np.asarray(summary["confusion_matrix"], dtype=float)
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    axis.imshow(shares, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"]), vmin=0, vmax=1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = SURFACE if shares[row, column] > 0.55 else INK
            axis.text(column, row, f"{int(matrix[row, column])}", ha="center", va="center", color=color, fontsize=9)
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Klasa przewidziana")
    axis.set_ylabel("Klasa rzeczywista")
    axis.set_title(title, fontsize=9.5)
    axis.grid(visible=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_controls(report: dict, threshold: float, path) -> None:
    main = report["evaluations"]["test_chronological"]["accuracy"]
    values = {name: report["controls"][name]["test_chronological"]["accuracy"] for name in CONTROL_LABELS}
    labels = ["k-NN: pierwsze N pakietów (model główny)", *CONTROL_LABELS.values()]
    accuracies = [main, *values.values()]
    positions = np.arange(len(labels))[::-1]
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 2.5))
    colors = [SERIES[0]] + [MUTED] * len(values)
    axis.barh(positions, np.asarray(accuracies) * 100, height=0.55, color=colors)
    for position, accuracy in zip(positions, accuracies):
        axis.text(accuracy * 100 + 1, position, f"{percent(accuracy * 100)}%", va="center", color=INK_SECONDARY)
    axis.axvline(threshold * 100, color=INK, linewidth=1)
    axis.text(threshold * 100 - 0.8, positions[0] + 0.48, "kryterium 90%", ha="right", color=INK, fontsize=8.5)
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 112)
    axis.set_xlabel("Accuracy na zbiorze testowym [%]")
    axis.grid(axis="y", visible=False)
    save_figure(figure, path)


def plot_external_validity(report: dict, threshold: float, path) -> None:
    names = list(TEST_LABELS)
    prefix = [report["evaluations"][name]["accuracy"] * 100 for name in names]
    volume = [report["controls"]["whole_flow_volume_knn"][name]["accuracy"] * 100 for name in names]
    positions = np.arange(len(names))
    width = 0.3
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 3.0))
    first = axis.bar(positions - width / 2 - 0.01, prefix, width, color=SERIES[0], label="k-NN: pierwsze N pakietów")
    second = axis.bar(positions + width / 2 + 0.01, volume, width, color=SERIES[1], label="k-NN: objętość całego przepływu")
    for bars in (first, second):
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, percent(bar.get_height()),
                      ha="center", color=INK_SECONDARY, fontsize=8.5)
    axis.axhline(threshold * 100, color=INK, linewidth=1)
    axis.text(positions[-1] + 0.48, threshold * 100 + 1.2, "kryterium 90%", ha="right", color=INK, fontsize=8.5)
    axis.set_xticks(positions, [TEST_LABELS[name] for name in names])
    axis.set_ylim(0, 112)
    axis.set_ylabel("Accuracy [%]")
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    save_figure(figure, path)


def main() -> None:
    config = load_config()
    apply_style()
    category = load_results(config, "category")
    application = load_results(config, "app")
    figures = config.paths.figures

    table = pd.read_csv(config.paths.tables / "cv_category.csv")
    plot_cv_by_packets(table, figures / "results_cv_by_packets.png")

    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.2))
    plot_confusion(category, axes[0], "4 kategorie ruchu")
    plot_confusion(application, axes[1], "5 aplikacji")
    figure.tight_layout(w_pad=2.0)
    save_figure(figure, figures / "results_confusion.png")

    plot_controls(category, config.accuracy_threshold, figures / "results_controls.png")
    plot_external_validity(category, config.accuracy_threshold, figures / "results_external_validity.png")


if __name__ == "__main__":
    main()
