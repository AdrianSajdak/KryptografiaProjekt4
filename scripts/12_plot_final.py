"""Wykresy etapu końcowego: padding, przesunięcie danych, modele referencyjne, CESNET-TLS22.

Każdy wykres powstaje tylko wtedy, gdy istnieje odpowiadający mu plik wyników.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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

APP_ORDER = ["Youtube", "Google Music", "Google Search", "Google Doc", "Google Drive"]
APP_LABELS = {"Youtube": "YouTube", "Google Music": "Google Music", "Google Search": "Google Search", "Google Doc": "Google Doc", "Google Drive": "Google Drive"}
SPLIT_LABELS = {
    "test_chronological": "test główny",
    "test_script_retraining": "retraining (skrypt)",
    "test_human_retraining": "retraining (człowiek)",
}
FEATURE_LABELS = {"sizes": "rozmiary", "timing": "odstępy czasu", "sizes+timing": "rozmiary + odstępy", "stats": "statystyki", "all": "wszystkie"}
MODEL_LABELS = {"knn": "k-NN", "svm": "SVM", "random_forest": "las losowy"}
GROUP_ORDER = ["odstępy czasu", "histogramy", "rozmiary", "ogólne"]
BLUES = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])


def percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def load(config, name):
    path = config.paths.tables / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def threshold_line(axis, threshold, horizontal=True, label_x=None):
    if horizontal:
        axis.axhline(threshold * 100, color=INK, linewidth=1)
        if label_x is not None:
            axis.text(label_x, threshold * 100 + 1.2, "kryterium 90%", ha="right", color=INK, fontsize=8.5)


# ---------------------------------------------------------------- przesunięcie danych
def plot_shift_distance(shift, path):
    figure, (left, right) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    names = list(SPLIT_LABELS)
    for row, name in enumerate(names[::-1]):
        q = shift["nearest_distance"][name]
        color = SERIES[names.index(name)]
        left.plot([q["p10"], q["p90"]], [row, row], color=color, linewidth=2, solid_capstyle="round")
        left.plot([q["p25"], q["p75"]], [row, row], color=color, linewidth=7, solid_capstyle="round")
        left.scatter([q["p50"]], [row], s=46, color=SURFACE, edgecolor=color, linewidth=2, zorder=3)
        left.text(q["p90"] * 1.12, row, f"mediana {percent(q['p50'], 1)}", va="center", color=INK_SECONDARY, fontsize=8.5)
    left.set_xscale("log")
    left.set_yticks(range(len(names)), [SPLIT_LABELS[name] for name in names[::-1]])
    left.set_xlim(1.2, 400)
    left.set_ylim(-0.6, len(names) - 0.4)
    left.set_xlabel("Odległość do najbliższego przepływu uczącego (skala log.)")
    left.grid(axis="y", visible=False)
    left.set_title("Jak daleko są przepływy testowe", fontsize=9.5)

    terciles = shift["human_accuracy_by_distance"]
    values = [item["accuracy"] * 100 for item in terciles]
    bars = right.bar(range(3), values, width=0.6, color=SERIES[2])
    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2, value + 2, percent(value), ha="center", color=INK_SECONDARY, fontsize=8.5)
    right.set_xticks(range(3), ["najbliższe", "środkowe", "najdalsze"])
    right.set_ylim(0, 112)
    right.set_ylabel("Accuracy [%]")
    right.grid(axis="x", visible=False)
    right.set_title("Ruch ludzi: trafność wg odległości", fontsize=9.5)
    figure.tight_layout(w_pad=2.5)
    save_figure(figure, path)


def plot_shift_coverage(shift, path):
    figure, (left, right) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.4))
    behaviour = shift["behaviour_by_decile"]
    deciles = np.arange(1, 11)
    for color, app in zip(SERIES, APP_ORDER):
        values = np.asarray(behaviour[app]["out_packet_share_median"]) * 100
        left.plot(deciles, values, color=color, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=0.8, label=APP_LABELS[app])
    left.set_xticks(deciles)
    left.set_ylim(0, 100)
    left.set_xlabel("Kolejne 10% przepływów (od najstarszych)")
    left.set_ylabel("Pakiety od klienta, mediana [%]")
    left.set_title("Zmiana trybu skryptu Google Drive", fontsize=9.5)
    left.legend(loc="upper center", fontsize=7.5, ncol=3, bbox_to_anchor=(0.5, -0.24), columnspacing=0.8)

    coverage = shift["class_coverage"]
    fractions = [item["train_fraction"] * 100 for item in coverage]
    overall = [item["accuracy"] * 100 for item in coverage]
    drive = [item["per_app"]["Google Drive"] * 100 for item in coverage]
    right.plot(fractions, drive, color=SERIES[4], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, label="Google Drive")
    right.plot(fractions, overall, color=SERIES[0], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, label="wszystkie kategorie")
    right.text(fractions[0] + 1.5, overall[0], percent(overall[0]), va="center", color=INK_SECONDARY, fontsize=8)
    right.text(fractions[0] + 1.5, drive[0], percent(drive[0]), va="center", color=INK_SECONDARY, fontsize=8)
    right.text(fractions[-1], min(overall[1:] + drive[1:]) - 14, f"od 60%: co najmniej {percent(min(overall[1:] + drive[1:]))}%", ha="right", color=INK_SECONDARY, fontsize=8)
    right.set_xticks(fractions, [f"{int(x)}%" for x in fractions])
    right.set_ylim(0, 108)
    right.set_xlabel("Najstarsze przepływy użyte do uczenia")
    right.set_ylabel("Accuracy na pozostałych [%]")
    right.set_title("Trafność a pokrycie zachowań klasy", fontsize=9.5)
    right.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.24))
    figure.tight_layout(w_pad=2.0)
    save_figure(figure, path)


# ---------------------------------------------------------------- modele referencyjne
def plot_baselines(baselines, threshold, path):
    figure, (left, right) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    models = list(MODEL_LABELS)
    splits = list(SPLIT_LABELS)
    width = 0.25
    positions = np.arange(len(splits))
    for index, model in enumerate(models):
        values = [baselines["models"][model][split]["accuracy"] * 100 for split in splits]
        offset = (index - 1) * (width + 0.02)
        bars = left.bar(positions + offset, values, width, color=SERIES[index], label=MODEL_LABELS[model])
        for bar, value in zip(bars, values):
            left.text(bar.get_x() + bar.get_width() / 2, value + 1, percent(value), ha="center", color=INK_SECONDARY, fontsize=6.8, rotation=90, va="bottom")
    left.axhline(threshold * 100, color=INK, linewidth=1)
    left.text(positions[-1] + 0.45, threshold * 100 + 1.5, "90%", ha="right", color=INK, fontsize=8)
    left.set_xticks(positions, ["test główny", "retraining\n(skrypt)", "retraining\n(człowiek)"])
    left.set_ylim(0, 125)
    left.set_yticks(range(0, 101, 20))
    left.set_ylabel("Accuracy [%]")
    left.grid(axis="x", visible=False)
    left.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))

    importance = baselines["group_importance"]
    rows = np.arange(len(GROUP_ORDER))[::-1]
    height = 0.36
    for index, model in enumerate(["knn", "random_forest"]):
        values = [importance[model]["groups"][group]["mean"] for group in GROUP_ORDER]
        offset = (0.5 - index) * (height + 0.03)
        right.barh(rows + offset, values, height, color=SERIES[0 if model == "knn" else 2], label=MODEL_LABELS[model])
    right.set_yticks(rows, GROUP_ORDER)
    right.set_xlabel("Spadek makro-F1 po przetasowaniu grupy")
    right.grid(axis="y", visible=False)
    right.legend(loc="lower right")
    figure.tight_layout(w_pad=2.0)
    save_figure(figure, path)


# ---------------------------------------------------------------- padding
def plot_padding_tradeoff(padding, threshold, path):
    main = padding["main_feature_set"]
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 3.4))
    base = padding["baseline"][main]["test_chronological"]["accuracy"] * 100
    overhead = [d["overhead"] * 100 for d in padding["defenses"]]
    naive = [d["naive"][main]["test_chronological"]["accuracy"] * 100 for d in padding["defenses"]]
    adaptive = [max(item["test_chronological"]["accuracy"] for item in d["adaptive"]["by_feature_set"].values()) * 100 for d in padding["defenses"]]
    axis.scatter([0], [base], s=60, color=INK, zorder=4, label="bez obrony")
    axis.scatter(overhead, naive, s=55, color=SERIES[1], edgecolor=SURFACE, linewidth=1.2, zorder=3, label="przeciwnik naiwny (statystyki)")
    axis.scatter(overhead, adaptive, s=55, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=3, label="przeciwnik adaptacyjny (najlepszy zestaw)")
    for d, x, y_naive, y_adaptive in zip(padding["defenses"], overhead, naive, adaptive):
        axis.plot([x, x], [y_naive, y_adaptive], color=MUTED, linewidth=0.8, zorder=1)
        axis.text(x, min(y_naive, y_adaptive) - 3.2, d["label"], ha="center", va="top", fontsize=7, color=INK_SECONDARY, rotation=0)
    axis.axhline(threshold * 100, color=INK, linewidth=1)
    axis.text(max(overhead) * 1.02, threshold * 100 + 1, "kryterium 90%", ha="right", color=INK, fontsize=8.5)
    axis.set_xlabel("Narzut przepustowości [% dodatkowych bajtów]")
    axis.set_ylabel("Accuracy na teście głównym [%]")
    axis.set_ylim(0, 108)
    axis.set_xscale("symlog", linthresh=10)
    axis.legend(loc="lower left", ncol=1)
    save_figure(figure, path)


def plot_padding_heatmap(padding, path):
    sets = list(FEATURE_LABELS)
    defenses = padding["defenses"]
    rows = ["brak obrony", *[d["label"] for d in defenses]]
    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.9), sharey=True)
    for axis, adversary, title in ((axes[0], "naive", "Przeciwnik naiwny"), (axes[1], "adaptive", "Przeciwnik adaptacyjny")):
        matrix = [[padding["baseline"][s]["test_chronological"]["accuracy"] for s in sets]]
        for d in defenses:
            source = d["naive"] if adversary == "naive" else d["adaptive"]["by_feature_set"]
            matrix.append([source[s]["test_chronological"]["accuracy"] for s in sets])
        matrix = np.asarray(matrix) * 100
        axis.imshow(matrix, cmap=BLUES, vmin=20, vmax=100, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, percent(matrix[i, j], 0), ha="center", va="center", fontsize=7.5, color=SURFACE if matrix[i, j] > 75 else INK)
        axis.set_xticks(range(len(sets)), [FEATURE_LABELS[s] for s in sets], rotation=30, ha="right")
        axis.set_yticks(range(len(rows)), rows)
        axis.set_title(title, fontsize=9.5)
        axis.grid(visible=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.tight_layout(w_pad=1.5)
    save_figure(figure, path)


def main() -> None:
    config = load_config()
    apply_style()
    figures = config.paths.figures
    threshold = config.accuracy_threshold

    shift = load(config, "shift_category.json")
    if shift:
        plot_shift_distance(shift, figures / "results_shift_distance.png")
        plot_shift_coverage(shift, figures / "results_shift_coverage.png")
    baselines = load(config, "baselines_category.json")
    if baselines:
        plot_baselines(baselines, threshold, figures / "results_baselines.png")
    padding = load(config, "padding_category.json")
    if padding:
        plot_padding_tradeoff(padding, threshold, figures / "results_padding_tradeoff.png")
        plot_padding_heatmap(padding, figures / "results_padding_feature_sets.png")


if __name__ == "__main__":
    main()
