"""Walidacja k-NN na ruchu TLS po TCP (CESNET-TLS22): uczenie na tygodniu 40, test na 41."""

import json
import time

from trafficfp.config import load_config
from trafficfp.data.dataset import FlowStore
from trafficfp.validation import evaluate_cesnet


def main() -> None:
    config = load_config()
    store = FlowStore.load(config.cesnet.processed)
    started = time.perf_counter()
    table, report = evaluate_cesnet(store, config)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)

    tables = config.paths.tables
    table.to_csv(tables / "cv_cesnet.csv", index=False, float_format="%.5f", encoding="utf-8")
    (tables / "results_cesnet.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = report["evaluations"]["test_week"]
    print("najlepsza konfiguracja:", report["best_configuration"], report["best_cv"])
    print(f"test (tydzień 41): accuracy {summary['accuracy']:.4f}, makro-F1 {summary['f1_macro']:.4f}")
    for name, splits in report["controls"].items():
        print(f"  [kontrola] {name}: {splits['test_week']['accuracy']:.4f}")
    for name, item in report["feature_set_comparison"].items():
        print(f"  [zestaw] {name}: {item['test_accuracy']:.4f} {item['configuration']}")
    print("czas [s]:", report["elapsed_seconds"])


if __name__ == "__main__":
    main()
