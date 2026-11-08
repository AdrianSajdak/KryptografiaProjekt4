"""Etap 3: strojenie k-NN, ocena na zbiorach testowych i grupa kontrolna."""

import json
import time

from trafficfp.config import load_config
from trafficfp.data.dataset import FlowStore
from trafficfp.experiment import TARGETS, FeatureCache, evaluate_target


def main() -> None:
    config = load_config()
    store = FlowStore.load(config.paths.processed)
    cache = FeatureCache(store)
    config.paths.tables.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        started = time.perf_counter()
        table, report = evaluate_target(store, cache, target, config)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 1)

        table.to_csv(config.paths.tables / f"cv_{target}.csv", index=False, float_format="%.5f", encoding="utf-8")
        results_path = config.paths.tables / f"results_{target}.json"
        results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(target, report["best_configuration"], report["best_cv"])
        for name, summary in report["evaluations"].items():
            print(f"  {name}: acc={summary['accuracy']:.4f} f1={summary['f1_macro']:.4f}")
        for name, splits in report["controls"].items():
            print(f"  [control] {name}: acc={splits['test_chronological']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
