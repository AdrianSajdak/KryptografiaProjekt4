"""Porównanie k-NN z lasem losowym i SVM oraz istotność grup cech (4 kategorie)."""

import json
import time

from trafficfp.comparison import compare_models
from trafficfp.config import load_config
from trafficfp.data.dataset import FlowStore

TARGET = "category"


def main() -> None:
    config = load_config()
    store = FlowStore.load(config.paths.processed)
    clean = json.loads((config.paths.tables / f"results_{TARGET}.json").read_text(encoding="utf-8"))

    started = time.perf_counter()
    result = compare_models(store, clean, TARGET, config)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    path = config.paths.tables / f"baselines_{TARGET}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, model in result["models"].items():
        tests = ", ".join(f"{split}: {model[split]['accuracy']:.4f}" for split in ("test_chronological", "test_script_retraining", "test_human_retraining"))
        print(f"{name}: CV {model['cv_f1_macro']:.4f}, {tests}, uczenie {model['fit_seconds']:.2f} s, test {model['predict_seconds']:.2f} s")
    for name, item in result["group_importance"].items():
        print(name, {group: round(value["mean"], 4) for group, value in item["groups"].items()})
    print("czas [s]:", result["elapsed_seconds"])


if __name__ == "__main__":
    main()
