"""Etap 4: testy odporności k-NN na padding dla czterech kategorii ruchu."""

import json
import time

from trafficfp.config import load_config
from trafficfp.data.dataset import FlowStore
from trafficfp.defenses.padding import defenses_from_config
from trafficfp.robustness import evaluate_padding, flatten

TARGET = "category"


def main() -> None:
    config = load_config()
    store = FlowStore.load(config.paths.processed)
    clean = json.loads((config.paths.tables / f"results_{TARGET}.json").read_text(encoding="utf-8"))

    started = time.perf_counter()
    result = evaluate_padding(store, defenses_from_config(config.padding), clean, TARGET, config, progress=print)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 1)

    tables = config.paths.tables
    (tables / f"padding_{TARGET}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    flatten(result).to_csv(tables / f"padding_{TARGET}.csv", index=False, float_format="%.5f", encoding="utf-8")

    main_set = result["main_feature_set"]
    for defense in result["defenses"]:
        naive = defense["naive"][main_set]["test_chronological"]["accuracy"]
        adaptive = defense["adaptive"]["by_feature_set"][main_set]["test_chronological"]["accuracy"]
        print(f"{defense['label']:>16}: narzut {defense['overhead']:.1%}, naiwny {naive:.4f}, adaptacyjny {adaptive:.4f}")
    print("czas [s]:", result["elapsed_seconds"])


if __name__ == "__main__":
    main()
