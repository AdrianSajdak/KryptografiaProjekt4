"""Etap 1: scalenie archiwów w magazyn przepływów i raport jakości danych."""

import json

from trafficfp.config import load_config
from trafficfp.data.dataset import build_flow_store
from trafficfp.data.download import verify_archives
from trafficfp.data.validate import quality_report


def main() -> None:
    config = load_config()
    store = build_flow_store(verify_archives(config.paths.raw))
    store.save(config.paths.processed)

    report = quality_report(store.meta)
    config.paths.tables.mkdir(parents=True, exist_ok=True)
    target = config.paths.tables / "data_quality.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
