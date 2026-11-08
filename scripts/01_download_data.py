"""Etap 1: pobranie archiwów UC Davis QUIC i kontrola ich integralności."""

from trafficfp.config import load_config
from trafficfp.data.download import download_dataset


def main() -> None:
    config = load_config()
    for archive in download_dataset(config.paths.raw):
        print(f"OK  {archive.name}  ({archive.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
