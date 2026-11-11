"""Pobiera zbiór walidacyjny CESNET-TLS22 (9,5 GB, CC BY 4.0) i sprawdza sumę MD5."""

from trafficfp.config import load_config
from trafficfp.data.cesnet import download_archive


def main() -> None:
    config = load_config()
    path = download_archive(config.cesnet.raw)
    print("Archiwum gotowe i zweryfikowane:", path)


if __name__ == "__main__":
    main()
