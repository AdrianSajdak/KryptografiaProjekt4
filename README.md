# Identyfikacja aplikacji w szyfrowanym kanale HTTPS/TLS (k-NN)

Projekt 4 z przedmiotu *Zaawansowane Techniki Kryptografii i Kryptoanalizy*.
Autorzy: Adrian Sajdak, Jan Wojnicki.

Bierny podsłuchujący widzi rozmiary, kierunki i czasy pakietów, ale nie zna kluczy
sesji. Projekt sprawdza, na ile te metadane wystarczają do rozpoznania rodzaju ruchu
(wideo, audio, www, pliki) za pomocą klasyfikatora k-NN.

Kryterium sukcesu z instrukcji: **Accuracy > 90%** na zbiorze testowym.
Pełny opis znajduje się w pliku `Sprawozdanie_Adrian_Sajdak__Jan_Wojnicki.docx`.

## Struktura repozytorium

| Ścieżka | Zawartość |
|---|---|
| `configs/default.yaml` | parametry eksperymentów |
| `src/trafficfp/data/` | pobieranie, parser, magazyn przepływów, kontrola jakości |
| `src/trafficfp/features/` | cechy sekwencyjne i statystyczne |
| `src/trafficfp/models/` | potok k-NN i grupa kontrolna |
| `src/trafficfp/evaluation/` | podział danych i metryki |
| `scripts/` | kolejne etapy potoku uruchamiane z wiersza poleceń |
| `tests/` | testy jednostkowe (`pytest`) |
| `reports/` | wygenerowane tabele i wykresy |

## Uruchomienie

Wymagany Python 3.11 lub nowszy. Polecenia dla Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest
python scripts/01_download_data.py
python scripts/02_build_dataset.py
python scripts/03_explore_data.py
python scripts/04_run_knn.py
python scripts/05_plot_results.py
```

## Dane

Zbiór UC Davis QUIC (Rezaei, Liu, 2018) nie jest przechowywany w repozytorium.
Skrypt `01_download_data.py` pobiera go z folderu Google Drive autorów i sprawdza
sumy SHA-256 archiwów. Link z instrukcji projektu (`github.com/ucdavis/quic-data`)
jest nieaktywny, dlatego użyto źródła wskazanego w repozytorium autorów zbioru:
<https://github.com/shrezaei/Semi-supervised-Learning-QUIC->.
