# Identyfikacja aplikacji w szyfrowanym kanale HTTPS/TLS (k-NN)

Projekt 4 z przedmiotu *Zaawansowane Techniki Kryptografii i Kryptoanalizy*.
Autorzy: Adrian Sajdak, Jan Wojnicki.

Bierny podsłuchujący widzi rozmiary, kierunki i czasy pakietów, ale nie zna kluczy
sesji. Projekt sprawdza, na ile te metadane wystarczają do rozpoznania rodzaju ruchu
(wideo, audio, www, pliki) za pomocą klasyfikatora k-NN, i czy chroni przed tym padding.

Kryterium sukcesu z instrukcji: **Accuracy > 90%** na zbiorze testowym.

## Najważniejsze wyniki

| Zbiór testowy | Accuracy |
|---|---|
| UC Davis QUIC, test chronologiczny (150 pierwszych pakietów) | 99,46% |
| CESNET-TLS22, TLS po TCP, drugi tydzień (10 pierwszych pakietów) | 92,52% |
| UC Davis QUIC, ruch wygenerowany przez ludzi | 80,72% |
| UC Davis, przeciwnik znający padding, najsłabsza z 9 obron | 96,90% |

Padding obniża trafność tylko przeciwnikowi, który nie zna obrony (do 37,4% przy
dopełnianiu do stałego rozmiaru). Spadek na ruchu ludzi wynika z przesunięcia danych.

## Dokumentacja

| Plik | Zawartość |
|---|---|
| `Sprawozdanie_Adrian_Sajdak__Jan_Wojnicki.docx` | sprawozdanie końcowe |
| `przewodnik.docx` | przewodnik po projekcie prostym językiem |
| `prezentacja.pptx` | prezentacja na obronę z notatkami prelegenta |

## Struktura repozytorium

| Ścieżka | Zawartość |
|---|---|
| `configs/default.yaml` | parametry wszystkich eksperymentów |
| `src/trafficfp/data/` | pobieranie, parser, magazyn przepływów, kontrola jakości, CESNET-TLS22 |
| `src/trafficfp/features/` | cechy sekwencyjne i statystyczne |
| `src/trafficfp/models/` | potok k-NN, grupa kontrolna, las losowy i SVM |
| `src/trafficfp/evaluation/` | podział danych i metryki |
| `src/trafficfp/defenses/` | symulacja obron typu padding |
| `src/trafficfp/experiment.py` | strojenie i ocena k-NN |
| `src/trafficfp/robustness.py` | przeciwnik naiwny i adaptacyjny na ruchu z paddingiem |
| `src/trafficfp/shift.py` | analiza przesunięcia danych |
| `src/trafficfp/comparison.py` | modele referencyjne i istotność grup cech |
| `src/trafficfp/validation.py` | walidacja na ruchu TLS po TCP |
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
python scripts/06_run_padding.py
python scripts/07_analyze_shift.py
python scripts/08_run_baselines.py
python scripts/09_download_cesnet.py
python scripts/10_build_cesnet.py
python scripts/11_run_cesnet.py
python scripts/12_plot_final.py
```

Skrypt `04_run_knn.py` trwa około 5 minut, a `06_run_padding.py` około 30 minut
na 16-wątkowym procesorze. Skrypty 09–11 są potrzebne tylko do walidacji na CESNET-TLS22.

## Dane

Danych nie przechowujemy w repozytorium, odtwarzają je skrypty.

**UC Davis QUIC** (Rezaei, Liu, 2018). Skrypt `01_download_data.py` pobiera zbiór z folderu
Google Drive autorów i sprawdza sumy SHA-256 archiwów. Link z instrukcji projektu
(`github.com/ucdavis/quic-data`) jest nieaktywny, dlatego użyto źródła wskazanego w
repozytorium autorów zbioru: <https://github.com/shrezaei/Semi-supervised-Learning-QUIC->.

**CESNET-TLS22** (Luxemburk, Čejka, 2023, licencja CC BY 4.0). Skrypt `09_download_cesnet.py`
pobiera archiwum (9,5 GB) z <https://zenodo.org/records/10610895> i sprawdza sumę MD5.
Pliki są czytane bezpośrednio z archiwum ZIP, bez rozpakowywania. Skrypt `10_build_cesnet.py`
losuje z każdego dnia po 400 przepływów każdej kategorii.
