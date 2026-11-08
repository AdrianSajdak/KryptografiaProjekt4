"""Mapowanie aplikacji ze zbioru UC Davis na kategorie ruchu z tematu projektu."""

from __future__ import annotations

#: Google Doc to interaktywna aplikacja webowa (krótkie żądania HTTP),
#: dlatego należy do kategorii www, a nie do transferu plików.
APP_TO_CATEGORY = {
    "Youtube": "wideo",
    "Google Music": "audio",
    "Google Search": "www",
    "Google Doc": "www",
    "Google Drive": "pliki",
}

APPS = tuple(sorted(APP_TO_CATEGORY))
CATEGORIES = ("wideo", "audio", "www", "pliki")


def category_of(app: str) -> str:
    """Zwraca kategorię ruchu dla nazwy aplikacji."""
    try:
        return APP_TO_CATEGORY[app]
    except KeyError as error:
        raise ValueError(f"Nieznana aplikacja: {app!r}") from error
