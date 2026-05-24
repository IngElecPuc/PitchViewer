# -*- coding: utf-8 -*-

"""Escalas musicales y utilidades de visualización de escala."""

from ..config.settings import AppSettings
from .notes import pitch_class_name

SCALE_INTERVALS = {
    "chromatic": list(range(12)),
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}

SCALE_LABELS = {
    "es": {
        "chromatic": "cromática",
        "major": "mayor",
        "natural_minor": "menor natural",
        "harmonic_minor": "menor armónica",
        "melodic_minor": "menor melódica",
        "major_pentatonic": "pentatónica mayor",
        "minor_pentatonic": "pentatónica menor",
        "blues": "blues",
    },
    "en": {
        "chromatic": "chromatic",
        "major": "major",
        "natural_minor": "natural minor",
        "harmonic_minor": "harmonic minor",
        "melodic_minor": "melodic minor",
        "major_pentatonic": "major pentatonic",
        "minor_pentatonic": "minor pentatonic",
        "blues": "blues",
    },
}


def scale_pitch_classes(root: int, scale_name: str) -> set[int]:
    intervals = SCALE_INTERVALS.get(scale_name, SCALE_INTERVALS["chromatic"])
    return {(root + interval) % 12 for interval in intervals}


def scale_display_name(settings: AppSettings) -> str:
    language = settings.note_language
    root = pitch_class_name(settings.scale_root, language)
    label = SCALE_LABELS.get(language, SCALE_LABELS["es"]).get(
        settings.scale_name,
        settings.scale_name,
    )

    if settings.scale_name == "chromatic":
        return label.capitalize()

    return f"{root} {label}"
