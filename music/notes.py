# -*- coding: utf-8 -*-

"""Conversión entre frecuencia, MIDI y nombres de notas."""

import math

from ..constants import MAX_MIDI_CHOICE, MIN_MIDI_CHOICE

NOTE_NAMES = {
    "es": [
        "Do",
        "Do♯",
        "Re",
        "Mi♭",
        "Mi",
        "Fa",
        "Fa♯",
        "Sol",
        "Sol♯",
        "La",
        "Si♭",
        "Si",
    ],
    "en": [
        "C",
        "C♯",
        "D",
        "E♭",
        "E",
        "F",
        "F♯",
        "G",
        "G♯",
        "A",
        "B♭",
        "B",
    ],
}

LANGUAGE_LABELS = {
    "es": "Español",
    "en": "English",
}


def freq_to_midi(freq_hz: float, a4_hz: float = 440.0) -> float:
    return 69.0 + 12.0 * math.log2(freq_hz / a4_hz)


def midi_to_freq(midi_value: float, a4_hz: float = 440.0) -> float:
    return a4_hz * (2.0 ** ((midi_value - 69.0) / 12.0))


def midi_to_octave(midi_value: int) -> int:
    return (midi_value // 12) - 1


def midi_to_note_name(midi_value: int, language: str = "es") -> str:
    names = NOTE_NAMES.get(language, NOTE_NAMES["es"])
    return f"{names[midi_value % 12]}{midi_to_octave(midi_value)}"


def pitch_class_name(pitch_class: int, language: str = "es") -> str:
    names = NOTE_NAMES.get(language, NOTE_NAMES["es"])
    return names[pitch_class % 12]


def cents_from_nearest_note(midi_float: float) -> float:
    return 100.0 * (midi_float - round(midi_float))


def build_note_choices(language: str = "es") -> list[str]:
    return [
        midi_to_note_name(midi, language)
        for midi in range(MIN_MIDI_CHOICE, MAX_MIDI_CHOICE + 1)
    ]


def build_note_to_midi(language: str = "es") -> dict[str, int]:
    return {
        midi_to_note_name(midi, language): midi
        for midi in range(MIN_MIDI_CHOICE, MAX_MIDI_CHOICE + 1)
    }
