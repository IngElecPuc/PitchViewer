# -*- coding: utf-8 -*-

"""Modelos de datos compartidos por la app."""

import math
from dataclasses import dataclass


def _cents_from_nearest_note(midi_float: float) -> float:
    return 100.0 * (midi_float - round(midi_float))


@dataclass
class PitchPoint:
    time_s: float
    freq_hz: float
    midi_float: float
    raw_midi_float: float
    confidence: float
    rms: float
    voiced: bool

    @property
    def cents(self) -> float:
        if not self.voiced or math.isnan(self.midi_float):
            return float("nan")
        return _cents_from_nearest_note(self.midi_float)


@dataclass
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: int

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name} ({self.channels} ch, {self.default_samplerate} Hz)"
