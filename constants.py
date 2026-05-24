# -*- coding: utf-8 -*-

"""Constantes generales de Pitch Viewer."""

# Rangos vocales aproximados extendidos ± media octava para visualización.
# No son límites anatómicos estrictos; sirven como presets de pantalla.
VOCAL_RANGE_PRESETS = [
    ("Bajo ±½ oct.: Si1 - La♯4", 35, 70),
    ("Barítono ±½ oct.: Re♯2 - Re♯5", 39, 75),
    ("Tenor ±½ oct.: Fa♯2 - Fa♯5", 42, 78),
    ("Contralto ±½ oct.: Si2 - Si5", 47, 83),
    ("Mezzo ±½ oct.: Re♯3 - Re♯6", 51, 87),
    ("Soprano ±½ oct.: Fa♯3 - Fa♯6", 54, 90),
]

RANGE_PRESETS = [
    ("Voz grave: Mi2 - Mi4", 40, 64),
    ("Voz media: La2 - La4", 45, 69),
    ("Voz aguda: Do3 - Do6", 48, 84),
    ("Amplio: Mi2 - Do6", 40, 84),
    ("Muy amplio: Do2 - Do7", 36, 96),
]

TIME_WINDOWS = [5, 10, 20, 30]
TOLERANCE_OPTIONS = [10, 20, 30, 40, 45]
A4_OPTIONS = [440.0, 441.0, 442.0]

DEFAULT_MIN_MIDI = 40
DEFAULT_MAX_MIDI = 84
MIN_MIDI_CHOICE = 24
MAX_MIDI_CHOICE = 108

MIN_DETECTABLE_HZ = 50.0
MAX_DETECTABLE_HZ = 2000.0
