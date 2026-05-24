# -*- coding: utf-8 -*-

"""Conversión de frames de pitch a segmentos musicales para karaoke."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np

from ..config.settings import AppSettings, settings_to_dict
from ..models import PitchPoint
from ..music.notes import cents_from_nearest_note, midi_to_note_name
from .models import KaraokeNoteSegment, KaraokePitchFrame


def pitch_points_to_frames(points: Iterable[PitchPoint]) -> list[KaraokePitchFrame]:
    frames: list[KaraokePitchFrame] = []
    for point in points:
        frames.append(
            KaraokePitchFrame(
                time_s=float(point.time_s),
                freq_hz=float(point.freq_hz),
                midi_float=float(point.midi_float),
                raw_midi_float=float(point.raw_midi_float),
                confidence=float(point.confidence),
                rms=float(point.rms),
                voiced=bool(point.voiced),
            )
        )
    return frames


def build_note_segments(
    points: Iterable[PitchPoint],
    settings: AppSettings,
    min_duration_s: float = 0.080,
    max_gap_s: float = 0.140,
) -> list[KaraokeNoteSegment]:
    """Agrupa puntos dentro de tolerancia en bloques de nota.

    La segmentación usa la misma idea visual de la app: solo se crea segmento
    cuando el pitch cae dentro de ±tolerancia de una nota. Las zonas entre notas
    no generan segmento.
    """
    segments: list[KaraokeNoteSegment] = []

    active_midi: Optional[int] = None
    start_s: Optional[float] = None
    last_s: Optional[float] = None
    cents_values: list[float] = []
    conf_values: list[float] = []
    rms_values: list[float] = []

    def matched_midi(point: PitchPoint) -> Optional[int]:
        if not point.voiced or math.isnan(point.midi_float):
            return None
        nearest = int(round(point.midi_float))
        cents = abs(cents_from_nearest_note(point.midi_float))
        if cents > settings.tolerance_cents:
            return None
        # Producción karaoke: se guarda la nota alcanzada cromática. La escala
        # queda en settings_snapshot y podrá usarse para filtrar más adelante.
        return nearest

    def flush() -> None:
        nonlocal active_midi, start_s, last_s, cents_values, conf_values, rms_values
        if active_midi is None or start_s is None or last_s is None:
            return
        duration = float(last_s) - float(start_s)
        if duration >= min_duration_s and cents_values:
            segments.append(
                KaraokeNoteSegment(
                    start_s=float(start_s),
                    end_s=float(last_s),
                    midi=int(active_midi),
                    note=midi_to_note_name(int(active_midi), settings.note_language),
                    mean_cents=float(np.mean(np.asarray(cents_values, dtype=np.float64))),
                    confidence=float(np.mean(np.asarray(conf_values, dtype=np.float64))) if conf_values else 0.0,
                    rms=float(np.mean(np.asarray(rms_values, dtype=np.float64))) if rms_values else 0.0,
                    frame_count=int(len(cents_values)),
                )
            )
        active_midi = None
        start_s = None
        last_s = None
        cents_values = []
        conf_values = []
        rms_values = []

    for point in points:
        midi = matched_midi(point)
        if midi is None:
            flush()
            continue

        if active_midi is None:
            active_midi = midi
            start_s = float(point.time_s)
            last_s = float(point.time_s)
        elif midi != active_midi or (last_s is not None and float(point.time_s) - last_s > max_gap_s):
            flush()
            active_midi = midi
            start_s = float(point.time_s)
            last_s = float(point.time_s)
        else:
            last_s = float(point.time_s)

        cents_values.append(float(cents_from_nearest_note(point.midi_float)))
        conf_values.append(float(point.confidence))
        rms_values.append(float(point.rms))

    flush()
    return segments


def settings_snapshot(settings: AppSettings) -> dict:
    return settings_to_dict(settings)
