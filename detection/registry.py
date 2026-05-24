# -*- coding: utf-8 -*-

"""Registro y fábrica de backends de detección de pitch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .autocorrelation import estimate_pitch_autocorrelation
from .base import PitchDetector, PitchEstimate
from .yin import estimate_pitch_yin_cmnd

BACKEND_AUTOCORRELATION = "autocorrelation_fft"
BACKEND_YIN_CMND = "yin_cmnd"
BACKEND_TORCHCREPE_TINY = "torchcrepe_tiny"
BACKEND_TORCHCREPE_FULL = "torchcrepe_full"

DEFAULT_BACKEND_ID = BACKEND_AUTOCORRELATION


@dataclass(frozen=True)
class BackendInfo:
    backend_id: str
    label: str
    description: str
    is_optional: bool = False


BACKENDS: tuple[BackendInfo, ...] = (
    BackendInfo(
        BACKEND_AUTOCORRELATION,
        "Autocorrelación FFT",
        "Backend portable con NumPy; rápido y sin dependencias pesadas.",
        False,
    ),
    BackendInfo(
        BACKEND_YIN_CMND,
        "YIN CMND",
        "Backend portable con NumPy; más conservador para voz monofónica sostenida.",
        False,
    ),
    BackendInfo(
        BACKEND_TORCHCREPE_TINY,
        "Torchcrepe tiny",
        "Backend CREPE opcional; requiere torch y torchcrepe.",
        True,
    ),
    BackendInfo(
        BACKEND_TORCHCREPE_FULL,
        "Torchcrepe full",
        "Backend CREPE opcional más pesado; requiere torch y torchcrepe.",
        True,
    ),
)

BACKEND_LABELS = {backend.backend_id: backend.label for backend in BACKENDS}
VALID_BACKEND_IDS = {backend.backend_id for backend in BACKENDS}


def normalize_backend_id(backend_id: str) -> str:
    value = str(backend_id or "").strip()
    if value in VALID_BACKEND_IDS:
        return value
    return DEFAULT_BACKEND_ID


def backend_label(backend_id: str) -> str:
    return BACKEND_LABELS.get(normalize_backend_id(backend_id), BACKEND_LABELS[DEFAULT_BACKEND_ID])


class AutocorrelationDetector(PitchDetector):
    backend_id = BACKEND_AUTOCORRELATION
    label = BACKEND_LABELS[BACKEND_AUTOCORRELATION]

    def estimate(self, frame: np.ndarray, min_hz: float, max_hz: float) -> PitchEstimate:
        freq_hz, confidence = estimate_pitch_autocorrelation(
            frame,
            self.sample_rate,
            min_hz,
            max_hz,
        )
        return PitchEstimate(freq_hz=freq_hz, confidence=confidence)


class YinCmndDetector(PitchDetector):
    backend_id = BACKEND_YIN_CMND
    label = BACKEND_LABELS[BACKEND_YIN_CMND]

    def estimate(self, frame: np.ndarray, min_hz: float, max_hz: float) -> PitchEstimate:
        freq_hz, confidence = estimate_pitch_yin_cmnd(
            frame,
            self.sample_rate,
            min_hz,
            max_hz,
        )
        return PitchEstimate(freq_hz=freq_hz, confidence=confidence)


def create_pitch_detector(backend_id: str, sample_rate: int, frame_size: int) -> PitchDetector:
    normalized = normalize_backend_id(backend_id)

    if normalized == BACKEND_AUTOCORRELATION:
        return AutocorrelationDetector(sample_rate=sample_rate, frame_size=frame_size)

    if normalized == BACKEND_YIN_CMND:
        return YinCmndDetector(sample_rate=sample_rate, frame_size=frame_size)

    if normalized in {BACKEND_TORCHCREPE_TINY, BACKEND_TORCHCREPE_FULL}:
        from .torchcrepe_detector import TorchcrepeDetector

        model = "full" if normalized == BACKEND_TORCHCREPE_FULL else "tiny"
        return TorchcrepeDetector(sample_rate=sample_rate, frame_size=frame_size, model=model)

    return AutocorrelationDetector(sample_rate=sample_rate, frame_size=frame_size)
