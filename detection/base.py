# -*- coding: utf-8 -*-

"""Interfaces comunes para detectores de F0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchEstimate:
    freq_hz: float
    confidence: float


class PitchDetector:
    """Contrato mínimo para cualquier backend de detección de pitch."""

    backend_id: str = "base"
    label: str = "Base"

    def __init__(self, sample_rate: int, frame_size: int) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)

    def reset(self) -> None:
        """Reinicia estado interno si el backend lo necesita."""

    def estimate(self, frame: np.ndarray, min_hz: float, max_hz: float) -> PitchEstimate:
        raise NotImplementedError
