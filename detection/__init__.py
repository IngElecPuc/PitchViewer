# -*- coding: utf-8 -*-

"""Backends de detección de pitch."""

from .base import PitchDetector, PitchEstimate
from .registry import (
    BACKENDS,
    BACKEND_AUTOCORRELATION,
    BACKEND_TORCHCREPE_FULL,
    BACKEND_TORCHCREPE_TINY,
    BACKEND_YIN_CMND,
    DEFAULT_BACKEND_ID,
    backend_label,
    create_pitch_detector,
    normalize_backend_id,
)

__all__ = [
    "BACKENDS",
    "BACKEND_AUTOCORRELATION",
    "BACKEND_TORCHCREPE_FULL",
    "BACKEND_TORCHCREPE_TINY",
    "BACKEND_YIN_CMND",
    "DEFAULT_BACKEND_ID",
    "PitchDetector",
    "PitchEstimate",
    "backend_label",
    "create_pitch_detector",
    "normalize_backend_id",
]
