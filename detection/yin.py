# -*- coding: utf-8 -*-

"""Detector F0 tipo YIN/CMND implementado solo con NumPy."""

from __future__ import annotations

import math

import numpy as np

from ..constants import MAX_DETECTABLE_HZ, MIN_DETECTABLE_HZ
from .autocorrelation import next_power_of_two


DEFAULT_YIN_THRESHOLD = 0.15


def estimate_pitch_yin_cmnd(
    frame: np.ndarray,
    sample_rate: int,
    min_hz: float,
    max_hz: float,
    threshold: float = DEFAULT_YIN_THRESHOLD,
) -> tuple[float, float]:
    """
    Estima F0 con una variante de YIN basada en CMND.

    Retorna:
        (freq_hz, confidence)

    Es un backend portable. Suele ser más conservador que la autocorrelación
    simple y puede producir menos saltos en voz sostenida, aunque sigue siendo
    inferior a implementaciones especializadas como pYIN/CREPE.
    """
    if frame.size < 64:
        return 0.0, 0.0

    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)

    energy = float(np.dot(x, x))
    if energy <= 1e-12:
        return 0.0, 0.0

    x = x * np.hanning(x.size)
    n = x.size

    min_hz = max(MIN_DETECTABLE_HZ, float(min_hz))
    max_hz = min(MAX_DETECTABLE_HZ, float(max_hz))

    if max_hz <= min_hz:
        min_hz = MIN_DETECTABLE_HZ
        max_hz = MAX_DETECTABLE_HZ

    min_lag = max(2, int(sample_rate / max_hz))
    max_lag = min(n - 2, int(sample_rate / min_hz))

    if max_lag <= min_lag + 2:
        return 0.0, 0.0

    nfft = next_power_of_two(n * 2)
    spectrum = np.fft.rfft(x, n=nfft)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:n]

    sq = x * x
    prefix = np.concatenate([[0.0], np.cumsum(sq)])
    taus = np.arange(0, max_lag + 1)

    # d(tau) = sum_j (x[j] - x[j + tau])^2, j = 0..N-tau-1
    first_energy = prefix[n - taus] - prefix[0]
    second_energy = prefix[n] - prefix[taus]
    difference = first_energy + second_energy - 2.0 * corr[: max_lag + 1]
    difference[0] = 0.0
    difference = np.maximum(difference, 0.0)

    cumulative = np.cumsum(difference)
    cmnd = np.ones_like(difference)
    valid = taus >= 1
    cmnd[valid] = difference[valid] * taus[valid] / np.maximum(cumulative[valid], 1e-12)
    cmnd[0] = 1.0

    search = cmnd[min_lag : max_lag + 1]
    if search.size < 3:
        return 0.0, 0.0

    threshold = float(np.clip(threshold, 0.03, 0.6))
    below = np.where(search < threshold)[0]

    if below.size:
        candidate = int(below[0] + min_lag)
        while candidate + 1 <= max_lag and cmnd[candidate + 1] < cmnd[candidate]:
            candidate += 1
    else:
        candidate = int(np.argmin(search) + min_lag)

    if candidate <= 0 or candidate >= cmnd.size - 1:
        return 0.0, 0.0

    y0 = float(cmnd[candidate - 1])
    y1 = float(cmnd[candidate])
    y2 = float(cmnd[candidate + 1])

    denominator = y0 - 2.0 * y1 + y2
    if abs(denominator) > 1e-12:
        delta = 0.5 * (y0 - y2) / denominator
        delta = max(-0.5, min(0.5, delta))
    else:
        delta = 0.0

    refined_lag = float(candidate) + delta
    if refined_lag <= 0.0:
        return 0.0, 0.0

    freq_hz = float(sample_rate) / refined_lag
    confidence = max(0.0, min(1.0, 1.0 - y1))

    if not math.isfinite(freq_hz):
        return 0.0, 0.0

    return freq_hz, confidence
