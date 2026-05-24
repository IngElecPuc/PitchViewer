# -*- coding: utf-8 -*-

"""Detector F0 portable basado en autocorrelación por FFT."""

import math

import numpy as np

from ..constants import MAX_DETECTABLE_HZ, MIN_DETECTABLE_HZ


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def estimate_pitch_autocorrelation(
    frame: np.ndarray,
    sample_rate: int,
    min_hz: float,
    max_hz: float,
) -> tuple[float, float]:
    """
    Estima F0 con autocorrelación FFT sobre un frame monofónico.

    Retorna:
        (freq_hz, confidence)

    Este detector es portable y no requiere aubio. Es suficiente para validar la app,
    aunque no reemplaza a YIN/pYIN/CREPE en casos difíciles.
    """
    if frame.size < 32:
        return 0.0, 0.0

    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)

    energy = float(np.dot(x, x))
    if energy <= 1e-12:
        return 0.0, 0.0

    x = x * np.hanning(x.size)

    min_hz = max(MIN_DETECTABLE_HZ, float(min_hz))
    max_hz = min(MAX_DETECTABLE_HZ, float(max_hz))

    if max_hz <= min_hz:
        min_hz = MIN_DETECTABLE_HZ
        max_hz = MAX_DETECTABLE_HZ

    min_lag = max(1, int(sample_rate / max_hz))
    max_lag = min(x.size - 2, int(sample_rate / min_hz))

    if max_lag <= min_lag + 2:
        return 0.0, 0.0

    nfft = next_power_of_two(x.size * 2)
    spectrum = np.fft.rfft(x, n=nfft)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[: x.size]

    normalization = np.arange(x.size, 0, -1, dtype=np.float64)
    corr = corr / normalization

    zero_lag = float(corr[0])
    if zero_lag <= 1e-12:
        return 0.0, 0.0

    search = corr[min_lag : max_lag + 1]

    if search.size < 3:
        return 0.0, 0.0

    local_max_mask = (search[1:-1] > search[:-2]) & (search[1:-1] >= search[2:])
    local_max_indices = np.where(local_max_mask)[0] + 1

    if local_max_indices.size == 0:
        peak_relative = int(np.argmax(search))
    else:
        local_values = search[local_max_indices]
        best_value = float(np.max(local_values))
        strong = local_max_indices[local_values >= best_value * 0.85]
        peak_relative = int(strong[0]) if strong.size else int(local_max_indices[np.argmax(local_values)])

    peak_lag = min_lag + peak_relative

    if peak_lag <= 0 or peak_lag >= corr.size - 1:
        return 0.0, 0.0

    y0 = float(corr[peak_lag - 1])
    y1 = float(corr[peak_lag])
    y2 = float(corr[peak_lag + 1])

    denominator = y0 - 2.0 * y1 + y2
    if abs(denominator) > 1e-12:
        delta = 0.5 * (y0 - y2) / denominator
        delta = max(-0.5, min(0.5, delta))
    else:
        delta = 0.0

    refined_lag = float(peak_lag) + delta
    freq_hz = float(sample_rate) / refined_lag
    confidence = max(0.0, min(1.0, y1 / zero_lag))

    if not math.isfinite(freq_hz):
        return 0.0, 0.0

    return freq_hz, confidence
