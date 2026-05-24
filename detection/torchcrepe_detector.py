# -*- coding: utf-8 -*-

"""Backend opcional basado en torchcrepe."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .base import PitchDetector, PitchEstimate


class TorchcrepeDetector(PitchDetector):
    """Detector CREPE opcional.

    Este backend solo se activa si el usuario instaló torch y torchcrepe.
    No se incluye en requirements.txt porque es pesado y puede no estar disponible
    para todas las versiones de Python en Windows.
    """

    def __init__(self, sample_rate: int, frame_size: int, model: str = "tiny") -> None:
        super().__init__(sample_rate=sample_rate, frame_size=frame_size)
        self.model = model
        self._torch = None
        self._torchcrepe = None
        self._device = "cpu"
        self._load_modules()

    def _load_modules(self) -> None:
        try:
            import torch  # type: ignore
            import torchcrepe  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "torchcrepe no está disponible. Instala las dependencias opcionales "
                "solo si tu versión de Python lo soporta."
            ) from exc

        self._torch = torch
        self._torchcrepe = torchcrepe
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def estimate(self, frame: np.ndarray, min_hz: float, max_hz: float) -> PitchEstimate:
        torch = self._torch
        torchcrepe = self._torchcrepe

        if torch is None or torchcrepe is None:
            return PitchEstimate(0.0, 0.0)

        x = np.asarray(frame, dtype=np.float32)
        if x.size < 512:
            return PitchEstimate(0.0, 0.0)

        x = x - float(np.mean(x))
        peak = float(np.max(np.abs(x)))
        if peak <= 1e-6:
            return PitchEstimate(0.0, 0.0)

        # CREPE espera una amplitud razonable. Normalizar frame por frame no es
        # perfecto, pero es suficiente para un backend opcional de tiempo real.
        x = x / max(peak, 1e-6)

        audio = torch.tensor(x, dtype=torch.float32, device=self._device).unsqueeze(0)
        hop_length = max(1, int(self.sample_rate / 100.0))  # 10 ms nominales

        try:
            with torch.no_grad():
                pitch, periodicity = torchcrepe.predict(
                    audio,
                    self.sample_rate,
                    hop_length,
                    float(min_hz),
                    float(max_hz),
                    model=self.model,
                    batch_size=2048,
                    device=self._device,
                    return_periodicity=True,
                )
        except Exception:
            return PitchEstimate(0.0, 0.0)

        try:
            pitch_np = pitch.detach().cpu().numpy().reshape(-1)
            periodicity_np = periodicity.detach().cpu().numpy().reshape(-1)
        except Exception:
            return PitchEstimate(0.0, 0.0)

        if pitch_np.size == 0 or periodicity_np.size == 0:
            return PitchEstimate(0.0, 0.0)

        valid = np.isfinite(pitch_np) & np.isfinite(periodicity_np) & (pitch_np > 0.0)
        if not np.any(valid):
            return PitchEstimate(0.0, 0.0)

        valid_indices = np.where(valid)[0]
        idx = int(valid_indices[-1])
        freq_hz = float(pitch_np[idx])
        confidence = float(periodicity_np[idx])

        if not math.isfinite(freq_hz):
            return PitchEstimate(0.0, 0.0)

        return PitchEstimate(freq_hz, max(0.0, min(1.0, confidence)))
