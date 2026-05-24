# -*- coding: utf-8 -*-

"""Backend opcional basado en torchcrepe.

Este backend usa CREPE mediante torchcrepe, pero mantiene el contrato simple de
la app: recibe un frame mono de NumPy y devuelve frecuencia/confianza.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .base import PitchDetector, PitchEstimate


class TorchcrepeDetector(PitchDetector):
    """Detector CREPE opcional.

    Notas de integración:
    - torchcrepe es caro en CPU, por eso se limita la frecuencia de inferencia.
    - CREPE funciona mejor a 16 kHz; se remuestrea internamente el frame recibido.
    - No se silencian los errores de inicialización. Los errores de runtime se
      guardan en ``last_error`` para diagnóstico.
    """

    def __init__(self, sample_rate: int, frame_size: int, model: str = "tiny") -> None:
        super().__init__(sample_rate=sample_rate, frame_size=frame_size)

        if model not in {"tiny", "full"}:
            model = "tiny"

        self.model = model
        self.target_sample_rate = 16000
        self.min_interval_s = 0.08 if model == "tiny" else 0.16
        self.batch_size = 128 if model == "tiny" else 64

        self._torch = None
        self._torchcrepe = None
        self._device = "cpu"
        self._last_run_time_s: Optional[float] = None
        self._last_estimate = PitchEstimate(0.0, 0.0)
        self.last_error = ""

        self._load_modules()

    def _load_modules(self) -> None:
        try:
            import torch  # type: ignore
            import torchcrepe  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "torchcrepe no está disponible. Instala PyTorch y torchcrepe en "
                "el mismo entorno virtual desde el que ejecutas la app."
            ) from exc

        self._torch = torch
        self._torchcrepe = torchcrepe
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def reset(self) -> None:
        self._last_run_time_s = None
        self._last_estimate = PitchEstimate(0.0, 0.0)
        self.last_error = ""

    def estimate(self, frame: np.ndarray, min_hz: float, max_hz: float) -> PitchEstimate:
        torch = self._torch
        torchcrepe = self._torchcrepe

        if torch is None or torchcrepe is None:
            return PitchEstimate(0.0, 0.0)

        now = time.perf_counter()
        if self._last_run_time_s is not None:
            if now - self._last_run_time_s < self.min_interval_s:
                return self._last_estimate

        x = np.asarray(frame, dtype=np.float32)
        if x.size < 512:
            self._last_estimate = PitchEstimate(0.0, 0.0)
            return self._last_estimate

        x = x - float(np.mean(x))
        peak = float(np.max(np.abs(x)))
        if peak <= 1e-6:
            self._last_estimate = PitchEstimate(0.0, 0.0)
            return self._last_estimate

        x = x / max(peak, 1e-6)
        x = self._resample_to_target(x)

        if x.size < 1024:
            x = np.pad(x, (0, 1024 - x.size), mode="constant")

        audio = torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0).to(self._device)
        hop_length = max(1, int(self.target_sample_rate / 100.0))

        safe_min_hz = max(40.0, float(min_hz))
        safe_max_hz = min(2000.0, float(max_hz))

        if safe_max_hz <= safe_min_hz:
            self._last_estimate = PitchEstimate(0.0, 0.0)
            return self._last_estimate

        try:
            with torch.no_grad():
                pitch, periodicity = torchcrepe.predict(
                    audio,
                    self.target_sample_rate,
                    hop_length,
                    safe_min_hz,
                    safe_max_hz,
                    self.model,
                    batch_size=self.batch_size,
                    device=self._device,
                    return_periodicity=True,
                )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._last_run_time_s = now
            self._last_estimate = PitchEstimate(0.0, 0.0)
            return self._last_estimate

        try:
            pitch_np = pitch.detach().cpu().numpy().reshape(-1)
            periodicity_np = periodicity.detach().cpu().numpy().reshape(-1)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._last_run_time_s = now
            self._last_estimate = PitchEstimate(0.0, 0.0)
            return self._last_estimate

        if pitch_np.size == 0 or periodicity_np.size == 0:
            self._last_run_time_s = now
            self._last_estimate = PitchEstimate(0.0, 0.0)
            return self._last_estimate

        valid = (
            np.isfinite(pitch_np)
            & np.isfinite(periodicity_np)
            & (pitch_np >= safe_min_hz)
            & (pitch_np <= safe_max_hz)
            & (periodicity_np > 0.0)
        )

        if not np.any(valid):
            self._last_run_time_s = now
            self._last_estimate = PitchEstimate(0.0, 0.0)
            return self._last_estimate

        valid_indices = np.where(valid)[0]
        best_local_idx = int(np.argmax(periodicity_np[valid_indices]))
        best_idx = int(valid_indices[best_local_idx])

        freq_hz = float(pitch_np[best_idx])
        confidence = float(periodicity_np[best_idx])

        if not math.isfinite(freq_hz):
            self._last_run_time_s = now
            self._last_estimate = PitchEstimate(0.0, 0.0)
            return self._last_estimate

        confidence = max(0.0, min(1.0, confidence))
        self.last_error = ""
        self._last_run_time_s = now
        self._last_estimate = PitchEstimate(freq_hz=freq_hz, confidence=confidence)
        return self._last_estimate

    def _resample_to_target(self, x: np.ndarray) -> np.ndarray:
        if self.sample_rate == self.target_sample_rate:
            return np.asarray(x, dtype=np.float32)

        if x.size <= 1:
            return np.asarray(x, dtype=np.float32)

        duration_s = float(x.size) / float(self.sample_rate)
        target_size = max(1, int(round(duration_s * self.target_sample_rate)))

        if target_size == x.size:
            return np.asarray(x, dtype=np.float32)

        src_positions = np.arange(x.size, dtype=np.float64)
        dst_positions = np.linspace(0.0, float(x.size - 1), num=target_size, dtype=np.float64)
        y = np.interp(dst_positions, src_positions, x.astype(np.float64, copy=False))
        return y.astype(np.float32, copy=False)
