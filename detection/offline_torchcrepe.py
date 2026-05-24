# -*- coding: utf-8 -*-

"""Análisis offline de pitch con Torchcrepe.

Este módulo no se usa para el flujo vivo. Está pensado para grabaciones o,
más adelante, para karaoke producción. Ejecuta CREPE sobre audio completo o por
bloques, y devuelve una secuencia temporal de frecuencia/confianza.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class OfflinePitchFrame:
    time_s: float
    freq_hz: float
    confidence: float
    rms: float


@dataclass(frozen=True)
class OfflineTorchcrepeInfo:
    model: str
    device: str
    input_sample_rate: int
    target_sample_rate: int
    hop_s: float
    frame_count: int
    duration_s: float


def torchcrepe_available() -> tuple[bool, str]:
    try:
        import torch  # type: ignore  # noqa: F401
        import torchcrepe  # type: ignore  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def analyze_audio_with_torchcrepe(
    audio: np.ndarray,
    sample_rate: int,
    min_hz: float,
    max_hz: float,
    model: str = "full",
    device: Optional[str] = None,
    target_sample_rate: int = 16000,
    hop_s: float = 0.05,
    batch_size: Optional[int] = None,
    chunk_duration_s: Optional[float] = 20.0,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[list[OfflinePitchFrame], OfflineTorchcrepeInfo]:
    """Analiza audio mono completo con torchcrepe.

    Args:
        audio: Señal mono float32/float64 en rango aproximado [-1, 1].
        sample_rate: Frecuencia de muestreo original.
        min_hz: Frecuencia mínima aceptada por CREPE.
        max_hz: Frecuencia máxima aceptada por CREPE.
        model: "tiny" o "full".
        device: "cpu", "cuda:0" o None para autodetectar.
        target_sample_rate: sample rate interno de CREPE.
        hop_s: salto temporal entre estimaciones.
        batch_size: batch size Torchcrepe. Si es None se elige por modelo.
        chunk_duration_s: duración de bloque para reportar progreso en audios largos.
        progress_callback: callback opcional con fracción 0..1 y mensaje.
    """
    if model not in {"tiny", "full"}:
        model = "full"

    import torch  # type: ignore

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if batch_size is None:
        batch_size = 256 if model == "tiny" else 64

    x_original = np.asarray(audio, dtype=np.float32)
    if x_original.ndim != 1:
        x_original = np.reshape(x_original, (-1,)).astype(np.float32, copy=False)

    duration_s = float(x_original.size) / float(max(1, sample_rate)) if x_original.size else 0.0

    if x_original.size == 0:
        info = OfflineTorchcrepeInfo(
            model=model,
            device=device,
            input_sample_rate=int(sample_rate),
            target_sample_rate=int(target_sample_rate),
            hop_s=float(hop_s),
            frame_count=0,
            duration_s=0.0,
        )
        if progress_callback is not None:
            progress_callback(1.0, "Audio vacío")
        return [], info

    if progress_callback is not None:
        progress_callback(0.0, f"Inicializando Torchcrepe {model} en {device}")

    if chunk_duration_s is None or chunk_duration_s <= 0.0 or duration_s <= max(1.0, chunk_duration_s * 1.1):
        frames, info = _analyze_audio_with_torchcrepe_single(
            audio=x_original,
            sample_rate=sample_rate,
            min_hz=min_hz,
            max_hz=max_hz,
            model=model,
            device=device,
            target_sample_rate=target_sample_rate,
            hop_s=hop_s,
            batch_size=batch_size,
            time_offset_s=0.0,
        )
        if progress_callback is not None:
            progress_callback(1.0, "Torchcrepe completado")
        return frames, info

    chunk_samples = max(int(round(float(chunk_duration_s) * float(sample_rate))), int(sample_rate))
    total_chunks = int(math.ceil(float(x_original.size) / float(chunk_samples)))
    all_frames: list[OfflinePitchFrame] = []
    actual_hop_s = float(hop_s)

    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_samples
        end = min(x_original.size, start + chunk_samples)
        chunk = x_original[start:end]
        offset_s = float(start) / float(max(1, sample_rate))

        if progress_callback is not None:
            progress_callback(
                float(chunk_index) / float(max(1, total_chunks)),
                f"Torchcrepe {model}: bloque {chunk_index + 1}/{total_chunks}",
            )

        frames, info = _analyze_audio_with_torchcrepe_single(
            audio=chunk,
            sample_rate=sample_rate,
            min_hz=min_hz,
            max_hz=max_hz,
            model=model,
            device=device,
            target_sample_rate=target_sample_rate,
            hop_s=hop_s,
            batch_size=batch_size,
            time_offset_s=offset_s,
        )
        actual_hop_s = float(info.hop_s)
        all_frames.extend(frames)

        if progress_callback is not None:
            progress_callback(
                float(chunk_index + 1) / float(max(1, total_chunks)),
                f"Torchcrepe {model}: bloque {chunk_index + 1}/{total_chunks} listo",
            )

    final_info = OfflineTorchcrepeInfo(
        model=model,
        device=device,
        input_sample_rate=int(sample_rate),
        target_sample_rate=int(target_sample_rate),
        hop_s=actual_hop_s,
        frame_count=len(all_frames),
        duration_s=duration_s,
    )
    return all_frames, final_info


def _analyze_audio_with_torchcrepe_single(
    audio: np.ndarray,
    sample_rate: int,
    min_hz: float,
    max_hz: float,
    model: str,
    device: str,
    target_sample_rate: int,
    hop_s: float,
    batch_size: int,
    time_offset_s: float = 0.0,
) -> tuple[list[OfflinePitchFrame], OfflineTorchcrepeInfo]:
    import torch  # type: ignore
    import torchcrepe  # type: ignore

    x_original = np.asarray(audio, dtype=np.float32)
    if x_original.ndim != 1:
        x_original = np.reshape(x_original, (-1,)).astype(np.float32, copy=False)

    if x_original.size == 0:
        info = OfflineTorchcrepeInfo(
            model=model,
            device=device,
            input_sample_rate=int(sample_rate),
            target_sample_rate=int(target_sample_rate),
            hop_s=float(hop_s),
            frame_count=0,
            duration_s=0.0,
        )
        return [], info

    x = x_original - float(np.mean(x_original))
    peak = float(np.max(np.abs(x)))
    if peak > 1e-6:
        x = x / peak

    x_resampled = _resample_linear(x, int(sample_rate), int(target_sample_rate))
    if x_resampled.size < target_sample_rate // 4:
        pad = max(1, target_sample_rate // 4 - x_resampled.size)
        x_resampled = np.pad(x_resampled, (0, pad), mode="constant")

    safe_min_hz = max(40.0, float(min_hz))
    safe_max_hz = min(2000.0, float(max_hz))
    if safe_max_hz <= safe_min_hz:
        safe_min_hz = 50.0
        safe_max_hz = 1200.0

    hop_length = max(1, int(round(float(hop_s) * float(target_sample_rate))))
    tensor = torch.from_numpy(x_resampled.astype(np.float32, copy=False)).unsqueeze(0).to(device)

    with torch.no_grad():
        pitch, periodicity = torchcrepe.predict(
            tensor,
            int(target_sample_rate),
            int(hop_length),
            safe_min_hz,
            safe_max_hz,
            model,
            batch_size=int(batch_size),
            device=device,
            return_periodicity=True,
        )

    pitch_np = pitch.detach().cpu().numpy().reshape(-1)
    periodicity_np = periodicity.detach().cpu().numpy().reshape(-1)

    local_duration_s = float(x_original.size) / float(max(1, sample_rate))
    frames: list[OfflinePitchFrame] = []

    for idx, freq in enumerate(pitch_np):
        local_time_s = min(local_duration_s, float(idx * hop_length) / float(target_sample_rate))
        time_s = float(time_offset_s) + local_time_s
        conf = float(periodicity_np[idx]) if idx < periodicity_np.size else 0.0
        if not math.isfinite(float(freq)):
            freq = 0.0
        if not math.isfinite(conf):
            conf = 0.0
        if float(freq) < safe_min_hz or float(freq) > safe_max_hz:
            freq = 0.0
        rms = _local_rms(x_original, sample_rate, local_time_s, window_s=0.05)
        frames.append(
            OfflinePitchFrame(
                time_s=float(time_s),
                freq_hz=float(freq),
                confidence=max(0.0, min(1.0, conf)),
                rms=float(rms),
            )
        )

    info = OfflineTorchcrepeInfo(
        model=model,
        device=device,
        input_sample_rate=int(sample_rate),
        target_sample_rate=int(target_sample_rate),
        hop_s=float(hop_length) / float(target_sample_rate),
        frame_count=len(frames),
        duration_s=local_duration_s,
    )
    return frames, info


def _resample_linear(x: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sample_rate == target_sample_rate or x.size <= 1:
        return x.astype(np.float32, copy=False)

    duration_s = float(x.size) / float(max(1, sample_rate))
    target_size = max(1, int(round(duration_s * float(target_sample_rate))))
    if target_size == x.size:
        return x.astype(np.float32, copy=False)

    src_positions = np.arange(x.size, dtype=np.float64)
    dst_positions = np.linspace(0.0, float(x.size - 1), num=target_size, dtype=np.float64)
    y = np.interp(dst_positions, src_positions, x.astype(np.float64, copy=False))
    return y.astype(np.float32, copy=False)


def _local_rms(audio: np.ndarray, sample_rate: int, center_s: float, window_s: float) -> float:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return 0.0

    half = max(1, int(round(window_s * sample_rate / 2.0)))
    center = int(round(center_s * sample_rate))
    start = max(0, center - half)
    end = min(x.size, center + half)
    if end <= start:
        return 0.0
    chunk = x[start:end]
    return float(np.sqrt(np.mean(chunk * chunk)))
