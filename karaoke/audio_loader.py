# -*- coding: utf-8 -*-

"""Carga básica de audio para modo karaoke producción."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LoadedAudio:
    path: str
    filename: str
    sample_rate: int
    channels: int
    selected_channel: str
    duration_s: float
    audio: np.ndarray


def load_audio_file(path: str, selected_channel: str = "mix") -> LoadedAudio:
    """Carga un archivo y devuelve mono float32.

    Soporte principal: WAV. Si soundfile está disponible, se intenta primero
    para WAV/FLAC/OGG y otros formatos soportados por libsndfile. Para MP3/MP4
    se intenta ffmpeg si está instalado en PATH.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    data: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    # 1) soundfile, si está disponible.
    try:
        import soundfile as sf  # type: ignore

        sf_data, sf_sr = sf.read(str(source), dtype="float32", always_2d=True)
        data = np.asarray(sf_data, dtype=np.float32)
        sample_rate = int(sf_sr)
        channels = int(data.shape[1])
    except Exception:
        data = None
        sample_rate = None
        channels = None

    # 2) WAV estándar con stdlib.
    if data is None and source.suffix.lower() == ".wav":
        data, sample_rate, channels = _load_wav_stdlib(source)

    # 3) ffmpeg como fallback para mp3/mp4/m4a y otros.
    if data is None:
        data, sample_rate, channels = _load_with_ffmpeg(source)

    if data is None or sample_rate is None or channels is None:
        raise RuntimeError(
            "No se pudo cargar el audio. Usa WAV o instala soundfile/ffmpeg para más formatos."
        )

    mono = select_channel(data, selected_channel)
    mono = np.asarray(mono, dtype=np.float32)
    if mono.size:
        peak = float(np.max(np.abs(mono)))
        if peak > 1.0:
            mono = mono / peak

    duration_s = float(mono.size) / float(max(1, sample_rate))
    return LoadedAudio(
        path=str(source),
        filename=source.name,
        sample_rate=int(sample_rate),
        channels=int(channels),
        selected_channel=str(selected_channel),
        duration_s=duration_s,
        audio=mono,
    )


def select_channel(data: np.ndarray, selected_channel: str) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        x = np.reshape(x, (x.shape[0], -1)).astype(np.float32, copy=False)

    channels = x.shape[1]
    mode = str(selected_channel or "mix").strip().lower()

    if channels <= 1:
        return x[:, 0]
    if mode in {"left", "izquierdo", "l"}:
        return x[:, 0]
    if mode in {"right", "derecho", "r"}:
        return x[:, min(1, channels - 1)]
    if mode in {"max_rms", "mayor_rms", "rms"}:
        rms = np.sqrt(np.mean(x * x, axis=0))
        return x[:, int(np.argmax(rms))]
    return np.mean(x, axis=1).astype(np.float32)


def _load_wav_stdlib(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_rate = int(wf.getframerate())
        sample_width = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        raw = wf.readframes(nframes)

    if sample_width == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"WAV con sample width no soportado: {sample_width}")

    if channels > 1:
        arr = arr.reshape(-1, channels)
    else:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float32, copy=False), sample_rate, channels


def _load_with_ffmpeg(path: Path) -> tuple[np.ndarray, int, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg no está instalado o no está en PATH.")

    target_sr = 44100
    # Se convierte a mono float32. Si se requiere elegir canal real para mp3/mp4,
    # conviene exportar WAV multicanal o usar ffmpeg externamente para separar pistas.
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg no pudo convertir el audio:\n{detail}")

    arr = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 1)
    return arr.astype(np.float32, copy=False), target_sr, 1
