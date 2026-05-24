# -*- coding: utf-8 -*-

"""Integración opcional con Demucs.

Se ejecuta como subprocess para mantener aislado el costo de importación y para
facilitar el uso con CPU/CUDA sin bloquear el thread principal de Tkinter.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .models import SeparationResult, StemAudio

ProgressCallback = Optional[Callable[[float, str], None]]

KNOWN_STEMS = ["vocals", "drums", "bass", "other", "no_vocals"]
STEM_DISPLAY_NAMES = {
	"vocals": "Voz",
	"drums": "Batería",
	"bass": "Bajo",
	"other": "Otros instrumentos",
	"no_vocals": "Instrumental",
}


def is_demucs_available() -> bool:
	return importlib.util.find_spec("demucs") is not None


def default_separation_dir() -> Path:
	base = Path(tempfile.gettempdir()) / "PitchViewer" / "separation"
	base.mkdir(parents=True, exist_ok=True)
	return base


def run_demucs_separation(
	input_path: str,
	output_root: str | Path,
	model_name: str = "htdemucs",
	device: str = "cpu",
	mode: str = "4stems",
	progress_callback: ProgressCallback = None,
) -> SeparationResult:
	if not is_demucs_available():
		raise RuntimeError(
			"Demucs no está instalado. Instala las dependencias opcionales con: "
			"pip install -r optional-requirements-separation.txt"
		)

	source = Path(input_path).expanduser().resolve()
	if not source.exists():
		raise FileNotFoundError(str(source))

	out_root = Path(output_root).expanduser().resolve()
	out_root.mkdir(parents=True, exist_ok=True)

	device = "cuda" if str(device).lower().startswith("cuda") else "cpu"
	mode = str(mode or "4stems").lower()
	model_name = str(model_name or "htdemucs")

	cmd = [
		sys.executable,
		"-m",
		"demucs",
		"-n",
		model_name,
		"--device",
		device,
		"-o",
		str(out_root),
	]
	if mode in {"2stems", "vocals", "voice"}:
		cmd.extend(["--two-stems", "vocals"])
	cmd.append(str(source))

	if progress_callback is not None:
		progress_callback(0.02, f"Demucs: iniciando separación en {device}")

	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		encoding="utf-8",
		errors="replace",
		bufsize=1,
	)

	last_message = "Demucs: procesando"
	if proc.stdout is not None:
		for line in proc.stdout:
			message = line.strip()
			if not message:
				continue
			last_message = message
			fraction = _parse_demucs_progress(message)
			if progress_callback is not None:
				progress_callback(fraction, f"Demucs: {message[:120]}")

	return_code = proc.wait()
	if return_code != 0:
		raise RuntimeError(f"Demucs terminó con código {return_code}. Último mensaje: {last_message}")

	if progress_callback is not None:
		progress_callback(0.96, "Demucs: cargando stems")

	return load_demucs_result(source, out_root, model_name, device)


def _parse_demucs_progress(message: str) -> float:
	# Demucs/tqdm puede emitir porcentajes. Si no hay porcentaje, se devuelve
	# una fracción conservadora para mantener la barra en movimiento.
	match = re.search(r"(\d{1,3})%", message)
	if match:
		value = max(0, min(100, int(match.group(1))))
		return 0.05 + 0.90 * (value / 100.0)
	return 0.15


def load_demucs_result(source: Path, output_root: Path, model_name: str, device: str) -> SeparationResult:
	track_dir = _find_track_dir(source, output_root, model_name)
	stems: dict[str, StemAudio] = {}
	for wav_path in sorted(track_dir.glob("*.wav")):
		name = wav_path.stem.lower()
		if name not in KNOWN_STEMS:
			continue
		stem = _load_wav_float(wav_path, name)
		stems[name] = stem

	if not stems:
		raise RuntimeError(f"No se encontraron stems WAV en {track_dir}")

	return SeparationResult(
		source_path=source,
		output_dir=track_dir,
		model_name=model_name,
		device=device,
		stems=stems,
	)


def _find_track_dir(source: Path, output_root: Path, model_name: str) -> Path:
	model_dir = output_root / model_name
	candidates = []
	if model_dir.exists():
		candidates.extend([p for p in model_dir.iterdir() if p.is_dir()])
	candidates.extend([p for p in output_root.rglob("*") if p.is_dir()])

	preferred_name = source.stem.lower()
	for candidate in candidates:
		if candidate.name.lower() == preferred_name and any(candidate.glob("*.wav")):
			return candidate
	for candidate in candidates:
		if (candidate / "vocals.wav").exists() or (candidate / "no_vocals.wav").exists():
			return candidate
	raise RuntimeError(f"No se encontró la carpeta de salida de Demucs en {output_root}")


def _load_wav_float(path: Path, name: str) -> StemAudio:
	try:
		import soundfile as sf  # type: ignore
	except Exception as exc:
		raise RuntimeError("Para cargar stems de Demucs se requiere soundfile.") from exc

	data, sr = sf.read(str(path), dtype="float32", always_2d=True)
	return StemAudio(name=name, path=path, sample_rate=int(sr), audio=np.asarray(data, dtype=np.float32))


def export_mix_with_gains(
	result: SeparationResult,
	gains: dict[str, float],
	output_path: str,
	allow_normalize: bool = True,
) -> Path:
	if not result.stems:
		raise RuntimeError("No hay stems cargados para mezclar.")

	try:
		import soundfile as sf  # type: ignore
	except Exception as exc:
		raise RuntimeError("Para exportar mezcla se requiere soundfile.") from exc

	stems = list(result.stems.values())
	sample_rate = stems[0].sample_rate
	channels = max(stem.channels for stem in stems)
	length = max(stem.audio.shape[0] for stem in stems)
	mix = np.zeros((length, channels), dtype=np.float32)

	for stem in stems:
		if stem.sample_rate != sample_rate:
			raise RuntimeError("Todos los stems deben tener el mismo sample rate para mezclar.")
		audio = np.asarray(stem.audio, dtype=np.float32)
		if audio.ndim == 1:
			audio = audio[:, None]
		if audio.shape[1] == 1 and channels > 1:
			audio = np.repeat(audio, channels, axis=1)
		elif audio.shape[1] != channels:
			audio = audio[:, :channels]

		gain = float(gains.get(stem.name, 1.0))
		mix[: audio.shape[0], : audio.shape[1]] += audio * gain

	peak = float(np.max(np.abs(mix))) if mix.size else 0.0
	if allow_normalize and peak > 1.0:
		mix = mix / peak

	out = Path(output_path).expanduser().resolve()
	out.parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(out), mix, sample_rate)
	return out
