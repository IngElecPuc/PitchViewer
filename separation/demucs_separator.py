# -*- coding: utf-8 -*-

"""Separación offline de fuentes para Pitch Viewer.

El nombre del módulo se conserva por compatibilidad con versiones previas,
pero desde v0.9.4 contiene un despachador para más de un motor:

- Demucs: PyTorch, 2 o 4 stems.
- Audio Separator / UVR: modelos MDX/MDXC/RoFormer, normalmente 2 stems.

Ambos se ejecutan como subprocess para mantener aislado el costo de importación,
las descargas de modelos y el consumo de memoria.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .models import SeparationResult, StemAudio
from ..runtime import find_ffmpeg_executable

ProgressCallback = Optional[Callable[[float, str], None]]

ENGINE_DEMUCS = "demucs"
ENGINE_AUDIO_SEPARATOR = "audio_separator"

ENGINE_LABELS = {
	ENGINE_DEMUCS: "Demucs",
	ENGINE_AUDIO_SEPARATOR: "Audio Separator / UVR",
}

DEMUCS_MODELS = ["htdemucs", "htdemucs_ft"]

# Modelos conocidos por audio-separator. La lista se mantiene corta para que la
# UI no se vuelva inmanejable. audio-separator puede listar muchos más con:
# audio-separator --list_models --list_filter=vocals
AUDIO_SEPARATOR_MODELS = [
	"model_bs_roformer_ep_317_sdr_12.9755.ckpt",
	"vocals_mel_band_roformer.ckpt",
	"UVR-MDX-NET-Inst_HQ_3.onnx",
	"UVR_MDXNET_KARA_2.onnx",
]

KNOWN_STEMS = [
	"vocals",
	"instrumental",
	"no_vocals",
	"accompaniment",
	"drums",
	"bass",
	"other",
	"guitar",
	"piano",
]

STEM_DISPLAY_NAMES = {
	"vocals": "Voz",
	"instrumental": "Instrumental",
	"no_vocals": "Instrumental",
	"accompaniment": "Acompañamiento",
	"drums": "Batería",
	"bass": "Bajo",
	"other": "Otros instrumentos",
	"guitar": "Guitarra",
	"piano": "Piano",
}


def is_demucs_available() -> bool:
	return importlib.util.find_spec("demucs") is not None


def is_audio_separator_available() -> bool:
	return importlib.util.find_spec("audio_separator") is not None or shutil.which("audio-separator") is not None


def is_ffmpeg_available() -> bool:
	available, _path, _source = find_ffmpeg_executable()
	return available


def ffmpeg_path() -> str:
	_available, path, _source = find_ffmpeg_executable()
	return path


def default_separation_dir() -> Path:
	base = Path(tempfile.gettempdir()) / "PitchViewer" / "separation"
	base.mkdir(parents=True, exist_ok=True)
	return base


def normalize_engine_id(engine: str) -> str:
	value = str(engine or ENGINE_DEMUCS).strip().lower()
	if value in {"audio", "uvr", "mdx", "mdxc", "roformer", "audio-separator"}:
		return ENGINE_AUDIO_SEPARATOR
	if value not in {ENGINE_DEMUCS, ENGINE_AUDIO_SEPARATOR}:
		return ENGINE_DEMUCS
	return value


def default_model_for_engine(engine: str) -> str:
	engine = normalize_engine_id(engine)
	if engine == ENGINE_AUDIO_SEPARATOR:
		return AUDIO_SEPARATOR_MODELS[0]
	return DEMUCS_MODELS[0]


def models_for_engine(engine: str) -> list[str]:
	engine = normalize_engine_id(engine)
	if engine == ENGINE_AUDIO_SEPARATOR:
		return list(AUDIO_SEPARATOR_MODELS)
	return list(DEMUCS_MODELS)


def run_ai_separation(
	input_path: str,
	output_root: str | Path,
	engine: str = ENGINE_DEMUCS,
	model_name: str = "htdemucs",
	device: str = "cpu",
	mode: str = "4stems",
	progress_callback: ProgressCallback = None,
) -> SeparationResult:
	engine = normalize_engine_id(engine)
	if engine == ENGINE_AUDIO_SEPARATOR:
		model = model_name if model_name else default_model_for_engine(engine)
		return run_audio_separator_separation(
			input_path=input_path,
			output_root=output_root,
			model_name=model,
			device=device,
			progress_callback=progress_callback,
		)
	model = model_name if model_name else default_model_for_engine(engine)
	return run_demucs_separation(
		input_path=input_path,
		output_root=output_root,
		model_name=model,
		device=device,
		mode=mode,
		progress_callback=progress_callback,
	)


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
		progress_callback(0.02, f"Demucs: iniciando separación offline en {device}")

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
			fraction = _parse_percent_progress(message, default=0.15)
			if progress_callback is not None:
				progress_callback(fraction, f"Demucs: {message[:120]}")

	return_code = proc.wait()
	if return_code != 0:
		raise RuntimeError(f"Demucs terminó con código {return_code}. Último mensaje: {last_message}")

	if progress_callback is not None:
		progress_callback(0.96, "Demucs: cargando stems")

	return load_demucs_result(source, out_root, model_name, device)


def run_audio_separator_separation(
	input_path: str,
	output_root: str | Path,
	model_name: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
	device: str = "cpu",
	progress_callback: ProgressCallback = None,
) -> SeparationResult:
	if not is_audio_separator_available():
		raise RuntimeError(
			"Audio Separator no está instalado. Instala las dependencias opcionales con: "
			"pip install -r optional-requirements-separation.txt"
		)

	source = Path(input_path).expanduser().resolve()
	if not source.exists():
		raise FileNotFoundError(str(source))

	out_root = Path(output_root).expanduser().resolve()
	track_out = out_root / "audio_separator" / source.stem
	track_out.mkdir(parents=True, exist_ok=True)

	model_name = str(model_name or default_model_for_engine(ENGINE_AUDIO_SEPARATOR))
	device = "cuda" if str(device).lower().startswith("cuda") else "cpu"

	# El CLI usa ONNX Runtime/PyTorch según el modelo instalado. No todos los
	# flags de device son uniformes entre versiones, por eso dejamos que el
	# paquete elija proveedor; la app solo informa CUDA detectada.
	cmd = [
		"audio-separator",
		str(source),
		"--model_filename",
		model_name,
		"--output_dir",
		str(track_out),
		"--output_format",
		"WAV",
		"--sample_rate",
		"44100",
		"--log_level",
		"info",
	]

	if progress_callback is not None:
		progress_callback(0.02, f"Audio Separator: iniciando modelo {model_name}")

	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		encoding="utf-8",
		errors="replace",
		bufsize=1,
	)

	last_message = "Audio Separator: procesando"
	seen_lines = 0
	if proc.stdout is not None:
		for line in proc.stdout:
			message = line.strip()
			if not message:
				continue
			seen_lines += 1
			last_message = message
			fraction = _parse_percent_progress(message, default=min(0.90, 0.08 + seen_lines * 0.015))
			if progress_callback is not None:
				progress_callback(fraction, f"Audio Separator: {message[:120]}")

	return_code = proc.wait()
	if return_code != 0:
		raise RuntimeError(f"Audio Separator terminó con código {return_code}. Último mensaje: {last_message}")

	if progress_callback is not None:
		progress_callback(0.96, "Audio Separator: cargando stems")

	return load_generic_stem_result(
		source=source,
		output_dir=track_out,
		model_name=f"audio_separator:{model_name}",
		device=device,
	)


def _parse_percent_progress(message: str, default: float) -> float:
	match = re.search(r"(\d{1,3})%", message)
	if match:
		value = max(0, min(100, int(match.group(1))))
		return 0.05 + 0.90 * (value / 100.0)
	return max(0.02, min(0.95, float(default)))


def load_demucs_result(source: Path, output_root: Path, model_name: str, device: str) -> SeparationResult:
	track_dir = _find_track_dir(source, output_root, model_name)
	return load_generic_stem_result(source=source, output_dir=track_dir, model_name=f"demucs:{model_name}", device=device)


def load_generic_stem_result(source: Path, output_dir: Path, model_name: str, device: str) -> SeparationResult:
	stems: dict[str, StemAudio] = {}
	for wav_path in sorted(output_dir.rglob("*.wav")):
		name = normalize_stem_name(wav_path)
		if name not in KNOWN_STEMS:
			continue
		stem = _load_wav_float(wav_path, name)
		# Evita duplicados: preferir archivos en la carpeta raíz de salida si los hay.
		if name not in stems or wav_path.parent == output_dir:
			stems[name] = stem

	if not stems:
		raise RuntimeError(f"No se encontraron stems WAV reconocibles en {output_dir}")

	return SeparationResult(
		source_path=source,
		output_dir=output_dir,
		model_name=model_name,
		device=device,
		stems=stems,
	)


def normalize_stem_name(path: Path) -> str:
	name = path.stem.lower()
	clean = re.sub(r"[^a-z0-9]+", "_", name)
	if "vocals" in clean or "vocal" in clean or "voice" in clean or "voz" in clean:
		return "vocals"
	if "instrumental" in clean:
		return "instrumental"
	if "no_vocals" in clean or "novocals" in clean or "no_vocal" in clean:
		return "no_vocals"
	if "accompaniment" in clean or "acomp" in clean:
		return "accompaniment"
	if "drums" in clean or "drum" in clean:
		return "drums"
	if "bass" in clean or "bajo" in clean:
		return "bass"
	if "other" in clean or "otros" in clean:
		return "other"
	if "guitar" in clean or "guit" in clean:
		return "guitar"
	if "piano" in clean:
		return "piano"
	return clean


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
		if any(normalize_stem_name(p) in KNOWN_STEMS for p in candidate.glob("*.wav")):
			return candidate
	raise RuntimeError(f"No se encontró la carpeta de salida de Demucs en {output_root}")


def _load_wav_float(path: Path, name: str) -> StemAudio:
	try:
		import soundfile as sf  # type: ignore
	except Exception as exc:
		raise RuntimeError("Para cargar stems se requiere soundfile.") from exc

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

	if out.suffix.lower() == ".mp3":
		return _write_mp3_via_ffmpeg(sf, mix, sample_rate, out)

	# WAV/FLAC/OGG dependen de libsndfile. WAV es la ruta principal.
	sf.write(str(out), mix, sample_rate)
	return out


def export_mix_pair_with_gains(
	result: SeparationResult,
	gains: dict[str, float],
	output_base_path: str,
	allow_normalize: bool = True,
) -> tuple[Path, Path]:
	base = Path(output_base_path).expanduser().resolve()
	if base.suffix:
		base = base.with_suffix("")
	wav_path = base.with_suffix(".wav")
	mp3_path = base.with_suffix(".mp3")
	export_mix_with_gains(result, gains, str(wav_path), allow_normalize=allow_normalize)
	export_mix_with_gains(result, gains, str(mp3_path), allow_normalize=allow_normalize)
	return wav_path, mp3_path


def _write_mp3_via_ffmpeg(sf_module, audio: np.ndarray, sample_rate: int, out: Path) -> Path:
	available, ffmpeg, source = find_ffmpeg_executable()
	if not available or not ffmpeg:
		raise RuntimeError(
			"Para exportar MP3 se requiere ffmpeg. Instala imageio-ffmpeg con:\n"
			"pip install imageio-ffmpeg\n\n"
			"O instala ffmpeg globalmente y agrégalo al PATH."
		)

	with tempfile.TemporaryDirectory(prefix="pitchviewer_mp3_") as tmpdir:
		tmp_wav = Path(tmpdir) / "mix.wav"
		sf_module.write(str(tmp_wav), audio, sample_rate)
		cmd = [
			ffmpeg,
			"-hide_banner",
			"-y",
			"-loglevel",
			"error",
			"-i",
			str(tmp_wav),
			"-codec:a",
			"libmp3lame",
			"-b:a",
			"320k",
			str(out),
		]
		proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
		if proc.returncode != 0:
			detail = proc.stderr.decode("utf-8", errors="replace")
			raise RuntimeError(f"ffmpeg no pudo exportar MP3:\n{detail}")
	return out
