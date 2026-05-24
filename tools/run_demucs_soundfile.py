# -*- coding: utf-8 -*-

"""Runner de Demucs para PitchViewer que evita torchaudio.load/save.

Demucs 4.0.1 usa torchaudio.load para leer audio y demucs.audio.save_audio
termina usando torchaudio.save para escribir stems. En Windows + Python reciente,
torchaudio puede intentar TorchCodec y fallar por compatibilidad/DLLs, incluso
cuando el WAV preparado es válido.

Este runner reemplaza:

- demucs.separate.load_track  -> lectura con soundfile
- demucs.audio.save_audio     -> escritura WAV con soundfile
- demucs.separate.save_audio  -> misma función parcheada, porque separate.py
                                 importa save_audio en su propio namespace

Uso interno:
    python tools/run_demucs_soundfile.py -n htdemucs --device cpu -o out --two-stems vocals input.wav
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _load_track_with_soundfile(track, channels: int, samplerate: int):
	try:
		import soundfile as sf  # type: ignore
		import torch  # type: ignore
		from demucs.audio import convert_audio  # type: ignore
	except Exception as exc:
		raise RuntimeError(
			"El runner de Demucs requiere soundfile, torch y demucs instalados. "
			"Ejecuta: python tools/install_separation_dependencies.py"
		) from exc

	path = Path(track).expanduser().resolve()
	if not path.exists():
		raise FileNotFoundError(str(path))

	try:
		data, sr = sf.read(str(path), dtype="float32", always_2d=True)
	except Exception as exc:
		raise RuntimeError(f"soundfile no pudo leer el WAV preparado: {path}") from exc

	# soundfile devuelve [samples, channels]; Demucs espera [channels, samples].
	wav = torch.from_numpy(data.T).contiguous()
	wav = convert_audio(wav, int(sr), int(samplerate), int(channels))
	return wav


def _save_audio_with_soundfile(
	wav,
	path,
	*,
	samplerate: int = 44100,
	sample_rate: int | None = None,
	encoding: str | None = None,
	bits_per_sample: int | None = None,
	clip: str | None = None,
	rescale: bool | None = None,
	**_: Any,
) -> None:
	"""Guarda stems WAV sin pasar por torchaudio.save/TorchCodec.

	Demucs suele llamar esta función como:
	    save_audio(wav, path, samplerate=..., encoding=..., bits_per_sample=...)

	En algunas versiones puede aparecer sample_rate en vez de samplerate. Aceptamos
	ambos nombres para que el runner sea tolerante.
	"""

	try:
		import numpy as np  # type: ignore
		import soundfile as sf  # type: ignore
		import torch  # type: ignore
	except Exception as exc:
		raise RuntimeError(
			"El guardado de stems requiere numpy, soundfile y torch instalados. "
			"Ejecuta: python tools/install_separation_dependencies.py"
		) from exc

	out_path = Path(path).expanduser().resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)

	if sample_rate is not None:
		samplerate = int(sample_rate)
	else:
		samplerate = int(samplerate)

	if isinstance(wav, torch.Tensor):
		array = wav.detach().cpu().float().numpy()
	else:
		array = np.asarray(wav, dtype=np.float32)

	# Demucs trabaja normalmente como [channels, samples]. soundfile escribe
	# [samples, channels]. Si recibimos mono plano, lo dejamos como [samples].
	if array.ndim == 2:
		if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
			array = array.T
	elif array.ndim == 1:
		pass
	else:
		array = np.squeeze(array)
		if array.ndim == 2 and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
			array = array.T

	array = np.asarray(array, dtype=np.float32)

	max_abs = float(np.max(np.abs(array))) if array.size else 0.0
	clip_mode = (clip or "rescale").lower()

	if max_abs > 1.0:
		if rescale is True or clip_mode == "rescale":
			array = array / max_abs
		else:
			array = np.clip(array, -1.0, 1.0)

	# Demucs normalmente guarda WAV. Para MP3/FLAC se prefiere que PitchViewer
	# exporte después con ffmpeg/imageio-ffmpeg; soundfile depende de libsndfile.
	suffix = out_path.suffix.lower()
	if suffix not in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
		out_path = out_path.with_suffix(".wav")

	subtype = None
	if bits_per_sample == 24:
		subtype = "PCM_24"
	elif bits_per_sample == 32 or (encoding and "float" in str(encoding).lower()):
		subtype = "FLOAT"
	elif suffix == ".wav":
		subtype = "PCM_16"

	sf.write(str(out_path), array, samplerate=samplerate, subtype=subtype)


def main() -> int:
	try:
		import demucs.audio as demucs_audio  # type: ignore
		import demucs.separate as separate  # type: ignore
	except Exception as exc:
		print(f"No se pudo importar Demucs: {type(exc).__name__}: {exc}", file=sys.stderr)
		return 2

	separate.load_track = _load_track_with_soundfile

	# Parchear ambos namespaces. separate.py importa save_audio directamente desde
	# demucs.audio, así que cambiar solo demucs_audio.save_audio no basta.
	demucs_audio.save_audio = _save_audio_with_soundfile
	separate.save_audio = _save_audio_with_soundfile

	# demucs.separate.main lee sys.argv igual que python -m demucs.
	sys.argv = ["demucs", *sys.argv[1:]]
	separate.main()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
