# -*- coding: utf-8 -*-

"""Estado de runtime detectado una vez al iniciar la aplicación.

La idea es evitar consultar repetidamente a PyTorch/CUDA/ffmpeg desde cada módulo.
Los componentes que necesitan saber si hay CUDA o ffmpeg disponible deben leer
RUNTIME_INFO o recibir una copia de RuntimeInfo desde la app.
"""

from __future__ import annotations

from dataclasses import dataclass
import shutil
from typing import Optional


@dataclass(frozen=True)
class RuntimeInfo:
	python_executable: str
	torch_available: bool
	torch_version: str
	cuda_available: bool
	cuda_version: str
	cuda_device_name: str
	cuda_device_count: int
	ffmpeg_available: bool
	ffmpeg_path: str
	ffmpeg_source: str


_RUNTIME_INFO: Optional[RuntimeInfo] = None


def find_ffmpeg_executable() -> tuple[bool, str, str]:
	"""Busca ffmpeg primero en PATH y luego vía imageio-ffmpeg.

	Devuelve:
		(available, path, source)
	"""

	ffmpeg = shutil.which("ffmpeg") or ""
	if ffmpeg:
		return True, ffmpeg, "PATH"

	try:
		import imageio_ffmpeg  # type: ignore

		ffmpeg = str(imageio_ffmpeg.get_ffmpeg_exe())
		if ffmpeg:
			return True, ffmpeg, "imageio-ffmpeg"
	except Exception:
		pass

	return False, "", "no disponible"


def detect_runtime_info() -> RuntimeInfo:
	import sys

	torch_available = False
	torch_version = "no instalado"
	cuda_available = False
	cuda_version = ""
	cuda_device_name = ""
	cuda_device_count = 0
	ffmpeg_available, ffmpeg_path, ffmpeg_source = find_ffmpeg_executable()

	try:
		import torch  # type: ignore

		torch_available = True
		torch_version = str(getattr(torch, "__version__", "desconocida"))
		cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
		cuda_available = bool(torch.cuda.is_available())
		if cuda_available:
			cuda_device_count = int(torch.cuda.device_count())
			if cuda_device_count > 0:
				cuda_device_name = str(torch.cuda.get_device_name(0))
	except Exception:
		pass

	return RuntimeInfo(
		python_executable=str(sys.executable),
		torch_available=torch_available,
		torch_version=torch_version,
		cuda_available=cuda_available,
		cuda_version=cuda_version,
		cuda_device_name=cuda_device_name,
		cuda_device_count=cuda_device_count,
		ffmpeg_available=ffmpeg_available,
		ffmpeg_path=ffmpeg_path,
		ffmpeg_source=ffmpeg_source,
	)


def get_runtime_info(refresh: bool = False) -> RuntimeInfo:
	global _RUNTIME_INFO
	if refresh or _RUNTIME_INFO is None:
		_RUNTIME_INFO = detect_runtime_info()
	return _RUNTIME_INFO


RUNTIME_INFO = get_runtime_info()
