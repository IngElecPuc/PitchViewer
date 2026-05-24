# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


def has_module(name: str) -> bool:
	return importlib.util.find_spec(name) is not None


def ffmpeg_info() -> tuple[bool, str, str, str]:
	path = shutil.which("ffmpeg") or ""
	if path:
		return True, path, "PATH", version_for(path)
	try:
		import imageio_ffmpeg  # type: ignore

		path = str(imageio_ffmpeg.get_ffmpeg_exe())
		return True, path, "imageio-ffmpeg", version_for(path)
	except Exception as exc:
		return False, "", "no disponible", str(exc)


def version_for(executable: str) -> str:
	try:
		proc = subprocess.run(
			[executable, "-version"],
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			encoding="utf-8",
			errors="replace",
			check=False,
			timeout=8,
		)
		line = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
		return line.strip()
	except Exception as exc:
		return f"no se pudo consultar versión: {exc}"


def main() -> None:
	print("PitchViewer - diagnóstico de separación IA")
	print(f"Python: {sys.executable}")
	print(f"Versión: {sys.version.split()[0]}")
	print()

	ff_ok, ff_path, ff_source, ff_version = ffmpeg_info()
	if ff_ok:
		print(f"[OK] ffmpeg | {ff_source}")
		print(f"    path: {ff_path}")
		print(f"    {ff_version}")
	else:
		print("[NO] ffmpeg | instala imageio-ffmpeg o ffmpeg global para MP3/M4A/MP4")
		print(f"    detalle: {ff_version}")

	if has_module("demucs"):
		print("[OK] demucs | motor recomendado")
		try:
			proc = subprocess.run(
				[sys.executable, "-m", "demucs", "--help"],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				encoding="utf-8",
				errors="replace",
				check=False,
				timeout=8,
			)
			first = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
			print(f"    {first}")
		except Exception as exc:
			print(f"    no se pudo ejecutar help: {exc}")
	else:
		print("[NO] demucs | ejecuta: python tools/install_separation_dependencies.py")

	if has_module("audio_separator") or shutil.which("audio-separator"):
		print("[OK] audio-separator / UVR | motor experimental disponible")
		cli = shutil.which("audio-separator")
		if cli:
			print(f"    cli: {cli}")
		try:
			proc = subprocess.run(
				["audio-separator", "--help"],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				encoding="utf-8",
				errors="replace",
				check=False,
				timeout=8,
			)
			first = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
			print(f"    {first}")
		except Exception as exc:
			print(f"    no se pudo ejecutar CLI: {exc}")
	else:
		print("[NO] audio-separator / UVR | experimental; Demucs no depende de esto")
		print("    Para intentarlo: python tools/install_separation_dependencies.py --force-uvr")

	print()
	if has_module("demucs") and ff_ok:
		print("Resultado: Demucs + exportación MP3 deberían estar disponibles.")
	elif has_module("demucs"):
		print("Resultado: Demucs está instalado, pero falta ffmpeg para MP3/M4A/MP4 robusto.")
	else:
		print("Resultado: faltan dependencias estables de separación.")


if __name__ == "__main__":
	main()
