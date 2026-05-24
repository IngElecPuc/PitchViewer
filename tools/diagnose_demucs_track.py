# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def bootstrap_package() -> str:
	tools_dir = Path(__file__).resolve().parent
	package_dir = tools_dir.parent
	parent_dir = package_dir.parent
	package_name = package_dir.name
	parent_dir_str = str(parent_dir)
	if parent_dir_str not in sys.path:
		sys.path.insert(0, parent_dir_str)
	return package_name


class Reporter:
	def __init__(self, report_path: Path) -> None:
		self.report_path = report_path
		self.report_path.parent.mkdir(parents=True, exist_ok=True)
		self._fh = self.report_path.open("w", encoding="utf-8")

	def close(self) -> None:
		self._fh.close()

	def write(self, text: str = "") -> None:
		print(text)
		self._fh.write(text + "\n")
		self._fh.flush()

	def section(self, title: str) -> None:
		self.write()
		self.write("=" * 90)
		self.write(title)
		self.write("=" * 90)


def run_command(cmd: list[str], reporter: Reporter, timeout: int | None = None, env: dict[str, str] | None = None) -> int:
	reporter.write("$ " + " ".join(cmd))
	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		encoding="utf-8",
		errors="replace",
		env=env,
	)
	assert proc.stdout is not None
	try:
		for line in proc.stdout:
			reporter.write(line.rstrip())
		return_code = proc.wait(timeout=timeout)
	except subprocess.TimeoutExpired:
		proc.kill()
		reporter.write(f"[ERROR] Timeout después de {timeout}s")
		return_code = -999
	reporter.write(f"[returncode] {return_code}")
	return int(return_code)


def main() -> int:
	parser = argparse.ArgumentParser(description="Diagnóstico reproducible de Demucs sobre una pista concreta.")
	parser.add_argument("input", help="Ruta del audio a separar, por ejemplo Amigo.mp3")
	parser.add_argument("--model", default="htdemucs", help="Modelo Demucs. Default: htdemucs")
	parser.add_argument("--mode", default="2stems", choices=["2stems", "4stems"], help="Modo de stems")
	parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device solicitado")
	parser.add_argument("--clip-seconds", type=float, default=0.0, help="Si es >0, prueba solo un clip de N segundos")
	parser.add_argument("--output", default="", help="Carpeta de salida para prueba Demucs")
	args = parser.parse_args()

	package_name = bootstrap_package()
	runtime = importlib.import_module(f"{package_name}.runtime")
	separator = importlib.import_module(f"{package_name}.separation.demucs_separator")

	input_path = Path(args.input).expanduser().resolve()
	timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
	report_path = Path.cwd() / "diagnostics" / f"demucs_track_{timestamp}.txt"
	reporter = Reporter(report_path)

	try:
		reporter.section("ENTORNO")
		reporter.write(f"Python executable : {sys.executable}")
		reporter.write(f"Python version    : {sys.version}")
		reporter.write(f"Package           : {package_name}")
		reporter.write(f"Input             : {input_path}")
		reporter.write(f"Input exists      : {input_path.exists()}")
		if input_path.exists():
			reporter.write(f"Input size        : {input_path.stat().st_size} bytes")

		info = runtime.get_runtime_info(refresh=True)
		reporter.write(f"Torch available   : {info.torch_available}")
		reporter.write(f"Torch version     : {info.torch_version}")
		reporter.write(f"CUDA available    : {info.cuda_available}")
		reporter.write(f"CUDA version      : {info.cuda_version}")
		reporter.write(f"CUDA device       : {info.cuda_device_name}")
		reporter.write(f"ffmpeg available  : {info.ffmpeg_available}")
		reporter.write(f"ffmpeg source     : {info.ffmpeg_source}")
		reporter.write(f"ffmpeg path       : {info.ffmpeg_path}")
		reporter.write(f"demucs module     : {separator.is_demucs_available()}")
		reporter.write(f"soundfile module  : {importlib.util.find_spec('soundfile') is not None}")
		reporter.write(f"demucs runner     : {separator._demucs_soundfile_runner_path()}")
		reporter.write(f"torchcodec module : {importlib.util.find_spec('torchcodec') is not None} (no requerido)")

		if not input_path.exists():
			reporter.write("[ERROR] El archivo de entrada no existe.")
			return 2

		available, ffmpeg, _source = runtime.find_ffmpeg_executable()
		if not available or not ffmpeg:
			reporter.write("[ERROR] No hay ffmpeg disponible.")
			return 3

		env = separator._build_subprocess_env()

		reporter.section("FFMPEG -VERSIÓN")
		run_command([ffmpeg, "-version"], reporter, timeout=10, env=env)

		reporter.section("FFMPEG -PROBE INPUT")
		run_command([ffmpeg, "-hide_banner", "-i", str(input_path)], reporter, timeout=20, env=env)

		reporter.section("FFMPEG -CONVERSIÓN DE PRUEBA A WAV")
		with tempfile.TemporaryDirectory(prefix="pitchviewer_demucs_diag_") as tmpdir:
			clip_source = input_path
			if args.clip_seconds and args.clip_seconds > 0:
				clip_source = Path(tmpdir) / f"{input_path.stem}_clip.wav"
				clip_cmd = [
					ffmpeg,
					"-hide_banner",
					"-y",
					"-loglevel",
					"error",
					"-i",
					str(input_path),
					"-t",
					str(float(args.clip_seconds)),
					"-vn",
					"-ac",
					"2",
					"-ar",
					"44100",
					"-c:a",
					"pcm_s16le",
					str(clip_source),
				]
				code = run_command(clip_cmd, reporter, timeout=120, env=env)
				if code != 0:
					reporter.write("[ERROR] No se pudo generar el clip WAV de diagnóstico.")
					return 4

			prepared = separator._prepare_input_for_demucs(clip_source, Path(tmpdir), None)
			reporter.write(f"Prepared WAV: {prepared}")
			reporter.write(f"Prepared exists: {prepared.exists()}")
			reporter.write(f"Prepared size: {prepared.stat().st_size if prepared.exists() else 0} bytes")

			reporter.section("DEMucs RUNNER -HELP")
			run_command([sys.executable, str(separator._demucs_soundfile_runner_path()), "--help"], reporter, timeout=30, env=env)

			reporter.section("DEMucs -SEPARACIÓN")
			if args.device == "auto":
				device = "cuda" if info.cuda_available else "cpu"
			else:
				device = args.device
			output_root = Path(args.output).expanduser().resolve() if args.output else Path.cwd() / "diagnostics" / f"demucs_output_{timestamp}"

			try:
				result = separator.run_demucs_separation(
					input_path=str(prepared),
					output_root=output_root,
					model_name=args.model,
					device=device,
					mode=args.mode,
					progress_callback=lambda fraction, message: reporter.write(f"[progress {fraction:.3f}] {message}"),
				)
				reporter.write("[OK] Demucs terminó correctamente.")
				reporter.write(f"Output dir: {result.output_dir}")
				reporter.write(f"Stems: {', '.join(sorted(result.stems))}")
			except Exception as exc:
				reporter.write("[ERROR] Falló la separación Demucs.")
				reporter.write(f"{type(exc).__name__}: {exc}")
				reporter.write(traceback.format_exc())
				return 5

		reporter.section("RESULTADO")
		reporter.write("Diagnóstico completo: OK.")
		return 0
	finally:
		reporter.write()
		reporter.write(f"Reporte guardado en: {report_path}")
		reporter.close()


if __name__ == "__main__":
	raise SystemExit(main())
