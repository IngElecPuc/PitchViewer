# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def bootstrap_package():
	tools_dir = Path(__file__).resolve().parent
	package_dir = tools_dir.parent
	parent_dir = package_dir.parent
	package_name = package_dir.name
	parent_dir_str = str(parent_dir)
	if parent_dir_str not in sys.path:
		sys.path.insert(0, parent_dir_str)
	return package_name


PACKAGE_NAME = bootstrap_package()

sd = importlib.import_module("sounddevice")
registry = importlib.import_module(f"{PACKAGE_NAME}.detection.registry")
notes = importlib.import_module(f"{PACKAGE_NAME}.music.notes")


BACKEND_IDS = [
	registry.BACKEND_AUTOCORRELATION,
	registry.BACKEND_YIN_CMND,
	registry.BACKEND_TORCHCREPE_TINY,
	registry.BACKEND_TORCHCREPE_FULL,
]


def fmt(value: float, digits: int = 3) -> str:
	if value is None or not math.isfinite(float(value)):
		return "nan"
	return f"{float(value):.{digits}f}"


def percentile(values: list[float], p: float) -> float:
	if not values:
		return float("nan")
	arr = np.asarray(values, dtype=np.float64)
	return float(np.percentile(arr, p))


def list_input_devices() -> None:
	print("\nDispositivos de entrada:")
	devices = sd.query_devices()
	for idx, device in enumerate(devices):
		max_inputs = int(device.get("max_input_channels", 0))
		if max_inputs <= 0:
			continue
		name = device.get("name", "Sin nombre")
		sr = int(float(device.get("default_samplerate", 44100)))
		print(f"  {idx:>3}: {name} | inputs={max_inputs} | default_sr={sr}")


def record_audio(device: int | None, seconds: float, sample_rate: int) -> np.ndarray:
	print(f"\nGrabando {seconds:.1f} s a {sample_rate} Hz. Canta una nota sostenida y luego cambia de nota.")
	print("Inicio en 1 segundo...")
	time.sleep(1.0)
	audio = sd.rec(
		frames=int(round(seconds * sample_rate)),
		samplerate=sample_rate,
		channels=1,
		dtype="float32",
		device=device,
	)
	sd.wait()
	x = np.asarray(audio[:, 0], dtype=np.float32)
	print("Grabación terminada.")
	return x


def analyze_backend(
	backend_id: str,
	audio: np.ndarray,
	sample_rate: int,
	frame_size: int,
	hop_size: int,
	min_hz: float,
	max_hz: float,
	confidence_threshold: float,
	rms_threshold: float,
) -> None:
	label = registry.backend_label(backend_id)
	print("\n" + "=" * 88)
	print(f"Backend: {backend_id} | {label}")
	print("=" * 88)

	try:
		detector = registry.create_pitch_detector(
			backend_id=backend_id,
			sample_rate=sample_rate,
			frame_size=frame_size,
		)
	except Exception as exc:
		print(f"create_pitch_detector: ERROR | {type(exc).__name__}: {exc}")
		return

	freqs: list[float] = []
	confs: list[float] = []
	rmss: list[float] = []
	elapsed_ms_values: list[float] = []
	voiced_count = 0
	valid_freq_count = 0
	slow_count = 0
	frames_count = 0
	last_error = ""

	for start in range(0, max(1, audio.size - frame_size + 1), hop_size):
		frame = audio[start : start + frame_size]
		if frame.size < frame_size:
			break

		rms = float(np.sqrt(np.mean(frame * frame)))
		t0 = time.perf_counter()
		estimate = detector.estimate(frame, min_hz, max_hz)
		elapsed_ms = (time.perf_counter() - t0) * 1000.0

		freq = float(estimate.freq_hz)
		conf = float(estimate.confidence)
		frames_count += 1
		rmss.append(rms)
		elapsed_ms_values.append(elapsed_ms)

		if elapsed_ms > (1000.0 * hop_size / sample_rate):
			slow_count += 1

		if math.isfinite(freq) and min_hz <= freq <= max_hz:
			valid_freq_count += 1
			freqs.append(freq)
			confs.append(conf)

		if min_hz <= freq <= max_hz and conf >= confidence_threshold and rms >= rms_threshold:
			voiced_count += 1

		last_error = getattr(detector, "last_error", "") or last_error

	if frames_count == 0:
		print("No se pudieron formar frames suficientes.")
		return

	frame_budget_ms = 1000.0 * hop_size / sample_rate
	valid_pct = 100.0 * valid_freq_count / frames_count
	voiced_pct = 100.0 * voiced_count / frames_count
	slow_pct = 100.0 * slow_count / frames_count

	print(f"frames                    : {frames_count}")
	print(f"frame_size / hop_size      : {frame_size} / {hop_size}")
	print(f"presupuesto por hop        : {frame_budget_ms:.2f} ms")
	print(f"rango Hz                   : {min_hz:.1f} - {max_hz:.1f}")
	print(f"umbrales conf/RMS          : {confidence_threshold:.3f} / {rms_threshold:.5f}")
	print(f"RMS mean / p50 / p95       : {fmt(statistics.mean(rmss), 5)} / {fmt(percentile(rmss, 50), 5)} / {fmt(percentile(rmss, 95), 5)}")
	print(f"tiempo ms mean / p50 / p95 : {fmt(statistics.mean(elapsed_ms_values), 2)} / {fmt(percentile(elapsed_ms_values, 50), 2)} / {fmt(percentile(elapsed_ms_values, 95), 2)}")
	print(f"frames lentos              : {slow_count}/{frames_count} ({slow_pct:.1f}%)")
	print(f"freq válida                : {valid_freq_count}/{frames_count} ({valid_pct:.1f}%)")
	print(f"voiced según app           : {voiced_count}/{frames_count} ({voiced_pct:.1f}%)")

	if freqs:
		print(f"freq Hz mean / p50 / p95   : {fmt(statistics.mean(freqs), 2)} / {fmt(percentile(freqs, 50), 2)} / {fmt(percentile(freqs, 95), 2)}")
		print(f"conf mean / p50 / p95      : {fmt(statistics.mean(confs), 3)} / {fmt(percentile(confs, 50), 3)} / {fmt(percentile(confs, 95), 3)}")
		print("primeras detecciones       :")
		for freq, conf in list(zip(freqs, confs))[:12]:
			midi = notes.freq_to_midi(freq, 440.0)
			print(f"  {freq:8.2f} Hz | midi={midi:7.2f} | conf={conf:.3f}")

	if last_error:
		print(f"last_error                 : {last_error}")

	if backend_id == registry.BACKEND_TORCHCREPE_FULL and slow_pct > 30.0:
		print("\nConclusión local: torchcrepe_full está corriendo más lento que el flujo de audio.")
		print("Para tiempo real conviene usar torchcrepe_tiny, YIN CMND o autocorrelación; full queda mejor para análisis offline/karaoke producción.")

	if valid_freq_count > 0 and voiced_count == 0:
		print("\nConclusión local: hay frecuencia válida, pero no pasa los umbrales de la app.")
		print("Baja Confianza mínima o RMS mínimo para probar.")



def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=int, default=None, help="Índice de dispositivo de entrada. Si se omite, usa el default.")
	parser.add_argument("--seconds", type=float, default=6.0)
	parser.add_argument("--sample-rate", type=int, default=44100)
	parser.add_argument("--frame-seconds", type=float, default=0.35)
	parser.add_argument("--hop-seconds", type=float, default=0.05)
	parser.add_argument("--min-hz", type=float, default=60.0)
	parser.add_argument("--max-hz", type=float, default=1200.0)
	parser.add_argument("--confidence", type=float, default=0.65)
	parser.add_argument("--rms", type=float, default=0.008)
	parser.add_argument("--backend", choices=BACKEND_IDS + ["all"], default="all")
	parser.add_argument("--list-devices", action="store_true")
	args = parser.parse_args()

	print(f"Package      : {PACKAGE_NAME}")
	print(f"Python       : {sys.executable}")

	if args.list_devices:
		list_input_devices()
		return

	frame_size = max(1024, int(round(args.sample_rate * args.frame_seconds)))
	hop_size = max(128, int(round(args.sample_rate * args.hop_seconds)))

	audio = record_audio(args.device, args.seconds, args.sample_rate)
	print(f"Audio samples: {audio.size}")
	print(f"Audio RMS    : {float(np.sqrt(np.mean(audio * audio))):.5f}")
	print(f"Audio peak   : {float(np.max(np.abs(audio))):.5f}")

	backend_ids = BACKEND_IDS if args.backend == "all" else [args.backend]
	for backend_id in backend_ids:
		analyze_backend(
			backend_id=backend_id,
			audio=audio,
			sample_rate=args.sample_rate,
			frame_size=frame_size,
			hop_size=hop_size,
			min_hz=args.min_hz,
			max_hz=args.max_hz,
			confidence_threshold=args.confidence,
			rms_threshold=args.rms,
		)


if __name__ == "__main__":
	main()
