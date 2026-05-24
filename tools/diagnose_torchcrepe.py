# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import importlib.metadata
import math
import platform
import sys
import traceback


def print_section(title: str) -> None:
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)


def package_version(package_name: str) -> str:
	try:
		return importlib.metadata.version(package_name)
	except importlib.metadata.PackageNotFoundError:
		return "NO INSTALADO"


def try_import(module_name: str):
	try:
		module = importlib.import_module(module_name)
		print(f"[OK] import {module_name}")
		return module
	except Exception as exc:
		print(f"[ERROR] import {module_name}")
		print(f"{type(exc).__name__}: {exc}")
		traceback.print_exc()
		return None


def main() -> None:
	print_section("ENTORNO PYTHON")

	print(f"Python executable : {sys.executable}")
	print(f"Python version    : {sys.version}")
	print(f"Platform          : {platform.platform()}")
	print(f"Machine           : {platform.machine()}")
	print(f"Architecture      : {platform.architecture()}")

	print_section("PAQUETES INSTALADOS")

	for package_name in [
		"numpy",
		"torch",
		"torchaudio",
		"torchvision",
		"torchcrepe",
		"librosa",
		"scipy",
		"soundfile",
	]:
		print(f"{package_name:12s}: {package_version(package_name)}")

	print_section("IMPORTS")

	np = try_import("numpy")
	torch = try_import("torch")
	torchcrepe = try_import("torchcrepe")

	if torch is None or torchcrepe is None or np is None:
		print("\nDiagnóstico detenido: falta importar torch, torchcrepe o numpy.")
		return

	print_section("TORCH DEVICE")

	print(f"torch.__version__       : {torch.__version__}")
	print(f"torch.version.cuda      : {torch.version.cuda}")
	print(f"torch.cuda.is_available : {torch.cuda.is_available()}")

	if torch.cuda.is_available():
		print(f"cuda device count       : {torch.cuda.device_count()}")

		for idx in range(torch.cuda.device_count()):
			print(f"cuda:{idx} name          : {torch.cuda.get_device_name(idx)}")

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	print(f"device usado en prueba  : {device}")

	print_section("PRUEBA MÍNIMA DE TENSOR")

	try:
		x = torch.tensor([1.0, 2.0, 3.0], device=device)
		print(f"[OK] tensor en {device}: {x}")
	except Exception as exc:
		print(f"[ERROR] no se pudo crear tensor en {device}")
		print(f"{type(exc).__name__}: {exc}")
		traceback.print_exc()
		return

	print_section("PRUEBA MÍNIMA TORCHCREPE")

	sample_rate = 16000
	duration_s = 1.0
	frequency_hz = 220.0
	hop_length = int(sample_rate / 200.0)
	fmin = 50.0
	fmax = 550.0
	batch_size = 128

	try:
		t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
		audio_np = 0.15 * np.sin(2.0 * math.pi * frequency_hz * t).astype(np.float32)

		audio = torch.from_numpy(audio_np).unsqueeze(0).to(device)

		print(f"audio shape       : {tuple(audio.shape)}")
		print(f"sample_rate       : {sample_rate}")
		print(f"hop_length        : {hop_length}")
		print(f"fmin/fmax         : {fmin} / {fmax}")
		print(f"model             : tiny")
		print(f"batch_size        : {batch_size}")

		result = torchcrepe.predict(
			audio,
			sample_rate,
			hop_length,
			fmin,
			fmax,
			"tiny",
			batch_size=batch_size,
			device=device,
			return_periodicity=True,
		)

		if isinstance(result, tuple):
			pitch, periodicity = result
		else:
			pitch = result
			periodicity = None

		pitch_cpu = pitch.detach().cpu().numpy()

		valid = pitch_cpu[pitch_cpu > 0]

		print(f"[OK] torchcrepe.predict ejecutado")
		print(f"pitch shape       : {tuple(pitch.shape)}")

		if valid.size:
			print(f"pitch mean válido : {float(valid.mean()):.2f} Hz")
			print(f"pitch min/max     : {float(valid.min()):.2f} / {float(valid.max()):.2f} Hz")
		else:
			print("pitch válido      : ninguno")

		if periodicity is not None:
			periodicity_cpu = periodicity.detach().cpu().numpy()
			print(f"periodicity shape : {tuple(periodicity.shape)}")
			print(f"periodicity mean  : {float(periodicity_cpu.mean()):.3f}")

	except Exception as exc:
		print("[ERROR] torchcrepe.predict falló")
		print(f"{type(exc).__name__}: {exc}")
		traceback.print_exc()
		return

	print_section("RESULTADO")

	print("Torchcrepe funciona fuera de la app.")
	print("Si la app no lo muestra, el problema está en detection/registry.py o detection/torchcrepe_detector.py.")


if __name__ == "__main__":
	main()