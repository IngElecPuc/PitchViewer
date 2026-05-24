# -*- coding: utf-8 -*-

"""Instalador tolerante para dependencias de separación IA.

Uso:
    python tools/install_separation_dependencies.py

Instala siempre la ruta estable:
    - demucs
    - soundfile
    - imageio-ffmpeg

Luego intenta instalar Audio Separator / UVR. Si falla, no aborta: informa el
error y deja Demucs funcionando. Esto es deliberado porque algunas dependencias
nativas de UVR/Audio Separator pueden fallar en Windows + Python reciente.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass
class StepResult:
    name: str
    ok: bool
    skipped: bool = False
    detail: str = ""


def run_pip(args: Iterable[str]) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", *args]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return proc.returncode == 0, proc.stdout


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def install_core() -> StepResult:
    print("\n== Instalando dependencias estables ==")
    ok, output = run_pip(["install", "demucs", "soundfile", "imageio-ffmpeg"])
    print(output)
    return StepResult("core", ok, detail="demucs + soundfile + imageio-ffmpeg")


def torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def install_audio_separator(force: bool = False) -> StepResult:
    print("\n== Intentando instalar Audio Separator / UVR ==")
    if module_exists("audio_separator"):
        print("audio_separator ya está instalado.")
        return StepResult("audio-separator", True, detail="ya instalado")

    if sys.version_info >= (3, 14) and sys.platform.startswith("win") and not force:
        detail = (
            "omitido por defecto en Windows + Python 3.14; "
            "usa --force-uvr si quieres intentar resolver dependencias nativas"
        )
        print(detail)
        return StepResult("audio-separator", False, skipped=True, detail=detail)

    extra = "gpu" if torch_cuda_available() else "cpu"
    package = f"audio-separator[{extra}]"
    print(f"Instalando {package}...")
    ok, output = run_pip(["install", package])
    print(output)

    if ok:
        return StepResult("audio-separator", True, detail=f"{package} instalado")
    return StepResult(
        "audio-separator",
        False,
        detail=(
            f"falló la instalación de {package}; Demucs sigue disponible. "
            "En Windows/Python reciente puede fallar por dependencias nativas."
        ),
    )


def print_summary(results: list[StepResult]) -> int:
    print("\n== Resumen ==")
    exit_code = 0
    for result in results:
        if result.ok:
            status = "OK"
        elif result.skipped:
            status = "OMITIDO"
        else:
            status = "FALLÓ"
            if result.name == "core":
                exit_code = 1
        print(f"[{status}] {result.name}: {result.detail}")

    if exit_code == 0:
        print("\nRuta estable lista: Demucs + soundfile + exportación MP3 vía imageio-ffmpeg.")
        print("Audio Separator / UVR puede quedar no disponible sin bloquear la app.")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-uvr",
        action="store_true",
        help="intenta instalar audio-separator aunque el entorno sea riesgoso",
    )
    args = parser.parse_args()

    results = [install_core(), install_audio_separator(force=args.force_uvr)]
    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
