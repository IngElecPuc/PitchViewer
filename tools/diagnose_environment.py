# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except Exception:
        return "NO INSTALADO"


def module_available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> int:
    print_section("ENTORNO")
    print(f"Proyecto      : {PROJECT_ROOT}")
    print(f"Python        : {sys.executable}")
    print(f"Versión       : {sys.version}")
    print(f"Sistema       : {platform.platform()}")
    print(f"Arquitectura  : {platform.machine()}")

    print_section("PAQUETES")
    for package in [
        "numpy",
        "sounddevice",
        "cffi",
        "pyinstaller",
        "torch",
        "torchcrepe",
        "torchaudio",
        "demucs",
        "soundfile",
        "imageio-ffmpeg",
        "audio-separator",
    ]:
        print(f"{package:18s}: {version(package)}")

    print_section("RUNTIME PITCHVIEWER")
    try:
        sys.path.insert(0, str(PROJECT_ROOT.parent))
        runtime = importlib.import_module(f"{PROJECT_ROOT.name}.runtime")
        info = runtime.get_runtime_info(refresh=True)
        print(f"torch available     : {info.torch_available}")
        print(f"torch version       : {info.torch_version}")
        print(f"cuda available      : {info.cuda_available}")
        print(f"cuda version        : {info.cuda_version}")
        print(f"cuda device         : {info.cuda_device_name}")
        print(f"ffmpeg available    : {info.ffmpeg_available}")
        print(f"ffmpeg source       : {info.ffmpeg_source}")
        print(f"ffmpeg path         : {info.ffmpeg_path}")
        print(f"torchcodec          : {info.torchcodec_available} ({info.torchcodec_version})")
    except Exception as exc:
        print(f"[ERROR] no se pudo importar runtime.py: {type(exc).__name__}: {exc}")

    print_section("HERRAMIENTAS")
    for executable in ["ffmpeg", "git"]:
        path = shutil.which(executable)
        print(f"{executable:18s}: {path or 'NO EN PATH'}")

    print_section("AUDIO")
    if module_available("sounddevice"):
        try:
            import sounddevice as sd  # type: ignore
            devices = sd.query_devices()
            input_count = sum(1 for item in devices if int(item.get("max_input_channels", 0)) > 0)
            print(f"Dispositivos input : {input_count}")
            for idx, item in enumerate(devices):
                if int(item.get("max_input_channels", 0)) > 0:
                    print(f"  {idx}: {item.get('name')} ({item.get('max_input_channels')} ch)")
        except Exception as exc:
            print(f"[ERROR] sounddevice query_devices: {type(exc).__name__}: {exc}")
    else:
        print("sounddevice no instalado")

    print_section("PYINSTALLER")
    if module_available("PyInstaller"):
        proc = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(proc.stdout.strip())
    else:
        print("PyInstaller no instalado. Ejecuta: python tools/install_project_dependencies.py --build")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
