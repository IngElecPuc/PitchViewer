# -*- coding: utf-8 -*-

"""Instala dependencias de PitchViewer de forma guiada y tolerante.

Uso típico:
    python tools/install_project_dependencies.py --base --build
    python tools/install_project_dependencies.py --torchcrepe
    python tools/install_project_dependencies.py --separation
    python tools/install_project_dependencies.py --full

Las dependencias base y de build fallan con código != 0 si no instalan. Las
opciones pesadas pueden continuar si el entorno no soporta alguna pieza.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pip(args: list[str], *, required: bool = True) -> bool:
    cmd = [sys.executable, "-m", "pip", *args]
    print("\n$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if proc.returncode != 0 and required:
        raise SystemExit(proc.returncode)
    return proc.returncode == 0


def install_requirements(path: Path, *, required: bool = True) -> bool:
    if not path.exists():
        print(f"[OMITIDO] no existe {path.relative_to(PROJECT_ROOT)}")
        return not required
    return run_pip(["install", "-r", str(path)], required=required)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="instala requirements.txt")
    parser.add_argument("--build", action="store_true", help="instala dependencias de build/PyInstaller")
    parser.add_argument("--torchcrepe", action="store_true", help="instala backends Torchcrepe")
    parser.add_argument("--separation", action="store_true", help="instala separación IA estable con el instalador tolerante")
    parser.add_argument("--force-uvr", action="store_true", help="fuerza intento de Audio Separator / UVR")
    parser.add_argument("--full", action="store_true", help="equivale a --base --build --torchcrepe --separation")
    args = parser.parse_args()

    if not any(vars(args).values()):
        args.base = True
        args.build = True

    if args.full:
        args.base = True
        args.build = True
        args.torchcrepe = True
        args.separation = True

    print(f"Proyecto: {PROJECT_ROOT}")
    run_pip(["install", "--upgrade", "pip"], required=True)

    if args.base:
        install_requirements(PROJECT_ROOT / "requirements.txt", required=True)

    if args.build:
        install_requirements(PROJECT_ROOT / "optional-requirements-build.txt", required=True)

    if args.torchcrepe:
        install_requirements(PROJECT_ROOT / "optional-requirements-torchcrepe.txt", required=False)

    if args.separation:
        cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "install_separation_dependencies.py")]
        if args.force_uvr:
            cmd.append("--force-uvr")
        print("\n$ " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
        if proc.returncode != 0:
            print("[ADVERTENCIA] falló alguna dependencia de separación IA.")

    print("\nInstalación solicitada terminada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
