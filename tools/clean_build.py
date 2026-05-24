# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        print(f"eliminado: {path}")
    elif path.exists():
        path.unlink()
        print(f"eliminado: {path}")


def main() -> int:
    for relative in ["build", "dist/PitchViewer", "PitchViewer.spec"]:
        remove(PROJECT_ROOT / relative)
    print("Limpieza terminada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
