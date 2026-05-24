# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_app_class():
    """
    Permite ejecutar la app de dos formas:

    1. Desde la carpeta padre:
        python -m PitchViewer.main

    2. Directamente desde esta carpeta:
        python main.py

    La segunda forma requiere este bootstrap, porque los imports relativos
    de app.py necesitan contexto de paquete.
    """

    if __package__:
        from .app import PitchViewerApp

        return PitchViewerApp

    package_dir = Path(__file__).resolve().parent
    parent_dir = package_dir.parent
    package_name = package_dir.name

    parent_dir_str = str(parent_dir)

    if parent_dir_str not in sys.path:
        sys.path.insert(0, parent_dir_str)

    module = importlib.import_module(f"{package_name}.app")

    return module.PitchViewerApp


def main() -> None:
    PitchViewerApp = _load_app_class()

    app = PitchViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
