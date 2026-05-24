# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import math
import sys
import traceback
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


def main() -> None:
    package_name = bootstrap_package()

    registry = importlib.import_module(f"{package_name}.detection.registry")

    sample_rate = 44100
    frame_size = max(8192, int(round(sample_rate * 0.35)))
    frequency_hz = 220.0
    min_hz = 60.0
    max_hz = 900.0

    t = np.arange(frame_size, dtype=np.float32) / float(sample_rate)
    frame = 0.15 * np.sin(2.0 * math.pi * frequency_hz * t).astype(np.float32)

    print("Package      :", package_name)
    print("Sample rate  :", sample_rate)
    print("Frame size   :", frame_size)
    print("Target sine  :", frequency_hz, "Hz")
    print()

    for backend in registry.BACKENDS:
        print("=" * 80)
        print(f"Backend: {backend.backend_id} | {backend.label}")

        try:
            detector = registry.create_pitch_detector(
                backend_id=backend.backend_id,
                sample_rate=sample_rate,
                frame_size=frame_size,
            )
            print("create_pitch_detector: OK")
        except Exception as exc:
            print("create_pitch_detector: ERROR")
            print(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        for attempt in range(1, 4):
            try:
                estimate = detector.estimate(frame, min_hz, max_hz)
                print(
                    f"attempt {attempt}: "
                    f"freq={estimate.freq_hz:.2f} Hz | confidence={estimate.confidence:.3f}"
                )
            except Exception as exc:
                print(f"attempt {attempt}: ERROR")
                print(f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
                break

        last_error = getattr(detector, "last_error", "")
        if last_error:
            print("last_error:", last_error)

    print("=" * 80)
    print("Listo.")


if __name__ == "__main__":
    main()
