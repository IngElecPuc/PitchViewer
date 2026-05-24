# -*- coding: utf-8 -*-

"""Configuración persistente de Pitch Viewer."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

from ..constants import (
    DEFAULT_MAX_MIDI,
    DEFAULT_MIN_MIDI,
    MAX_MIDI_CHOICE,
    MIN_MIDI_CHOICE,
    TIME_WINDOWS,
)
from ..detection.registry import DEFAULT_BACKEND_ID, normalize_backend_id

SETTINGS_VERSION = 1
APP_DIR_NAME = "PitchViewer"
SETTINGS_FILENAME = "settings.json"

VALID_NOTE_LANGUAGES = {"es", "en"}
VALID_SCALE_NAMES = {
    "chromatic",
    "major",
    "natural_minor",
    "harmonic_minor",
    "melodic_minor",
    "major_pentatonic",
    "minor_pentatonic",
    "blues",
}
GEOMETRY_PATTERN = re.compile(r"^\d+x\d+(?:[+-]\d+[+-]\d+)?$")


@dataclass
class AppSettings:
    a4_hz: float = 440.0
    note_language: str = "es"
    time_window_s: int = 10
    min_midi: int = DEFAULT_MIN_MIDI
    max_midi: int = DEFAULT_MAX_MIDI
    scale_name: str = "chromatic"
    scale_root: int = 0
    tolerance_cents: int = 30
    show_out_of_scale: bool = True
    show_tolerance_bands: bool = True
    show_center_lines: bool = True
    show_achieved_blocks: bool = True
    dynamic_tracking: bool = False
    pitch_line_width: int = 2
    theme_name: str = "dark"
    confidence_threshold: float = 0.35
    rms_threshold: float = 0.006
    smoothing_factor: float = 0.35
    median_window: int = 5
    max_jump_semitones: float = 7.0
    octave_guard: bool = True
    detector_backend: str = DEFAULT_BACKEND_ID
    selected_input_device_index: Optional[int] = None
    selected_input_device_name: str = ""
    window_geometry: str = "1120x780"


@dataclass
class SettingsLoadResult:
    settings: AppSettings
    path: Path
    loaded: bool
    error: Optional[str] = None


def get_settings_dir() -> Path:
    """Devuelve la carpeta de configuración de usuario para la app."""
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_DIR_NAME
        return Path.home() / "AppData" / "Roaming" / APP_DIR_NAME

    if sys_config := os.environ.get("XDG_CONFIG_HOME"):
        return Path(sys_config) / APP_DIR_NAME

    return Path.home() / ".config" / APP_DIR_NAME


def get_settings_path() -> Path:
    return get_settings_dir() / SETTINGS_FILENAME


def default_settings() -> AppSettings:
    return AppSettings()


def load_settings(path: Optional[Path] = None) -> SettingsLoadResult:
    settings_path = path or get_settings_path()

    if not settings_path.exists():
        return SettingsLoadResult(
            settings=default_settings(),
            path=settings_path,
            loaded=False,
            error=None,
        )

    try:
        with settings_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        return SettingsLoadResult(
            settings=default_settings(),
            path=settings_path,
            loaded=False,
            error=f"No se pudo leer la configuración: {exc}",
        )

    try:
        settings = settings_from_dict(raw)
    except Exception as exc:
        return SettingsLoadResult(
            settings=default_settings(),
            path=settings_path,
            loaded=False,
            error=f"La configuración estaba dañada o incompleta: {exc}",
        )

    return SettingsLoadResult(settings=settings, path=settings_path, loaded=True, error=None)


def save_settings(settings: AppSettings, path: Optional[Path] = None) -> Path:
    settings_path = path or get_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": SETTINGS_VERSION,
        "settings": settings_to_dict(settings),
    }

    tmp_path = settings_path.with_suffix(settings_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")

    tmp_path.replace(settings_path)
    return settings_path


def delete_settings(path: Optional[Path] = None) -> None:
    settings_path = path or get_settings_path()
    try:
        settings_path.unlink()
    except FileNotFoundError:
        pass


def settings_to_dict(settings: AppSettings) -> dict[str, Any]:
    data = asdict(settings)
    data["a4_hz"] = float(data["a4_hz"])
    data["scale_root"] = int(data["scale_root"]) % 12
    data["selected_input_device_index"] = settings.selected_input_device_index
    return data


def settings_from_dict(raw: dict[str, Any]) -> AppSettings:
    if "settings" in raw and isinstance(raw["settings"], dict):
        raw_settings = raw["settings"]
    else:
        raw_settings = raw

    allowed = {field.name for field in fields(AppSettings)}
    filtered = {key: value for key, value in raw_settings.items() if key in allowed}
    base = AppSettings(**filtered)
    return normalize_settings(base)


def normalize_settings(settings: AppSettings) -> AppSettings:
    settings.a4_hz = _clamp_float(settings.a4_hz, 400.0, 480.0, 440.0)

    if settings.note_language not in VALID_NOTE_LANGUAGES:
        settings.note_language = "es"

    settings.time_window_s = _clamp_int(settings.time_window_s, 1, 60, 10)

    settings.min_midi = _clamp_int(settings.min_midi, MIN_MIDI_CHOICE, MAX_MIDI_CHOICE - 1, DEFAULT_MIN_MIDI)
    settings.max_midi = _clamp_int(settings.max_midi, MIN_MIDI_CHOICE + 1, MAX_MIDI_CHOICE, DEFAULT_MAX_MIDI)
    if settings.max_midi <= settings.min_midi:
        settings.min_midi = DEFAULT_MIN_MIDI
        settings.max_midi = DEFAULT_MAX_MIDI

    if settings.scale_name not in VALID_SCALE_NAMES:
        settings.scale_name = "chromatic"

    settings.scale_root = _coerce_int(settings.scale_root, 0) % 12
    settings.tolerance_cents = _clamp_int(settings.tolerance_cents, 1, 49, 30)

    settings.show_out_of_scale = bool(settings.show_out_of_scale)
    settings.show_tolerance_bands = bool(settings.show_tolerance_bands)
    settings.show_center_lines = bool(settings.show_center_lines)
    settings.show_achieved_blocks = bool(settings.show_achieved_blocks)
    settings.dynamic_tracking = bool(settings.dynamic_tracking)
    settings.pitch_line_width = _clamp_int(settings.pitch_line_width, 1, 8, 2)
    if settings.theme_name not in {"dark", "light"}:
        settings.theme_name = "dark"

    settings.confidence_threshold = _clamp_float(settings.confidence_threshold, 0.0, 1.0, 0.35)
    settings.rms_threshold = _clamp_float(settings.rms_threshold, 0.0, 1.0, 0.006)
    settings.smoothing_factor = _clamp_float(settings.smoothing_factor, 0.0, 1.0, 0.35)

    settings.median_window = _clamp_int(settings.median_window, 1, 21, 5)
    if settings.median_window % 2 == 0:
        settings.median_window += 1
        if settings.median_window > 21:
            settings.median_window = 21

    settings.max_jump_semitones = _clamp_float(settings.max_jump_semitones, 1.0, 24.0, 7.0)
    settings.octave_guard = bool(settings.octave_guard)
    settings.detector_backend = normalize_backend_id(settings.detector_backend)

    if settings.selected_input_device_index is not None:
        settings.selected_input_device_index = _coerce_optional_device_index(settings.selected_input_device_index)

    settings.selected_input_device_name = str(settings.selected_input_device_name or "")[:300]

    if not isinstance(settings.window_geometry, str) or not GEOMETRY_PATTERN.match(settings.window_geometry):
        settings.window_geometry = "1120x780"

    return settings


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp_int(value: Any, min_value: int, max_value: int, default: int) -> int:
    result = _coerce_int(value, default)
    return max(min_value, min(max_value, result))


def _coerce_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except Exception:
        return default

    if not result == result:
        return default

    if result in (float("inf"), float("-inf")):
        return default

    return result


def _clamp_float(value: Any, min_value: float, max_value: float, default: float) -> float:
    result = _coerce_float(value, default)
    return max(min_value, min(max_value, result))


def _coerce_optional_device_index(value: Any) -> Optional[int]:
    try:
        result = int(value)
    except Exception:
        return None

    if result < 0:
        return None

    return result
