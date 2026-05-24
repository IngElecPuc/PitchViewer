# -*- coding: utf-8 -*-

"""Ventana principal de Pitch Viewer."""

import csv
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .version import APP_DISPLAY_NAME, APP_VERSION
from .config.settings import (
    AppSettings,
    default_settings,
    get_settings_dir,
    load_settings,
    normalize_settings,
    save_settings,
)
from .constants import (
    A4_OPTIONS,
    MAX_DETECTABLE_HZ,
    MAX_MIDI_CHOICE,
    MIN_DETECTABLE_HZ,
    MIN_MIDI_CHOICE,
    RANGE_PRESETS,
    VOCAL_RANGE_PRESETS,
    TIME_WINDOWS,
    TOLERANCE_OPTIONS,
)
from .karaoke.analyzer import build_note_segments, pitch_points_to_frames, settings_snapshot
from .karaoke.audio_loader import LoadedAudio, load_audio_file
from .karaoke.lyrics import LyricsLine, current_lyric_line, load_lyrics_file, parse_lrc
from .karaoke.models import KaraokeAudioInfo, KaraokeProject, KaraokeNoteSegment
from .karaoke.project_file import load_pvk, save_pvk
from .runtime import RUNTIME_INFO
from .separation.demucs_separator import (
    AUDIO_SEPARATOR_MODELS,
    DEMUCS_MODELS,
    ENGINE_AUDIO_SEPARATOR,
    ENGINE_DEMUCS,
    ENGINE_LABELS,
    STEM_DISPLAY_NAMES,
    default_model_for_engine,
    default_separation_dir,
    export_mix_pair_with_gains,
    export_mix_with_gains,
    is_audio_separator_available,
    is_demucs_available,
    is_ffmpeg_available,
    models_for_engine,
    normalize_engine_id,
    run_ai_separation,
)
from .separation.models import SeparationResult
from .detection.registry import (
    BACKENDS,
    BACKEND_AUTOCORRELATION,
    BACKEND_TORCHCREPE_FULL,
    BACKEND_TORCHCREPE_TINY,
    BACKEND_YIN_CMND,
    backend_label,
    create_pitch_detector,
    normalize_backend_id,
)
from .models import InputDevice, PitchPoint
from .music.notes import (
    LANGUAGE_LABELS,
    NOTE_NAMES,
    cents_from_nearest_note,
    freq_to_midi,
    midi_to_freq,
    midi_to_note_name,
    pitch_class_name,
)
from .music.scales import (
    SCALE_INTERVALS,
    SCALE_LABELS,
    scale_display_name,
    scale_pitch_classes,
)
from .ui.dialogs import (
    DetectorSettingsDialog,
    InputDeviceDialog,
    RangeDialog,
    StabilitySettingsDialog,
)


class PitchViewerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title(APP_DISPLAY_NAME)

        self.settings_load_result = load_settings()
        self.settings = self.settings_load_result.settings
        self.settings_path = self.settings_load_result.path

        self.geometry(self.settings.window_geometry)
        self.minsize(900, 600)

        self.settings_lock = threading.Lock()

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=30)
        self.pitch_queue: queue.Queue[PitchPoint] = queue.Queue(maxsize=300)

        self.stream = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_running = False

        self.sample_rate = 44100
        self.block_size = 512
        self.frame_size = 4096
        self.pitch_detector = None
        self.active_backend_id = normalize_backend_id(self.settings.detector_backend)
        self.runtime_info = RUNTIME_INFO
        self.cuda_available = self.runtime_info.cuda_available
        self.torchcrepe_full_realtime_enabled = self.cuda_available
        if (
            not self.torchcrepe_full_realtime_enabled
            and self.active_backend_id == BACKEND_TORCHCREPE_FULL
        ):
            self.active_backend_id = BACKEND_TORCHCREPE_TINY
            self.settings.detector_backend = BACKEND_TORCHCREPE_TINY

        self.is_audio_paused = False
        self.paused_audio_elapsed_s = 0.0
        self.capture_mode = "idle"

        self.is_recording = False
        self.recording_lock = threading.Lock()
        self.recorded_chunks: deque[np.ndarray] = deque()
        self.recorded_sample_count = 0
        self.offline_analysis_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.offline_analysis_running = False
        self.calibration_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.calibration_running = False
        self.offline_mode = False
        self.offline_duration_s = 0.0
        self.offline_cursor_s = 0.0

        self.time_origin = time.perf_counter()
        self.points: deque[PitchPoint] = deque(maxlen=20000)
        self.current_point: Optional[PitchPoint] = None
        self.last_smoothed_midi: Optional[float] = None
        self.recent_midi_values: deque[float] = deque(maxlen=self.settings.median_window)

        self.audio_devices: list[InputDevice] = []
        self.selected_device_index: Optional[int] = self.settings.selected_input_device_index

        self.language_var = tk.StringVar(value=self.settings.note_language)
        self.scale_name_var = tk.StringVar(value=self.settings.scale_name)
        self.scale_root_var = tk.IntVar(value=self.settings.scale_root)
        self.time_window_var = tk.IntVar(value=self.settings.time_window_s)
        self.tolerance_var = tk.IntVar(value=self.settings.tolerance_cents)
        self.a4_var = tk.StringVar(value=f"{self.settings.a4_hz:.1f}")
        self.show_out_of_scale_var = tk.BooleanVar(value=self.settings.show_out_of_scale)
        self.show_tolerance_bands_var = tk.BooleanVar(value=self.settings.show_tolerance_bands)
        self.show_center_lines_var = tk.BooleanVar(value=self.settings.show_center_lines)
        self.detector_backend_var = tk.StringVar(value=normalize_backend_id(self.settings.detector_backend))
        self.offline_detector_backend_var = tk.StringVar(value=normalize_backend_id(self.settings.offline_detector_backend))
        self.dynamic_tracking_var = tk.BooleanVar(value=self.settings.dynamic_tracking)
        self.show_achieved_blocks_var = tk.BooleanVar(value=self.settings.show_achieved_blocks)
        self.theme_var = tk.StringVar(value=self.settings.theme_name)
        self.help_overlay_var = tk.BooleanVar(value=True)
        self.help_overlay_until_s: Optional[float] = time.perf_counter() + 5.0

        self.visual_paused = False
        self.paused_display_time_s: Optional[float] = None
        self.dynamic_center_midi: Optional[float] = None
        self._range_drag_start_y: Optional[int] = None
        self._range_drag_start_min_midi: Optional[int] = None
        self._range_drag_start_max_midi: Optional[int] = None
        self.calibration_visual_info: Optional[dict[str, object]] = None
        self.calibration_visual_until_s: float = 0.0

        self.karaoke_panel_visible = False
        self.karaoke_audio: Optional[LoadedAudio] = None
        self.karaoke_project: Optional[KaraokeProject] = None
        self.karaoke_segments: list[KaraokeNoteSegment] = []
        self.karaoke_lyrics_lines: list[LyricsLine] = []
        self.karaoke_analysis_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.karaoke_analysis_running = False
        self.karaoke_timeline_updating = False
        self.karaoke_timeline_var = tk.DoubleVar(value=0.0)
        self.karaoke_status_var = tk.StringVar(value="Karaoke: sin pista cargada")
        self.karaoke_time_var = tk.StringVar(value="00:00.000 / 00:00.000")
        self.karaoke_current_lyric_var = tk.StringVar(value="")
        self.karaoke_progress_var = tk.DoubleVar(value=0.0)
        self.karaoke_progress_text_var = tk.StringVar(value="Progreso: —")
        self.karaoke_play_active = False
        self.karaoke_play_paused = False
        self.karaoke_score = {
            "evaluated": 0,
            "hits": 0,
            "misses": 0,
            "no_voice": 0,
            "rest": 0,
            "sum_abs_cents": 0.0,
            "sum_signed_cents": 0.0,
        }
        self.karaoke_score_var = tk.StringVar(value="Karaoke play: —")

        self.separation_panel_visible = False
        self.separation_source_path: Optional[str] = None
        self.separation_result: Optional[SeparationResult] = None
        self.separation_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.separation_running = False
        self.separation_output_root = default_separation_dir()
        self.separation_engine_var = tk.StringVar(value=ENGINE_DEMUCS)
        self.separation_model_var = tk.StringVar(value="htdemucs")
        self.separation_mode_var = tk.StringVar(value="4stems")
        self.separation_device_var = tk.StringVar(value="cuda" if self.cuda_available else "cpu")
        self.separation_status_var = tk.StringVar(value="Separación IA: sin mezcla cargada")
        self.separation_progress_var = tk.DoubleVar(value=0.0)
        self.separation_progress_text_var = tk.StringVar(value="Progreso: —")
        self.separation_stem_gain_vars: dict[str, tk.DoubleVar] = {}
        self.separation_stem_rows: dict[str, ttk.Frame] = {}

        self.note_status_var = tk.StringVar(value="Nota: —")
        self.freq_status_var = tk.StringVar(value="Frecuencia: —")
        self.cents_status_var = tk.StringVar(value="Desviación: —")
        self.conf_status_var = tk.StringVar(value="Confianza: —")
        self.assessment_status_var = tk.StringVar(value="Estado: —")
        self.scale_status_var = tk.StringVar(value="Escala: cromática")
        self.range_status_var = tk.StringVar(value="Rango: —")
        self.audio_status_var = tk.StringVar(value="Audio: detenido")
        self.backend_status_var = tk.StringVar(value="Backend: —")
        self.device_status_var = tk.StringVar(value="Entrada: —")
        self.settings_status_var = tk.StringVar(value="")
        self.config_status_var = tk.StringVar(value="")

        self._build_menu()
        self._build_ui()
        self._load_audio_devices(select_default=True)
        self._refresh_status_labels()
        self._refresh_config_status_after_load()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._ui_loop)

    @staticmethod
    def _detect_cuda_available() -> bool:
        return bool(RUNTIME_INFO.cuda_available)

    def _is_realtime_backend_enabled(self, backend_id: str) -> bool:
        backend_id = normalize_backend_id(backend_id)
        if backend_id == BACKEND_TORCHCREPE_FULL and not self.torchcrepe_full_realtime_enabled:
            return False
        return True

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Limpiar historial", command=self._clear_history)
        file_menu.add_command(label="Exportar historial CSV...", command=self._export_history_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Guardar configuración", command=self._save_settings_now)
        file_menu.add_command(label="Recargar configuración", command=self._reload_settings_from_disk)
        file_menu.add_command(label="Restaurar configuración predeterminada...", command=self._reset_settings_to_defaults)
        file_menu.add_command(label="Abrir carpeta de configuración", command=self._open_settings_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self._on_close)
        menubar.add_cascade(label="Archivo", menu=file_menu)

        audio_menu = tk.Menu(menubar, tearoff=False)
        audio_menu.add_command(label="▶ Play / iniciar captura", command=self.start_audio)
        audio_menu.add_command(label="⏸ Pausar captura", command=self.pause_audio)
        audio_menu.add_command(label="⏹ Stop / detener captura", command=self.stop_audio)
        audio_menu.add_command(label="⏺ Grabar para análisis offline", command=self.start_recording)
        audio_menu.add_separator()
        audio_menu.add_command(label="Fuente de entrada...", command=self._choose_input_device)
        audio_menu.add_command(label="Actualizar dispositivos", command=lambda: self._load_audio_devices(select_default=False))
        audio_menu.add_separator()

        live_backend_menu = tk.Menu(audio_menu, tearoff=False)
        for backend in BACKENDS:
            suffix = " (opcional)" if backend.is_optional else ""
            state = tk.NORMAL
            if backend.backend_id == BACKEND_TORCHCREPE_FULL and not self.torchcrepe_full_realtime_enabled:
                suffix = " (requiere CUDA para vivo)"
                state = tk.DISABLED
            live_backend_menu.add_radiobutton(
                label=f"{backend.label}{suffix}",
                variable=self.detector_backend_var,
                value=backend.backend_id,
                state=state,
                command=lambda value=backend.backend_id: self._set_detector_backend(value),
            )
        audio_menu.add_cascade(label="Backend en vivo", menu=live_backend_menu)

        offline_backend_menu = tk.Menu(audio_menu, tearoff=False)
        for backend in BACKENDS:
            suffix = " (opcional)" if backend.is_optional else ""
            if backend.backend_id == BACKEND_TORCHCREPE_FULL and not self.cuda_available:
                suffix += " (CPU: puede tardar)"
            offline_backend_menu.add_radiobutton(
                label=f"{backend.label}{suffix}",
                variable=self.offline_detector_backend_var,
                value=backend.backend_id,
                command=lambda value=backend.backend_id: self._set_offline_detector_backend(value),
            )
        audio_menu.add_cascade(label="Backend offline / record", menu=offline_backend_menu)
        audio_menu.add_separator()
        audio_menu.add_command(label="Parámetros de detección...", command=self._open_detector_settings)
        audio_menu.add_command(label="Estabilidad de pitch...", command=self._open_stability_settings)
        menubar.add_cascade(label="Audio", menu=audio_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        time_menu = tk.Menu(view_menu, tearoff=False)
        for seconds in TIME_WINDOWS:
            time_menu.add_radiobutton(
                label=f"{seconds} segundos",
                variable=self.time_window_var,
                value=seconds,
                command=lambda value=seconds: self._set_time_window(value),
            )
        view_menu.add_cascade(label="Ventana temporal", menu=time_menu)

        range_menu = tk.Menu(view_menu, tearoff=False)
        for label, min_midi, max_midi in VOCAL_RANGE_PRESETS:
            range_menu.add_command(
                label=label,
                command=lambda lo=min_midi, hi=max_midi: self._set_visible_range(lo, hi),
            )
        range_menu.add_separator()
        for label, min_midi, max_midi in RANGE_PRESETS:
            range_menu.add_command(
                label=label,
                command=lambda lo=min_midi, hi=max_midi: self._set_visible_range(lo, hi),
            )
        range_menu.add_separator()
        range_menu.add_command(label="Personalizado por nota...", command=self._open_range_dialog)
        range_menu.add_command(label="Personalizado por Hz...", command=self._open_range_hz_dialog)
        view_menu.add_cascade(label="Rango visible", menu=range_menu)
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label="Seguimiento dinámico de voz",
            variable=self.dynamic_tracking_var,
            command=self._toggle_dynamic_tracking,
        )
        view_menu.add_command(label="Centrar rango en nota actual", command=self._center_range_on_current_pitch)
        view_menu.add_command(label="Congelar/reanudar vista", command=self._toggle_visual_pause)
        view_menu.add_checkbutton(
            label="Mostrar panel instructivo",
            variable=self.help_overlay_var,
            command=self._toggle_help_overlay,
        )
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label="Mostrar bandas de tolerancia",
            variable=self.show_tolerance_bands_var,
            command=self._toggle_tolerance_bands,
        )
        view_menu.add_checkbutton(
            label="Mostrar centro exacto de nota",
            variable=self.show_center_lines_var,
            command=self._toggle_center_lines,
        )
        view_menu.add_checkbutton(
            label="Mostrar bloques alcanzados",
            variable=self.show_achieved_blocks_var,
            command=self._toggle_achieved_blocks,
        )
        view_menu.add_command(label="Grosor de línea de pitch...", command=self._open_pitch_line_width_dialog)
        theme_menu = tk.Menu(view_menu, tearoff=False)
        theme_menu.add_radiobutton(
            label="Oscuro",
            variable=self.theme_var,
            value="dark",
            command=lambda: self._set_theme("dark"),
        )
        theme_menu.add_radiobutton(
            label="Claro",
            variable=self.theme_var,
            value="light",
            command=lambda: self._set_theme("light"),
        )
        view_menu.add_cascade(label="Apariencia", menu=theme_menu)
        menubar.add_cascade(label="Vista", menu=view_menu)

        scale_menu = tk.Menu(menubar, tearoff=False)

        language_menu = tk.Menu(scale_menu, tearoff=False)
        for language, label in LANGUAGE_LABELS.items():
            language_menu.add_radiobutton(
                label=label,
                variable=self.language_var,
                value=language,
                command=lambda value=language: self._set_note_language(value),
            )
        scale_menu.add_cascade(label="Idioma de notas", menu=language_menu)
        scale_menu.add_separator()

        type_menu = tk.Menu(scale_menu, tearoff=False)
        labels = SCALE_LABELS.get(self.settings.note_language, SCALE_LABELS["es"])
        for scale_name in SCALE_INTERVALS:
            type_menu.add_radiobutton(
                label=labels.get(scale_name, scale_name).capitalize(),
                variable=self.scale_name_var,
                value=scale_name,
                command=lambda value=scale_name: self._set_scale_name(value),
            )
        scale_menu.add_cascade(label="Tipo de escala", menu=type_menu)

        root_menu = tk.Menu(scale_menu, tearoff=False)
        for pitch_class in range(12):
            root_menu.add_radiobutton(
                label=pitch_class_name(pitch_class, self.settings.note_language),
                variable=self.scale_root_var,
                value=pitch_class,
                command=lambda value=pitch_class: self._set_scale_root(value),
            )
        scale_menu.add_cascade(label="Tonalidad", menu=root_menu)

        scale_menu.add_separator()
        scale_menu.add_checkbutton(
            label="Mostrar notas fuera de escala",
            variable=self.show_out_of_scale_var,
            command=self._toggle_out_of_scale,
        )

        menubar.add_cascade(label="Escala", menu=scale_menu)

        tuning_menu = tk.Menu(menubar, tearoff=False)
        a4_menu = tk.Menu(tuning_menu, tearoff=False)
        for hz in A4_OPTIONS:
            a4_menu.add_radiobutton(
                label=f"A4 = {hz:.0f} Hz",
                variable=self.a4_var,
                value=f"{hz:.1f}",
                command=lambda value=hz: self._set_a4(value),
            )
        a4_menu.add_separator()
        a4_menu.add_command(label="Personalizado...", command=self._open_a4_dialog)
        tuning_menu.add_cascade(label="Referencia A4", menu=a4_menu)

        tolerance_menu = tk.Menu(tuning_menu, tearoff=False)
        for cents in TOLERANCE_OPTIONS:
            tolerance_menu.add_radiobutton(
                label=f"±{cents} cents",
                variable=self.tolerance_var,
                value=cents,
                command=lambda value=cents: self._set_tolerance(value),
            )
        tolerance_menu.add_separator()
        tolerance_menu.add_command(label="Personalizada...", command=self._open_tolerance_dialog)
        tuning_menu.add_cascade(label="Tolerancia", menu=tolerance_menu)

        menubar.add_cascade(label="Afinación", menu=tuning_menu)

        calibration_menu = tk.Menu(menubar, tearoff=False)
        calibration_menu.add_command(
            label="Diagnóstico rápido de micrófono y voz...",
            command=self._start_voice_calibration_diagnostic,
        )
        calibration_menu.add_command(
            label="Estado Torchcrepe / CUDA",
            command=self._show_backend_runtime_report,
        )
        calibration_menu.add_separator()
        calibration_menu.add_command(
            label="Preset: micrófono USB / baja latencia",
            command=lambda: self._apply_calibration_preset("usb"),
        )
        calibration_menu.add_command(
            label="Preset: micrófono integrado / más filtro",
            command=lambda: self._apply_calibration_preset("laptop"),
        )
        calibration_menu.add_command(
            label="Preset: vibrato o glissando",
            command=lambda: self._apply_calibration_preset("vibrato"),
        )
        calibration_menu.add_command(
            label="Preset: ambiente ruidoso",
            command=lambda: self._apply_calibration_preset("noisy"),
        )
        menubar.add_cascade(label="Calibración", menu=calibration_menu)

        karaoke_menu = tk.Menu(menubar, tearoff=False)
        karaoke_menu.add_command(label="Mostrar/ocultar panel karaoke", command=self._toggle_karaoke_panel)
        karaoke_menu.add_separator()
        karaoke_menu.add_command(label="Abrir proyecto .pvk...", command=self._load_karaoke_project_pvk)
        karaoke_menu.add_command(label="Play karaoke", command=self._start_karaoke_play)
        karaoke_menu.add_command(label="Pausar karaoke", command=self.pause_audio)
        karaoke_menu.add_command(label="Stop karaoke", command=self.stop_audio)
        karaoke_menu.add_separator()
        karaoke_menu.add_command(label="Nuevo proyecto desde audio...", command=self._load_karaoke_audio)
        karaoke_menu.add_command(label="Importar letra...", command=self._import_karaoke_lyrics)
        karaoke_menu.add_command(label="Analizar pista vocal", command=self._analyze_karaoke_audio)
        karaoke_menu.add_separator()
        karaoke_menu.add_command(label="Guardar proyecto .pvk...", command=self._save_karaoke_project)
        karaoke_menu.add_command(label="Exportar segmentos CSV...", command=self._export_karaoke_segments_csv)
        menubar.add_cascade(label="Karaoke", menu=karaoke_menu)

        separation_menu = tk.Menu(menubar, tearoff=False)
        separation_menu.add_command(label="Mostrar/ocultar panel separación IA", command=self._toggle_separation_panel)
        separation_menu.add_separator()
        separation_menu.add_command(label="Abrir canción/mezcla...", command=self._load_separation_source)
        separation_menu.add_command(label="Separar pistas offline", command=self._start_ai_separation)
        separation_menu.add_separator()
        separation_menu.add_command(label="Usar stem de voz como pista karaoke", command=self._use_vocals_stem_for_karaoke)
        separation_menu.add_command(label="Exportar mezcla MP3...", command=lambda: self._export_separation_mix("mp3"))
        separation_menu.add_command(label="Exportar mezcla WAV...", command=lambda: self._export_separation_mix("wav"))
        separation_menu.add_command(label="Exportar mezcla WAV + MP3...", command=lambda: self._export_separation_mix("both"))
        menubar.add_cascade(label="Separación IA", menu=separation_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Acerca de", command=self._show_about)
        menubar.add_cascade(label="Ayuda", menu=help_menu)

        self.config(menu=menubar)

    def _rebuild_menu(self) -> None:
        self._build_menu()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(root)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        self.to_start_button = self._make_transport_button(toolbar, "⏮︎", self._jump_offline_start, "Ir al inicio offline")
        self.to_start_button.pack(side=tk.LEFT, padx=(0, 4))

        self.backward_button = self._make_transport_button(toolbar, "⏪︎", self._transport_backward, "Retroceder / aumentar ventana")
        self.backward_button.pack(side=tk.LEFT, padx=(0, 6))

        self.play_button = self._make_transport_button(toolbar, "▶︎", self.start_audio, "Play / iniciar o reanudar")
        self.play_button.pack(side=tk.LEFT, padx=(0, 4))

        self.pause_button = self._make_transport_button(toolbar, "⏸︎", self.pause_audio, "Pausar captura")
        self.pause_button.pack(side=tk.LEFT, padx=(0, 4))

        self.stop_button = self._make_transport_button(toolbar, "⏹︎", self.stop_audio, "Stop / detener")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 4))

        self.record_button = self._make_transport_button(toolbar, "⏺︎", self.start_recording, "Grabar para análisis offline")
        self.record_button.pack(side=tk.LEFT, padx=(0, 6))

        self.forward_button = self._make_transport_button(toolbar, "⏩︎", self._transport_forward, "Avanzar / reducir ventana")
        self.forward_button.pack(side=tk.LEFT, padx=(0, 4))

        self.to_end_button = self._make_transport_button(toolbar, "⏭︎", self._jump_offline_end, "Ir al final offline")
        self.to_end_button.pack(side=tk.LEFT, padx=(0, 14))

        self.pause_view_button = None

        ttk.Button(toolbar, text="Fuente...", command=self._choose_input_device).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(toolbar, textvariable=self.device_status_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(toolbar, textvariable=self.backend_status_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(toolbar, textvariable=self.audio_status_var).pack(side=tk.RIGHT)

        status = ttk.Frame(root)
        status.pack(fill=tk.X, side=tk.TOP, pady=(8, 6))

        ttk.Label(status, textvariable=self.note_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(status, textvariable=self.freq_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(status, textvariable=self.cents_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(status, textvariable=self.conf_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(status, textvariable=self.assessment_status_var).pack(side=tk.LEFT, padx=(0, 18))

        settings_row = ttk.Frame(root)
        settings_row.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))

        ttk.Label(settings_row, textvariable=self.scale_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(settings_row, textvariable=self.range_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(settings_row, textvariable=self.settings_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(settings_row, textvariable=self.config_status_var).pack(side=tk.RIGHT, padx=(0, 0))

        self.main_view = ttk.Frame(root)
        self.main_view.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.main_view,
            bg="#172026",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._on_canvas_wheel_linux(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self._on_canvas_wheel_linux(event, -1))
        self.canvas.bind("<ButtonPress-2>", self._on_range_drag_start)
        self.canvas.bind("<B2-Motion>", self._on_range_drag_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_range_drag_end)
        self.canvas.bind("<ButtonPress-3>", self._on_range_drag_start)
        self.canvas.bind("<B3-Motion>", self._on_range_drag_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_range_drag_end)

        self._build_karaoke_panel(root)
        self._build_separation_panel(root)

        self.footer = ttk.Label(
            root,
            text=(
                f"v{APP_VERSION}: karaoke play contra targets .pvk; "
                "captura de voz y scoring simple por tolerancia."
            ),
            anchor="w",
        )
        self.footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

    def _make_transport_button(self, parent: tk.Widget, symbol: str, command, tooltip: str = "") -> tk.Button:
        """Crea botones de transporte con glifos en modo texto, sin recuadro interno tipo emoji."""
        button = tk.Button(
            parent,
            text=symbol,
            command=command,
            font=("Segoe UI Symbol", 15, "bold"),
            width=3,
            height=1,
            padx=4,
            pady=2,
            bd=0,
            relief=tk.FLAT,
            overrelief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#b8c2cc",
            highlightcolor="#8aa0b4",
            background="#f5f7fa",
            foreground="#0f1720",
            activebackground="#e5eaf0",
            activeforeground="#0f1720",
            cursor="hand2",
            takefocus=True,
        )
        if tooltip:
            button.bind("<Enter>", lambda _event, text=tooltip: self.audio_status_var.set(text))
        return button

    def _set_visual_pause_label(self, text: str) -> None:
        button = getattr(self, "pause_view_button", None)
        if button is not None:
            try:
                button.configure(text=text)
            except Exception:
                pass

    def _refresh_config_status_after_load(self) -> None:
        if self.settings_load_result.error:
            self.config_status_var.set(
                f"Config: defaults por error; revisar {self.settings_path}"
            )
            messagebox.showwarning(
                "Configuración",
                f"No se pudo cargar la configuración guardada. Se usarán valores por defecto.\n\n"
                f"Detalle: {self.settings_load_result.error}",
                parent=self,
            )
            return

        if self.settings_load_result.loaded:
            self.config_status_var.set(f"Config: cargada desde {self.settings_path}")
        else:
            self.config_status_var.set(f"Config: se guardará en {self.settings_path}")

    def _capture_persistent_state(self) -> None:
        geometry = self.geometry()
        device = self._selected_device()

        with self.settings_lock:
            self.settings.window_geometry = geometry
            if device is not None:
                self.settings.selected_input_device_index = int(device.index)
                self.settings.selected_input_device_name = device.name
            else:
                self.settings.selected_input_device_index = self.selected_device_index
                self.settings.selected_input_device_name = ""
            self.settings = normalize_settings(self.settings)

    def _save_settings_now(self, show_message: bool = True) -> bool:
        self._capture_persistent_state()

        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        try:
            path = save_settings(settings, self.settings_path)
        except Exception as exc:
            self.config_status_var.set("Config: error al guardar")
            if show_message:
                messagebox.showerror(
                    "Guardar configuración",
                    f"No se pudo guardar la configuración:\n\n{exc}",
                    parent=self,
                )
            return False

        self.config_status_var.set(f"Config: guardada en {path}")
        if show_message:
            messagebox.showinfo("Guardar configuración", f"Configuración guardada:\n{path}", parent=self)
        return True

    def _autosave_settings(self) -> None:
        self._save_settings_now(show_message=False)

    def _reload_settings_from_disk(self) -> None:
        result = load_settings(self.settings_path)
        if result.error:
            messagebox.showerror(
                "Recargar configuración",
                f"No se pudo recargar la configuración:\n\n{result.error}",
                parent=self,
            )
            return

        was_running = self.is_running
        if was_running:
            self.stop_audio()

        with self.settings_lock:
            self.settings = result.settings

        self.selected_device_index = result.settings.selected_input_device_index
        self._apply_settings_to_variables()
        self._load_audio_devices(select_default=True)
        self._refresh_status_labels()
        self._rebuild_menu()
        self.config_status_var.set(f"Config: recargada desde {result.path}")

        if was_running:
            self.start_audio()

    def _reset_settings_to_defaults(self) -> None:
        answer = messagebox.askyesno(
            "Restaurar configuración",
            "Esto reemplazará la configuración actual por valores predeterminados.\n\n¿Continuar?",
            parent=self,
        )
        if not answer:
            return

        was_running = self.is_running
        if was_running:
            self.stop_audio()

        with self.settings_lock:
            self.settings = default_settings()

        self.selected_device_index = None
        self._apply_settings_to_variables()
        self._load_audio_devices(select_default=True)
        self._refresh_status_labels()
        self._rebuild_menu()
        self._save_settings_now(show_message=False)
        self.config_status_var.set("Config: restaurada a valores predeterminados")

        if was_running:
            self.start_audio()

    def _open_settings_folder(self) -> None:
        folder = get_settings_dir()
        folder.mkdir(parents=True, exist_ok=True)

        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            messagebox.showerror(
                "Carpeta de configuración",
                f"No se pudo abrir la carpeta:\n\n{folder}\n\n{exc}",
                parent=self,
            )

    def _apply_settings_to_variables(self) -> None:
        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        self.language_var.set(settings.note_language)
        self.scale_name_var.set(settings.scale_name)
        self.scale_root_var.set(settings.scale_root)
        self.time_window_var.set(settings.time_window_s)
        self.tolerance_var.set(settings.tolerance_cents)
        self.a4_var.set(f"{settings.a4_hz:.1f}")
        self.show_out_of_scale_var.set(settings.show_out_of_scale)
        self.show_tolerance_bands_var.set(settings.show_tolerance_bands)
        self.show_center_lines_var.set(settings.show_center_lines)
        self.show_achieved_blocks_var.set(settings.show_achieved_blocks)
        self.dynamic_tracking_var.set(settings.dynamic_tracking)
        self.theme_var.set(settings.theme_name)
        self.detector_backend_var.set(normalize_backend_id(settings.detector_backend))
        self.offline_detector_backend_var.set(normalize_backend_id(settings.offline_detector_backend))

        try:
            self.geometry(settings.window_geometry)
        except Exception:
            pass

    def _resolve_saved_device_index(self, devices: list[InputDevice]) -> Optional[int]:
        if not devices:
            return None

        with self.settings_lock:
            saved_name = self.settings.selected_input_device_name.strip().casefold()
            saved_index = self.settings.selected_input_device_index

        if saved_name:
            for device in devices:
                if device.name.strip().casefold() == saved_name:
                    return device.index

            for device in devices:
                name = device.name.strip().casefold()
                if saved_name in name or name in saved_name:
                    return device.index

        if saved_index is not None:
            for device in devices:
                if device.index == saved_index:
                    return device.index

        return None

    def _load_audio_devices(self, select_default: bool) -> None:
        if sd is None:
            self.audio_devices = []
            self.selected_device_index = None
            self.device_status_var.set("Entrada: falta instalar sounddevice")
            self.audio_status_var.set("Audio: falta instalar sounddevice")
            return

        try:
            raw_devices = sd.query_devices()
        except Exception as exc:
            messagebox.showerror("Error de audio", f"No se pudieron consultar los dispositivos:\n{exc}")
            return

        devices: list[InputDevice] = []
        for idx, device in enumerate(raw_devices):
            channels = int(device.get("max_input_channels", 0))
            if channels <= 0:
                continue

            name = str(device.get("name", "Dispositivo sin nombre"))
            samplerate = int(float(device.get("default_samplerate", 44100)))
            devices.append(InputDevice(idx, name, channels, samplerate))

        self.audio_devices = devices

        if not devices:
            self.selected_device_index = None
            self.device_status_var.set("Entrada: no hay dispositivos")
            self.audio_status_var.set("Audio: no hay dispositivos de entrada")
            return

        if select_default or self.selected_device_index is None:
            selected = self._resolve_saved_device_index(devices)

            if selected is None:
                default_input_idx = None
                try:
                    default_input_idx = int(sd.default.device[0])
                except Exception:
                    default_input_idx = None

                selected = devices[0].index
                if default_input_idx is not None and default_input_idx >= 0:
                    for device in devices:
                        if device.index == default_input_idx:
                            selected = device.index
                            break

            self.selected_device_index = selected
        else:
            valid_indices = {device.index for device in devices}
            if self.selected_device_index not in valid_indices:
                selected = self._resolve_saved_device_index(devices)
                self.selected_device_index = selected if selected is not None else devices[0].index

        self._refresh_device_label()

    def _refresh_device_label(self) -> None:
        device = self._selected_device()
        if device is None:
            self.device_status_var.set("Entrada: —")
        else:
            self.device_status_var.set(f"Entrada: {device.name}")

    def _selected_device(self) -> Optional[InputDevice]:
        for device in self.audio_devices:
            if device.index == self.selected_device_index:
                return device
        return None

    def _choose_input_device(self) -> None:
        if sd is None:
            messagebox.showerror(
                "Dependencia faltante",
                "No está instalado sounddevice. Ejecuta: pip install sounddevice",
            )
            return

        self._load_audio_devices(select_default=False)

        if not self.audio_devices:
            messagebox.showerror("Audio", "No se encontraron dispositivos de entrada.")
            return

        dialog = InputDeviceDialog(self, self.audio_devices, self.selected_device_index)
        if dialog.result is None:
            return

        was_running = self.is_running
        if was_running:
            self.stop_audio()

        self.selected_device_index = dialog.result
        self._refresh_device_label()
        self._autosave_settings()

        if was_running:
            self.start_audio()

    def start_audio(self) -> None:
        """Inicia o reanuda captura desde micrófono.

        Si hay un proyecto karaoke cargado y visible, ▶ funciona como modo play:
        avanza el cursor de la canción y evalúa la voz contra los targets.
        """
        if self.karaoke_segments and self.karaoke_panel_visible and self.capture_mode != "record":
            self._start_karaoke_play()
            return

        mode = "record" if self.is_audio_paused and self.capture_mode == "record" else "online"
        self._start_live_capture(mode=mode)

    def start_recording(self) -> None:
        """Inicia captura para análisis offline.

        La grabación se mantiene en memoria y se limita a la ventana temporal
        configurada. Al presionar stop se analiza con el backend offline seleccionado.
        """
        if self.offline_analysis_running:
            messagebox.showinfo(
                "Análisis offline",
                "Ya hay un análisis offline en curso.",
                parent=self,
            )
            return

        self._reset_recording_buffer()
        self.is_recording = True
        self.offline_mode = False
        self.offline_duration_s = 0.0
        self.offline_cursor_s = 0.0
        self._start_live_capture(mode="record")

    def _start_live_capture(self, mode: str = "online") -> None:
        if sd is None:
            messagebox.showerror(
                "Dependencia faltante",
                "No está instalado sounddevice. Ejecuta: pip install sounddevice",
            )
            return

        if self.selected_device_index is None:
            self._load_audio_devices(select_default=True)

        if self.selected_device_index is None:
            messagebox.showerror("Entrada no válida", "Selecciona un dispositivo de entrada.")
            return

        if self.is_running:
            return

        mode = "record" if mode == "record" else "online"
        resume_elapsed = self.paused_audio_elapsed_s if self.is_audio_paused and mode == self.capture_mode else 0.0
        clear_history = not self.is_audio_paused or mode != self.capture_mode

        if mode == "online":
            self.is_recording = False
            self.offline_mode = False
            if not self.karaoke_play_active:
                self.offline_duration_s = 0.0
                self.offline_cursor_s = 0.0

        backend_id = normalize_backend_id(self.settings.detector_backend)
        if not self._is_realtime_backend_enabled(backend_id):
            messagebox.showwarning(
                "Backend de detección",
                "Torchcrepe full queda deshabilitado para tiempo real porque no se detectó CUDA.\n\n"
                "Usa Torchcrepe tiny/YIN/Autocorrelación en vivo, o el botón ⏺ para grabar "
                "y analizar offline con el backend offline seleccionado.",
                parent=self,
            )
            self._set_detector_backend(BACKEND_TORCHCREPE_TINY, autosave=True, restart_if_running=False)

        try:
            device_info = sd.query_devices(self.selected_device_index, "input")
            self.sample_rate = int(float(device_info.get("default_samplerate", 44100)))
        except Exception:
            self.sample_rate = 44100

        self._configure_frame_size_for_backend()

        self.audio_queue = queue.Queue(maxsize=30)
        self.pitch_queue = queue.Queue(maxsize=300)

        try:
            self.pitch_detector = self._create_active_pitch_detector()
        except Exception as exc:
            self.pitch_detector = None
            self.audio_status_var.set("Audio: detenido")
            messagebox.showerror(
                "Backend de detección",
                f"No se pudo iniciar el backend seleccionado:\n\n{backend_label(self.settings.detector_backend)}\n\n{exc}\n\n"
                "Se volverá a Autocorrelación FFT.",
                parent=self,
            )
            self._set_detector_backend(BACKEND_AUTOCORRELATION, autosave=True, restart_if_running=False)
            self.is_recording = False
            return

        self.stop_event.clear()

        if clear_history:
            self.points.clear()
            self.current_point = None
            self.last_smoothed_midi = None
            self.recent_midi_values = deque(maxlen=max(1, self.settings.median_window))
            self.time_origin = time.perf_counter()
        else:
            self.time_origin = time.perf_counter() - float(resume_elapsed)

        self.visual_paused = False
        self.paused_display_time_s = None
        self._set_visual_pause_label("Pausar vista")
        self.is_audio_paused = False
        self.paused_audio_elapsed_s = 0.0
        self.capture_mode = mode

        try:
            self.worker_thread = threading.Thread(target=self._pitch_worker, daemon=True)
            self.worker_thread.start()

            self.stream = sd.InputStream(
                device=self.selected_device_index,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()

            self.is_running = True
            if mode == "record":
                self.audio_status_var.set(
                    f"Grabando para offline: {self.sample_rate} Hz | máx. {self.settings.time_window_s}s"
                )
            else:
                self.audio_status_var.set(f"Audio: capturando a {self.sample_rate} Hz")
            self._refresh_device_label()
            self._autosave_settings()

        except Exception as exc:
            self.is_recording = False
            self.stop_audio()
            messagebox.showerror("Error al iniciar audio", f"No se pudo iniciar la captura:\n\n{exc}")

    def pause_audio(self) -> None:
        """Pausa captura sin borrar puntos ni reiniciar el reloj visual."""
        if not self.is_running:
            return

        self.paused_audio_elapsed_s = self._current_time_s()
        self.is_audio_paused = True

        if self.karaoke_play_active:
            self.karaoke_play_active = False
            self.karaoke_play_paused = True
            self.offline_mode = True
            self.offline_cursor_s = max(0.0, min(self.offline_duration_s, self.paused_audio_elapsed_s))
            self.current_point = self._point_at_time(self.offline_cursor_s)

        self._stop_stream_only()
        self.audio_status_var.set(f"Audio: pausado en {self.paused_audio_elapsed_s:.1f}s")

    def stop_audio(self) -> None:
        was_recording = bool(self.is_recording)
        was_karaoke_play = bool(self.karaoke_play_active or self.karaoke_play_paused)
        stop_time = self._current_time_s()
        self._stop_stream_only()
        self.is_audio_paused = False
        self.paused_audio_elapsed_s = 0.0
        self.capture_mode = "idle"

        if was_recording:
            self.is_recording = False
            self._finish_recording_and_analyze()
        elif was_karaoke_play:
            self.karaoke_play_active = False
            self.karaoke_play_paused = False
            self.offline_mode = True
            self.offline_cursor_s = max(0.0, min(self.offline_duration_s, stop_time))
            self.current_point = self._point_at_time(self.offline_cursor_s)
            self.audio_status_var.set("Karaoke play: detenido")
            self._update_karaoke_panel_state(force=True)
        else:
            self.audio_status_var.set("Audio: detenido")

    def _stop_stream_only(self) -> None:
        self.is_running = False
        self.stop_event.set()

        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass

            try:
                self.stream.close()
            except Exception:
                pass

            self.stream = None

        self.pitch_detector = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if indata is None or len(indata) == 0:
            return

        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()

        if self.is_recording:
            self._append_recorded_samples(samples)

        try:
            self.audio_queue.put_nowait(samples)
        except queue.Full:
            pass

    def _reset_recording_buffer(self) -> None:
        with self.recording_lock:
            self.recorded_chunks.clear()
            self.recorded_sample_count = 0

    def _append_recorded_samples(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return

        max_samples = max(1, int(self.sample_rate * max(1, int(self.settings.time_window_s))))
        chunk = np.asarray(samples, dtype=np.float32).copy()

        with self.recording_lock:
            self.recorded_chunks.append(chunk)
            self.recorded_sample_count += int(chunk.size)

            while self.recorded_sample_count > max_samples and self.recorded_chunks:
                overflow = self.recorded_sample_count - max_samples
                first = self.recorded_chunks[0]

                if overflow >= first.size:
                    removed = self.recorded_chunks.popleft()
                    self.recorded_sample_count -= int(removed.size)
                else:
                    self.recorded_chunks[0] = first[int(overflow) :].copy()
                    self.recorded_sample_count -= int(overflow)
                    break

    def _get_recorded_audio(self) -> np.ndarray:
        with self.recording_lock:
            if not self.recorded_chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(list(self.recorded_chunks)).astype(np.float32, copy=False)

    def _finish_recording_and_analyze(self) -> None:
        audio = self._get_recorded_audio()
        if audio.size < max(1024, self.sample_rate // 4):
            self.audio_status_var.set("Grabación: demasiado corta para analizar")
            return

        if self.offline_analysis_running:
            self.audio_status_var.set("Análisis offline: ya en curso")
            return

        self.offline_analysis_running = True
        with self.settings_lock:
            offline_backend = normalize_backend_id(self.settings.offline_detector_backend)
        backend_text = backend_label(offline_backend)
        if offline_backend in {BACKEND_TORCHCREPE_TINY, BACKEND_TORCHCREPE_FULL}:
            device_text = "CUDA" if self.cuda_available else "CPU"
            self.audio_status_var.set(
                f"Análisis offline: {backend_text} ({device_text}) sobre {audio.size / max(1, self.sample_rate):.1f}s"
            )
        else:
            self.audio_status_var.set(
                f"Análisis offline: {backend_text} sobre {audio.size / max(1, self.sample_rate):.1f}s"
            )

        thread = threading.Thread(
            target=self._offline_analysis_worker,
            args=(audio, int(self.sample_rate)),
            daemon=True,
        )
        thread.start()

    def _offline_analysis_worker(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            points, info = self._analyze_audio_offline(audio, sample_rate)
            duration_s = float(audio.size) / float(max(1, sample_rate))
            self.offline_analysis_queue.put(("ok", (points, duration_s, info)))
        except Exception as exc:
            self.offline_analysis_queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _consume_offline_analysis_results(self) -> None:
        while True:
            try:
                status, payload = self.offline_analysis_queue.get_nowait()
            except queue.Empty:
                break

            self.offline_analysis_running = False

            if status == "error":
                self.audio_status_var.set("Análisis offline: error")
                messagebox.showerror(
                    "Análisis offline",
                    f"No se pudo analizar la grabación offline:\n\n{payload}",
                    parent=self,
                )
                return

            points, duration_s, info = payload  # type: ignore[misc]
            self.points = deque(points, maxlen=20000)
            self.offline_mode = True
            self.offline_duration_s = float(duration_s)
            self.offline_cursor_s = float(duration_s)
            self.current_point = self._point_at_time(self.offline_cursor_s)
            self.audio_status_var.set(
                f"Offline: {len(points)} puntos | {self._offline_info_label(info)} | duración {duration_s:.1f}s"
            )

    def _analyze_audio_offline(self, audio: np.ndarray, sample_rate: int, progress_callback: Optional[Callable[[float, str], None]] = None):
        """Analiza una grabación con el backend offline seleccionado.

        A diferencia del backend vivo, esta ruta no tiene presupuesto temporal
        por frame. Por eso Torchcrepe full puede correr en CPU si no hay CUDA,
        pero el usuario también puede elegir YIN, Autocorrelación o Torchcrepe tiny.
        """
        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        backend_id = normalize_backend_id(settings.offline_detector_backend)
        if progress_callback is not None:
            progress_callback(0.03, f"Backend offline: {backend_label(backend_id)}")
        detect_min_hz = max(MIN_DETECTABLE_HZ, midi_to_freq(settings.min_midi - 8, settings.a4_hz))
        detect_max_hz = min(MAX_DETECTABLE_HZ, midi_to_freq(settings.max_midi + 8, settings.a4_hz))

        if backend_id in {BACKEND_TORCHCREPE_TINY, BACKEND_TORCHCREPE_FULL}:
            return self._analyze_audio_offline_with_torchcrepe(
                audio=audio,
                sample_rate=sample_rate,
                settings=settings,
                backend_id=backend_id,
                detect_min_hz=detect_min_hz,
                detect_max_hz=detect_max_hz,
                progress_callback=progress_callback,
            )

        return self._analyze_audio_offline_with_frame_detector(
            audio=audio,
            sample_rate=sample_rate,
            settings=settings,
            backend_id=backend_id,
            detect_min_hz=detect_min_hz,
            detect_max_hz=detect_max_hz,
            progress_callback=progress_callback,
        )

    def _analyze_audio_offline_with_torchcrepe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        settings: AppSettings,
        backend_id: str,
        detect_min_hz: float,
        detect_max_hz: float,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        from .detection.offline_torchcrepe import analyze_audio_with_torchcrepe

        model = "full" if backend_id == BACKEND_TORCHCREPE_FULL else "tiny"
        device = "cuda:0" if self.cuda_available else "cpu"
        frames, info = analyze_audio_with_torchcrepe(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=int(sample_rate),
            min_hz=detect_min_hz,
            max_hz=detect_max_hz,
            model=model,
            device=device,
            target_sample_rate=16000,
            hop_s=0.05,
            batch_size=64 if model == "full" else 256,
            chunk_duration_s=20.0,
            progress_callback=(
                (lambda fraction, message: progress_callback(0.05 + 0.82 * float(fraction), message))
                if progress_callback is not None
                else None
            ),
        )

        if progress_callback is not None:
            progress_callback(0.88, "Aplicando filtros de voz")

        points = self._offline_torchcrepe_frames_to_points(
            frames=frames,
            settings=settings,
            detect_min_hz=detect_min_hz,
            detect_max_hz=detect_max_hz,
        )

        return points, info

    def _analyze_audio_offline_with_frame_detector(
        self,
        audio: np.ndarray,
        sample_rate: int,
        settings: AppSettings,
        backend_id: str,
        detect_min_hz: float,
        detect_max_hz: float,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            x = np.reshape(x, (-1)).astype(np.float32, copy=False)

        frame_size = 4096
        hop_size = max(1, int(round(float(sample_rate) * 0.05)))
        detector = create_pitch_detector(backend_id, int(sample_rate), frame_size)

        points: list[PitchPoint] = []
        last_smoothed: Optional[float] = None
        recent_values: deque[float] = deque(maxlen=max(1, int(settings.median_window)))

        if x.size == 0:
            duration_s = 0.0
        else:
            duration_s = float(x.size) / float(max(1, sample_rate))

        starts = list(range(0, max(1, x.size), hop_size))
        total_frames = max(1, len(starts))

        for frame_index, start in enumerate(starts):
            if progress_callback is not None and (frame_index == 0 or frame_index % 20 == 0 or frame_index == total_frames - 1):
                fraction = 0.05 + 0.82 * (float(frame_index) / float(max(1, total_frames - 1)))
                progress_callback(fraction, f"Analizando frame {frame_index + 1}/{total_frames}")

            frame = x[start:start + frame_size]
            if frame.size == 0:
                continue
            if frame.size < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.size), mode="constant")

            time_s = min(duration_s, float(start) / float(max(1, sample_rate)))
            point, last_smoothed = self._estimate_pitch_point_from_frame(
                frame=frame,
                time_s=time_s,
                detector=detector,
                settings=settings,
                detect_min_hz=detect_min_hz,
                detect_max_hz=detect_max_hz,
                last_smoothed_midi=last_smoothed,
                recent_values=recent_values,
            )
            points.append(point)

        class OfflineFrameDetectorInfo:
            pass

        info = OfflineFrameDetectorInfo()
        info.backend_id = backend_id
        info.label = backend_label(backend_id)
        info.model = backend_id
        info.device = "cpu"
        info.frame_count = len(points)
        info.duration_s = duration_s
        info.hop_s = float(hop_size) / float(max(1, sample_rate))
        return points, info

    def _offline_torchcrepe_frames_to_points(
        self,
        frames,
        settings: AppSettings,
        detect_min_hz: float,
        detect_max_hz: float,
    ) -> list[PitchPoint]:
        points: list[PitchPoint] = []
        last_smoothed: Optional[float] = None
        recent_values: deque[float] = deque(maxlen=max(1, int(settings.median_window)))

        for frame in frames:
            freq_hz = float(frame.freq_hz)
            confidence = float(frame.confidence)
            rms = float(frame.rms)
            voiced = (
                detect_min_hz <= freq_hz <= detect_max_hz
                and confidence >= settings.confidence_threshold
                and rms >= settings.rms_threshold
            )

            raw_midi = float("nan")
            if voiced:
                raw_midi = freq_to_midi(freq_hz, settings.a4_hz)

                if settings.octave_guard and last_smoothed is not None:
                    raw_midi = self._correct_octave_jump(raw_midi, last_smoothed)

                recent_values.append(raw_midi)
                midi_stable = float(np.median(np.asarray(recent_values, dtype=np.float64)))

                if last_smoothed is None:
                    midi_smooth = midi_stable
                else:
                    jump = abs(midi_stable - last_smoothed)
                    if jump > settings.max_jump_semitones and confidence < max(0.75, settings.confidence_threshold):
                        midi_smooth = last_smoothed
                    else:
                        midi_smooth = (
                            settings.smoothing_factor * midi_stable
                            + (1.0 - settings.smoothing_factor) * last_smoothed
                        )

                last_smoothed = midi_smooth
            else:
                midi_smooth = float("nan")
                if rms < settings.rms_threshold * 0.5:
                    last_smoothed = None
                    recent_values.clear()

            points.append(
                PitchPoint(
                    time_s=float(frame.time_s),
                    freq_hz=freq_hz,
                    midi_float=midi_smooth,
                    raw_midi_float=raw_midi,
                    confidence=confidence,
                    rms=rms,
                    voiced=voiced,
                )
            )

        return points

    @staticmethod
    def _offline_info_label(info) -> str:
        label = getattr(info, "label", None)
        if not label:
            model = str(getattr(info, "model", "backend offline"))
            if model == "full":
                label = "Torchcrepe full"
            elif model == "tiny":
                label = "Torchcrepe tiny"
            else:
                label = model

        device = getattr(info, "device", "")
        if device:
            return f"{label} en {device}"
        return str(label)

    def _estimate_pitch_point_from_frame(
        self,
        frame: np.ndarray,
        time_s: float,
        detector,
        settings: AppSettings,
        detect_min_hz: float,
        detect_max_hz: float,
        last_smoothed_midi: Optional[float],
        recent_values: deque[float],
    ) -> tuple[PitchPoint, Optional[float]]:
        rms = float(np.sqrt(np.mean(frame * frame)))
        estimate = detector.estimate(frame, detect_min_hz, detect_max_hz)
        freq_hz = float(estimate.freq_hz)
        confidence = float(estimate.confidence)

        voiced = (
            detect_min_hz <= freq_hz <= detect_max_hz
            and confidence >= settings.confidence_threshold
            and rms >= settings.rms_threshold
        )

        raw_midi = float("nan")
        if voiced:
            raw_midi = freq_to_midi(freq_hz, settings.a4_hz)

            if settings.octave_guard and last_smoothed_midi is not None:
                raw_midi = self._correct_octave_jump(raw_midi, last_smoothed_midi)

            recent_values.append(raw_midi)
            midi_stable = float(np.median(np.asarray(recent_values, dtype=np.float64)))

            if last_smoothed_midi is None:
                midi_smooth = midi_stable
            else:
                jump = abs(midi_stable - last_smoothed_midi)
                if jump > settings.max_jump_semitones and confidence < max(0.75, settings.confidence_threshold):
                    midi_smooth = last_smoothed_midi
                else:
                    midi_smooth = (
                        settings.smoothing_factor * midi_stable
                        + (1.0 - settings.smoothing_factor) * last_smoothed_midi
                    )

            last_smoothed_midi = midi_smooth
        else:
            midi_smooth = float("nan")
            if rms < settings.rms_threshold * 0.5:
                last_smoothed_midi = None
                recent_values.clear()

        return (
            PitchPoint(
                time_s=float(time_s),
                freq_hz=freq_hz,
                midi_float=midi_smooth,
                raw_midi_float=raw_midi,
                confidence=confidence,
                rms=rms,
                voiced=voiced,
            ),
            last_smoothed_midi,
        )

    def _point_at_time(self, time_s: float) -> Optional[PitchPoint]:
        best: Optional[PitchPoint] = None
        for point in self.points:
            if point.time_s <= time_s:
                best = point
            else:
                break
        return best

    def _pitch_worker(self) -> None:
        rolling = np.zeros(0, dtype=np.float32)

        while not self.stop_event.is_set():
            try:
                samples = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rolling = np.concatenate([rolling, samples])
            if rolling.size > self.frame_size:
                rolling = rolling[-self.frame_size :]

            if rolling.size < self.frame_size:
                continue

            with self.settings_lock:
                a4_hz = self.settings.a4_hz
                min_midi = self.settings.min_midi
                max_midi = self.settings.max_midi
                confidence_threshold = self.settings.confidence_threshold
                rms_threshold = self.settings.rms_threshold
                smoothing_factor = self.settings.smoothing_factor
                median_window = max(1, int(self.settings.median_window))
                max_jump_semitones = float(self.settings.max_jump_semitones)
                octave_guard = bool(self.settings.octave_guard)

            detect_min_hz = max(MIN_DETECTABLE_HZ, midi_to_freq(min_midi - 8, a4_hz))
            detect_max_hz = min(MAX_DETECTABLE_HZ, midi_to_freq(max_midi + 8, a4_hz))

            frame = rolling.copy()
            rms = float(np.sqrt(np.mean(frame * frame)))
            detector = self.pitch_detector
            if detector is None:
                continue

            estimate = detector.estimate(frame, detect_min_hz, detect_max_hz)
            freq_hz = float(estimate.freq_hz)
            confidence = float(estimate.confidence)

            voiced = (
                detect_min_hz <= freq_hz <= detect_max_hz
                and confidence >= confidence_threshold
                and rms >= rms_threshold
            )

            raw_midi = float("nan")

            if voiced:
                raw_midi = freq_to_midi(freq_hz, a4_hz)

                if octave_guard and self.last_smoothed_midi is not None:
                    raw_midi = self._correct_octave_jump(raw_midi, self.last_smoothed_midi)

                if self.recent_midi_values.maxlen != median_window:
                    recent_tail = list(self.recent_midi_values)[-median_window:]
                    self.recent_midi_values = deque(recent_tail, maxlen=median_window)

                self.recent_midi_values.append(raw_midi)
                midi_stable = float(np.median(np.asarray(self.recent_midi_values, dtype=np.float64)))

                if self.last_smoothed_midi is None:
                    midi_smooth = midi_stable
                else:
                    jump = abs(midi_stable - self.last_smoothed_midi)
                    if jump > max_jump_semitones and confidence < max(0.75, confidence_threshold):
                        midi_smooth = self.last_smoothed_midi
                    else:
                        midi_smooth = (
                            smoothing_factor * midi_stable
                            + (1.0 - smoothing_factor) * self.last_smoothed_midi
                        )

                self.last_smoothed_midi = midi_smooth
            else:
                midi_smooth = float("nan")

                if rms < rms_threshold * 0.5:
                    self.last_smoothed_midi = None
                    self.recent_midi_values.clear()

            point = PitchPoint(
                time_s=time.perf_counter() - self.time_origin,
                freq_hz=freq_hz,
                midi_float=midi_smooth,
                raw_midi_float=raw_midi,
                confidence=confidence,
                rms=rms,
                voiced=voiced,
            )

            try:
                self.pitch_queue.put_nowait(point)
            except queue.Full:
                try:
                    self.pitch_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    self.pitch_queue.put_nowait(point)
                except queue.Full:
                    pass

    def _create_active_pitch_detector(self):
        with self.settings_lock:
            backend_id = normalize_backend_id(self.settings.detector_backend)

        detector = create_pitch_detector(
            backend_id=backend_id,
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
        )
        self.active_backend_id = normalize_backend_id(backend_id)
        self.backend_status_var.set(f"Backend: {backend_label(self.active_backend_id)}")
        return detector

    def _configure_frame_size_for_backend(self) -> None:
        with self.settings_lock:
            backend_id = normalize_backend_id(self.settings.detector_backend)

        if backend_id in {BACKEND_TORCHCREPE_TINY, BACKEND_TORCHCREPE_FULL}:
            # CREPE necesita una ventana de análisis más larga que la
            # autocorrelación/YIN para ser estable. Se mantiene el bloque de
            # audio corto, pero el buffer rodante crece para este backend.
            self.frame_size = max(8192, int(round(self.sample_rate * 0.35)))
        else:
            self.frame_size = 4096


    def _set_detector_backend(
        self,
        backend_id: str,
        autosave: bool = True,
        restart_if_running: bool = True,
    ) -> None:
        backend_id = normalize_backend_id(backend_id)
        if not self._is_realtime_backend_enabled(backend_id):
            messagebox.showwarning(
                "Backend de detección",
                "Torchcrepe full está disponible para análisis offline, pero queda deshabilitado "
                "como backend vivo porque no se detectó CUDA.",
                parent=self,
            )
            backend_id = BACKEND_TORCHCREPE_TINY

        was_running = self.is_running and restart_if_running

        if was_running:
            self.stop_audio()

        with self.settings_lock:
            self.settings.detector_backend = backend_id

        self.detector_backend_var.set(backend_id)
        self.active_backend_id = backend_id
        self.backend_status_var.set(f"Backend: {backend_label(backend_id)}")
        self._refresh_status_labels()

        if autosave:
            self._autosave_settings()

        if was_running:
            self.start_audio()

    def _set_offline_detector_backend(self, backend_id: str, autosave: bool = True) -> None:
        backend_id = normalize_backend_id(backend_id)

        with self.settings_lock:
            self.settings.offline_detector_backend = backend_id

        self.offline_detector_backend_var.set(backend_id)
        self._refresh_status_labels()

        if autosave:
            self._autosave_settings()

    @staticmethod
    def _correct_octave_jump(midi_raw: float, previous_midi: float) -> float:
        candidates = [midi_raw + 12.0 * k for k in range(-3, 4)]
        return min(candidates, key=lambda value: abs(value - previous_midi))

    def _ui_loop(self) -> None:
        self._update_transient_overlays()
        self._consume_offline_analysis_results()
        self._consume_calibration_results()
        self._consume_karaoke_analysis_results()
        self._consume_separation_results()
        self._consume_pitch_points()
        self._update_karaoke_panel_state()
        self._update_status()
        self._redraw_canvas()
        self.after(33, self._ui_loop)

    def _update_transient_overlays(self) -> None:
        now = time.perf_counter()
        if self.help_overlay_until_s is not None and now >= self.help_overlay_until_s:
            self.help_overlay_var.set(False)
            self.help_overlay_until_s = None

        if self.calibration_visual_info is not None and self.calibration_visual_until_s > 0:
            if now >= self.calibration_visual_until_s:
                self.calibration_visual_info = None
                self.calibration_visual_until_s = 0.0

    def _toggle_help_overlay(self) -> None:
        # Cuando el usuario lo invoca desde el menú, queda fijo hasta que lo desactive.
        self.help_overlay_until_s = None

    def _consume_pitch_points(self) -> None:
        while True:
            try:
                point = self.pitch_queue.get_nowait()
            except queue.Empty:
                break

            self.points.append(point)
            self._evaluate_karaoke_point(point)
            if point.voiced:
                self.current_point = point
                if not math.isnan(point.midi_float):
                    self._update_dynamic_center(point.midi_float)

        if self.offline_mode:
            return

        now = self._current_time_s()
        with self.settings_lock:
            max_history = max(90.0, self.settings.time_window_s * 4.0)

        while self.points and self.points[0].time_s < now - max_history:
            self.points.popleft()

    def _current_time_s(self) -> float:
        if self.offline_mode:
            return self.offline_cursor_s

        if self.is_running:
            return time.perf_counter() - self.time_origin

        if self.is_audio_paused:
            return self.paused_audio_elapsed_s

        if self.points:
            return self.points[-1].time_s

        return 0.0

    def _display_time_s(self) -> float:
        if self.visual_paused and self.paused_display_time_s is not None:
            return self.paused_display_time_s
        return self._current_time_s()

    def _update_dynamic_center(self, midi_value: float) -> None:
        with self.settings_lock:
            enabled = self.settings.dynamic_tracking
            min_midi = self.settings.min_midi
            max_midi = self.settings.max_midi

        if not enabled:
            return

        span = max(12.0, float(max_midi - min_midi))
        low_limit = MIN_MIDI_CHOICE + span / 2.0
        high_limit = MAX_MIDI_CHOICE - span / 2.0
        target = max(low_limit, min(high_limit, midi_value))

        if self.dynamic_center_midi is None:
            self.dynamic_center_midi = target
        else:
            self.dynamic_center_midi = 0.12 * target + 0.88 * self.dynamic_center_midi

    def _effective_visible_range(self, settings: AppSettings) -> tuple[int, int]:
        min_midi = int(settings.min_midi)
        max_midi = int(settings.max_midi)

        if not settings.dynamic_tracking or self.dynamic_center_midi is None:
            return min_midi, max_midi

        span = max(12, max_midi - min_midi)
        center = int(round(self.dynamic_center_midi))
        dynamic_min = center - span // 2
        dynamic_max = dynamic_min + span
        return self._clamp_visible_range(dynamic_min, dynamic_max)

    @staticmethod
    def _clamp_visible_range(min_midi: int, max_midi: int) -> tuple[int, int]:
        span = max(1, max_midi - min_midi)
        if min_midi < MIN_MIDI_CHOICE:
            min_midi = MIN_MIDI_CHOICE
            max_midi = min_midi + span
        if max_midi > MAX_MIDI_CHOICE:
            max_midi = MAX_MIDI_CHOICE
            min_midi = max_midi - span
        return int(min_midi), int(max_midi)

    def _clear_history(self) -> None:
        self.points.clear()
        self.current_point = None
        self.last_smoothed_midi = None
        self.recent_midi_values.clear()

    def _set_time_window(self, seconds: int) -> None:
        seconds = max(1, min(60, int(seconds)))
        with self.settings_lock:
            self.settings.time_window_s = seconds
        self.time_window_var.set(seconds)
        self._refresh_status_labels()
        self._autosave_settings()

    def _transport_forward(self) -> None:
        if self.offline_mode:
            with self.settings_lock:
                step = max(0.5, self.settings.time_window_s / 2.0)
            self.offline_cursor_s = min(self.offline_duration_s, self.offline_cursor_s + step)
            self.current_point = self._point_at_time(self.offline_cursor_s)
            self.audio_status_var.set(f"Offline: cursor {self.offline_cursor_s:.1f}s / {self.offline_duration_s:.1f}s")
            return

        with self.settings_lock:
            seconds = max(1, int(self.settings.time_window_s) - 1)
        self._set_time_window(seconds)

    def _transport_backward(self) -> None:
        if self.offline_mode:
            with self.settings_lock:
                step = max(0.5, self.settings.time_window_s / 2.0)
            self.offline_cursor_s = max(0.0, self.offline_cursor_s - step)
            self.current_point = self._point_at_time(self.offline_cursor_s)
            self.audio_status_var.set(f"Offline: cursor {self.offline_cursor_s:.1f}s / {self.offline_duration_s:.1f}s")
            return

        with self.settings_lock:
            seconds = min(60, int(self.settings.time_window_s) + 1)
        self._set_time_window(seconds)

    def _jump_offline_start(self) -> None:
        if not self.offline_mode:
            return
        self.offline_cursor_s = 0.0
        self.current_point = self._point_at_time(self.offline_cursor_s)
        self.audio_status_var.set(f"Offline: cursor 0.0s / {self.offline_duration_s:.1f}s")

    def _jump_offline_end(self) -> None:
        if not self.offline_mode:
            return
        self.offline_cursor_s = self.offline_duration_s
        self.current_point = self._point_at_time(self.offline_cursor_s)
        self.audio_status_var.set(f"Offline: cursor {self.offline_cursor_s:.1f}s / {self.offline_duration_s:.1f}s")

    def _set_visible_range(self, min_midi: int, max_midi: int) -> None:
        if max_midi <= min_midi:
            return

        with self.settings_lock:
            self.settings.min_midi = int(min_midi)
            self.settings.max_midi = int(max_midi)

        self.dynamic_center_midi = None
        self._refresh_status_labels()
        self._autosave_settings()

    def _on_canvas_mousewheel(self, event) -> None:
        delta = 1 if event.delta > 0 else -1
        self._zoom_visible_range(delta)

    def _on_canvas_wheel_linux(self, event, delta: int) -> None:
        self._zoom_visible_range(delta)

    def _zoom_visible_range(self, delta: int) -> None:
        with self.settings_lock:
            min_midi = self.settings.min_midi
            max_midi = self.settings.max_midi

        span = max(2, max_midi - min_midi)
        center = (min_midi + max_midi) / 2.0

        if delta > 0:
            new_span = max(8, span - 2)
        else:
            new_span = min(MAX_MIDI_CHOICE - MIN_MIDI_CHOICE, span + 2)

        new_min = int(round(center - new_span / 2.0))
        new_max = new_min + int(new_span)
        new_min, new_max = self._clamp_visible_range(new_min, new_max)
        self._set_visible_range(new_min, new_max)

    def _on_range_drag_start(self, event) -> None:
        self._range_drag_start_y = int(event.y)
        with self.settings_lock:
            self._range_drag_start_min_midi = int(self.settings.min_midi)
            self._range_drag_start_max_midi = int(self.settings.max_midi)

    def _on_range_drag_move(self, event) -> None:
        if (
            self._range_drag_start_y is None
            or self._range_drag_start_min_midi is None
            or self._range_drag_start_max_midi is None
        ):
            return

        height = max(1, self.canvas.winfo_height() - 56)
        span = self._range_drag_start_max_midi - self._range_drag_start_min_midi
        semitone_delta = int(round((int(event.y) - self._range_drag_start_y) / height * span))
        new_min = self._range_drag_start_min_midi + semitone_delta
        new_max = self._range_drag_start_max_midi + semitone_delta
        new_min, new_max = self._clamp_visible_range(new_min, new_max)

        with self.settings_lock:
            self.settings.min_midi = new_min
            self.settings.max_midi = new_max

        self._refresh_status_labels()

    def _on_range_drag_end(self, event) -> None:
        self._range_drag_start_y = None
        self._range_drag_start_min_midi = None
        self._range_drag_start_max_midi = None
        self._autosave_settings()

    def _open_range_dialog(self) -> None:
        dialog = RangeDialog(self, self.settings)
        if dialog.result is not None:
            self._set_visible_range(*dialog.result)

    def _open_range_hz_dialog(self) -> None:
        with self.settings_lock:
            a4_hz = self.settings.a4_hz
            current_min = midi_to_freq(self.settings.min_midi, a4_hz)
            current_max = midi_to_freq(self.settings.max_midi, a4_hz)

        min_hz = simpledialog.askfloat(
            "Rango visible por Hz",
            "Frecuencia inferior visible en Hz:",
            initialvalue=round(current_min, 2),
            minvalue=MIN_DETECTABLE_HZ,
            maxvalue=MAX_DETECTABLE_HZ,
            parent=self,
        )
        if min_hz is None:
            return

        max_hz = simpledialog.askfloat(
            "Rango visible por Hz",
            "Frecuencia superior visible en Hz:",
            initialvalue=round(current_max, 2),
            minvalue=MIN_DETECTABLE_HZ,
            maxvalue=MAX_DETECTABLE_HZ,
            parent=self,
        )
        if max_hz is None:
            return

        if max_hz <= min_hz:
            messagebox.showerror("Rango inválido", "La frecuencia superior debe ser mayor que la inferior.")
            return

        min_midi = int(math.floor(freq_to_midi(min_hz, a4_hz)))
        max_midi = int(math.ceil(freq_to_midi(max_hz, a4_hz)))
        min_midi = max(MIN_MIDI_CHOICE, min_midi)
        max_midi = min(MAX_MIDI_CHOICE, max_midi)

        if max_midi <= min_midi:
            max_midi = min_midi + 1

        self._set_visible_range(min_midi, max_midi)

    def _set_note_language(self, language: str) -> None:
        if language not in NOTE_NAMES:
            return

        with self.settings_lock:
            self.settings.note_language = language

        self.language_var.set(language)
        self._refresh_status_labels()
        self._rebuild_menu()
        self._autosave_settings()

    def _set_scale_name(self, scale_name: str) -> None:
        if scale_name not in SCALE_INTERVALS:
            return

        with self.settings_lock:
            self.settings.scale_name = scale_name

        self.scale_name_var.set(scale_name)
        self._refresh_status_labels()
        self._autosave_settings()

    def _set_scale_root(self, root: int) -> None:
        with self.settings_lock:
            self.settings.scale_root = int(root) % 12

        self.scale_root_var.set(int(root) % 12)
        self._refresh_status_labels()
        self._autosave_settings()

    def _toggle_out_of_scale(self) -> None:
        with self.settings_lock:
            self.settings.show_out_of_scale = bool(self.show_out_of_scale_var.get())

        self._refresh_status_labels()
        self._autosave_settings()

    def _toggle_tolerance_bands(self) -> None:
        with self.settings_lock:
            self.settings.show_tolerance_bands = bool(self.show_tolerance_bands_var.get())

        self._refresh_status_labels()
        self._autosave_settings()

    def _toggle_center_lines(self) -> None:
        with self.settings_lock:
            self.settings.show_center_lines = bool(self.show_center_lines_var.get())

        self._refresh_status_labels()
        self._autosave_settings()


    def _toggle_achieved_blocks(self) -> None:
        with self.settings_lock:
            self.settings.show_achieved_blocks = bool(self.show_achieved_blocks_var.get())

        self._refresh_status_labels()
        self._autosave_settings()

    def _toggle_dynamic_tracking(self) -> None:
        enabled = bool(self.dynamic_tracking_var.get())
        with self.settings_lock:
            self.settings.dynamic_tracking = enabled

        if not enabled:
            self.dynamic_center_midi = None

        self._refresh_status_labels()
        self._autosave_settings()

    def _set_theme(self, theme_name: str) -> None:
        if theme_name not in {"dark", "light"}:
            return

        with self.settings_lock:
            self.settings.theme_name = theme_name

        self.theme_var.set(theme_name)
        self.canvas.configure(bg=self._palette(self.settings)["canvas_bg"])
        self._refresh_status_labels()
        self._autosave_settings()

    def _open_pitch_line_width_dialog(self) -> None:
        with self.settings_lock:
            initial = self.settings.pitch_line_width

        width = simpledialog.askinteger(
            "Grosor de línea de pitch",
            "Grosor visual de la línea de pitch, entre 1 y 8:",
            initialvalue=initial,
            minvalue=1,
            maxvalue=8,
            parent=self,
        )
        if width is None:
            return

        with self.settings_lock:
            self.settings.pitch_line_width = max(1, min(8, int(width)))

        self._refresh_status_labels()
        self._autosave_settings()

    def _toggle_visual_pause(self) -> None:
        if self.visual_paused:
            self.visual_paused = False
            self.paused_display_time_s = None
            self._set_visual_pause_label("Pausar vista")
            return

        self.visual_paused = True
        self.paused_display_time_s = self._current_time_s()
        self._set_visual_pause_label("Reanudar vista")

    def _center_range_on_current_pitch(self) -> None:
        point = self.current_point
        if point is None or not point.voiced or math.isnan(point.midi_float):
            messagebox.showinfo("Centrar rango", "No hay una nota actual confiable para centrar.", parent=self)
            return

        with self.settings_lock:
            span = max(12, self.settings.max_midi - self.settings.min_midi)

        center = int(round(point.midi_float))
        min_midi = center - span // 2
        max_midi = min_midi + span
        min_midi, max_midi = self._clamp_visible_range(min_midi, max_midi)
        self._set_visible_range(min_midi, max_midi)

    def _set_a4(self, hz: float) -> None:
        if hz <= 0:
            return

        with self.settings_lock:
            self.settings.a4_hz = float(hz)

        self.a4_var.set(f"{float(hz):.1f}")
        self._refresh_status_labels()
        self._autosave_settings()

    def _open_a4_dialog(self) -> None:
        with self.settings_lock:
            initial = self.settings.a4_hz

        hz = simpledialog.askfloat(
            "Referencia A4",
            "Frecuencia de A4 en Hz:",
            initialvalue=initial,
            minvalue=400.0,
            maxvalue=480.0,
            parent=self,
        )
        if hz is None:
            return

        self._set_a4(hz)

    def _set_tolerance(self, cents: int) -> None:
        cents = int(cents)
        cents = max(1, min(49, cents))

        with self.settings_lock:
            self.settings.tolerance_cents = cents

        self.tolerance_var.set(cents)
        self._refresh_status_labels()
        self._autosave_settings()

    def _open_tolerance_dialog(self) -> None:
        with self.settings_lock:
            initial = self.settings.tolerance_cents

        cents = simpledialog.askinteger(
            "Tolerancia",
            "Tolerancia de afinación en cents:",
            initialvalue=initial,
            minvalue=1,
            maxvalue=49,
            parent=self,
        )
        if cents is None:
            return

        self._set_tolerance(cents)

    def _open_detector_settings(self) -> None:
        dialog = DetectorSettingsDialog(self, self.settings)
        if dialog.result is None:
            return

        confidence, rms, smoothing = dialog.result
        with self.settings_lock:
            self.settings.confidence_threshold = confidence
            self.settings.rms_threshold = rms
            self.settings.smoothing_factor = smoothing

        self._refresh_status_labels()
        self._autosave_settings()

    def _open_stability_settings(self) -> None:
        dialog = StabilitySettingsDialog(self, self.settings)
        if dialog.result is None:
            return

        median_window, max_jump_semitones, octave_guard = dialog.result
        with self.settings_lock:
            self.settings.median_window = median_window
            self.settings.max_jump_semitones = max_jump_semitones
            self.settings.octave_guard = octave_guard

        self.recent_midi_values = deque(list(self.recent_midi_values)[-median_window:], maxlen=median_window)
        self._refresh_status_labels()
        self._autosave_settings()

    def _export_history_csv(self) -> None:
        if not self.points:
            messagebox.showinfo("Exportar historial", "No hay puntos para exportar.")
            return

        path = filedialog.asksaveasfilename(
            title="Exportar historial CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")],
            parent=self,
        )
        if not path:
            return

        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)
            scale_pcs = scale_pitch_classes(settings.scale_root, settings.scale_name)

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time_s",
                    "freq_hz",
                    "midi_smooth",
                    "midi_raw",
                    "note",
                    "cents",
                    "confidence",
                    "rms",
                    "voiced",
                    "in_scale",
                ])
                for point in self.points:
                    if point.voiced and not math.isnan(point.midi_float):
                        nearest = int(round(point.midi_float))
                        note = midi_to_note_name(nearest, settings.note_language)
                        cents = cents_from_nearest_note(point.midi_float)
                        in_scale = nearest % 12 in scale_pcs
                    else:
                        note = ""
                        cents = ""
                        in_scale = ""

                    writer.writerow([
                        f"{point.time_s:.6f}",
                        f"{point.freq_hz:.6f}",
                        "" if math.isnan(point.midi_float) else f"{point.midi_float:.6f}",
                        "" if math.isnan(point.raw_midi_float) else f"{point.raw_midi_float:.6f}",
                        note,
                        cents,
                        f"{point.confidence:.6f}",
                        f"{point.rms:.6f}",
                        int(point.voiced),
                        in_scale,
                    ])
        except Exception as exc:
            messagebox.showerror("Exportar historial", f"No se pudo guardar el CSV:\n{exc}")
            return

        messagebox.showinfo("Exportar historial", f"Historial exportado:\n{path}")

    def _refresh_status_labels(self) -> None:
        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        min_note = midi_to_note_name(settings.min_midi, settings.note_language)
        max_note = midi_to_note_name(settings.max_midi, settings.note_language)
        min_hz = midi_to_freq(settings.min_midi, settings.a4_hz)
        max_hz = midi_to_freq(settings.max_midi, settings.a4_hz)

        self.scale_status_var.set(f"Escala: {scale_display_name(settings)}")
        self.range_status_var.set(
            f"Rango: {min_note} - {max_note} ({min_hz:.0f} - {max_hz:.0f} Hz)"
        )
        dynamic = "on" if settings.dynamic_tracking else "off"
        self.settings_status_var.set(
            f"A4: {settings.a4_hz:.1f} Hz | Ventana: {settings.time_window_s}s | "
            f"Tolerancia: ±{settings.tolerance_cents} cents | Línea: {settings.pitch_line_width}px | "
            f"Seguimiento: {dynamic} | Vivo: {backend_label(settings.detector_backend)} | Offline: {backend_label(settings.offline_detector_backend)}"
        )
        cuda_text = "CUDA: sí" if self.cuda_available else "CUDA: no"
        full_text = "full vivo" if self.torchcrepe_full_realtime_enabled else "full offline"
        self.backend_status_var.set(f"Vivo: {backend_label(settings.detector_backend)} | Offline: {backend_label(settings.offline_detector_backend)} | {cuda_text}")

    def _update_status(self) -> None:
        now = self._current_time_s()
        point = self._point_at_time(now) if self.offline_mode else self.current_point

        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        stale = False if self.offline_mode else (point is not None and now - point.time_s > 0.75)
        if point is None or not point.voiced or stale:
            self.note_status_var.set("Nota alcanzada: —")
            self.freq_status_var.set("Frecuencia: —")
            self.cents_status_var.set("Desviación: —")
            self.conf_status_var.set("Confianza: —")
            self.assessment_status_var.set("Estado: —")
            return

        nearest_midi = int(round(point.midi_float))
        note_name = midi_to_note_name(nearest_midi, settings.note_language)
        cents = cents_from_nearest_note(point.midi_float)
        in_scale = nearest_midi % 12 in scale_pitch_classes(settings.scale_root, settings.scale_name)
        matched_midi = self._matched_midi(point.midi_float, settings)

        if matched_midi is not None:
            self.note_status_var.set(f"Nota alcanzada: {midi_to_note_name(matched_midi, settings.note_language)}")
        elif in_scale:
            self.note_status_var.set(f"Nota alcanzada: — | cerca de {note_name} ({cents:+.1f}c)")
        else:
            self.note_status_var.set(f"Nota alcanzada: — | {note_name} fuera de escala")

        self.freq_status_var.set(f"Frecuencia: {point.freq_hz:7.2f} Hz")
        self.cents_status_var.set(f"Desviación: {cents:+6.1f} cents")
        self.conf_status_var.set(f"Confianza: {point.confidence:0.2f} | RMS: {point.rms:0.4f}")

        if self.karaoke_play_active:
            target = self._active_karaoke_segment_at(point.time_s)
            if target is not None:
                target_error = 100.0 * (float(point.midi_float) - float(target.midi))
                hit_text = "ok" if abs(target_error) <= settings.tolerance_cents else ("alto" if target_error > 0 else "bajo")
                target_note = midi_to_note_name(int(target.midi), settings.note_language)
                self.assessment_status_var.set(f"Target: {target_note} | error {target_error:+.1f}c | {hit_text}")
            else:
                self.assessment_status_var.set("Target: descanso")
        else:
            self.assessment_status_var.set(f"Estado: {self._assessment_text(cents, in_scale, settings)}")

    @staticmethod
    def _assessment_text(cents: float, in_scale: bool, settings: AppSettings) -> str:
        if not in_scale:
            return "fuera de escala"

        abs_cents = abs(cents)
        if abs_cents <= max(3.0, settings.tolerance_cents * 0.25):
            return "centrado"
        if abs_cents <= settings.tolerance_cents:
            return "dentro de tolerancia"
        if cents > 0:
            return "desafinado alto"
        return "desafinado bajo"

    @staticmethod
    def _palette(settings: AppSettings) -> dict[str, str]:
        if settings.theme_name == "light":
            return {
                "canvas_bg": "#edf2f7",
                "plot_bg": "#dce6ee",
                "lane_natural": "#d8e1e8",
                "lane_accidental": "#cfd9e1",
                "lane_out": "#c5cdd5",
                "label_bg": "#eef3f7",
                "label_dim_bg": "#c8d0d8",
                "label_fg": "#111827",
                "grid": "#b7c3cc",
                "time_grid": "#c1ccd5",
                "center": "#7d8a95",
                "tolerance_fill": "#b8dcc7",
                "tolerance_edge": "#6aa87e",
                "current_band": "#f3d58a",
                "pitch": "#111827",
                "pitch_marker": "#ffffff",
                "block_fill": "#d97706",
                "block_outline": "#92400e",
                "text": "#111827",
                "legend_bg": "#ffffff",
                "invalid": "#aeb8c2",
            }

        return {
            "canvas_bg": "#172026",
            "plot_bg": "#11181d",
            "lane_natural": "#202b31",
            "lane_accidental": "#192329",
            "lane_out": "#10171c",
            "label_bg": "#dfe7ee",
            "label_dim_bg": "#697782",
            "label_fg": "#111820",
            "grid": "#2f3b43",
            "time_grid": "#25323a",
            "center": "#52616b",
            "tolerance_fill": "#263f36",
            "tolerance_edge": "#3a7657",
            "current_band": "#3a3822",
            "pitch": "#e5edf3",
            "pitch_marker": "#ffffff",
            "block_fill": "#d97706",
            "block_outline": "#f59e0b",
            "text": "#e5edf3",
            "legend_bg": "#0f1720",
            "invalid": "#2a333a",
        }

    def _redraw_canvas(self) -> None:
        canvas = self.canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()

        if width <= 2 or height <= 2:
            return

        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        visible_min_midi, visible_max_midi = self._effective_visible_range(settings)
        palette = self._palette(settings)
        canvas.configure(bg=palette["canvas_bg"])

        left_margin = 86
        right_margin = 28
        top_margin = 18
        bottom_margin = 38

        plot_left = left_margin
        plot_right = width - right_margin
        plot_top = top_margin
        plot_bottom = height - bottom_margin

        if plot_right <= plot_left or plot_bottom <= plot_top:
            return

        if visible_max_midi <= visible_min_midi:
            return

        def midi_to_y(midi_value: float) -> float:
            ratio = (midi_value - visible_min_midi) / (visible_max_midi - visible_min_midi)
            return plot_bottom - ratio * (plot_bottom - plot_top)

        canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, fill=palette["plot_bg"], outline="")

        self._draw_pitch_grid(
            canvas=canvas,
            settings=settings,
            palette=palette,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            left_margin=left_margin,
            visible_min_midi=visible_min_midi,
            visible_max_midi=visible_max_midi,
            midi_to_y=midi_to_y,
        )

        self._draw_time_grid(
            canvas=canvas,
            settings=settings,
            palette=palette,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
        )

        if settings.show_achieved_blocks:
            self._draw_reached_note_blocks(
                canvas=canvas,
                settings=settings,
                palette=palette,
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
                visible_min_midi=visible_min_midi,
                visible_max_midi=visible_max_midi,
                midi_to_y=midi_to_y,
            )

        self._draw_karaoke_segments(
            canvas=canvas,
            settings=settings,
            palette=palette,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            visible_min_midi=visible_min_midi,
            visible_max_midi=visible_max_midi,
            midi_to_y=midi_to_y,
        )

        self._draw_pitch_curve(
            canvas=canvas,
            settings=settings,
            palette=palette,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            visible_min_midi=visible_min_midi,
            visible_max_midi=visible_max_midi,
            midi_to_y=midi_to_y,
        )

        canvas.create_line(plot_right, plot_top, plot_right, plot_bottom, fill=palette["text"], width=1)
        self._draw_calibration_overlay(
            canvas=canvas,
            settings=settings,
            palette=palette,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            visible_min_midi=visible_min_midi,
            visible_max_midi=visible_max_midi,
            midi_to_y=midi_to_y,
        )
        self._draw_legend(canvas, settings, palette, plot_left, plot_top)

    def _draw_pitch_grid(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        palette: dict[str, str],
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        left_margin: int,
        visible_min_midi: int,
        visible_max_midi: int,
        midi_to_y: Callable[[float], float],
    ) -> None:
        current_match = None
        now = self._current_time_s()
        if self.current_point is not None and self.current_point.voiced:
            if now - self.current_point.time_s <= 0.75:
                current_match = self._matched_midi(self.current_point.midi_float, settings)

        semitone_height = abs(midi_to_y(visible_min_midi + 1) - midi_to_y(visible_min_midi))
        label_every = 1 if semitone_height >= 13 else 2 if semitone_height >= 7 else 3
        scale_pcs = scale_pitch_classes(settings.scale_root, settings.scale_name)
        tolerance = settings.tolerance_cents / 100.0

        for midi_value in range(visible_min_midi, visible_max_midi + 1):
            y_top = midi_to_y(midi_value + 0.5)
            y_bottom = midi_to_y(midi_value - 0.5)
            y1 = max(plot_top, min(plot_bottom, y_top))
            y2 = max(plot_top, min(plot_bottom, y_bottom))

            if y2 < plot_top or y1 > plot_bottom:
                continue

            pitch_class = midi_value % 12
            is_scale_note = pitch_class in scale_pcs
            is_natural = pitch_class in {0, 2, 4, 5, 7, 9, 11}

            if is_scale_note:
                fill = palette["lane_natural"] if is_natural else palette["lane_accidental"]
            elif settings.show_out_of_scale:
                fill = palette["lane_out"]
            else:
                fill = palette["plot_bg"]

            canvas.create_rectangle(plot_left, y1, plot_right, y2, fill=fill, outline="")

            if is_scale_note and settings.show_tolerance_bands:
                tolerance_top = midi_to_y(midi_value + tolerance)
                tolerance_bottom = midi_to_y(midi_value - tolerance)
                band_y1 = max(plot_top, min(plot_bottom, tolerance_top))
                band_y2 = max(plot_top, min(plot_bottom, tolerance_bottom))
                band_fill = palette["current_band"] if midi_value == current_match else palette["tolerance_fill"]
                canvas.create_rectangle(plot_left, band_y1, plot_right, band_y2, fill=band_fill, outline="")
                canvas.create_line(plot_left, band_y1, plot_right, band_y1, fill=palette["tolerance_edge"], width=1, dash=(2, 4))
                canvas.create_line(plot_left, band_y2, plot_right, band_y2, fill=palette["tolerance_edge"], width=1, dash=(2, 4))

            label_fill = palette["label_bg"] if is_scale_note else palette["label_dim_bg"]
            if not settings.show_out_of_scale and not is_scale_note:
                label_fill = palette["label_dim_bg"]

            canvas.create_rectangle(0, y1, left_margin, y2, fill=label_fill, outline="#1e2930")

            center_y = midi_to_y(midi_value)
            if settings.show_center_lines or is_scale_note:
                width = 2 if is_scale_note and settings.show_center_lines else 1
                canvas.create_line(plot_left, center_y, plot_right, center_y, fill=palette["center"], width=width)

            # Marcas de separación de zona inválida: los espacios entre bandas de tolerancia quedan oscuros.
            if is_scale_note and settings.show_tolerance_bands and settings.tolerance_cents < 49:
                upper_invalid = midi_to_y(midi_value + 0.5)
                lower_invalid = midi_to_y(midi_value - 0.5)
                canvas.create_line(plot_left, upper_invalid, plot_right, upper_invalid, fill=palette["invalid"], width=1)
                canvas.create_line(plot_left, lower_invalid, plot_right, lower_invalid, fill=palette["invalid"], width=1)

            should_label = (midi_value - visible_min_midi) % label_every == 0
            if should_label and (settings.show_out_of_scale or is_scale_note):
                canvas.create_text(
                    left_margin - 8,
                    center_y,
                    text=midi_to_note_name(midi_value, settings.note_language),
                    fill=palette["label_fg"],
                    font=("TkDefaultFont", 9),
                    anchor="e",
                )

    def _draw_time_grid(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        palette: dict[str, str],
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
    ) -> None:
        now = self._display_time_s()
        window_s = float(settings.time_window_s)
        visible_start = now - window_s

        if window_s <= 10:
            step = 1.0
        elif window_s <= 20:
            step = 2.0
        else:
            step = 5.0

        first_tick = math.ceil(visible_start / step) * step
        tick = first_tick

        while tick <= now:
            ratio = (tick - visible_start) / window_s
            x = plot_left + ratio * (plot_right - plot_left)

            if plot_left <= x <= plot_right:
                canvas.create_line(x, plot_top, x, plot_bottom, fill=palette["time_grid"], width=1)
                seconds_ago = now - tick
                canvas.create_text(
                    x,
                    plot_bottom + 16,
                    text=f"-{seconds_ago:0.0f}s",
                    fill=palette["center"],
                    font=("TkDefaultFont", 8),
                    anchor="n",
                )

            tick += step

    def _draw_reached_note_blocks(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        palette: dict[str, str],
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        visible_min_midi: int,
        visible_max_midi: int,
        midi_to_y: Callable[[float], float],
    ) -> None:
        now = self._display_time_s()
        window_s = float(settings.time_window_s)
        visible_start = now - window_s
        plot_width = plot_right - plot_left
        tolerance = settings.tolerance_cents / 100.0

        active_midi: Optional[int] = None
        start_t: Optional[float] = None
        last_t: Optional[float] = None

        def flush_segment() -> None:
            nonlocal active_midi, start_t, last_t
            if active_midi is None or start_t is None or last_t is None:
                return
            if last_t < visible_start or start_t > now:
                active_midi = None
                start_t = None
                last_t = None
                return
            if last_t - start_t < 0.035:
                active_midi = None
                start_t = None
                last_t = None
                return

            x1 = plot_left + ((max(start_t, visible_start) - visible_start) / window_s) * plot_width
            x2 = plot_left + ((min(last_t, now) - visible_start) / window_s) * plot_width
            if x2 - x1 < 2.0:
                x2 = x1 + 2.0

            y1 = max(plot_top, min(plot_bottom, midi_to_y(active_midi + tolerance)))
            y2 = max(plot_top, min(plot_bottom, midi_to_y(active_midi - tolerance)))

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=palette["block_fill"],
                outline=palette["block_outline"],
                width=1,
                stipple="gray50",
            )

            if x2 - x1 >= 34:
                canvas.create_text(
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                    text=midi_to_note_name(active_midi, settings.note_language),
                    fill=palette["text"],
                    font=("TkDefaultFont", 8, "bold"),
                    anchor="center",
                )

            active_midi = None
            start_t = None
            last_t = None

        for point in self.points:
            if point.time_s < visible_start - 0.30:
                continue
            if point.time_s > now:
                break

            matched = self._matched_midi(point.midi_float, settings) if point.voiced else None
            if matched is not None and not (visible_min_midi <= matched <= visible_max_midi):
                matched = None

            if matched is None:
                flush_segment()
                continue

            if active_midi is None:
                active_midi = matched
                start_t = point.time_s
                last_t = point.time_s
                continue

            if matched == active_midi and last_t is not None and point.time_s - last_t <= 0.25:
                last_t = point.time_s
                continue

            flush_segment()
            active_midi = matched
            start_t = point.time_s
            last_t = point.time_s

        flush_segment()

    def _draw_pitch_curve(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        palette: dict[str, str],
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        visible_min_midi: int,
        visible_max_midi: int,
        midi_to_y: Callable[[float], float],
    ) -> None:
        now = self._display_time_s()
        window_s = float(settings.time_window_s)
        visible_start = now - window_s
        plot_width = plot_right - plot_left

        previous: Optional[tuple[PitchPoint, float, float]] = None

        for point in self.points:
            if point.time_s < visible_start or point.time_s > now:
                continue

            valid = (
                point.voiced
                and not math.isnan(point.midi_float)
                and visible_min_midi - 0.5 <= point.midi_float <= visible_max_midi + 0.5
            )

            if not valid:
                previous = None
                continue

            ratio = (point.time_s - visible_start) / window_s
            x = plot_left + ratio * plot_width
            y = midi_to_y(point.midi_float)

            if previous is not None:
                previous_point, px, py = previous
                gap_ok = point.time_s - previous_point.time_s <= 0.25
                jump_ok = abs(point.midi_float - previous_point.midi_float) <= 12.0

                if gap_ok and jump_ok:
                    canvas.create_line(
                        px,
                        py,
                        x,
                        y,
                        fill=palette["pitch"],
                        width=settings.pitch_line_width,
                        capstyle=tk.ROUND,
                        joinstyle=tk.ROUND,
                    )

            previous = (point, x, y)

        marker_point = self._point_at_time(self.offline_cursor_s) if self.offline_mode else self.current_point
        marker_allowed = self.offline_mode or not self.visual_paused
        if marker_allowed and marker_point is not None and marker_point.voiced:
            if self.offline_mode or self._current_time_s() - marker_point.time_s <= 0.75:
                midi_value = marker_point.midi_float
                if visible_min_midi - 0.5 <= midi_value <= visible_max_midi + 0.5:
                    y = midi_to_y(midi_value)
                    canvas.create_oval(
                        plot_right - 5,
                        y - 5,
                        plot_right + 5,
                        y + 5,
                        fill=palette["pitch_marker"],
                        outline=palette["pitch"],
                        width=2,
                    )
                    nearest = int(round(midi_value))
                    note_text = midi_to_note_name(nearest, settings.note_language)
                    cents = cents_from_nearest_note(midi_value)
                    matched = self._matched_midi(midi_value, settings)
                    prefix = "✓" if matched is not None else "—"
                    label = f"{prefix} {note_text} {cents:+.0f}c"
                    canvas.create_rectangle(
                        plot_right - 102,
                        y - 14,
                        plot_right - 10,
                        y + 14,
                        fill=palette["legend_bg"],
                        outline=palette["pitch"],
                        width=1,
                    )
                    canvas.create_text(
                        plot_right - 56,
                        y,
                        text=label,
                        fill=palette["text"],
                        font=("TkDefaultFont", 9, "bold"),
                        anchor="center",
                    )

    def _draw_calibration_overlay(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        palette: dict[str, str],
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        visible_min_midi: int,
        visible_max_midi: int,
        midi_to_y: Callable[[float], float],
    ) -> None:
        info = self.calibration_visual_info
        if not info:
            return

        min_midi = info.get("min_midi")
        max_midi = info.get("max_midi")
        message = str(info.get("message", "Calibración"))
        detail = str(info.get("detail", ""))
        applied = bool(info.get("applied", False))

        accent = "#22c55e" if applied else "#f59e0b"

        if isinstance(min_midi, int) and isinstance(max_midi, int):
            if max_midi >= visible_min_midi and min_midi <= visible_max_midi:
                y_top = max(plot_top, min(plot_bottom, midi_to_y(max_midi)))
                y_bottom = max(plot_top, min(plot_bottom, midi_to_y(min_midi)))
                if y_bottom < y_top:
                    y_top, y_bottom = y_bottom, y_top
                canvas.create_rectangle(
                    plot_left,
                    y_top,
                    plot_right,
                    y_bottom,
                    outline=accent,
                    width=2,
                    dash=(6, 4),
                )
                canvas.create_line(plot_right - 12, y_top, plot_right - 12, y_bottom, fill=accent, width=3)
                canvas.create_line(plot_right - 24, y_top, plot_right, y_top, fill=accent, width=2)
                canvas.create_line(plot_right - 24, y_bottom, plot_right, y_bottom, fill=accent, width=2)

        x2 = plot_right - 12
        x1 = max(plot_left + 16, x2 - 330)
        y1 = plot_top + 10
        y2 = y1 + (54 if detail else 36)
        canvas.create_rectangle(x1, y1, x2, y2, fill=palette["legend_bg"], outline=accent, width=2)
        canvas.create_text(x1 + 10, y1 + 14, text=message, fill=palette["text"], font=("TkDefaultFont", 9, "bold"), anchor="w")
        if detail:
            canvas.create_text(x1 + 10, y1 + 34, text=detail, fill=palette["text"], font=("TkDefaultFont", 8), anchor="w")

    def _set_calibration_visual_result(self, result: dict[str, object], applied: bool) -> None:
        with self.settings_lock:
            language = self.settings.note_language

        min_midi = int(result["recommended_min_midi"])
        max_midi = int(result["recommended_max_midi"])
        min_note = midi_to_note_name(min_midi, language)
        max_note = midi_to_note_name(max_midi, language)
        conf = float(result["recommended_confidence"])
        rms = float(result["recommended_rms"])
        status = "Calibración aplicada" if applied else "Calibración recomendada"
        self.calibration_visual_info = {
            "message": status,
            "detail": f"Rango {min_note}-{max_note} | conf {conf:.2f} | RMS {rms:.4f}",
            "min_midi": min_midi,
            "max_midi": max_midi,
            "applied": applied,
        }
        self.calibration_visual_until_s = time.perf_counter() + 12.0

    def _set_calibration_visual_message(self, message: str, detail: str = "", seconds: float = 8.0) -> None:
        self.calibration_visual_info = {
            "message": message,
            "detail": detail,
            "applied": False,
        }
        self.calibration_visual_until_s = time.perf_counter() + max(1.0, float(seconds))

    def _draw_legend(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        palette: dict[str, str],
        plot_left: int,
        plot_top: int,
    ) -> None:
        if not bool(self.help_overlay_var.get()):
            return

        lines = [
            f"Línea pitch: grosor {settings.pitch_line_width}",
            f"Banda válida: ±{settings.tolerance_cents} cents",
            "Bloque: nota alcanzada",
            "Zona oscura: entre notas / desafinado",
        ]
        if settings.dynamic_tracking:
            lines.append("Seguimiento dinámico activo")
        if self.is_recording:
            lines.append("Grabando para análisis offline")
        if self.offline_analysis_running:
            lines.append(f"Analizando offline con {backend_label(settings.offline_detector_backend)}")
        if self.offline_mode:
            lines.append(f"Offline: {self.offline_cursor_s:.1f}s / {self.offline_duration_s:.1f}s")
        if self.calibration_running:
            lines.append("Diagnóstico de calibración en curso")
        if self.visual_paused:
            lines.append("Vista pausada")

        x1 = plot_left + 10
        y1 = plot_top + 8
        x2 = x1 + 245
        y2 = y1 + 18 * len(lines) + 10
        canvas.create_rectangle(x1, y1, x2, y2, fill=palette["legend_bg"], outline=palette["center"], width=1)

        y = y1 + 13
        for idx, text in enumerate(lines):
            if idx == 0:
                canvas.create_line(x1 + 10, y, x1 + 34, y, fill=palette["pitch"], width=settings.pitch_line_width)
                tx = x1 + 42
            elif idx == 1:
                canvas.create_rectangle(x1 + 10, y - 5, x1 + 34, y + 5, fill=palette["tolerance_fill"], outline=palette["tolerance_edge"])
                tx = x1 + 42
            elif idx == 2:
                canvas.create_rectangle(x1 + 10, y - 5, x1 + 34, y + 5, fill=palette["block_fill"], outline=palette["block_outline"], stipple="gray50")
                tx = x1 + 42
            else:
                tx = x1 + 10
            canvas.create_text(tx, y, text=text, fill=palette["text"], font=("TkDefaultFont", 8), anchor="w")
            y += 18

    @staticmethod
    def _matched_midi(midi_float: float, settings: AppSettings) -> Optional[int]:
        if math.isnan(midi_float):
            return None
        nearest_midi = int(round(midi_float))
        cents = abs(cents_from_nearest_note(midi_float))
        if cents > settings.tolerance_cents:
            return None
        scale_pcs = scale_pitch_classes(settings.scale_root, settings.scale_name)
        if nearest_midi % 12 not in scale_pcs:
            return None
        return nearest_midi


    def _build_separation_panel(self, parent: tk.Widget) -> None:
        self.separation_panel = ttk.LabelFrame(parent, text="Separación IA / offline", padding=8)

        controls = ttk.Frame(self.separation_panel)
        controls.pack(fill=tk.X, side=tk.TOP)

        ttk.Button(controls, text="Abrir mezcla...", command=self._load_separation_source).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Separar offline", command=self._start_ai_separation).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Usar voz en karaoke", command=self._use_vocals_stem_for_karaoke).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Exportar MP3...", command=lambda: self._export_separation_mix("mp3")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Exportar WAV...", command=lambda: self._export_separation_mix("wav")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="WAV+MP3...", command=lambda: self._export_separation_mix("both")).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(controls, textvariable=self.separation_status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        options = ttk.Frame(self.separation_panel)
        options.pack(fill=tk.X, side=tk.TOP, pady=(8, 0))

        ttk.Label(options, text="Motor:").pack(side=tk.LEFT, padx=(0, 4))
        self.separation_engine_combo = ttk.Combobox(
            options,
            textvariable=self.separation_engine_var,
            values=[ENGINE_DEMUCS, ENGINE_AUDIO_SEPARATOR],
            state="readonly",
            width=18,
        )
        self.separation_engine_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.separation_engine_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_separation_engine_changed())

        ttk.Label(options, text="Modelo:").pack(side=tk.LEFT, padx=(0, 4))
        self.separation_model_combo = ttk.Combobox(
            options,
            textvariable=self.separation_model_var,
            values=DEMUCS_MODELS,
            state="readonly",
            width=38,
        )
        self.separation_model_combo.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(options, text="Salida:").pack(side=tk.LEFT, padx=(0, 4))
        self.separation_mode_combo = ttk.Combobox(
            options,
            textvariable=self.separation_mode_var,
            values=["4stems", "2stems"],
            state="readonly",
            width=10,
        )
        self.separation_mode_combo.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(options, text="Device:").pack(side=tk.LEFT, padx=(0, 4))
        device_values = ["cpu"]
        if self.cuda_available:
            device_values = ["cuda", "cpu"]
        ttk.Combobox(
            options,
            textvariable=self.separation_device_var,
            values=device_values,
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 12))

        ffmpeg_text = f"ffmpeg: sí ({self.runtime_info.ffmpeg_source})" if self.runtime_info.ffmpeg_available else "ffmpeg: no"
        cuda_text = "CUDA: sí"
        if self.cuda_available and self.runtime_info.cuda_device_name:
            cuda_text = f"CUDA: sí | {self.runtime_info.cuda_device_name}"
        elif not self.cuda_available:
            cuda_text = "CUDA: no"
        env_text = f"{cuda_text} | {ffmpeg_text}"
        ttk.Label(options, text=(env_text if len(env_text) <= 90 else env_text[:87] + "...")).pack(side=tk.LEFT, fill=tk.X, expand=True)

        progress_row = ttk.Frame(self.separation_panel)
        progress_row.pack(fill=tk.X, side=tk.TOP, pady=(8, 0))
        ttk.Label(progress_row, textvariable=self.separation_progress_text_var, width=44, anchor="w").pack(side=tk.LEFT, padx=(0, 8))
        self.separation_progress_bar = ttk.Progressbar(
            progress_row,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.separation_progress_var,
            maximum=100.0,
        )
        self.separation_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.separation_stems_frame = ttk.LabelFrame(self.separation_panel, text="Ganancia por pista detectada", padding=8)
        self.separation_stems_frame.pack(fill=tk.X, side=tk.TOP, pady=(8, 0))
        ttk.Label(
            self.separation_stems_frame,
            text="Después de separar, ajusta las ganancias: 100% conserva, 0% silencia. Exporta MP3/WAV o usa vocals como pista karaoke.",
        ).pack(anchor="w")
        self._on_separation_engine_changed()

    def _on_separation_engine_changed(self) -> None:
        engine = normalize_engine_id(self.separation_engine_var.get())
        self.separation_engine_var.set(engine)

        models = models_for_engine(engine)
        if hasattr(self, "separation_model_combo"):
            self.separation_model_combo.configure(values=models)
        if self.separation_model_var.get() not in models:
            self.separation_model_var.set(default_model_for_engine(engine))

        if hasattr(self, "separation_mode_combo"):
            if engine == ENGINE_AUDIO_SEPARATOR:
                self.separation_mode_var.set("2stems")
                self.separation_mode_combo.configure(values=["2stems"], state="disabled")
            else:
                self.separation_mode_combo.configure(values=["4stems", "2stems"], state="readonly")
                if self.separation_mode_var.get() not in {"4stems", "2stems"}:
                    self.separation_mode_var.set("4stems")

        label = ENGINE_LABELS.get(engine, engine)
        self.audio_status_var.set(f"Separación IA: motor seleccionado {label}")

    def _toggle_separation_panel(self) -> None:
        if self.separation_panel_visible:
            self.separation_panel.pack_forget()
            self.separation_panel_visible = False
            return
        self._show_separation_panel()

    def _show_separation_panel(self) -> None:
        if not self.separation_panel_visible:
            self.separation_panel.pack(fill=tk.X, side=tk.BOTTOM, before=self.footer, pady=(8, 0))
            self.separation_panel_visible = True

    def _load_separation_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Abrir canción o mezcla para separar",
            filetypes=[
                ("Audio", "*.wav *.flac *.ogg *.mp3 *.m4a *.mp4"),
                ("WAV", "*.wav"),
                ("Todos los archivos", "*.*"),
            ],
            parent=self,
        )
        if not path:
            return
        self.separation_source_path = path
        self.separation_result = None
        self._clear_separation_stem_sliders()
        self._set_separation_progress(0.0, "Progreso: mezcla cargada")
        self.separation_status_var.set(f"Separación IA: {os.path.basename(path)}")
        self.audio_status_var.set("Separación IA: mezcla cargada; falta separar pistas")
        self._show_separation_panel()

    def _start_ai_separation(self) -> None:
        if self.separation_running:
            messagebox.showinfo("Separación IA", "Ya hay una separación en curso.", parent=self)
            return
        if not self.separation_source_path:
            messagebox.showinfo("Separación IA", "Primero abre una canción o mezcla.", parent=self)
            self._show_separation_panel()
            return
        engine = normalize_engine_id(self.separation_engine_var.get())
        if engine == ENGINE_DEMUCS and not is_demucs_available():
            messagebox.showerror(
                "Separación IA",
                "Demucs no está instalado. Instala las dependencias estables:\n\n"
                "pip install -r optional-requirements-separation.txt",
                parent=self,
            )
            self._show_separation_panel()
            return
        if engine == ENGINE_AUDIO_SEPARATOR and not is_audio_separator_available():
            messagebox.showerror(
                "Separación IA",
                "Audio Separator / UVR no está instalado. Instálalo solo como extra experimental:\n\n"
                "python tools/install_separation_dependencies.py\n\n"
                "Audio Separator / UVR es experimental. Si falla la instalación, "
                "Demucs queda como motor estable.",
                parent=self,
            )
            self._show_separation_panel()
            return

        self.separation_running = True
        self.separation_result = None
        self._clear_separation_stem_sliders()
        self._set_separation_progress(0.01, "Progreso: preparando separación offline")
        self.separation_status_var.set("Separación IA: ejecutando separación offline")
        self.audio_status_var.set("Separación IA: proceso en curso")
        self._show_separation_panel()

        thread = threading.Thread(target=self._separation_worker, daemon=True)
        thread.start()

    def _separation_worker(self) -> None:
        try:
            source_path = str(self.separation_source_path or "")
            engine = normalize_engine_id(self.separation_engine_var.get())
            model_name = str(self.separation_model_var.get() or default_model_for_engine(engine))
            mode = str(self.separation_mode_var.get() or "4stems")
            requested_device = str(self.separation_device_var.get() or "cpu").lower()
            device = "cuda" if requested_device.startswith("cuda") and self.cuda_available else "cpu"

            def report(fraction: float, message: str) -> None:
                self.separation_queue.put(("progress", {"fraction": float(fraction), "message": str(message)}))

            result = run_ai_separation(
                input_path=source_path,
                output_root=self.separation_output_root,
                engine=engine,
                model_name=model_name,
                device=device,
                mode=mode,
                progress_callback=report,
            )
            self.separation_queue.put(("ok", result))
        except Exception as exc:
            self.separation_queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _consume_separation_results(self) -> None:
        while True:
            try:
                status, payload = self.separation_queue.get_nowait()
            except queue.Empty:
                break

            if status == "progress":
                data = payload if isinstance(payload, dict) else {}
                fraction = float(data.get("fraction", 0.0))
                message = str(data.get("message", "Procesando"))
                self._set_separation_progress(fraction, message)
                self.separation_status_var.set(message)
                continue

            self.separation_running = False
            if status == "error":
                self._set_separation_progress(0.0, "Progreso: error de separación")
                self.separation_status_var.set("Separación IA: error")
                self.audio_status_var.set("Separación IA: error")
                messagebox.showerror("Separación IA", f"No se pudo separar la canción:\n\n{payload}", parent=self)
                return

            result = payload
            if not isinstance(result, SeparationResult):
                self._set_separation_progress(0.0, "Progreso: resultado inválido")
                return

            self.separation_result = result
            self._set_separation_progress(1.0, "Progreso: separación completa")
            stems_text = ", ".join(sorted(result.stems))
            self.separation_status_var.set(
                f"Separación IA: {len(result.stems)} pistas | {stems_text} | {result.device}"
            )
            self.audio_status_var.set("Separación IA: separación lista; ajusta ganancias o usa voz en karaoke")
            self._refresh_separation_stem_sliders()

    def _set_separation_progress(self, fraction: float, message: str) -> None:
        percent = max(0.0, min(100.0, 100.0 * float(fraction)))
        self.separation_progress_var.set(percent)
        if percent <= 0.0:
            self.separation_progress_text_var.set(str(message))
        elif percent >= 100.0:
            self.separation_progress_text_var.set(f"{message} | 100%")
        else:
            self.separation_progress_text_var.set(f"{message} | {percent:0.1f}%")

    def _clear_separation_stem_sliders(self) -> None:
        if not hasattr(self, "separation_stems_frame"):
            return
        for row in list(self.separation_stem_rows.values()):
            try:
                row.destroy()
            except Exception:
                pass
        self.separation_stem_rows.clear()
        self.separation_stem_gain_vars.clear()

    def _refresh_separation_stem_sliders(self) -> None:
        self._clear_separation_stem_sliders()
        if self.separation_result is None:
            return

        for stem_name in sorted(self.separation_result.stems):
            stem = self.separation_result.stems[stem_name]
            row = ttk.Frame(self.separation_stems_frame)
            row.pack(fill=tk.X, side=tk.TOP, pady=(4, 0))
            label = STEM_DISPLAY_NAMES.get(stem_name, stem_name)
            ttk.Label(row, text=f"{label}:", width=18, anchor="w").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=100.0)
            self.separation_stem_gain_vars[stem_name] = var
            ttk.Scale(
                row,
                from_=0.0,
                to=150.0,
                orient=tk.HORIZONTAL,
                variable=var,
                length=360,
            ).pack(side=tk.LEFT, padx=(0, 8))
            value_label = ttk.Label(row, text="100%", width=7, anchor="e")
            value_label.pack(side=tk.LEFT)

            def update_label(_value=None, variable=var, label_widget=value_label) -> None:
                label_widget.configure(text=f"{float(variable.get()):0.0f}%")

            var.trace_add("write", lambda *_args, callback=update_label: callback())
            ttk.Label(
                row,
                text=f"{stem.duration_s:.1f}s | {stem.sample_rate} Hz | {stem.channels} ch",
                width=26,
                anchor="e",
            ).pack(side=tk.LEFT, padx=(8, 0))
            self.separation_stem_rows[stem_name] = row

    def _use_vocals_stem_for_karaoke(self) -> None:
        if self.separation_result is None or "vocals" not in self.separation_result.stems:
            messagebox.showinfo("Separación IA", "No hay stem vocals disponible. Primero separa una canción.", parent=self)
            return

        vocals = self.separation_result.stems["vocals"]
        try:
            loaded = load_audio_file(str(vocals.path), selected_channel="mix")
        except Exception as exc:
            messagebox.showerror("Separación IA", f"No se pudo cargar vocals.wav como pista karaoke:\n\n{exc}", parent=self)
            return

        self.karaoke_audio = loaded
        self.karaoke_segments = []
        self.karaoke_project = KaraokeProject(
            title=os.path.splitext(loaded.filename)[0],
            artist="",
            audio_info=KaraokeAudioInfo(
                path=loaded.path,
                filename=loaded.filename,
                sample_rate=loaded.sample_rate,
                channels=loaded.channels,
                selected_channel=loaded.selected_channel,
                duration_s=loaded.duration_s,
            ),
        )
        self.points.clear()
        self.current_point = None
        self.offline_mode = True
        self.offline_duration_s = loaded.duration_s
        self.offline_cursor_s = 0.0
        self.karaoke_timeline_var.set(0.0)
        try:
            self.karaoke_scale.configure(to=max(0.001, loaded.duration_s))
        except Exception:
            pass
        self.karaoke_status_var.set(
            f"Audio desde separación IA: {loaded.filename} | {loaded.sample_rate} Hz | {loaded.duration_s:.1f}s"
        )
        self._set_karaoke_progress(0.0, "Progreso: stem de voz cargado, falta analizar")
        self.audio_status_var.set("Karaoke: stem de voz cargado desde separación IA")
        self._show_karaoke_panel()
        self._update_karaoke_panel_state(force=True)

    def _export_separation_mix(self, output_kind: str = "mp3") -> None:
        if self.separation_result is None:
            messagebox.showinfo("Separación IA", "No hay stems para mezclar. Primero separa una canción.", parent=self)
            return

        output_kind = str(output_kind or "mp3").lower()

        if output_kind == "both":
            path = filedialog.asksaveasfilename(
                title="Exportar mezcla WAV + MP3 con ganancias",
                defaultextension=".wav",
                filetypes=[("Base de salida", "*.wav"), ("Todos los archivos", "*.*")],
                initialfile="pitchviewer_mix.wav",
                parent=self,
            )
            if not path:
                return
        else:
            extension = ".mp3" if output_kind == "mp3" else ".wav"
            title = "Exportar mezcla MP3 con ganancias" if output_kind == "mp3" else "Exportar mezcla WAV con ganancias"
            filetypes = [("MP3", "*.mp3"), ("WAV", "*.wav"), ("Todos los archivos", "*.*")] if output_kind == "mp3" else [("WAV", "*.wav"), ("MP3", "*.mp3"), ("Todos los archivos", "*.*")]
            path = filedialog.asksaveasfilename(
                title=title,
                defaultextension=extension,
                filetypes=filetypes,
                initialfile=f"pitchviewer_mix{extension}",
                parent=self,
            )
            if not path:
                return

        if (output_kind in {"mp3", "both"} or str(path).lower().endswith(".mp3")) and not is_ffmpeg_available():
            messagebox.showerror(
                "Separación IA",
                "Para exportar MP3 se requiere ffmpeg. Instala las dependencias de separación con:\n\npython tools/install_separation_dependencies.py\n\nTambién puedes instalar ffmpeg globalmente y agregarlo al PATH.",
                parent=self,
            )
            return

        gains = {
            name: max(0.0, float(var.get()) / 100.0)
            for name, var in self.separation_stem_gain_vars.items()
        }
        try:
            if output_kind == "both":
                wav_path, mp3_path = export_mix_pair_with_gains(self.separation_result, gains, path)
                saved_message = f"{wav_path}\n{mp3_path}"
                status_name = f"{wav_path.name} + {mp3_path.name}"
            else:
                saved = export_mix_with_gains(self.separation_result, gains, path)
                saved_message = str(saved)
                status_name = saved.name
        except Exception as exc:
            messagebox.showerror("Separación IA", f"No se pudo exportar la mezcla:\n\n{exc}", parent=self)
            return

        self.separation_status_var.set(f"Mezcla exportada: {status_name}")
        self.audio_status_var.set(f"Separación IA: mezcla exportada {status_name}")
        messagebox.showinfo("Separación IA", f"Mezcla exportada:\n{saved_message}", parent=self)



    def _load_karaoke_project_pvk(self) -> None:
        path = filedialog.askopenfilename(
            title="Abrir proyecto Pitch Viewer Karaoke",
            filetypes=[("Pitch Viewer Karaoke", "*.pvk"), ("Todos los archivos", "*.*")],
            parent=self,
        )
        if not path:
            return

        try:
            project = load_pvk(path)
        except Exception as exc:
            messagebox.showerror("Karaoke", f"No se pudo cargar el .pvk:\n\n{exc}", parent=self)
            return

        self.stop_audio()
        self.karaoke_project = project
        self.karaoke_audio = None
        self.karaoke_segments = list(project.note_segments)
        self.karaoke_lyrics_lines = parse_lrc(project.lyrics_lrc) if project.lyrics_lrc.strip() else []
        self.offline_mode = True
        self.offline_duration_s = float(project.audio_info.duration_s or self._infer_karaoke_duration())
        self.offline_cursor_s = 0.0
        self.points.clear()
        self.current_point = None
        self._reset_karaoke_score()

        lyrics_text = project.lyrics_lrc if project.lyrics_lrc.strip() else project.lyrics_text
        self._set_karaoke_lyrics_text(lyrics_text or "Proyecto .pvk sin letra.")
        self.karaoke_timeline_var.set(0.0)
        try:
            self.karaoke_scale.configure(to=max(0.001, self.offline_duration_s))
        except Exception:
            pass

        title = project.title or Path(path).stem
        artist = f" - {project.artist}" if project.artist else ""
        self.karaoke_status_var.set(
            f"Karaoke play: {title}{artist} | {len(self.karaoke_segments)} targets"
        )
        self.audio_status_var.set("Karaoke: .pvk cargado; usa ▶ Play para cantar contra el target")
        self._set_karaoke_progress(1.0, "Progreso: .pvk cargado")
        self._show_karaoke_panel()
        self._update_karaoke_panel_state(force=True)

    def _infer_karaoke_duration(self) -> float:
        durations: list[float] = []
        if self.karaoke_project is not None:
            if self.karaoke_project.frames:
                durations.append(max(frame.time_s for frame in self.karaoke_project.frames))
            if self.karaoke_project.note_segments:
                durations.append(max(segment.end_s for segment in self.karaoke_project.note_segments))
        if self.karaoke_segments:
            durations.append(max(segment.end_s for segment in self.karaoke_segments))
        return max(durations) if durations else 0.0

    def _start_karaoke_play(self) -> None:
        if not self.karaoke_segments:
            messagebox.showinfo(
                "Karaoke play",
                "Carga un proyecto .pvk o analiza una pista vocal antes de iniciar karaoke play.",
                parent=self,
            )
            return
        if self.is_running:
            return

        duration = self.karaoke_audio.duration_s if self.karaoke_audio is not None else self.offline_duration_s
        if duration <= 0:
            duration = self._infer_karaoke_duration()
        self.offline_duration_s = max(0.0, float(duration))

        start_cursor = self.offline_cursor_s if self.offline_mode or self.karaoke_play_paused else 0.0
        start_cursor = max(0.0, min(self.offline_duration_s, start_cursor))

        if not self.karaoke_play_paused:
            self._reset_karaoke_score()
            self.points.clear()
            self.current_point = None

        self.karaoke_play_active = True
        self.karaoke_play_paused = False
        self.offline_mode = False
        self.offline_cursor_s = start_cursor
        self.paused_audio_elapsed_s = start_cursor
        self.is_audio_paused = False

        self._start_live_capture(mode="online")
        if self.is_running:
            self.time_origin = time.perf_counter() - start_cursor
            self.capture_mode = "karaoke_play"
            self.audio_status_var.set(f"Karaoke play: evaluando desde {self._format_time(start_cursor)}")
            self.karaoke_status_var.set("Karaoke play: en curso")

    def _reset_karaoke_score(self) -> None:
        self.karaoke_score = {
            "evaluated": 0,
            "hits": 0,
            "misses": 0,
            "no_voice": 0,
            "rest": 0,
            "sum_abs_cents": 0.0,
            "sum_signed_cents": 0.0,
        }
        self.karaoke_score_var.set("Karaoke play: —")

    def _active_karaoke_segment_at(self, time_s: float) -> Optional[KaraokeNoteSegment]:
        if not self.karaoke_segments:
            return None
        t = float(time_s)
        for segment in self.karaoke_segments:
            if segment.start_s <= t <= segment.end_s:
                return segment
            if segment.start_s > t:
                break
        return None

    def _evaluate_karaoke_point(self, point: PitchPoint) -> None:
        if not self.karaoke_play_active:
            return

        segment = self._active_karaoke_segment_at(point.time_s)
        if segment is None:
            self.karaoke_score["rest"] += 1
            return

        if not point.voiced or math.isnan(point.midi_float):
            self.karaoke_score["no_voice"] += 1
            self._update_karaoke_score_label()
            return

        error_cents = 100.0 * (float(point.midi_float) - float(segment.midi))
        abs_error = abs(error_cents)

        self.karaoke_score["evaluated"] += 1
        self.karaoke_score["sum_abs_cents"] += abs_error
        self.karaoke_score["sum_signed_cents"] += error_cents

        with self.settings_lock:
            tolerance = float(self.settings.tolerance_cents)
        if abs_error <= tolerance:
            self.karaoke_score["hits"] += 1
        else:
            self.karaoke_score["misses"] += 1

        self._update_karaoke_score_label()

    def _update_karaoke_score_label(self) -> None:
        evaluated = int(self.karaoke_score.get("evaluated", 0))
        hits = int(self.karaoke_score.get("hits", 0))
        misses = int(self.karaoke_score.get("misses", 0))
        no_voice = int(self.karaoke_score.get("no_voice", 0))
        if evaluated <= 0:
            self.karaoke_score_var.set(
                f"Karaoke play: sin frames evaluables | sin voz target: {no_voice}"
            )
            return

        accuracy = 100.0 * hits / max(1, evaluated)
        mean_abs = float(self.karaoke_score.get("sum_abs_cents", 0.0)) / max(1, evaluated)
        mean_signed = float(self.karaoke_score.get("sum_signed_cents", 0.0)) / max(1, evaluated)
        if mean_signed > 5.0:
            tendency = "alto"
        elif mean_signed < -5.0:
            tendency = "bajo"
        else:
            tendency = "centrado"
        self.karaoke_score_var.set(
            f"Karaoke play: {accuracy:0.1f}% dentro de tolerancia | "
            f"error medio {mean_signed:+0.1f}c | abs {mean_abs:0.1f}c | "
            f"hits {hits} / fallos {misses} / sin voz {no_voice} | tendencia {tendency}"
        )

    def _build_karaoke_panel(self, parent: tk.Widget) -> None:
        self.karaoke_panel = ttk.LabelFrame(parent, text="Karaoke producción / play", padding=8)

        controls = ttk.Frame(self.karaoke_panel)
        controls.pack(fill=tk.X, side=tk.TOP)

        ttk.Button(controls, text="Abrir .pvk...", command=self._load_karaoke_project_pvk).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="▶ Play", command=self._start_karaoke_play).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(controls, text="⏸ Pausa", command=self.pause_audio).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(controls, text="⏹ Stop", command=self.stop_audio).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(controls, text="Abrir audio...", command=self._load_karaoke_audio).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Importar letra...", command=self._import_karaoke_lyrics).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Analizar pista", command=self._analyze_karaoke_audio).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Guardar .pvk...", command=self._save_karaoke_project).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(controls, textvariable=self.karaoke_status_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(controls, textvariable=self.karaoke_time_var).pack(side=tk.RIGHT)

        score_row = ttk.Frame(self.karaoke_panel)
        score_row.pack(fill=tk.X, side=tk.TOP, pady=(8, 0))
        ttk.Label(score_row, textvariable=self.karaoke_score_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        progress_row = ttk.Frame(self.karaoke_panel)
        progress_row.pack(fill=tk.X, side=tk.TOP, pady=(8, 0))
        ttk.Label(progress_row, textvariable=self.karaoke_progress_text_var, width=34, anchor="w").pack(side=tk.LEFT, padx=(0, 8))
        self.karaoke_progress_bar = ttk.Progressbar(
            progress_row,
            orient=tk.HORIZONTAL,
            mode="determinate",
            variable=self.karaoke_progress_var,
            maximum=100.0,
        )
        self.karaoke_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        timeline = ttk.Frame(self.karaoke_panel)
        timeline.pack(fill=tk.X, side=tk.TOP, pady=(8, 4))
        ttk.Button(timeline, text="⏮", width=4, command=self._jump_karaoke_start).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(timeline, text="⏪", width=4, command=self._transport_backward).pack(side=tk.LEFT, padx=(0, 4))
        self.karaoke_scale = ttk.Scale(
            timeline,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.karaoke_timeline_var,
            command=self._set_karaoke_cursor_from_slider,
        )
        self.karaoke_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(timeline, text="⏩", width=4, command=self._transport_forward).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(timeline, text="⏭", width=4, command=self._jump_karaoke_end).pack(side=tk.LEFT)

        self.karaoke_lyrics_panel = ttk.LabelFrame(self.main_view, text="Letra", padding=8)
        ttk.Label(
            self.karaoke_lyrics_panel,
            textvariable=self.karaoke_current_lyric_var,
            anchor="center",
            wraplength=300,
        ).pack(fill=tk.X, pady=(0, 6))

        self.karaoke_lyrics_text = tk.Text(
            self.karaoke_lyrics_panel,
            width=34,
            height=18,
            wrap=tk.WORD,
        )
        self.karaoke_lyrics_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        lyrics_scrollbar = ttk.Scrollbar(
            self.karaoke_lyrics_panel,
            orient=tk.VERTICAL,
            command=self.karaoke_lyrics_text.yview,
        )
        lyrics_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.karaoke_lyrics_text.configure(yscrollcommand=lyrics_scrollbar.set)
        self.karaoke_lyrics_text.insert("1.0", "Carga una pista vocal y, opcionalmente, una letra .txt o .lrc.")
        self.karaoke_lyrics_text.configure(state=tk.DISABLED)

    def _toggle_karaoke_panel(self) -> None:
        if self.karaoke_panel_visible:
            self.karaoke_panel.pack_forget()
            self.karaoke_lyrics_panel.pack_forget()
            self.karaoke_panel_visible = False
            return
        self._show_karaoke_panel()

    def _show_karaoke_panel(self) -> None:
        if not self.karaoke_panel_visible:
            self.karaoke_panel.pack(fill=tk.X, side=tk.BOTTOM, before=self.footer, pady=(8, 0))
            self.karaoke_lyrics_panel.pack(fill=tk.Y, side=tk.RIGHT, padx=(8, 0))
            self.karaoke_panel_visible = True

    def _format_time(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        rem = seconds - 60 * minutes
        return f"{minutes:02d}:{rem:06.3f}"

    def _load_karaoke_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Abrir pista vocal para karaoke",
            filetypes=[
                ("Audio", "*.wav *.flac *.ogg *.mp3 *.m4a *.mp4"),
                ("WAV", "*.wav"),
                ("Todos los archivos", "*.*"),
            ],
            parent=self,
        )
        if not path:
            return

        try:
            loaded = load_audio_file(path, selected_channel="mix")
        except Exception as exc:
            messagebox.showerror(
                "Karaoke",
                f"No se pudo cargar el audio:\n\n{exc}\n\n"
                "Recomendación para esta etapa: usa WAV. Para MP3/MP4 instala ffmpeg y deja ffmpeg en PATH.",
                parent=self,
            )
            return

        channel_mode = loaded.selected_channel
        if loaded.channels > 1:
            answer = simpledialog.askstring(
                "Canal de audio",
                "El archivo tiene más de un canal. Escribe una opción:\n\n"
                "mix      = promedio mono\n"
                "left     = canal izquierdo\n"
                "right    = canal derecho\n"
                "max_rms  = canal con mayor RMS",
                initialvalue="mix",
                parent=self,
            )
            if answer:
                channel_mode = answer.strip().lower()
                try:
                    loaded = load_audio_file(path, selected_channel=channel_mode)
                except Exception as exc:
                    messagebox.showerror("Karaoke", f"No se pudo recargar con ese canal:\n\n{exc}", parent=self)
                    return

        self.karaoke_audio = loaded
        self.karaoke_segments = []
        self.karaoke_lyrics_lines = []
        self.karaoke_project = KaraokeProject(
            title=os.path.splitext(loaded.filename)[0],
            artist="",
            audio_info=KaraokeAudioInfo(
                path=loaded.path,
                filename=loaded.filename,
                sample_rate=loaded.sample_rate,
                channels=loaded.channels,
                selected_channel=loaded.selected_channel,
                duration_s=loaded.duration_s,
            ),
        )

        self.points.clear()
        self.current_point = None
        self.offline_mode = True
        self.offline_duration_s = loaded.duration_s
        self.offline_cursor_s = 0.0
        self.karaoke_timeline_var.set(0.0)
        try:
            self.karaoke_scale.configure(to=max(0.001, loaded.duration_s))
        except Exception:
            pass

        self.karaoke_status_var.set(
            f"Audio: {loaded.filename} | {loaded.sample_rate} Hz | {loaded.duration_s:.1f}s | canal {loaded.selected_channel}"
        )
        self._set_karaoke_progress(0.0, "Progreso: pista cargada, falta analizar")
        self.audio_status_var.set("Karaoke: audio cargado; falta analizar pista")
        self._show_karaoke_panel()
        self._update_karaoke_panel_state(force=True)

    def _import_karaoke_lyrics(self) -> None:
        if self.karaoke_project is None:
            self.karaoke_project = KaraokeProject(title="Proyecto karaoke")

        path = filedialog.askopenfilename(
            title="Importar letra",
            filetypes=[("Letras", "*.txt *.lrc"), ("Todos los archivos", "*.*")],
            parent=self,
        )
        if not path:
            return

        try:
            lyrics_text, lyrics_lrc, lines = load_lyrics_file(path)
        except Exception as exc:
            messagebox.showerror("Karaoke", f"No se pudo cargar la letra:\n\n{exc}", parent=self)
            return

        self.karaoke_project.lyrics_text = lyrics_text
        self.karaoke_project.lyrics_lrc = lyrics_lrc
        self.karaoke_lyrics_lines = lines
        self._set_karaoke_lyrics_text(lyrics_lrc if lyrics_lrc.strip() else lyrics_text)
        self.karaoke_status_var.set(f"Letra importada: {os.path.basename(path)}")
        self._show_karaoke_panel()
        self._update_karaoke_panel_state(force=True)

    def _set_karaoke_lyrics_text(self, text: str) -> None:
        self.karaoke_lyrics_text.configure(state=tk.NORMAL)
        self.karaoke_lyrics_text.delete("1.0", tk.END)
        self.karaoke_lyrics_text.insert("1.0", text or "Sin letra cargada.")
        self.karaoke_lyrics_text.configure(state=tk.DISABLED)

    def _analyze_karaoke_audio(self) -> None:
        if self.karaoke_audio is None:
            messagebox.showinfo("Karaoke", "Primero carga una pista vocal.", parent=self)
            return
        if self.karaoke_analysis_running:
            messagebox.showinfo("Karaoke", "Ya hay un análisis karaoke en curso.", parent=self)
            return

        audio = np.asarray(self.karaoke_audio.audio, dtype=np.float32)
        sample_rate = int(self.karaoke_audio.sample_rate)
        if audio.size < max(1024, sample_rate // 4):
            messagebox.showerror("Karaoke", "El audio es demasiado corto para analizar.", parent=self)
            return

        self.karaoke_analysis_running = True
        self._set_karaoke_progress(0.0, "Progreso: preparando análisis")
        with self.settings_lock:
            backend_id = normalize_backend_id(self.settings.offline_detector_backend)
        self.audio_status_var.set(
            f"Karaoke: analizando con {backend_label(backend_id)} | {audio.size / max(1, sample_rate):.1f}s"
        )
        self.karaoke_status_var.set("Analizando pista vocal...")
        self._show_karaoke_panel()

        thread = threading.Thread(
            target=self._karaoke_analysis_worker,
            args=(audio.copy(), sample_rate),
            daemon=True,
        )
        thread.start()

    def _karaoke_analysis_worker(self, audio: np.ndarray, sample_rate: int) -> None:
        def progress_callback(fraction: float, message: str) -> None:
            safe_fraction = max(0.0, min(1.0, float(fraction)))
            self.karaoke_analysis_queue.put((
                "progress",
                {"fraction": safe_fraction, "message": str(message)},
            ))

        try:
            progress_callback(0.01, "Preparando detector offline")
            points, info = self._analyze_audio_offline(audio, sample_rate, progress_callback=progress_callback)
            progress_callback(0.90, "Convirtiendo frames de pitch")
            with self.settings_lock:
                settings = AppSettings(**self.settings.__dict__)
            frames = pitch_points_to_frames(points)
            progress_callback(0.94, "Construyendo segmentos de notas")
            segments = build_note_segments(points, settings)
            progress_callback(0.98, "Preparando proyecto .pvk")
            snapshot = settings_snapshot(settings)
            payload = (points, frames, segments, snapshot, info)
            self.karaoke_analysis_queue.put(("ok", payload))
        except Exception as exc:
            self.karaoke_analysis_queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _consume_karaoke_analysis_results(self) -> None:
        while True:
            try:
                status, payload = self.karaoke_analysis_queue.get_nowait()
            except queue.Empty:
                break

            if status == "progress":
                data = payload if isinstance(payload, dict) else {}
                fraction = float(data.get("fraction", 0.0))
                message = str(data.get("message", "Analizando"))
                self._set_karaoke_progress(fraction, message)
                self.karaoke_status_var.set(message)
                continue

            self.karaoke_analysis_running = False
            if status == "error":
                self._set_karaoke_progress(0.0, "Progreso: error de análisis")
                self.karaoke_status_var.set("Karaoke: error de análisis")
                self.audio_status_var.set("Karaoke: error de análisis")
                messagebox.showerror("Karaoke", f"No se pudo analizar la pista:\n\n{payload}", parent=self)
                return

            points, frames, segments, snapshot, info = payload  # type: ignore[misc]
            self._set_karaoke_progress(1.0, "Progreso: análisis completo")
            self._apply_karaoke_analysis_result(points, frames, segments, snapshot, info)

    def _set_karaoke_progress(self, fraction: float, message: str) -> None:
        percent = max(0.0, min(100.0, 100.0 * float(fraction)))
        self.karaoke_progress_var.set(percent)
        if percent <= 0.0:
            self.karaoke_progress_text_var.set(str(message))
        elif percent >= 100.0:
            self.karaoke_progress_text_var.set(f"{message} | 100%")
        else:
            self.karaoke_progress_text_var.set(f"{message} | {percent:0.1f}%")

    def _apply_karaoke_analysis_result(self, points, frames, segments, snapshot, info) -> None:
        if self.karaoke_audio is None:
            return
        if self.karaoke_project is None:
            self.karaoke_project = KaraokeProject(title=os.path.splitext(self.karaoke_audio.filename)[0])

        self.karaoke_project.audio_info = KaraokeAudioInfo(
            path=self.karaoke_audio.path,
            filename=self.karaoke_audio.filename,
            sample_rate=self.karaoke_audio.sample_rate,
            channels=self.karaoke_audio.channels,
            selected_channel=self.karaoke_audio.selected_channel,
            duration_s=self.karaoke_audio.duration_s,
        )
        self.karaoke_project.frames = list(frames)
        self.karaoke_project.note_segments = list(segments)
        self.karaoke_project.settings_snapshot = dict(snapshot)
        self.karaoke_segments = list(segments)

        self.points = deque(points, maxlen=max(20000, len(points) + 100))
        self.offline_mode = True
        self.offline_duration_s = self.karaoke_audio.duration_s
        self.offline_cursor_s = 0.0
        self.current_point = self._point_at_time(0.0)
        self.karaoke_timeline_var.set(0.0)
        try:
            self.karaoke_scale.configure(to=max(0.001, self.offline_duration_s))
        except Exception:
            pass

        self.karaoke_status_var.set(
            f"Karaoke: {len(points)} frames | {len(segments)} segmentos | {self._offline_info_label(info)}"
        )
        self.audio_status_var.set("Karaoke: análisis listo; puedes guardar .pvk")
        self._update_karaoke_panel_state(force=True)

    def _save_karaoke_project(self) -> None:
        if self.karaoke_project is None or not self.karaoke_project.frames:
            messagebox.showinfo("Karaoke", "No hay proyecto analizado para guardar. Carga audio y analiza la pista.", parent=self)
            return

        title = simpledialog.askstring(
            "Título",
            "Título de la canción:",
            initialvalue=self.karaoke_project.title or "Canción",
            parent=self,
        )
        if title is not None:
            self.karaoke_project.title = title.strip() or self.karaoke_project.title

        artist = simpledialog.askstring(
            "Artista",
            "Artista/intérprete (opcional):",
            initialvalue=self.karaoke_project.artist,
            parent=self,
        )
        if artist is not None:
            self.karaoke_project.artist = artist.strip()

        path = filedialog.asksaveasfilename(
            title="Guardar proyecto Pitch Viewer Karaoke",
            defaultextension=".pvk",
            filetypes=[("Pitch Viewer Karaoke", "*.pvk"), ("Todos los archivos", "*.*")],
            initialfile=f"{self.karaoke_project.title or 'cancion'}.pvk",
            parent=self,
        )
        if not path:
            return

        try:
            saved = save_pvk(self.karaoke_project, path)
        except Exception as exc:
            messagebox.showerror("Karaoke", f"No se pudo guardar el .pvk:\n\n{exc}", parent=self)
            return

        self.karaoke_status_var.set(f"Guardado: {saved.name}")
        self.audio_status_var.set(f"Karaoke: guardado {saved}")
        messagebox.showinfo("Karaoke", f"Proyecto guardado:\n{saved}", parent=self)

    def _export_karaoke_segments_csv(self) -> None:
        if not self.karaoke_segments:
            messagebox.showinfo("Karaoke", "No hay segmentos para exportar.", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="Exportar segmentos karaoke CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")],
            parent=self,
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["start_s", "end_s", "duration_s", "midi", "note", "mean_cents", "confidence", "rms", "frame_count"])
                for segment in self.karaoke_segments:
                    writer.writerow([
                        f"{segment.start_s:.6f}",
                        f"{segment.end_s:.6f}",
                        f"{segment.duration_s:.6f}",
                        int(segment.midi),
                        segment.note,
                        f"{segment.mean_cents:.6f}",
                        f"{segment.confidence:.6f}",
                        f"{segment.rms:.6f}",
                        int(segment.frame_count),
                    ])
        except Exception as exc:
            messagebox.showerror("Karaoke", f"No se pudo exportar CSV:\n\n{exc}", parent=self)
            return

        messagebox.showinfo("Karaoke", f"Segmentos exportados:\n{path}", parent=self)

    def _set_karaoke_cursor_from_slider(self, value) -> None:
        if self.karaoke_timeline_updating:
            return
        try:
            cursor = float(value)
        except Exception:
            return
        duration = self.karaoke_audio.duration_s if self.karaoke_audio is not None else self.offline_duration_s
        if duration <= 0:
            return
        self.offline_mode = True
        self.offline_duration_s = max(self.offline_duration_s, duration)
        self.offline_cursor_s = max(0.0, min(duration, cursor))
        self.current_point = self._point_at_time(self.offline_cursor_s)
        self._update_karaoke_panel_state(force=True)

    def _jump_karaoke_start(self) -> None:
        self.offline_mode = True
        self.offline_cursor_s = 0.0
        self.current_point = self._point_at_time(0.0)
        self._update_karaoke_panel_state(force=True)

    def _jump_karaoke_end(self) -> None:
        duration = self.karaoke_audio.duration_s if self.karaoke_audio is not None else self.offline_duration_s
        self.offline_mode = True
        self.offline_duration_s = duration
        self.offline_cursor_s = duration
        self.current_point = self._point_at_time(duration)
        self._update_karaoke_panel_state(force=True)

    def _update_karaoke_panel_state(self, force: bool = False) -> None:
        duration = self.karaoke_audio.duration_s if self.karaoke_audio is not None else self.offline_duration_s
        if duration <= 0:
            self.karaoke_time_var.set("00:00.000 / 00:00.000")
            self.karaoke_current_lyric_var.set("")
            return

        cursor = self.offline_cursor_s if self.offline_mode else self._display_time_s()
        if self.karaoke_play_active and duration > 0 and cursor >= duration:
            self.stop_audio()
            cursor = duration
        cursor = max(0.0, min(duration, cursor))
        self.karaoke_time_var.set(f"{self._format_time(cursor)} / {self._format_time(duration)}")

        if self.karaoke_lyrics_lines:
            self.karaoke_current_lyric_var.set(current_lyric_line(self.karaoke_lyrics_lines, cursor))
        else:
            self.karaoke_current_lyric_var.set("")

        if self.karaoke_panel_visible or force:
            try:
                self.karaoke_timeline_updating = True
                self.karaoke_scale.configure(to=max(0.001, duration))
                self.karaoke_timeline_var.set(cursor)
            finally:
                self.karaoke_timeline_updating = False

    def _draw_karaoke_segments(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        palette: dict[str, str],
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        visible_min_midi: int,
        visible_max_midi: int,
        midi_to_y: Callable[[float], float],
    ) -> None:
        if not self.karaoke_segments:
            return

        now = self._display_time_s()
        window_s = float(settings.time_window_s)
        visible_start = now - window_s
        plot_width = plot_right - plot_left
        tolerance = settings.tolerance_cents / 100.0

        target_fill = "#2563eb" if settings.theme_name == "dark" else "#60a5fa"
        target_outline = "#93c5fd" if settings.theme_name == "dark" else "#1d4ed8"

        for segment in self.karaoke_segments:
            if segment.end_s < visible_start or segment.start_s > now:
                continue
            if not (visible_min_midi <= segment.midi <= visible_max_midi):
                continue

            x1 = plot_left + ((max(segment.start_s, visible_start) - visible_start) / window_s) * plot_width
            x2 = plot_left + ((min(segment.end_s, now) - visible_start) / window_s) * plot_width
            if x2 - x1 < 2.0:
                x2 = x1 + 2.0
            y1 = max(plot_top, min(plot_bottom, midi_to_y(segment.midi + tolerance)))
            y2 = max(plot_top, min(plot_bottom, midi_to_y(segment.midi - tolerance)))

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=target_fill,
                outline=target_outline,
                width=1,
                stipple="gray25",
            )
            if x2 - x1 >= 44:
                canvas.create_text(
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                    text=segment.note,
                    fill=palette["text"],
                    font=("TkDefaultFont", 8, "bold"),
                    anchor="center",
                )

    def _show_backend_runtime_report(self) -> None:
        try:
            from .detection.offline_torchcrepe import torchcrepe_available

            available, error = torchcrepe_available()
        except Exception as exc:
            available = False
            error = f"{type(exc).__name__}: {exc}"

        cuda_text = "sí" if self.cuda_available else "no"
        realtime_full = "habilitado" if self.torchcrepe_full_realtime_enabled else "deshabilitado"
        torchcrepe_text = "disponible" if available else f"no disponible: {error}"
        demucs_text = "disponible" if is_demucs_available() else "no disponible"
        device_name = self.runtime_info.cuda_device_name or "—"

        messagebox.showinfo(
            "Estado runtime / CUDA",
            "Estado de backends pesados:\n\n"
            f"Torch: {self.runtime_info.torch_version}\n"
            f"CUDA detectada: {cuda_text}\n"
            f"Dispositivo CUDA: {device_name}\n"
            f"Torchcrepe: {torchcrepe_text}\n"
            f"Torchcrepe full en vivo: {realtime_full}\n"
            f"Demucs: {demucs_text}\n\n"
            "Regla actual:\n"
            "- La detección CUDA se realiza una vez al iniciar la app y queda cacheada.\n"
            "- Torchcrepe full se puede usar en vivo solo con CUDA.\n"
            "- El backend offline se elige libremente desde Audio > Backend offline / record.\n"
            "- Separación IA usa CUDA si está disponible; si no, CPU.",
            parent=self,
        )

    def _apply_calibration_preset(self, preset_name: str) -> None:
        preset = str(preset_name).strip().lower()

        if preset == "usb":
            values = {
                "confidence_threshold": 0.30,
                "rms_threshold": 0.004,
                "smoothing_factor": 0.30,
                "median_window": 5,
                "max_jump_semitones": 8.0,
                "octave_guard": True,
                "detector_backend": BACKEND_YIN_CMND,
            }
            label = "micrófono USB / baja latencia"
        elif preset == "laptop":
            values = {
                "confidence_threshold": 0.50,
                "rms_threshold": 0.012,
                "smoothing_factor": 0.45,
                "median_window": 7,
                "max_jump_semitones": 5.0,
                "octave_guard": True,
                "detector_backend": BACKEND_YIN_CMND,
            }
            label = "micrófono integrado / más filtro"
        elif preset == "vibrato":
            values = {
                "confidence_threshold": 0.35,
                "rms_threshold": 0.006,
                "smoothing_factor": 0.18,
                "median_window": 3,
                "max_jump_semitones": 12.0,
                "octave_guard": True,
                "detector_backend": BACKEND_YIN_CMND,
            }
            label = "vibrato o glissando"
        elif preset == "noisy":
            values = {
                "confidence_threshold": 0.58,
                "rms_threshold": 0.018,
                "smoothing_factor": 0.50,
                "median_window": 9,
                "max_jump_semitones": 4.0,
                "octave_guard": True,
                "detector_backend": BACKEND_YIN_CMND,
            }
            label = "ambiente ruidoso"
        else:
            return

        was_running = self.is_running
        if was_running:
            self.stop_audio()

        with self.settings_lock:
            for key, value in values.items():
                setattr(self.settings, key, value)
            self.settings = normalize_settings(self.settings)

        self.recent_midi_values = deque(maxlen=max(1, int(self.settings.median_window)))
        self.detector_backend_var.set(self.settings.detector_backend)
        self.offline_detector_backend_var.set(self.settings.offline_detector_backend)
        self.active_backend_id = normalize_backend_id(self.settings.detector_backend)
        self._refresh_status_labels()
        self._rebuild_menu()
        self._autosave_settings()

        self.calibration_visual_info = {
            "message": f"Preset aplicado: {label}",
            "detail": f"conf {self.settings.confidence_threshold:.2f} | RMS {self.settings.rms_threshold:.4f}",
            "min_midi": int(self.settings.min_midi),
            "max_midi": int(self.settings.max_midi),
            "applied": True,
        }
        self.calibration_visual_until_s = time.perf_counter() + 10.0

        if was_running:
            self.start_audio()

        messagebox.showinfo(
            "Preset de calibración",
            f"Preset aplicado: {label}\n\n"
            f"Backend vivo: {backend_label(self.settings.detector_backend)}\n"
            f"Confianza mínima: {self.settings.confidence_threshold:.2f}\n"
            f"RMS mínimo: {self.settings.rms_threshold:.4f}\n"
            f"Suavizado: {self.settings.smoothing_factor:.2f}\n"
            f"Mediana: {self.settings.median_window}\n"
            f"Salto máximo: {self.settings.max_jump_semitones:.1f} semitonos",
            parent=self,
        )

    def _start_voice_calibration_diagnostic(self) -> None:
        if sd is None:
            messagebox.showerror(
                "Diagnóstico de micrófono",
                "No está instalado sounddevice. Ejecuta: pip install sounddevice",
                parent=self,
            )
            return

        if self.calibration_running:
            messagebox.showinfo(
                "Diagnóstico de micrófono",
                "Ya hay un diagnóstico en curso.",
                parent=self,
            )
            return

        if self.is_running:
            messagebox.showinfo(
                "Diagnóstico de micrófono",
                "Detén o pausa la captura antes de iniciar el diagnóstico.",
                parent=self,
            )
            return

        if self.selected_device_index is None:
            self._load_audio_devices(select_default=True)

        if self.selected_device_index is None:
            messagebox.showerror(
                "Diagnóstico de micrófono",
                "No hay dispositivo de entrada seleccionado.",
                parent=self,
            )
            return

        duration_s = simpledialog.askfloat(
            "Diagnóstico de micrófono y voz",
            "Duración de la toma en segundos. Canta una nota sostenida y luego cambia de nota:",
            initialvalue=5.0,
            minvalue=2.0,
            maxvalue=15.0,
            parent=self,
        )
        if duration_s is None:
            return

        try:
            device_info = sd.query_devices(self.selected_device_index, "input")
            sample_rate = int(float(device_info.get("default_samplerate", 44100)))
        except Exception:
            sample_rate = 44100

        self.calibration_running = True
        self.audio_status_var.set(f"Calibración: grabando {duration_s:.1f}s")
        self._set_calibration_visual_message(
            "Calibración: grabando voz",
            f"Duración {duration_s:.1f}s | canta una nota y cambia de nota",
            seconds=float(duration_s) + 3.0,
        )

        thread = threading.Thread(
            target=self._voice_calibration_worker,
            args=(float(duration_s), int(sample_rate), int(self.selected_device_index)),
            daemon=True,
        )
        thread.start()

    def _voice_calibration_worker(self, duration_s: float, sample_rate: int, device_index: int) -> None:
        try:
            assert sd is not None
            sample_count = max(1, int(round(duration_s * sample_rate)))
            audio = sd.rec(
                sample_count,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device_index,
            )
            sd.wait()
            mono = np.asarray(audio[:, 0], dtype=np.float32)
            result = self._analyze_calibration_audio(mono, sample_rate)
            self.calibration_queue.put(("ok", result))
        except Exception as exc:
            self.calibration_queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _analyze_calibration_audio(self, audio: np.ndarray, sample_rate: int) -> dict[str, object]:
        from .detection.registry import create_pitch_detector

        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            raise RuntimeError("No se capturó audio.")

        rms_global = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        duration_s = float(x.size) / float(max(1, sample_rate))

        frame_size = max(4096, int(round(sample_rate * 0.12)))
        hop_size = max(256, int(round(sample_rate * 0.05)))
        detector = create_pitch_detector(BACKEND_YIN_CMND, sample_rate, frame_size)

        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        detect_min_hz = max(MIN_DETECTABLE_HZ, midi_to_freq(MIN_MIDI_CHOICE, settings.a4_hz))
        detect_max_hz = min(MAX_DETECTABLE_HZ, midi_to_freq(MAX_MIDI_CHOICE, settings.a4_hz))

        freqs: list[float] = []
        confs: list[float] = []
        rmss: list[float] = []
        midis: list[float] = []
        voiced_count = 0
        total_count = 0

        if x.size < frame_size:
            x = np.pad(x, (0, frame_size - x.size), mode="constant")

        for end_idx in range(frame_size, x.size + 1, hop_size):
            frame = x[end_idx - frame_size : end_idx]
            total_count += 1
            frame_rms = float(np.sqrt(np.mean(frame * frame)))
            estimate = detector.estimate(frame, detect_min_hz, detect_max_hz)
            freq = float(estimate.freq_hz)
            conf = float(estimate.confidence)

            if detect_min_hz <= freq <= detect_max_hz and frame_rms >= 0.001 and conf >= 0.15:
                voiced_count += 1
                freqs.append(freq)
                confs.append(conf)
                rmss.append(frame_rms)
                midis.append(freq_to_midi(freq, settings.a4_hz))

        if midis:
            midi_arr = np.asarray(midis, dtype=np.float64)
            conf_arr = np.asarray(confs, dtype=np.float64)
            rms_arr = np.asarray(rmss, dtype=np.float64)
            recommended_min = int(max(MIN_MIDI_CHOICE, math.floor(float(np.percentile(midi_arr, 5)) - 6)))
            recommended_max = int(min(MAX_MIDI_CHOICE, math.ceil(float(np.percentile(midi_arr, 95)) + 6)))
            recommended_conf = float(max(0.20, min(0.75, np.percentile(conf_arr, 25) * 0.75)))
            recommended_rms = float(max(0.0020, min(0.0800, np.percentile(rms_arr, 20) * 0.35)))
            midi_p50 = float(np.percentile(midi_arr, 50))
            freq_p50 = float(np.percentile(np.asarray(freqs, dtype=np.float64), 50))
            conf_p50 = float(np.percentile(conf_arr, 50))
            rms_p50 = float(np.percentile(rms_arr, 50))
        else:
            recommended_min = self.settings.min_midi
            recommended_max = self.settings.max_midi
            recommended_conf = self.settings.confidence_threshold
            recommended_rms = self.settings.rms_threshold
            midi_p50 = float("nan")
            freq_p50 = float("nan")
            conf_p50 = 0.0
            rms_p50 = 0.0

        if recommended_max <= recommended_min:
            recommended_max = min(MAX_MIDI_CHOICE, recommended_min + 12)

        voiced_ratio = float(voiced_count) / float(max(1, total_count))
        clipping = peak >= 0.98
        too_low = rms_global < 0.01
        too_noisy_or_unvoiced = voiced_ratio < 0.20

        notes = []
        if clipping:
            notes.append("La señal parece saturar. Baja la ganancia o aléjate del micrófono.")
        if too_low:
            notes.append("La señal parece baja. Sube la ganancia o acércate al micrófono.")
        if too_noisy_or_unvoiced:
            notes.append("Hay pocos frames con voz clara. Revisa ruido, distancia o umbrales.")
        if not notes:
            notes.append("La señal parece utilizable para análisis de pitch.")

        return {
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "rms_global": rms_global,
            "peak": peak,
            "frame_count": total_count,
            "voiced_count": voiced_count,
            "voiced_ratio": voiced_ratio,
            "freq_p50": freq_p50,
            "midi_p50": midi_p50,
            "conf_p50": conf_p50,
            "rms_p50": rms_p50,
            "recommended_min_midi": recommended_min,
            "recommended_max_midi": recommended_max,
            "recommended_confidence": recommended_conf,
            "recommended_rms": recommended_rms,
            "notes": notes,
        }

    def _consume_calibration_results(self) -> None:
        while True:
            try:
                status, payload = self.calibration_queue.get_nowait()
            except queue.Empty:
                break

            self.calibration_running = False

            if status == "error":
                self.audio_status_var.set("Calibración: error")
                messagebox.showerror(
                    "Diagnóstico de micrófono",
                    f"No se pudo ejecutar el diagnóstico:\n\n{payload}",
                    parent=self,
                )
                return

            result = payload  # type: ignore[assignment]
            self._show_calibration_result(result)  # type: ignore[arg-type]

    def _show_calibration_result(self, result: dict[str, object]) -> None:
        with self.settings_lock:
            language = self.settings.note_language

        min_midi = int(result["recommended_min_midi"])
        max_midi = int(result["recommended_max_midi"])
        min_note = midi_to_note_name(min_midi, language)
        max_note = midi_to_note_name(max_midi, language)
        midi_p50 = float(result["midi_p50"])
        if math.isnan(midi_p50):
            note_p50 = "—"
        else:
            note_p50 = midi_to_note_name(int(round(midi_p50)), language)

        notes_text = "\n".join(f"- {line}" for line in result["notes"])  # type: ignore[index]
        summary = (
            "Resultado del diagnóstico:\n\n"
            f"Duración: {float(result['duration_s']):.1f}s | Sample rate: {int(result['sample_rate'])} Hz\n"
            f"RMS global: {float(result['rms_global']):.4f} | Peak: {float(result['peak']):.4f}\n"
            f"Frames con voz clara: {int(result['voiced_count'])}/{int(result['frame_count'])} "
            f"({100.0 * float(result['voiced_ratio']):.1f}%)\n"
            f"Nota mediana detectada: {note_p50}\n"
            f"Confianza p50: {float(result['conf_p50']):.3f} | RMS p50: {float(result['rms_p50']):.4f}\n\n"
            "Recomendación:\n"
            f"- Rango visible: {min_note} - {max_note}\n"
            f"- Confianza mínima: {float(result['recommended_confidence']):.2f}\n"
            f"- RMS mínimo: {float(result['recommended_rms']):.4f}\n\n"
            f"Notas:\n{notes_text}\n\n"
            "¿Aplicar rango y umbrales recomendados?"
        )

        self.audio_status_var.set("Calibración: resultado listo")
        self._set_calibration_visual_result(result, applied=False)
        apply = messagebox.askyesno("Diagnóstico de micrófono y voz", summary, parent=self)
        if not apply:
            return

        with self.settings_lock:
            self.settings.min_midi = min_midi
            self.settings.max_midi = max_midi
            self.settings.confidence_threshold = float(result["recommended_confidence"])
            self.settings.rms_threshold = float(result["recommended_rms"])
            self.settings = normalize_settings(self.settings)

        self.dynamic_center_midi = None
        self._set_calibration_visual_result(result, applied=True)
        self._refresh_status_labels()
        self._autosave_settings()
        self.audio_status_var.set("Calibración: recomendaciones aplicadas")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "Acerca de",
            f"Monitor de afinación vocal - v{APP_VERSION}\n\n"
            "Incluye karaoke producción/play, separación IA offline, motores Demucs y Audio Separator / UVR, y exportación WAV/MP3.\n"
            "Si no hay CUDA, los procesos offline pueden correr en CPU; CUDA solo acelera.\n"
            "Para MP3/M4A/MP4 y exportar MP3 se requiere ffmpeg en PATH.\n\n"
            f"Archivo de configuración:\n{self.settings_path}",
        )

    def _on_close(self) -> None:
        self._save_settings_now(show_message=False)
        self.is_recording = False
        self._stop_stream_only()
        self.destroy()
