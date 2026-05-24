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
from typing import Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

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

        self.title("Monitor de afinación vocal - v0.8.1")

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
        self.cuda_available = self._detect_cuda_available()
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

        self.visual_paused = False
        self.paused_display_time_s: Optional[float] = None
        self.dynamic_center_midi: Optional[float] = None
        self._range_drag_start_y: Optional[int] = None
        self._range_drag_start_min_midi: Optional[int] = None
        self._range_drag_start_max_midi: Optional[int] = None

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
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

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
        view_menu.add_command(label="Pausar/reanudar visualización", command=self._toggle_visual_pause)
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

        self.to_start_button = ttk.Button(toolbar, text="⏮", width=3, command=self._jump_offline_start)
        self.to_start_button.pack(side=tk.LEFT, padx=(0, 2))

        self.backward_button = ttk.Button(toolbar, text="⏪", width=3, command=self._transport_backward)
        self.backward_button.pack(side=tk.LEFT, padx=(0, 4))

        self.play_button = ttk.Button(toolbar, text="▶", width=3, command=self.start_audio)
        self.play_button.pack(side=tk.LEFT, padx=(0, 2))

        self.pause_button = ttk.Button(toolbar, text="⏸", width=3, command=self.pause_audio)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 2))

        self.stop_button = ttk.Button(toolbar, text="⏹", width=3, command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 2))

        self.record_button = ttk.Button(toolbar, text="⏺", width=3, command=self.start_recording)
        self.record_button.pack(side=tk.LEFT, padx=(0, 4))

        self.forward_button = ttk.Button(toolbar, text="⏩", width=3, command=self._transport_forward)
        self.forward_button.pack(side=tk.LEFT, padx=(0, 2))

        self.to_end_button = ttk.Button(toolbar, text="⏭", width=3, command=self._jump_offline_end)
        self.to_end_button.pack(side=tk.LEFT, padx=(0, 12))

        self.pause_view_button = ttk.Button(toolbar, text="Pausar vista", command=self._toggle_visual_pause)
        self.pause_view_button.pack(side=tk.LEFT, padx=(0, 12))

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

        self.canvas = tk.Canvas(
            root,
            bg="#172026",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._on_canvas_wheel_linux(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self._on_canvas_wheel_linux(event, -1))
        self.canvas.bind("<ButtonPress-2>", self._on_range_drag_start)
        self.canvas.bind("<B2-Motion>", self._on_range_drag_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_range_drag_end)
        self.canvas.bind("<ButtonPress-3>", self._on_range_drag_start)
        self.canvas.bind("<B3-Motion>", self._on_range_drag_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_range_drag_end)

        footer = ttk.Label(
            root,
            text=(
                "v0.8.1: calibración, backends en vivo/offline separados y análisis offline seleccionable. "
                "Usa audífonos para evitar que el micrófono capture los parlantes."
            ),
            anchor="w",
        )
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

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

        Si estaba pausada una grabación offline, ▶ reanuda esa grabación.
        En cualquier otro caso funciona como captura online.
        """
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
        self.pause_view_button.configure(text="Pausar vista")
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
        self._stop_stream_only()
        self.audio_status_var.set(f"Audio: pausado en {self.paused_audio_elapsed_s:.1f}s")

    def stop_audio(self) -> None:
        was_recording = bool(self.is_recording)
        self._stop_stream_only()
        self.is_audio_paused = False
        self.paused_audio_elapsed_s = 0.0
        self.capture_mode = "idle"

        if was_recording:
            self.is_recording = False
            self._finish_recording_and_analyze()
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

    def _analyze_audio_offline(self, audio: np.ndarray, sample_rate: int):
        """Analiza una grabación con el backend offline seleccionado.

        A diferencia del backend vivo, esta ruta no tiene presupuesto temporal
        por frame. Por eso Torchcrepe full puede correr en CPU si no hay CUDA,
        pero el usuario también puede elegir YIN, Autocorrelación o Torchcrepe tiny.
        """
        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        backend_id = normalize_backend_id(settings.offline_detector_backend)
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
            )

        return self._analyze_audio_offline_with_frame_detector(
            audio=audio,
            sample_rate=sample_rate,
            settings=settings,
            backend_id=backend_id,
            detect_min_hz=detect_min_hz,
            detect_max_hz=detect_max_hz,
        )

    def _analyze_audio_offline_with_torchcrepe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        settings: AppSettings,
        backend_id: str,
        detect_min_hz: float,
        detect_max_hz: float,
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
        )

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

        for start in range(0, max(1, x.size), hop_size):
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
        self._consume_offline_analysis_results()
        self._consume_calibration_results()
        self._consume_pitch_points()
        self._update_status()
        self._redraw_canvas()
        self.after(33, self._ui_loop)

    def _consume_pitch_points(self) -> None:
        while True:
            try:
                point = self.pitch_queue.get_nowait()
            except queue.Empty:
                break

            self.points.append(point)
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
            self.pause_view_button.configure(text="Pausar vista")
            return

        self.visual_paused = True
        self.paused_display_time_s = self._current_time_s()
        self.pause_view_button.configure(text="Reanudar vista")

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

    def _draw_legend(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        palette: dict[str, str],
        plot_left: int,
        plot_top: int,
    ) -> None:
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

        messagebox.showinfo(
            "Estado Torchcrepe / CUDA",
            "Estado de backends pesados:\n\n"
            f"Torchcrepe: {torchcrepe_text}\n"
            f"CUDA detectada: {cuda_text}\n"
            f"Torchcrepe full en vivo: {realtime_full}\n\n"
            "Regla actual:\n"
            "- Torchcrepe full se puede usar en vivo solo con CUDA.\n"
            "- El backend offline se elige libremente desde Audio > Backend offline / record.\n"
            "- Torchcrepe full se puede usar offline con ⏺ Record aunque no haya CUDA.\n"
            "- Torchcrepe tiny, YIN CMND y Autocorrelación siguen disponibles para tiempo real.",
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
        self._refresh_status_labels()
        self._autosave_settings()
        self.audio_status_var.set("Calibración: recomendaciones aplicadas")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "Acerca de",
            "Monitor de afinación vocal - v0.8.0\n\n"
            "Incluye calibración de micrófono/voz, presets de estabilidad y backends separados para vivo/offline.\n"
            "Si no hay CUDA, Torchcrepe full queda deshabilitado en vivo, pero sigue disponible como backend offline/record seleccionable.\n"
            "La configuración guarda también el backend seleccionado.\n\n"
            f"Archivo de configuración:\n{self.settings_path}",
        )

    def _on_close(self) -> None:
        self._save_settings_now(show_message=False)
        self.is_recording = False
        self._stop_stream_only()
        self.destroy()
