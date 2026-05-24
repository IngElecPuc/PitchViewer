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
    TIME_WINDOWS,
    TOLERANCE_OPTIONS,
)
from .detection.autocorrelation import estimate_pitch_autocorrelation
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

        self.title("Monitor de afinación vocal - Etapa 5")

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

        self.note_status_var = tk.StringVar(value="Nota: —")
        self.freq_status_var = tk.StringVar(value="Frecuencia: —")
        self.cents_status_var = tk.StringVar(value="Desviación: —")
        self.conf_status_var = tk.StringVar(value="Confianza: —")
        self.assessment_status_var = tk.StringVar(value="Estado: —")
        self.scale_status_var = tk.StringVar(value="Escala: cromática")
        self.range_status_var = tk.StringVar(value="Rango: —")
        self.audio_status_var = tk.StringVar(value="Audio: detenido")
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
        audio_menu.add_command(label="Iniciar captura", command=self.start_audio)
        audio_menu.add_command(label="Detener captura", command=self.stop_audio)
        audio_menu.add_separator()
        audio_menu.add_command(label="Fuente de entrada...", command=self._choose_input_device)
        audio_menu.add_command(label="Actualizar dispositivos", command=lambda: self._load_audio_devices(select_default=False))
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
            label="Mostrar bandas de tolerancia",
            variable=self.show_tolerance_bands_var,
            command=self._toggle_tolerance_bands,
        )
        view_menu.add_checkbutton(
            label="Mostrar centro exacto de nota",
            variable=self.show_center_lines_var,
            command=self._toggle_center_lines,
        )
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

        self.start_button = ttk.Button(toolbar, text="Iniciar", command=self.start_audio)
        self.start_button.pack(side=tk.LEFT, padx=(0, 4))

        self.stop_button = ttk.Button(toolbar, text="Detener", command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(toolbar, text="Fuente...", command=self._choose_input_device).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(toolbar, textvariable=self.device_status_var).pack(side=tk.LEFT, padx=(0, 16))
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

        footer = ttk.Label(
            root,
            text=(
                "Etapa 5: configuración persistente en settings.json; mantiene estabilización, tolerancia, evaluación de afinación y exportación CSV. "
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

        self.stop_audio()

        try:
            device_info = sd.query_devices(self.selected_device_index, "input")
            self.sample_rate = int(float(device_info.get("default_samplerate", 44100)))
        except Exception:
            self.sample_rate = 44100

        self.audio_queue = queue.Queue(maxsize=30)
        self.pitch_queue = queue.Queue(maxsize=300)
        self.stop_event.clear()
        self.points.clear()
        self.current_point = None
        self.last_smoothed_midi = None
        self.recent_midi_values = deque(maxlen=max(1, self.settings.median_window))
        self.time_origin = time.perf_counter()

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
            self.audio_status_var.set(f"Audio: capturando a {self.sample_rate} Hz")
            self._refresh_device_label()
            self._autosave_settings()

        except Exception as exc:
            self.stop_audio()
            messagebox.showerror("Error al iniciar audio", f"No se pudo iniciar la captura:\n\n{exc}")

    def stop_audio(self) -> None:
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

        self.audio_status_var.set("Audio: detenido")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if indata is None or len(indata) == 0:
            return

        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()

        try:
            self.audio_queue.put_nowait(samples)
        except queue.Full:
            pass

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
            freq_hz, confidence = estimate_pitch_autocorrelation(
                frame,
                self.sample_rate,
                detect_min_hz,
                detect_max_hz,
            )

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

    @staticmethod
    def _correct_octave_jump(midi_raw: float, previous_midi: float) -> float:
        candidates = [midi_raw + 12.0 * k for k in range(-3, 4)]
        return min(candidates, key=lambda value: abs(value - previous_midi))

    def _ui_loop(self) -> None:
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

        now = self._current_time_s()
        with self.settings_lock:
            max_history = max(90.0, self.settings.time_window_s * 4.0)

        while self.points and self.points[0].time_s < now - max_history:
            self.points.popleft()

    def _current_time_s(self) -> float:
        if self.is_running:
            return time.perf_counter() - self.time_origin

        if self.points:
            return self.points[-1].time_s

        return 0.0

    def _clear_history(self) -> None:
        self.points.clear()
        self.current_point = None
        self.last_smoothed_midi = None
        self.recent_midi_values.clear()

    def _set_time_window(self, seconds: int) -> None:
        with self.settings_lock:
            self.settings.time_window_s = int(seconds)
        self.time_window_var.set(int(seconds))
        self._refresh_status_labels()
        self._autosave_settings()

    def _set_visible_range(self, min_midi: int, max_midi: int) -> None:
        if max_midi <= min_midi:
            return

        with self.settings_lock:
            self.settings.min_midi = int(min_midi)
            self.settings.max_midi = int(max_midi)

        self._refresh_status_labels()
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
        cents = max(1, min(99, cents))

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
            maxvalue=99,
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
        self.settings_status_var.set(
            f"A4: {settings.a4_hz:.1f} Hz | Ventana: {settings.time_window_s}s | "
            f"Tolerancia: ±{settings.tolerance_cents} cents | Mediana: {settings.median_window}"
        )

    def _update_status(self) -> None:
        now = self._current_time_s()
        point = self.current_point

        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

        if point is None or not point.voiced or now - point.time_s > 0.75:
            self.note_status_var.set("Nota: —")
            self.freq_status_var.set("Frecuencia: —")
            self.cents_status_var.set("Desviación: —")
            self.conf_status_var.set("Confianza: —")
            self.assessment_status_var.set("Estado: —")
            return

        nearest_midi = int(round(point.midi_float))
        note_name = midi_to_note_name(nearest_midi, settings.note_language)
        cents = cents_from_nearest_note(point.midi_float)
        in_scale = nearest_midi % 12 in scale_pitch_classes(settings.scale_root, settings.scale_name)
        scale_suffix = "" if in_scale else " | fuera de escala"

        self.note_status_var.set(f"Nota: {note_name}{scale_suffix}")
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
            return "alto"
        return "bajo"

    def _redraw_canvas(self) -> None:
        canvas = self.canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()

        if width <= 2 or height <= 2:
            return

        with self.settings_lock:
            settings = AppSettings(**self.settings.__dict__)

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

        if settings.max_midi <= settings.min_midi:
            return

        def midi_to_y(midi_value: float) -> float:
            ratio = (midi_value - settings.min_midi) / (settings.max_midi - settings.min_midi)
            return plot_bottom - ratio * (plot_bottom - plot_top)

        self._draw_pitch_grid(
            canvas=canvas,
            settings=settings,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            left_margin=left_margin,
            midi_to_y=midi_to_y,
        )

        self._draw_time_grid(
            canvas=canvas,
            settings=settings,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
        )

        self._draw_pitch_curve(
            canvas=canvas,
            settings=settings,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            midi_to_y=midi_to_y,
        )

        canvas.create_line(
            plot_right,
            plot_top,
            plot_right,
            plot_bottom,
            fill="#d9e3ea",
            width=1,
        )

    def _draw_pitch_grid(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        left_margin: int,
        midi_to_y: Callable[[float], float],
    ) -> None:
        current_midi = None
        now = self._current_time_s()
        if self.current_point is not None and self.current_point.voiced:
            if now - self.current_point.time_s <= 0.75:
                current_midi = int(round(self.current_point.midi_float))

        semitone_height = abs(midi_to_y(settings.min_midi + 1) - midi_to_y(settings.min_midi))
        label_every = 1 if semitone_height >= 13 else 2 if semitone_height >= 7 else 3
        scale_pcs = scale_pitch_classes(settings.scale_root, settings.scale_name)
        tolerance = settings.tolerance_cents / 100.0

        for midi_value in range(settings.min_midi, settings.max_midi + 1):
            y_top = midi_to_y(midi_value + 0.5)
            y_bottom = midi_to_y(midi_value - 0.5)
            y1 = max(plot_top, min(plot_bottom, y_top))
            y2 = max(plot_top, min(plot_bottom, y_bottom))

            if y2 < plot_top or y1 > plot_bottom:
                continue

            pitch_class = midi_value % 12
            is_scale_note = pitch_class in scale_pcs
            is_natural = pitch_class in {0, 2, 4, 5, 7, 9, 11}

            if midi_value == current_midi:
                fill = "#3a3822"
            elif is_scale_note:
                fill = "#22313a" if is_natural else "#1d2a32"
            elif settings.show_out_of_scale:
                fill = "#151d22"
            else:
                fill = "#11181d"

            canvas.create_rectangle(plot_left, y1, plot_right, y2, fill=fill, outline="")

            if is_scale_note and settings.show_tolerance_bands:
                tolerance_top = midi_to_y(midi_value + tolerance)
                tolerance_bottom = midi_to_y(midi_value - tolerance)
                canvas.create_rectangle(
                    plot_left,
                    max(plot_top, min(plot_bottom, tolerance_top)),
                    plot_right,
                    max(plot_top, min(plot_bottom, tolerance_bottom)),
                    fill="#263f36",
                    outline="",
                )
                canvas.create_line(
                    plot_left,
                    max(plot_top, min(plot_bottom, tolerance_top)),
                    plot_right,
                    max(plot_top, min(plot_bottom, tolerance_top)),
                    fill="#335747",
                    width=1,
                    dash=(2, 4),
                )
                canvas.create_line(
                    plot_left,
                    max(plot_top, min(plot_bottom, tolerance_bottom)),
                    plot_right,
                    max(plot_top, min(plot_bottom, tolerance_bottom)),
                    fill="#335747",
                    width=1,
                    dash=(2, 4),
                )

            label_fill = "#dfe7ee" if is_scale_note else "#87929b"
            if not settings.show_out_of_scale and not is_scale_note:
                label_fill = "#53606a"

            canvas.create_rectangle(
                0,
                y1,
                left_margin,
                y2,
                fill=label_fill,
                outline="#1e2930",
            )

            center_y = midi_to_y(midi_value)
            center_line = "#52616b" if is_scale_note else "#2a333a"
            if settings.show_center_lines or is_scale_note:
                width = 2 if is_scale_note and settings.show_center_lines else 1
                canvas.create_line(plot_left, center_y, plot_right, center_y, fill=center_line, width=width)

            should_label = (midi_value - settings.min_midi) % label_every == 0
            if should_label and (settings.show_out_of_scale or is_scale_note):
                canvas.create_text(
                    left_margin - 8,
                    center_y,
                    text=midi_to_note_name(midi_value, settings.note_language),
                    fill="#111820",
                    font=("TkDefaultFont", 9),
                    anchor="e",
                )

    def _draw_time_grid(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
    ) -> None:
        now = self._current_time_s()
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
                canvas.create_line(x, plot_top, x, plot_bottom, fill="#25323a", width=1)
                seconds_ago = now - tick
                canvas.create_text(
                    x,
                    plot_bottom + 16,
                    text=f"-{seconds_ago:0.0f}s",
                    fill="#9aa7b0",
                    font=("TkDefaultFont", 8),
                    anchor="n",
                )

            tick += step

    def _draw_pitch_curve(
        self,
        canvas: tk.Canvas,
        settings: AppSettings,
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        midi_to_y: Callable[[float], float],
    ) -> None:
        now = self._current_time_s()
        window_s = float(settings.time_window_s)
        visible_start = now - window_s
        plot_width = plot_right - plot_left

        previous: Optional[tuple[PitchPoint, float, float]] = None
        scale_pcs = scale_pitch_classes(settings.scale_root, settings.scale_name)

        for point in self.points:
            if point.time_s < visible_start or point.time_s > now:
                continue

            valid = (
                point.voiced
                and not math.isnan(point.midi_float)
                and settings.min_midi - 0.5 <= point.midi_float <= settings.max_midi + 0.5
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
                    color = self._curve_color(point, settings, scale_pcs)
                    canvas.create_line(
                        px,
                        py,
                        x,
                        y,
                        fill=color,
                        width=3,
                        capstyle=tk.ROUND,
                        joinstyle=tk.ROUND,
                    )

            previous = (point, x, y)

        if self.current_point is not None and self.current_point.voiced:
            if now - self.current_point.time_s <= 0.75:
                midi_value = self.current_point.midi_float
                if settings.min_midi - 0.5 <= midi_value <= settings.max_midi + 0.5:
                    y = midi_to_y(midi_value)
                    color = self._curve_color(self.current_point, settings, scale_pcs)
                    canvas.create_oval(
                        plot_right - 6,
                        y - 6,
                        plot_right + 6,
                        y + 6,
                        fill="#ffffff",
                        outline=color,
                        width=2,
                    )
                    nearest = int(round(midi_value))
                    note_text = midi_to_note_name(nearest, settings.note_language)
                    cents = cents_from_nearest_note(midi_value)
                    label = f"{note_text} {cents:+.0f}c"
                    canvas.create_rectangle(
                        plot_right - 88,
                        y - 14,
                        plot_right - 10,
                        y + 14,
                        fill="#0f1720",
                        outline=color,
                        width=1,
                    )
                    canvas.create_text(
                        plot_right - 49,
                        y,
                        text=label,
                        fill="#e5edf3",
                        font=("TkDefaultFont", 9, "bold"),
                        anchor="center",
                    )

    @staticmethod
    def _curve_color(point: PitchPoint, settings: AppSettings, scale_pcs: set[int]) -> str:
        nearest_midi = int(round(point.midi_float))
        cents = abs(cents_from_nearest_note(point.midi_float))
        in_scale = nearest_midi % 12 in scale_pcs

        if not in_scale:
            return "#ef4444"

        if cents <= settings.tolerance_cents:
            return "#22c55e"

        return "#f59e0b"

    def _show_about(self) -> None:
        messagebox.showinfo(
            "Acerca de",
            "Monitor de afinación vocal - Etapa 5\n\n"
            "Backend actual: autocorrelación FFT con NumPy.\n"
            "Esta etapa agrega persistencia automática de configuración en settings.json.\n"
            "La configuración incluye escala, tonalidad, rango, tolerancia, A4, idioma, \n"
            "parámetros de detección, estabilidad, ventana y fuente de entrada.\n\n"
            f"Archivo de configuración:\n{self.settings_path}",
        )

    def _on_close(self) -> None:
        self._save_settings_now(show_message=False)
        self.stop_audio()
        self.destroy()
