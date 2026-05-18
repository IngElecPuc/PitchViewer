# -*- coding: utf-8 -*-

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


A4_HZ = 440.0

NOTE_NAMES_ES = [
    "Do",
    "Do♯",
    "Re",
    "Mi♭",
    "Mi",
    "Fa",
    "Fa♯",
    "Sol",
    "Sol♯",
    "La",
    "Si♭",
    "Si",
]

DEFAULT_MIN_MIDI = 40  # Mi2
DEFAULT_MAX_MIDI = 84  # Do6

MIN_DETECTABLE_HZ = 50.0
MAX_DETECTABLE_HZ = 1100.0


@dataclass
class PitchPoint:
    time_s: float
    freq_hz: float
    midi_float: float
    confidence: float
    rms: float
    voiced: bool


def freq_to_midi(freq_hz: float, a4_hz: float = A4_HZ) -> float:
    return 69.0 + 12.0 * math.log2(freq_hz / a4_hz)


def midi_to_freq(midi_value: float, a4_hz: float = A4_HZ) -> float:
    return a4_hz * (2.0 ** ((midi_value - 69.0) / 12.0))


def midi_to_note_name_es(midi_value: int) -> str:
    note = NOTE_NAMES_ES[midi_value % 12]
    octave = (midi_value // 12) - 1
    return f"{note}{octave}"


def cents_from_nearest_note(midi_float: float) -> float:
    return 100.0 * (midi_float - round(midi_float))


def build_note_choices(min_midi: int = 24, max_midi: int = 96) -> list[str]:
    return [midi_to_note_name_es(midi) for midi in range(min_midi, max_midi + 1)]


NOTE_CHOICES = build_note_choices()
NOTE_TO_MIDI = {
    midi_to_note_name_es(midi): midi
    for midi in range(0, 128)
}


class AutoCorrelationPitchDetector:
    """
    Detector simple de F0 para el MVP.

    Usa una ventana temporal, autocorrelación por FFT y selección de peaks.
    No es tan robusto como YIN/pYIN/CREPE, pero evita aubio y no requiere
    compilar extensiones C en Windows.
    """

    def __init__(
        self,
        sample_rate: int,
        window_size: int = 4096,
        min_freq_hz: float = MIN_DETECTABLE_HZ,
        max_freq_hz: float = MAX_DETECTABLE_HZ,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.buffer = np.zeros(window_size, dtype=np.float32)
        self.hann = np.hanning(window_size).astype(np.float32)

    def process(self, samples: np.ndarray) -> tuple[float, float]:
        samples = np.asarray(samples, dtype=np.float32)

        if samples.size >= self.window_size:
            self.buffer[:] = samples[-self.window_size:]
        else:
            self.buffer[:-samples.size] = self.buffer[samples.size:]
            self.buffer[-samples.size:] = samples

        x = self.buffer.astype(np.float64, copy=True)
        x -= np.mean(x)

        rms = float(np.sqrt(np.mean(x * x)))
        if rms <= 1e-8:
            return 0.0, 0.0

        x *= self.hann

        n = x.size
        fft_size = 1 << ((2 * n - 1).bit_length())

        spectrum = np.fft.rfft(x, n=fft_size)
        autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[:n]

        if autocorr[0] <= 1e-12:
            return 0.0, 0.0

        autocorr = autocorr / autocorr[0]

        min_lag = max(1, int(self.sample_rate / self.max_freq_hz))
        max_lag = min(n - 2, int(self.sample_rate / self.min_freq_hz))

        if max_lag <= min_lag + 2:
            return 0.0, 0.0

        region = autocorr[min_lag:max_lag + 1]

        left = region[1:-1] > region[:-2]
        right = region[1:-1] >= region[2:]
        peak_offsets = np.where(left & right)[0] + 1

        if peak_offsets.size == 0:
            return 0.0, 0.0

        peak_values = region[peak_offsets]
        best_value = float(np.max(peak_values))

        if best_value < 0.15:
            return 0.0, best_value

        candidate_mask = peak_values >= max(0.15, 0.80 * best_value)
        candidate_offsets = peak_offsets[candidate_mask]

        # Tomamos el primer peak fuerte. Esto favorece la fundamental percibida
        # frente a lags dobles que producen errores de octava hacia abajo.
        lag = int(min_lag + candidate_offsets[0])

        if 1 <= lag < len(autocorr) - 1:
            y0 = float(autocorr[lag - 1])
            y1 = float(autocorr[lag])
            y2 = float(autocorr[lag + 1])
            denom = y0 - 2.0 * y1 + y2

            if abs(denom) > 1e-12:
                correction = 0.5 * (y0 - y2) / denom
                correction = max(-0.5, min(0.5, correction))
                lag_float = lag + correction
            else:
                lag_float = float(lag)
        else:
            lag_float = float(lag)

        freq_hz = self.sample_rate / lag_float

        if not (self.min_freq_hz <= freq_hz <= self.max_freq_hz):
            return 0.0, best_value

        confidence = max(0.0, min(1.0, float(autocorr[lag])))
        return float(freq_hz), confidence


class PitchViewerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Monitor de afinación vocal - MVP")
        self.geometry("1100x760")
        self.minsize(880, 560)

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)
        self.pitch_queue: queue.Queue[PitchPoint] = queue.Queue(maxsize=200)

        self.stream = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.is_running = False
        self.time_origin = time.perf_counter()

        self.points: deque[PitchPoint] = deque(maxlen=10000)
        self.current_point: Optional[PitchPoint] = None

        self.sample_rate = 44100
        self.hop_size = 1024
        self.buffer_size = 4096
        self.pitch_detector: Optional[AutoCorrelationPitchDetector] = None

        self.last_smoothed_midi: Optional[float] = None

        self.confidence_threshold = 0.35
        self.rms_threshold = 0.008
        self.smoothing_factor = 0.35

        self.device_var = tk.StringVar()
        self.time_window_var = tk.StringVar(value="10")
        self.min_note_var = tk.StringVar(value=midi_to_note_name_es(DEFAULT_MIN_MIDI))
        self.max_note_var = tk.StringVar(value=midi_to_note_name_es(DEFAULT_MAX_MIDI))
        self.confidence_var = tk.StringVar(value="0.35")
        self.rms_var = tk.StringVar(value="0.008")

        self.note_status_var = tk.StringVar(value="Nota: —")
        self.freq_status_var = tk.StringVar(value="Frecuencia: —")
        self.cents_status_var = tk.StringVar(value="Desviación: —")
        self.conf_status_var = tk.StringVar(value="Confianza: —")
        self.audio_status_var = tk.StringVar(value="Audio: detenido")

        self.device_entries: list[str] = []

        self._build_ui()
        self._load_audio_devices()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._ui_loop)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(controls, text="Entrada:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.device_combo = ttk.Combobox(
            controls,
            textvariable=self.device_var,
            state="readonly",
            width=56,
        )
        self.device_combo.grid(row=0, column=1, sticky="we", padx=(0, 8))

        ttk.Button(controls, text="Actualizar", command=self._load_audio_devices).grid(
            row=0,
            column=2,
            padx=(0, 8),
        )

        self.start_button = ttk.Button(controls, text="Iniciar", command=self.start_audio)
        self.start_button.grid(row=0, column=3, padx=(0, 4))

        self.stop_button = ttk.Button(controls, text="Detener", command=self.stop_audio)
        self.stop_button.grid(row=0, column=4, padx=(0, 12))

        ttk.Label(controls, text="Ventana:").grid(row=0, column=5, sticky="e", padx=(0, 4))
        self.window_combo = ttk.Combobox(
            controls,
            textvariable=self.time_window_var,
            values=["5", "10", "20", "30"],
            state="readonly",
            width=5,
        )
        self.window_combo.grid(row=0, column=6, sticky="w")
        ttk.Label(controls, text="s").grid(row=0, column=7, sticky="w", padx=(2, 12))

        ttk.Label(controls, text="Rango:").grid(row=0, column=8, sticky="e", padx=(0, 4))
        self.min_note_combo = ttk.Combobox(
            controls,
            textvariable=self.min_note_var,
            values=NOTE_CHOICES,
            state="readonly",
            width=8,
        )
        self.min_note_combo.grid(row=0, column=9, padx=(0, 2))

        ttk.Label(controls, text="a").grid(row=0, column=10, padx=(2, 2))

        self.max_note_combo = ttk.Combobox(
            controls,
            textvariable=self.max_note_var,
            values=NOTE_CHOICES,
            state="readonly",
            width=8,
        )
        self.max_note_combo.grid(row=0, column=11, padx=(2, 12))

        ttk.Label(controls, text="Conf.:").grid(row=0, column=12, sticky="e", padx=(0, 4))
        ttk.Entry(controls, textvariable=self.confidence_var, width=6).grid(
            row=0,
            column=13,
            padx=(0, 8),
        )

        ttk.Label(controls, text="RMS:").grid(row=0, column=14, sticky="e", padx=(0, 4))
        ttk.Entry(controls, textvariable=self.rms_var, width=7).grid(
            row=0,
            column=15,
            padx=(0, 0),
        )

        controls.columnconfigure(1, weight=1)

        status = ttk.Frame(root)
        status.pack(fill=tk.X, side=tk.TOP, pady=(8, 6))

        ttk.Label(status, textvariable=self.note_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(status, textvariable=self.freq_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(status, textvariable=self.cents_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(status, textvariable=self.conf_status_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(status, textvariable=self.audio_status_var).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            root,
            bg="#172026",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        footer = ttk.Label(
            root,
            text=(
                "MVP: detección monofónica de F0 por autocorrelación. "
                "Usa audífonos para evitar que el micrófono capture los parlantes."
            ),
            anchor="w",
        )
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

    def _load_audio_devices(self) -> None:
        if sd is None:
            self.device_entries = []
            self.device_combo["values"] = []
            self.device_var.set("")
            self.audio_status_var.set("Audio: falta instalar sounddevice")
            return

        try:
            devices = sd.query_devices()
        except Exception as exc:
            messagebox.showerror("Error de audio", f"No se pudieron consultar los dispositivos:\n{exc}")
            return

        entries = []
        default_input_idx = None

        try:
            default_input_idx = sd.default.device[0]
        except Exception:
            default_input_idx = None

        for idx, device in enumerate(devices):
            max_inputs = int(device.get("max_input_channels", 0))
            if max_inputs <= 0:
                continue

            name = str(device.get("name", "Dispositivo sin nombre"))
            default_sr = int(float(device.get("default_samplerate", 44100)))
            label = f"{idx}: {name} ({max_inputs} ch, {default_sr} Hz)"
            entries.append(label)

        self.device_entries = entries
        self.device_combo["values"] = entries

        if not entries:
            self.device_var.set("")
            self.audio_status_var.set("Audio: no hay dispositivos de entrada")
            return

        selected = None

        if default_input_idx is not None and default_input_idx >= 0:
            prefix = f"{default_input_idx}:"
            for entry in entries:
                if entry.startswith(prefix):
                    selected = entry
                    break

        if selected is None:
            selected = entries[0]

        self.device_var.set(selected)
        self.audio_status_var.set("Audio: dispositivo listo")

    def _selected_device_index(self) -> Optional[int]:
        value = self.device_var.get().strip()

        if not value:
            return None

        try:
            return int(value.split(":", 1)[0])
        except Exception:
            return None

    def start_audio(self) -> None:
        if sd is None:
            messagebox.showerror(
                "Dependencia faltante",
                "No está instalado sounddevice.\n\nEjecuta:\n\npip install sounddevice",
            )
            return

        device_index = self._selected_device_index()

        if device_index is None:
            messagebox.showerror("Entrada no válida", "Selecciona un dispositivo de entrada.")
            return

        self.stop_audio()

        self._sync_runtime_settings()

        try:
            device_info = sd.query_devices(device_index, "input")
            self.sample_rate = int(float(device_info.get("default_samplerate", 44100)))
        except Exception:
            self.sample_rate = 44100

        self.hop_size = 1024
        self.buffer_size = 4096

        self.audio_queue = queue.Queue(maxsize=20)
        self.pitch_queue = queue.Queue(maxsize=200)

        self.points.clear()
        self.current_point = None
        self.last_smoothed_midi = None

        self.time_origin = time.perf_counter()
        self.stop_event.clear()

        try:
            self.pitch_detector = AutoCorrelationPitchDetector(
                sample_rate=self.sample_rate,
                window_size=self.buffer_size,
                min_freq_hz=MIN_DETECTABLE_HZ,
                max_freq_hz=MAX_DETECTABLE_HZ,
            )

            self.worker_thread = threading.Thread(
                target=self._pitch_worker,
                daemon=True,
            )
            self.worker_thread.start()

            self.stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.hop_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()

            self.is_running = True
            self.audio_status_var.set(
                f"Audio: capturando a {self.sample_rate} Hz"
            )

        except Exception as exc:
            self.stop_audio()
            messagebox.showerror(
                "Error al iniciar audio",
                f"No se pudo iniciar la captura:\n\n{exc}",
            )

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
        if status:
            pass

        if indata is None or len(indata) == 0:
            return

        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()

        try:
            self.audio_queue.put_nowait(samples)
        except queue.Full:
            pass

    def _pitch_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                samples = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.pitch_detector is None:
                continue

            rms = float(np.sqrt(np.mean(samples * samples)))

            try:
                freq_hz, confidence = self.pitch_detector.process(samples)
            except Exception:
                continue

            voiced = (
                MIN_DETECTABLE_HZ <= freq_hz <= MAX_DETECTABLE_HZ
                and confidence >= self.confidence_threshold
                and rms >= self.rms_threshold
            )

            if voiced:
                midi_raw = freq_to_midi(freq_hz)

                if self.last_smoothed_midi is None:
                    midi_smooth = midi_raw
                elif abs(midi_raw - self.last_smoothed_midi) > 8.0:
                    midi_smooth = midi_raw
                else:
                    midi_smooth = (
                        self.smoothing_factor * midi_raw
                        + (1.0 - self.smoothing_factor) * self.last_smoothed_midi
                    )

                self.last_smoothed_midi = midi_smooth
            else:
                midi_smooth = float("nan")

                if rms < self.rms_threshold * 0.5:
                    self.last_smoothed_midi = None

            point = PitchPoint(
                time_s=time.perf_counter() - self.time_origin,
                freq_hz=freq_hz,
                midi_float=midi_smooth,
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

    def _ui_loop(self) -> None:
        self._sync_runtime_settings()
        self._consume_pitch_points()
        self._update_status()
        self._redraw_canvas()

        self.after(33, self._ui_loop)

    def _sync_runtime_settings(self) -> None:
        self.confidence_threshold = self._safe_float(
            self.confidence_var.get(),
            default=0.35,
            min_value=0.0,
            max_value=1.0,
        )

        self.rms_threshold = self._safe_float(
            self.rms_var.get(),
            default=0.008,
            min_value=0.0,
            max_value=1.0,
        )

    @staticmethod
    def _safe_float(
        value: str,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        try:
            result = float(value)
        except Exception:
            result = default

        if min_value is not None:
            result = max(min_value, result)

        if max_value is not None:
            result = min(max_value, result)

        return result

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
        max_history = max(90.0, self._get_time_window_s() * 4.0)

        while self.points and self.points[0].time_s < now - max_history:
            self.points.popleft()

    def _current_time_s(self) -> float:
        if self.is_running:
            return time.perf_counter() - self.time_origin

        if self.points:
            return self.points[-1].time_s

        return 0.0

    def _get_time_window_s(self) -> float:
        try:
            return float(self.time_window_var.get())
        except Exception:
            return 10.0

    def _get_visible_midi_range(self) -> tuple[int, int]:
        min_note = self.min_note_var.get()
        max_note = self.max_note_var.get()

        min_midi = NOTE_TO_MIDI.get(min_note, DEFAULT_MIN_MIDI)
        max_midi = NOTE_TO_MIDI.get(max_note, DEFAULT_MAX_MIDI)

        if max_midi <= min_midi:
            max_midi = min_midi + 12

        return min_midi, max_midi

    def _update_status(self) -> None:
        now = self._current_time_s()

        point = self.current_point

        if point is None or not point.voiced or now - point.time_s > 0.75:
            self.note_status_var.set("Nota: —")
            self.freq_status_var.set("Frecuencia: —")
            self.cents_status_var.set("Desviación: —")
            self.conf_status_var.set("Confianza: —")
            return

        nearest_midi = int(round(point.midi_float))
        note_name = midi_to_note_name_es(nearest_midi)
        cents = cents_from_nearest_note(point.midi_float)

        self.note_status_var.set(f"Nota: {note_name}")
        self.freq_status_var.set(f"Frecuencia: {point.freq_hz:7.2f} Hz")
        self.cents_status_var.set(f"Desviación: {cents:+6.1f} cents")
        self.conf_status_var.set(f"Confianza: {point.confidence:0.2f} | RMS: {point.rms:0.4f}")

    def _redraw_canvas(self) -> None:
        canvas = self.canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()

        if width <= 2 or height <= 2:
            return

        left_margin = 76
        right_margin = 24
        top_margin = 18
        bottom_margin = 34

        plot_left = left_margin
        plot_right = width - right_margin
        plot_top = top_margin
        plot_bottom = height - bottom_margin

        min_midi, max_midi = self._get_visible_midi_range()

        if max_midi <= min_midi:
            return

        def midi_to_y(midi_value: float) -> float:
            ratio = (midi_value - min_midi) / (max_midi - min_midi)
            return plot_bottom - ratio * (plot_bottom - plot_top)

        self._draw_pitch_grid(
            canvas=canvas,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            left_margin=left_margin,
            min_midi=min_midi,
            max_midi=max_midi,
            midi_to_y=midi_to_y,
        )

        self._draw_time_grid(
            canvas=canvas,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
        )

        self._draw_pitch_curve(
            canvas=canvas,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            min_midi=min_midi,
            max_midi=max_midi,
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
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        left_margin: int,
        min_midi: int,
        max_midi: int,
        midi_to_y,
    ) -> None:
        current_midi = None

        now = self._current_time_s()
        if self.current_point is not None and self.current_point.voiced:
            if now - self.current_point.time_s <= 0.75:
                current_midi = int(round(self.current_point.midi_float))

        semitone_height = abs(midi_to_y(min_midi + 1) - midi_to_y(min_midi))
        label_every = 1 if semitone_height >= 13 else 2

        for midi_value in range(min_midi, max_midi + 1):
            y_top = midi_to_y(midi_value + 0.5)
            y_bottom = midi_to_y(midi_value - 0.5)

            y1 = max(plot_top, min(plot_bottom, y_top))
            y2 = max(plot_top, min(plot_bottom, y_bottom))

            if y2 < plot_top or y1 > plot_bottom:
                continue

            pitch_class = midi_value % 12
            is_natural = pitch_class in {0, 2, 4, 5, 7, 9, 11}

            if midi_value == current_midi:
                fill = "#3a3822"
            elif is_natural:
                fill = "#202b31"
            else:
                fill = "#192329"

            canvas.create_rectangle(
                plot_left,
                y1,
                plot_right,
                y2,
                fill=fill,
                outline="",
            )

            label_fill = "#dfe7ee" if is_natural else "#c9d2da"
            canvas.create_rectangle(
                0,
                y1,
                left_margin,
                y2,
                fill=label_fill,
                outline="#1e2930",
            )

            center_y = midi_to_y(midi_value)

            canvas.create_line(
                plot_left,
                center_y,
                plot_right,
                center_y,
                fill="#2f3b43",
                width=1,
            )

            if (midi_value - min_midi) % label_every == 0:
                canvas.create_text(
                    left_margin - 8,
                    center_y,
                    text=midi_to_note_name_es(midi_value),
                    fill="#111820",
                    font=("TkDefaultFont", 9),
                    anchor="e",
                )

    def _draw_time_grid(
        self,
        canvas: tk.Canvas,
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
    ) -> None:
        now = self._current_time_s()
        window_s = self._get_time_window_s()
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
                canvas.create_line(
                    x,
                    plot_top,
                    x,
                    plot_bottom,
                    fill="#25323a",
                    width=1,
                )

                seconds_ago = now - tick
                label = f"-{seconds_ago:0.0f}s"

                canvas.create_text(
                    x,
                    plot_bottom + 16,
                    text=label,
                    fill="#9aa7b0",
                    font=("TkDefaultFont", 8),
                    anchor="n",
                )

            tick += step

    def _draw_pitch_curve(
        self,
        canvas: tk.Canvas,
        plot_left: int,
        plot_right: int,
        plot_top: int,
        plot_bottom: int,
        min_midi: int,
        max_midi: int,
        midi_to_y,
    ) -> None:
        now = self._current_time_s()
        window_s = self._get_time_window_s()
        visible_start = now - window_s

        plot_width = plot_right - plot_left

        segments: list[list[tuple[float, float]]] = []
        current_segment: list[tuple[float, float]] = []
        last_time: Optional[float] = None

        for point in self.points:
            if point.time_s < visible_start:
                continue

            if point.time_s > now:
                continue

            valid = (
                point.voiced
                and not math.isnan(point.midi_float)
                and min_midi - 0.5 <= point.midi_float <= max_midi + 0.5
            )

            if not valid:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                last_time = None
                continue

            ratio = (point.time_s - visible_start) / window_s
            x = plot_left + ratio * plot_width
            y = midi_to_y(point.midi_float)

            if last_time is not None and point.time_s - last_time > 0.25:
                if current_segment:
                    segments.append(current_segment)
                current_segment = []

            current_segment.append((x, y))
            last_time = point.time_s

        if current_segment:
            segments.append(current_segment)

        for segment in segments:
            if len(segment) >= 2:
                coords = []
                for x, y in segment:
                    coords.extend([x, y])

                canvas.create_line(
                    *coords,
                    fill="#f59e0b",
                    width=3,
                    smooth=True,
                    splinesteps=12,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                )

            elif len(segment) == 1:
                x, y = segment[0]
                canvas.create_oval(
                    x - 2,
                    y - 2,
                    x + 2,
                    y + 2,
                    fill="#f59e0b",
                    outline="",
                )

        if self.current_point is not None and self.current_point.voiced:
            if now - self.current_point.time_s <= 0.75:
                midi_value = self.current_point.midi_float

                if min_midi - 0.5 <= midi_value <= max_midi + 0.5:
                    y = midi_to_y(midi_value)

                    canvas.create_oval(
                        plot_right - 6,
                        y - 6,
                        plot_right + 6,
                        y + 6,
                        fill="#ffffff",
                        outline="#f59e0b",
                        width=2,
                    )

    def _on_close(self) -> None:
        self.stop_audio()
        self.destroy()


def main() -> None:
    app = PitchViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
