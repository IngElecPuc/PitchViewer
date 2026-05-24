# -*- coding: utf-8 -*-

"""Diálogos Tkinter de configuración."""

from typing import Optional

import tkinter as tk
from tkinter import messagebox, ttk

from ..config.settings import AppSettings
from ..models import InputDevice
from ..music.notes import build_note_choices, build_note_to_midi, midi_to_note_name


class RangeDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        settings: AppSettings,
    ) -> None:
        super().__init__(parent)

        self.title("Rango visible por nota")
        self.transient(parent)
        self.resizable(False, False)
        self.result: Optional[tuple[int, int]] = None
        self.settings = settings

        self.note_choices = build_note_choices(settings.note_language)
        self.note_to_midi = build_note_to_midi(settings.note_language)

        self.min_note_var = tk.StringVar(
            value=midi_to_note_name(settings.min_midi, settings.note_language)
        )
        self.max_note_var = tk.StringVar(
            value=midi_to_note_name(settings.max_midi, settings.note_language)
        )

        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="Nota inferior:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Combobox(
            body,
            textvariable=self.min_note_var,
            values=self.note_choices,
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="w", pady=6)

        ttk.Label(body, text="Nota superior:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Combobox(
            body,
            textvariable=self.max_note_var,
            values=self.note_choices,
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", pady=6)

        buttons = ttk.Frame(body)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))

        ttk.Button(buttons, text="Cancelar", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Aceptar", command=self._accept).pack(side=tk.RIGHT)

        self.bind("<Escape>", lambda _event: self._cancel())
        self.bind("<Return>", lambda _event: self._accept())

        self.grab_set()
        self.update_idletasks()
        self._center_over_parent(parent)
        self.wait_window(self)

    def _center_over_parent(self, parent: tk.Tk) -> None:
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_reqwidth()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_reqheight()) // 2)
        self.geometry(f"+{x}+{y}")

    def _accept(self) -> None:
        min_midi = self.note_to_midi.get(self.min_note_var.get())
        max_midi = self.note_to_midi.get(self.max_note_var.get())

        if min_midi is None or max_midi is None:
            messagebox.showerror("Rango inválido", "Selecciona dos notas válidas.", parent=self)
            return

        if max_midi <= min_midi:
            messagebox.showerror(
                "Rango inválido",
                "La nota superior debe estar por encima de la nota inferior.",
                parent=self,
            )
            return

        self.result = (min_midi, max_midi)
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class InputDeviceDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        devices: list[InputDevice],
        selected_index: Optional[int],
    ) -> None:
        super().__init__(parent)

        self.title("Fuente de entrada")
        self.transient(parent)
        self.resizable(True, False)
        self.devices = devices
        self.result: Optional[int] = None

        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="Selecciona el dispositivo de entrada:").pack(anchor="w", pady=(0, 6))

        self.listbox = tk.Listbox(body, height=10, width=78, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        for device in devices:
            self.listbox.insert(tk.END, device.label)

        if devices:
            selection = 0
            for idx, device in enumerate(devices):
                if device.index == selected_index:
                    selection = idx
                    break

            self.listbox.selection_set(selection)
            self.listbox.see(selection)

        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(buttons, text="Cancelar", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Aceptar", command=self._accept).pack(side=tk.RIGHT)

        self.listbox.bind("<Double-Button-1>", lambda _event: self._accept())
        self.bind("<Escape>", lambda _event: self._cancel())
        self.bind("<Return>", lambda _event: self._accept())

        self.grab_set()
        self.update_idletasks()
        self._center_over_parent(parent)
        self.wait_window(self)

    def _center_over_parent(self, parent: tk.Tk) -> None:
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_reqwidth()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_reqheight()) // 2)
        self.geometry(f"+{x}+{y}")

    def _accept(self) -> None:
        selection = self.listbox.curselection()

        if not selection:
            messagebox.showerror("Entrada no válida", "Selecciona un dispositivo.", parent=self)
            return

        self.result = self.devices[int(selection[0])].index
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class DetectorSettingsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, settings: AppSettings) -> None:
        super().__init__(parent)

        self.title("Parámetros de detección")
        self.transient(parent)
        self.resizable(False, False)
        self.result: Optional[tuple[float, float, float]] = None

        self.confidence_var = tk.StringVar(value=f"{settings.confidence_threshold:.3f}")
        self.rms_var = tk.StringVar(value=f"{settings.rms_threshold:.4f}")
        self.smoothing_var = tk.StringVar(value=f"{settings.smoothing_factor:.3f}")

        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        rows = [
            ("Confianza mínima:", self.confidence_var, "0.00 a 1.00"),
            ("RMS mínimo:", self.rms_var, "por ejemplo 0.004 a 0.020"),
            ("Suavizado visual:", self.smoothing_var, "0.00 a 1.00"),
        ]

        for row, (label, var, hint) in enumerate(rows):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=5)
            ttk.Entry(body, textvariable=var, width=12).grid(row=row, column=1, sticky="w", pady=5)
            ttk.Label(body, text=hint).grid(row=row, column=2, sticky="w", padx=(8, 0), pady=5)

        buttons = ttk.Frame(body)
        buttons.grid(row=len(rows), column=0, columnspan=3, sticky="e", pady=(12, 0))

        ttk.Button(buttons, text="Cancelar", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Aceptar", command=self._accept).pack(side=tk.RIGHT)

        self.bind("<Escape>", lambda _event: self._cancel())
        self.bind("<Return>", lambda _event: self._accept())

        self.grab_set()
        self.update_idletasks()
        self._center_over_parent(parent)
        self.wait_window(self)

    def _center_over_parent(self, parent: tk.Tk) -> None:
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_reqwidth()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_reqheight()) // 2)
        self.geometry(f"+{x}+{y}")

    def _accept(self) -> None:
        try:
            confidence = float(self.confidence_var.get())
            rms = float(self.rms_var.get())
            smoothing = float(self.smoothing_var.get())
        except ValueError:
            messagebox.showerror("Valor inválido", "Todos los valores deben ser numéricos.", parent=self)
            return

        if not 0.0 <= confidence <= 1.0:
            messagebox.showerror("Valor inválido", "La confianza debe estar entre 0 y 1.", parent=self)
            return

        if not 0.0 <= rms <= 1.0:
            messagebox.showerror("Valor inválido", "El RMS debe estar entre 0 y 1.", parent=self)
            return

        if not 0.0 <= smoothing <= 1.0:
            messagebox.showerror("Valor inválido", "El suavizado debe estar entre 0 y 1.", parent=self)
            return

        self.result = (confidence, rms, smoothing)
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class StabilitySettingsDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, settings: AppSettings) -> None:
        super().__init__(parent)

        self.title("Estabilidad de pitch")
        self.transient(parent)
        self.resizable(False, False)
        self.result: Optional[tuple[int, float, bool]] = None

        self.median_var = tk.StringVar(value=str(settings.median_window))
        self.max_jump_var = tk.StringVar(value=f"{settings.max_jump_semitones:.1f}")
        self.octave_guard_var = tk.BooleanVar(value=settings.octave_guard)

        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="Ventana de mediana:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Entry(body, textvariable=self.median_var, width=12).grid(row=0, column=1, sticky="w", pady=5)
        ttk.Label(body, text="frames; recomendado: 3, 5 o 7").grid(row=0, column=2, sticky="w", padx=(8, 0), pady=5)

        ttk.Label(body, text="Salto máximo:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Entry(body, textvariable=self.max_jump_var, width=12).grid(row=1, column=1, sticky="w", pady=5)
        ttk.Label(body, text="semitonos por frame; recomendado: 5 a 9").grid(row=1, column=2, sticky="w", padx=(8, 0), pady=5)

        ttk.Checkbutton(
            body,
            text="Corregir saltos falsos de octava",
            variable=self.octave_guard_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 4))

        hint = ttk.Label(
            body,
            text=(
                "La mediana reduce temblores del detector. La guardia de octava busca \"plegar\" "
                "saltos ±12 semitonos cuando el frame anterior sugiere que son errores."
            ),
            wraplength=520,
        )
        hint.grid(row=3, column=0, columnspan=3, sticky="we", pady=(4, 0))

        buttons = ttk.Frame(body)
        buttons.grid(row=4, column=0, columnspan=3, sticky="e", pady=(12, 0))

        ttk.Button(buttons, text="Cancelar", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Aceptar", command=self._accept).pack(side=tk.RIGHT)

        self.bind("<Escape>", lambda _event: self._cancel())
        self.bind("<Return>", lambda _event: self._accept())

        self.grab_set()
        self.update_idletasks()
        self._center_over_parent(parent)
        self.wait_window(self)

    def _center_over_parent(self, parent: tk.Tk) -> None:
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_reqwidth()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_reqheight()) // 2)
        self.geometry(f"+{x}+{y}")

    def _accept(self) -> None:
        try:
            median_window = int(self.median_var.get())
            max_jump = float(self.max_jump_var.get())
        except ValueError:
            messagebox.showerror("Valor inválido", "Los valores deben ser numéricos.", parent=self)
            return

        if median_window < 1 or median_window > 21:
            messagebox.showerror("Valor inválido", "La ventana de mediana debe estar entre 1 y 21.", parent=self)
            return

        if median_window % 2 == 0:
            median_window += 1

        if not 1.0 <= max_jump <= 24.0:
            messagebox.showerror("Valor inválido", "El salto máximo debe estar entre 1 y 24 semitonos.", parent=self)
            return

        self.result = (median_window, max_jump, bool(self.octave_guard_var.get()))
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


