# -*- coding: utf-8 -*-

"""Guardar proyectos Pitch Viewer Karaoke (.pvk)."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

from .models import KaraokeProject, PVK_FORMAT_NAME, PVK_FORMAT_VERSION


def save_pvk(project: KaraokeProject, path: str) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".pvk":
        target = target.with_suffix(".pvk")

    manifest = project.manifest_dict()
    manifest["format"] = PVK_FORMAT_NAME
    manifest["format_version"] = PVK_FORMAT_VERSION

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=4, sort_keys=True))
        zf.writestr(
            "settings_snapshot.json",
            json.dumps(project.settings_snapshot, ensure_ascii=False, indent=4, sort_keys=True),
        )
        zf.writestr("pitch_frames.csv", _frames_csv(project))
        zf.writestr(
            "note_segments.json",
            json.dumps([asdict(segment) for segment in project.note_segments], ensure_ascii=False, indent=4),
        )
        if project.lyrics_text.strip():
            zf.writestr("lyrics.txt", project.lyrics_text)
        if project.lyrics_lrc.strip():
            zf.writestr("lyrics.lrc", project.lyrics_lrc)
    return target


def _frames_csv(project: KaraokeProject) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time_s", "freq_hz", "midi_float", "raw_midi_float", "confidence", "rms", "voiced"])
    for frame in project.frames:
        writer.writerow([
            f"{frame.time_s:.6f}",
            f"{frame.freq_hz:.6f}",
            "" if _is_nan(frame.midi_float) else f"{frame.midi_float:.6f}",
            "" if _is_nan(frame.raw_midi_float) else f"{frame.raw_midi_float:.6f}",
            f"{frame.confidence:.6f}",
            f"{frame.rms:.6f}",
            int(frame.voiced),
        ])
    return buf.getvalue()


def _is_nan(value: float) -> bool:
    return value != value
