# -*- coding: utf-8 -*-

"""Guardar y cargar proyectos Pitch Viewer Karaoke (.pvk)."""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import (
    KaraokeAudioInfo,
    KaraokeNoteSegment,
    KaraokePitchFrame,
    KaraokeProject,
    PVK_FORMAT_NAME,
    PVK_FORMAT_VERSION,
)


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


def load_pvk(path: str) -> KaraokeProject:
    """Carga un archivo .pvk creado por Pitch Viewer.

    El formato es un ZIP con manifest/settings/frames/segmentos/letras. Esta
    función es tolerante a archivos parcialmente incompletos: si faltan frames,
    pero existen segmentos, el proyecto sigue cargando para modo play.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(source))

    with zipfile.ZipFile(source, "r") as zf:
        names = set(zf.namelist())

        manifest = _read_json(zf, "manifest.json") if "manifest.json" in names else {}
        fmt = str(manifest.get("format", ""))
        if fmt and fmt != PVK_FORMAT_NAME:
            raise ValueError(f"Formato .pvk no reconocido: {fmt}")

        settings_snapshot = _read_json(zf, "settings_snapshot.json") if "settings_snapshot.json" in names else {}
        segments = _read_segments(zf) if "note_segments.json" in names else []
        frames = _read_frames(zf) if "pitch_frames.csv" in names else []
        lyrics_text = _read_text(zf, "lyrics.txt") if "lyrics.txt" in names else ""
        lyrics_lrc = _read_text(zf, "lyrics.lrc") if "lyrics.lrc" in names else ""

    source_audio = manifest.get("source_audio", {}) if isinstance(manifest.get("source_audio", {}), dict) else {}
    audio_info = KaraokeAudioInfo(
        path=str(source_audio.get("path", "")),
        filename=str(source_audio.get("filename", "")),
        sample_rate=int(_number(source_audio.get("sample_rate"), 44100)),
        channels=int(_number(source_audio.get("channels"), 1)),
        selected_channel=str(source_audio.get("selected_channel", "mix")),
        duration_s=float(_number(source_audio.get("duration_s", manifest.get("duration_s")), _infer_duration(frames, segments))),
    )

    return KaraokeProject(
        title=str(manifest.get("title", source.stem)),
        artist=str(manifest.get("artist", "")),
        audio_info=audio_info,
        settings_snapshot=dict(settings_snapshot) if isinstance(settings_snapshot, dict) else {},
        frames=frames,
        note_segments=segments,
        lyrics_text=lyrics_text,
        lyrics_lrc=lyrics_lrc,
    )


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


def _read_json(zf: zipfile.ZipFile, name: str) -> Any:
    with zf.open(name, "r") as f:
        return json.loads(f.read().decode("utf-8"))


def _read_text(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name, "r") as f:
        return f.read().decode("utf-8-sig")


def _read_segments(zf: zipfile.ZipFile) -> list[KaraokeNoteSegment]:
    raw = _read_json(zf, "note_segments.json")
    if not isinstance(raw, list):
        return []

    segments: list[KaraokeNoteSegment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            segments.append(
                KaraokeNoteSegment(
                    start_s=float(item.get("start_s", 0.0)),
                    end_s=float(item.get("end_s", 0.0)),
                    midi=int(item.get("midi", 0)),
                    note=str(item.get("note", "")),
                    mean_cents=float(item.get("mean_cents", 0.0)),
                    confidence=float(item.get("confidence", 0.0)),
                    rms=float(item.get("rms", 0.0)),
                    frame_count=int(item.get("frame_count", 0)),
                )
            )
        except Exception:
            continue
    return segments


def _read_frames(zf: zipfile.ZipFile) -> list[KaraokePitchFrame]:
    text = _read_text(zf, "pitch_frames.csv")
    reader = csv.DictReader(io.StringIO(text))
    frames: list[KaraokePitchFrame] = []
    for row in reader:
        try:
            midi = _optional_float(row.get("midi_float"))
            raw_midi = _optional_float(row.get("raw_midi_float"))
            frames.append(
                KaraokePitchFrame(
                    time_s=float(row.get("time_s") or 0.0),
                    freq_hz=float(row.get("freq_hz") or 0.0),
                    midi_float=midi,
                    raw_midi_float=raw_midi,
                    confidence=float(row.get("confidence") or 0.0),
                    rms=float(row.get("rms") or 0.0),
                    voiced=str(row.get("voiced", "0")).strip().lower() in {"1", "true", "yes", "sí", "si"},
                )
            )
        except Exception:
            continue
    return frames


def _optional_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def _number(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _infer_duration(frames: list[KaraokePitchFrame], segments: list[KaraokeNoteSegment]) -> float:
    candidates: list[float] = []
    if frames:
        candidates.append(max(frame.time_s for frame in frames))
    if segments:
        candidates.append(max(segment.end_s for segment in segments))
    return max(candidates) if candidates else 0.0


def _is_nan(value: float) -> bool:
    return value != value
