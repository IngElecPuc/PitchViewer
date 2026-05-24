# -*- coding: utf-8 -*-

"""Modelos para proyectos de karaoke de Pitch Viewer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


PVK_FORMAT_NAME = "PitchViewerKaraoke"
PVK_FORMAT_VERSION = "1.0"


@dataclass
class KaraokeAudioInfo:
    path: str = ""
    filename: str = ""
    sample_rate: int = 44100
    channels: int = 1
    selected_channel: str = "mix"
    duration_s: float = 0.0


@dataclass
class KaraokePitchFrame:
    time_s: float
    freq_hz: float
    midi_float: float
    raw_midi_float: float
    confidence: float
    rms: float
    voiced: bool


@dataclass
class KaraokeNoteSegment:
    start_s: float
    end_s: float
    midi: int
    note: str
    mean_cents: float
    confidence: float
    rms: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))


@dataclass
class KaraokeProject:
    title: str = ""
    artist: str = ""
    audio_info: KaraokeAudioInfo = field(default_factory=KaraokeAudioInfo)
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    frames: list[KaraokePitchFrame] = field(default_factory=list)
    note_segments: list[KaraokeNoteSegment] = field(default_factory=list)
    lyrics_text: str = ""
    lyrics_lrc: str = ""

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "format": PVK_FORMAT_NAME,
            "format_version": PVK_FORMAT_VERSION,
            "title": self.title,
            "artist": self.artist,
            "duration_s": self.audio_info.duration_s,
            "source_audio": asdict(self.audio_info),
            "frames_count": len(self.frames),
            "note_segments_count": len(self.note_segments),
            "has_lyrics_text": bool(self.lyrics_text.strip()),
            "has_lyrics_lrc": bool(self.lyrics_lrc.strip()),
        }
