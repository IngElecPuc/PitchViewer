# -*- coding: utf-8 -*-

"""Utilidades mínimas para letras de karaoke."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


LRC_LINE_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)")


@dataclass
class LyricsLine:
    time_s: float
    text: str


def load_lyrics_file(path: str) -> tuple[str, str, list[LyricsLine]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    suffix = p.suffix.lower()
    if suffix == ".lrc":
        return "", text, parse_lrc(text)
    return text, "", []


def parse_lrc(text: str) -> list[LyricsLine]:
    lines: list[LyricsLine] = []
    for raw_line in text.splitlines():
        match = LRC_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        lyric = match.group(3).strip()
        lines.append(LyricsLine(time_s=minutes * 60.0 + seconds, text=lyric))
    return sorted(lines, key=lambda item: item.time_s)


def current_lyric_line(lines: list[LyricsLine], time_s: float) -> str:
    current = ""
    for line in lines:
        if line.time_s <= time_s:
            current = line.text
        else:
            break
    return current
