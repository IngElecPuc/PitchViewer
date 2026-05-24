# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class StemAudio:
	name: str
	path: Path
	sample_rate: int
	audio: np.ndarray

	@property
	def duration_s(self) -> float:
		if self.sample_rate <= 0:
			return 0.0
		return float(self.audio.shape[0]) / float(self.sample_rate)

	@property
	def channels(self) -> int:
		if self.audio.ndim == 1:
			return 1
		return int(self.audio.shape[1])


@dataclass
class SeparationResult:
	source_path: Path
	output_dir: Path
	model_name: str
	device: str
	stems: Dict[str, StemAudio] = field(default_factory=dict)

	def has_stem(self, name: str) -> bool:
		return name in self.stems
