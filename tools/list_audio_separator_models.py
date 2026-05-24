# -*- coding: utf-8 -*-

"""Lista modelos disponibles para Audio Separator / UVR si el paquete está instalado.

Uso:
    python tools/list_audio_separator_models.py

El comando depende del CLI de audio-separator. Si el paquete no está instalado,
usa primero:
    python tools/install_separation_dependencies.py --force-uvr
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
	cli = shutil.which("audio-separator")
	if not cli:
		print("audio-separator no está disponible en PATH.")
		print("Instalación experimental:")
		print("    python tools/install_separation_dependencies.py --force-uvr")
		return 1

	commands = [
		[cli, "--list_models", "--list_filter", "vocals"],
		[cli, "--list_models"],
	]

	last_output = ""
	for cmd in commands:
		proc = subprocess.run(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			encoding="utf-8",
			errors="replace",
			check=False,
		)
		last_output = proc.stdout or ""
		if proc.returncode == 0:
			print(last_output)
			return 0

	print("No se pudieron listar modelos con el CLI instalado.")
	print(last_output)
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
