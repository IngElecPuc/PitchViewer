# CHANGELOG

## v1.0.0

Primera versión empaquetable.

Incluye:

- Pitch tracking en vivo con backends seleccionables.
- Record/offline con backend separado.
- Visualización por notas, octavas, tolerancia y bloques alcanzados.
- Configuración persistente.
- Karaoke producción: análisis de pista vocal y generación `.pvk`.
- Karaoke play: evaluación contra targets `.pvk`.
- Separación IA offline con Demucs, exportación WAV/MP3 y sliders por stem.
- Diagnósticos de entorno, separación y Demucs.
- Build idempotente Windows/Ubuntu con `tools/build_app.py`.
- Workflow CI/CD base con GitHub Actions.
