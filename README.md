# Pitch Viewer v0.9.0

Monitor de afinación vocal en escritorio con Tkinter.

## Ejecución

```bash
pip install -r requirements.txt
python main.py
```

Para usar Torchcrepe:

```bash
pip install -r optional-requirements-torchcrepe.txt
```

## Karaoke producción

La etapa 9 agrega creación de proyectos karaoke `.pvk`.

Flujo básico:

1. `Karaoke > Nuevo proyecto desde audio...`
2. Cargar una pista vocal. Recomendado en esta etapa: `.wav`.
3. Opcional: `Karaoke > Importar letra...` con `.txt` o `.lrc`.
4. Elegir backend offline en `Audio > Backend offline / record`.
5. `Karaoke > Analizar pista vocal`.
6. Revisar los bloques de segmentos en el piano-roll y navegar con el slider del panel karaoke.
7. `Karaoke > Guardar proyecto .pvk...`.

El `.pvk` es un ZIP con:

```text
manifest.json
settings_snapshot.json
pitch_frames.csv
note_segments.json
lyrics.txt    opcional
lyrics.lrc    opcional
```

## Formatos de audio

- WAV: soporte principal.
- FLAC/OGG/otros: pueden funcionar si `soundfile`/libsndfile los soporta.
- MP3/MP4/M4A: se intenta usar `ffmpeg` si está disponible en PATH.

Para archivos multicanal, el programa permite elegir:

```text
mix      promedio mono
left     canal izquierdo
right    canal derecho
max_rms  canal con mayor energía RMS
```

## Backends

- En vivo: se usa `Audio > Backend en vivo`.
- Offline/record/karaoke producción: se usa `Audio > Backend offline / record`.

Torchcrepe full queda deshabilitado para tiempo real si no hay CUDA, pero sigue disponible para análisis offline en CPU.
