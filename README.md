# Pitch Viewer v0.9.5

Aplicación de escritorio en Tkinter para monitoreo de afinación vocal, análisis offline de pitch, producción de archivos karaoke `.pvk` y separación offline de fuentes de audio.

## 1. Instalación base

Desde la carpeta del proyecto:

```bash
cd "E:\Felpipe\Proyectos propios\PitchViewer"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python .\main.py
```

También puede ejecutarse como paquete desde la carpeta padre:

```bash
python -m PitchViewer.main
```

## 2. Dependencias opcionales

### 2.1 Torchcrepe

Torchcrepe se usa para detección de pitch más robusta. En vivo se recomienda `torchcrepe_tiny`. `torchcrepe_full` queda deshabilitado en vivo si no hay CUDA, pero puede usarse como backend offline.

```bash
pip install -r optional-requirements-torchcrepe.txt
```

Diagnósticos:

```bash
python .\tools\diagnose_torchcrepe.py
python .\tools\diagnose_pitchviewer_backends.py
python .\tools\diagnose_live_torchcrepe.py --backend all
```

### 2.2 Separación IA

La separación de canciones completas es un proceso **offline**. CUDA solo acelera; no cambia el flujo.

Instalación recomendada:

```bash
python .\tools\install_separation_dependencies.py
```

Ese instalador deja lista la ruta estable:

```text
Demucs + soundfile + imageio-ffmpeg
```

La app no depende de TorchCodec para ejecutar Demucs. En Windows + Python reciente, `torchaudio.load` o `torchaudio.save` pueden intentar usar TorchCodec y fallar por DLLs o compatibilidad binaria. Para evitarlo, Pitch Viewer ejecuta Demucs mediante un runner propio:

```text
tools/run_demucs_soundfile.py
```

Ese runner reemplaza la carga y el guardado de audio de Demucs por `soundfile`. Por eso la app primero convierte MP3/M4A/MP4/FLAC a WAV temporal usando FFmpeg, Demucs procesa ese WAV limpio, y los stems se escriben sin pasar por TorchCodec.

Instalación directa mínima equivalente:

```bash
pip install -r optional-requirements-separation.txt
```

Diagnóstico general:

```bash
python .\tools\diagnose_separation_dependencies.py
```

Diagnóstico sobre una canción concreta:

```bash
python .\tools\diagnose_demucs_track.py "E:\Felpipe\Escritorio\Amigo.mp3" --mode 2stems --device cpu --clip-seconds 20
```

Si el clip funciona, prueba el archivo completo:

```bash
python .\tools\diagnose_demucs_track.py "E:\Felpipe\Escritorio\Amigo.mp3" --mode 2stems --device cpu
```

Los reportes quedan en:

```text
diagnostics/demucs_track_YYYYMMDD_HHMMSS.txt
```

## 3. FFmpeg en Windows

La app busca FFmpeg en este orden:

```text
1. ffmpeg global en PATH
2. ffmpeg empaquetado por imageio-ffmpeg
```

Para la app basta con `imageio-ffmpeg`, instalado por `tools/install_separation_dependencies.py`. Instalar FFmpeg globalmente sigue siendo útil para pruebas de consola y otras herramientas.

Instalación global opcional con Winget:

```powershell
winget install Gyan.FFmpeg
```

O con Chocolatey:

```powershell
choco install ffmpeg
```

Después abre una terminal nueva y prueba:

```bash
ffmpeg -version
```

## 4. Uso general de la app

### Modo vivo

```text
▶ Play
    captura micrófono y usa Audio > Backend en vivo

⏸ Pausa
    pausa captura sin borrar el historial visual

⏹ Stop
    detiene captura
```

### Modo record/offline corto

```text
⏺ Record
    graba en memoria dentro de la ventana temporal vigente

⏹ Stop
    detiene grabación y analiza offline con Audio > Backend offline / record
```

Navegación:

```text
⏪ / ⏩
    offline: retrocede/avanza media ventana
    online: aumenta/reduce la ventana temporal en 1 segundo

⏮ / ⏭
    offline: inicio/final de la grabación
```

## 5. Karaoke producción

Panel:

```text
Karaoke > Mostrar/ocultar panel karaoke
```

Flujo:

```text
1. Abrir audio...
2. Importar letra... opcional, .txt o .lrc
3. Elegir Audio > Backend offline / record
4. Analizar pista
5. Revisar segmentos con el slider
6. Guardar .pvk
```

Formato `.pvk`:

```text
manifest.json
settings_snapshot.json
pitch_frames.csv
note_segments.json
lyrics.txt    opcional
lyrics.lrc    opcional
```

## 6. Separación IA offline

Panel:

```text
Separación IA > Mostrar/ocultar panel separación IA
```

Flujo recomendado con una canción MP3:

```text
1. Abrir mezcla...
2. Motor: Demucs
3. Modelo: htdemucs
4. Modo: 2stems si quieres vocals + no_vocals
5. Separar offline
6. Ajustar ganancias por stem
7. Exportar MP3, WAV o WAV+MP3
8. Usar stem de voz como pista karaoke
```

Para karaoke normalmente basta con:

```text
2stems: vocals + no_vocals
```

Para mezcla más flexible:

```text
4stems: vocals + drums + bass + other
```

## 7. Audio Separator / UVR

Audio Separator / UVR queda como motor experimental. En Windows + Python 3.14 puede fallar por dependencias nativas como `diffq-fixed`. Por eso no está en el requirements principal de separación.

Para intentarlo explícitamente:

```bash
python .\tools\install_separation_dependencies.py --force-uvr
```

Si queda instalado, puedes listar modelos:

```bash
python .\tools\list_audio_separator_models.py
```

Si falla, Demucs sigue siendo la ruta estable.

## 8. Diagnósticos útiles

```bash
python .\tools\diagnose_separation_dependencies.py
python .\tools\diagnose_demucs_track.py "ruta\cancion.mp3" --clip-seconds 20
python .\tools\diagnose_pitchviewer_backends.py
python .\tools\diagnose_live_torchcrepe.py --backend all
```

## 9. Control de versiones

Esta entrega queda como `v0.9.5`, porque corrige la visualización y clasificación de stems sobre la base funcional de `v0.9.4`.

```bash
git add .
git commit -m "fix: show gain slider for each separated stem"
git tag v0.9.5
```

## 10. Versiones relevantes

```text
v0.7.1: transporte offline corto
v0.8.0: calibración vocal y análisis offline
v0.8.1: backend vivo y backend offline separados
v0.8.2: ajustes visuales y overlay de calibración
v0.9.0: karaoke producción .pvk
v0.9.1: letras en panel lateral
v0.9.2: barra de progreso para análisis karaoke
v0.9.3: panel de separación IA
v0.9.4: separación offline robusta, preconversión a WAV, runner Demucs con soundfile para lectura/escritura, exportación MP3/WAV y diagnóstico por canción
v0.9.5: corrección de stems 2stems; no_vocals ya no se clasifica como vocals y la UI muestra una barra de ganancia por cada stem
```

### Ganancias por stem

Después de separar una canción, el panel de separación muestra una barra por cada stem detectado.

- En modo `2stems`: `Voz` y `Instrumental/no_vocals`.
- En modo `4stems`: `Voz`, `Batería`, `Bajo` y `Otros instrumentos`.

Cada barra controla la ganancia de ese stem en la mezcla exportada. `100%` conserva el volumen original, `0%` silencia el stem y valores mayores amplifican antes de normalizar si hay clipping.
