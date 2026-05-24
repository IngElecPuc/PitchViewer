# Pitch Viewer v0.9.4

Aplicación Tkinter para monitoreo de afinación vocal, análisis offline, producción karaoke y separación offline de fuentes.

## Ejecución

```bash
cd "E:\Felpipe\Proyectos propios\PitchViewer"
pip install -r requirements.txt
python .\main.py
```

## Dependencias de separación IA

La separación de canciones completas es un proceso offline. CUDA solo acelera; no cambia el flujo.

Instalación recomendada:

```bash
python .\tools\install_separation_dependencies.py
```

Ese instalador hace lo siguiente:

```text
1. Instala siempre la ruta estable:
   - demucs
   - soundfile
   - imageio-ffmpeg

2. Intenta dejar disponible Audio Separator / UVR.
   Si falla, no aborta: Demucs queda usable.
```

Instalación mínima equivalente:

```bash
pip install -r optional-requirements-separation.txt
```

## ffmpeg en Windows

La app puede usar dos rutas:

```text
1. ffmpeg global en PATH.
2. ffmpeg empaquetado por imageio-ffmpeg.
```

Por eso `imageio-ffmpeg` está incluido en `optional-requirements-separation.txt`. Para uso fuera de Python, o si otra herramienta exige `ffmpeg.exe` global, instala ffmpeg con uno de estos métodos:

```powershell
winget install Gyan.FFmpeg
```

O con Chocolatey:

```powershell
choco install ffmpeg
```

Luego abre una nueva terminal y revisa:

```bash
ffmpeg -version
```

## Diagnóstico

```bash
python .\tools\diagnose_separation_dependencies.py
```

Debe indicar, como mínimo:

```text
[OK] ffmpeg
[OK] demucs
```

Audio Separator / UVR puede aparecer como no disponible sin bloquear el flujo principal.

Si Audio Separator / UVR queda instalado, puedes listar modelos con:

```bash
python .\tools\list_audio_separator_models.py
```

## Separación IA offline

Panel:

```text
Separación IA > Mostrar/ocultar panel separación IA
```

Motores:

```text
Demucs                 recomendado, estable
Audio Separator / UVR  experimental, depende del entorno
```

Flujo básico con una canción MP3:

```text
1. Separación IA > Mostrar/ocultar panel separación IA.
2. Abrir mezcla...
3. Elegir motor.
4. Elegir modelo.
5. Separar offline.
6. Ajustar ganancias por stem.
7. Exportar MP3, WAV o ambos.
8. Usar voz en karaoke.
```

## Versiones relevantes

```text
v0.9.0: karaoke producción .pvk
v0.9.1: letras en panel lateral
v0.9.2: barra de progreso para análisis karaoke
v0.9.3: panel de separación IA
v0.9.4: separación offline más robusta, exportación MP3/WAV y motor experimental Audio Separator / UVR
```
