# PitchViewer v1.0.0

PitchViewer es una aplicación de escritorio en Python/Tkinter para monitorear afinación vocal, visualizar pitch en una grilla musical, grabar análisis offline, producir archivos karaoke `.pvk`, reproducir karaoke contra targets y separar voz/instrumentos de forma offline.

La aplicación se puede ejecutar como código Python o empaquetar como ejecutable para Windows/Ubuntu usando `tools/build_app.py`.

---

## 1. Estructura general

```text
PitchViewer/
    main.py
    app.py
    version.py
    requirements.txt
    optional-requirements-build.txt
    optional-requirements-torchcrepe.txt
    optional-requirements-separation.txt

    audio/
    config/
    detection/
    karaoke/
    music/
    separation/
    ui/
    tools/
    assets/
```

Punto de entrada normal:

```bash
python main.py
```

También funciona desde la carpeta padre:

```bash
python -m PitchViewer.main
```

---

## 2. Crear entorno virtual

### Windows

```powershell
cd "E:\Felpipe\Proyectos propios\PitchViewer"
python -m venv venv
.\venv\Scripts\activate
```

### Ubuntu

```bash
cd ~/PitchViewer
python3 -m venv venv
source venv/bin/activate
```

En Ubuntu puede ser necesario instalar dependencias del sistema:

```bash
sudo apt update
sudo apt install -y python3-tk portaudio19-dev ffmpeg
```

---

## 3. Instalación recomendada

### Instalación base + build

Esta es la instalación mínima recomendada para usar la app y generar ejecutables:

```bash
python tools/install_project_dependencies.py --base --build
```

Equivale a instalar:

```bash
pip install -r requirements.txt
pip install -r optional-requirements-build.txt
```

### Backends Torchcrepe opcionales

Torchcrepe se usa para detección de pitch más robusta y análisis offline. En Windows sin CUDA, `torchcrepe_full` puede ser demasiado lento en tiempo real, pero puede servir offline.

```bash
python tools/install_project_dependencies.py --torchcrepe
```

O directamente:

```bash
pip install -r optional-requirements-torchcrepe.txt
```

### Separación IA offline

Para separar voz/instrumentos con Demucs y exportar WAV/MP3:

```bash
python tools/install_project_dependencies.py --separation
```

Ese comando llama a:

```bash
python tools/install_separation_dependencies.py
```

La ruta estable instala:

```text
demucs
soundfile
imageio-ffmpeg
```

`imageio-ffmpeg` permite usar FFmpeg desde Python aunque `ffmpeg.exe` no esté globalmente en `PATH`.

Audio Separator / UVR queda como experimental. Si quieres intentarlo:

```bash
python tools/install_project_dependencies.py --separation --force-uvr
```

En Windows + Python reciente puede fallar por dependencias nativas. Si falla, Demucs sigue usable.

### Instalación completa

```bash
python tools/install_project_dependencies.py --full
```

Esto intenta instalar base, build, Torchcrepe y separación IA. Si una dependencia opcional pesada falla, revisa el diagnóstico antes de insistir.

---

## 4. Diagnóstico del entorno

Antes de reportar errores o construir ejecutables, corre:

```bash
python tools/diagnose_environment.py
```

Para separación IA:

```bash
python tools/diagnose_separation_dependencies.py
```

Para probar una canción concreta con Demucs:

```bash
python tools/diagnose_demucs_track.py "ruta\cancion.mp3" --mode 2stems --device cpu --clip-seconds 20
```

Si el clip funciona, prueba el archivo completo:

```bash
python tools/diagnose_demucs_track.py "ruta\cancion.mp3" --mode 2stems --device cpu
```

---

## 5. Uso básico: pitch en vivo

1. Ejecuta:

```bash
python main.py
```

2. Selecciona dispositivo de entrada.
3. Presiona `▶` para iniciar captura.
4. Canta o habla una nota sostenida.
5. Ajusta:

```text
Audio > Backend en vivo
Vista > Rango visible
Vista > Seguimiento dinámico de voz
Afinación > Tolerancia
```

Backends vivos recomendados:

```text
YIN CMND
Autocorrelación FFT
Torchcrepe tiny, si responde bien en tu equipo
```

`Torchcrepe full` se deshabilita para tiempo real si no hay CUDA.

---

## 6. Modo record/offline

El botón `⏺` graba en memoria dentro de la ventana temporal configurada. Al presionar `⏹`, la grabación se analiza con el backend offline seleccionado:

```text
Audio > Backend offline / record
```

A diferencia del modo vivo, el modo offline puede usar detectores más pesados porque no depende del presupuesto temporal del micrófono.

Botones:

```text
▶   play/captura en vivo
⏸   pausa
⏹   stop
⏺   record offline
⏪   retroceder offline / agrandar ventana online
⏩   avanzar offline / achicar ventana online
⏮   inicio de grabación offline
⏭   final de grabación offline
```

---

## 7. Karaoke producción

Permite crear archivos `.pvk` desde una pista vocal.

Flujo:

```text
Karaoke > Mostrar/ocultar panel karaoke
Karaoke > Nuevo proyecto desde audio...
Karaoke > Importar letra...      opcional
Audio > Backend offline / record
Karaoke > Analizar pista vocal
Karaoke > Guardar proyecto .pvk
```

Formato `.pvk`:

```text
.pvk = archivo ZIP con:
    manifest.json
    settings_snapshot.json
    pitch_frames.csv
    note_segments.json
    lyrics.txt / lyrics.lrc opcionales
```

---

## 8. Karaoke play

Permite abrir un `.pvk` y cantar contra los targets.

Flujo:

```text
Karaoke > Abrir proyecto .pvk...
Mover slider al punto deseado
▶ Play
Cantar contra los segmentos target
Revisar score
```

La tolerancia usada para evaluar aciertos es la misma de:

```text
Afinación > Tolerancia
```

---

## 9. Separación IA offline

El panel de separación IA sirve para separar una canción completa en stems y luego usar la voz como entrada karaoke.

Flujo:

```text
Separación IA > Mostrar/ocultar panel separación IA
Abrir mezcla...
Motor: Demucs
Modo: 2stems o 4stems
Device: cpu/cuda
Separar offline
Ajustar ganancias por stem
Exportar WAV/MP3
Usar vocals.wav como pista karaoke
```

Modo 2 stems:

```text
vocals
no_vocals / instrumental
```

Modo 4 stems:

```text
vocals
drums
bass
other
```

En Windows sin CUDA, Demucs puede tardar bastante. En Ubuntu con CUDA debería ser más rápido.

La app usa un runner propio para Demucs:

```text
tools/run_demucs_soundfile.py
```

Este runner evita depender de TorchCodec para leer/guardar audio, usando `soundfile` sobre WAV temporal limpio.

---

## 10. Build local de ejecutables

El build se hace con un único script idempotente:

```bash
python tools/build_app.py
```

El script detecta el sistema operativo y genera el artefacto correspondiente.

### Windows

```powershell
python tools/install_project_dependencies.py --base --build
python tools/build_app.py
```

Salida esperada:

```text
dist/
    PitchViewer-windows-x64/
        PitchViewer.exe
    releases/
        PitchViewer-v1.0.0-windows-x64.zip
```

### Ubuntu

```bash
sudo apt update
sudo apt install -y python3-tk portaudio19-dev ffmpeg
python tools/install_project_dependencies.py --base --build
python tools/build_app.py
```

Salida esperada:

```text
dist/
    PitchViewer-ubuntu-x64/
        PitchViewer
    releases/
        PitchViewer-v1.0.0-ubuntu-x64.tar.gz
```

### Perfil base y perfil full

Por defecto se genera un build base:

```bash
python tools/build_app.py --profile base
```

El build base no intenta empaquetar IA pesada. Es el recomendado para v1.0.0.

Perfil full:

```bash
python tools/build_app.py --profile full
```

Incluye módulos opcionales instalados como Torchcrepe/Demucs si están disponibles. Puede ser más pesado y frágil.

### Build sin comprimir

```bash
python tools/build_app.py --no-archive
```

### Limpiar build

```bash
python tools/clean_build.py
```

El build también limpia automáticamente `build/` y la carpeta de salida del SO actual antes de generar el nuevo ejecutable.

---

## 11. CI/CD

El workflow de GitHub Actions está en:

```text
.github/workflows/build.yml
```

Corre en:

```text
windows-latest
ubuntu-latest
```

Usa el mismo script local:

```bash
python tools/build_app.py
```

Para generar releases:

```bash
git tag v1.0.0
git push origin main --tags
```

El workflow sube los ejecutables como artifacts. Puedes descargar:

```text
PitchViewer-v1.0.0-windows-x64.zip
PitchViewer-v1.0.0-ubuntu-x64.tar.gz
```

---

## 12. Versionamiento

La versión central está en:

```text
version.py
```

Ejemplo:

```python
APP_VERSION = "1.0.0"
```

Antes de una nueva release:

```bash
# editar version.py
git add .
git commit -m "chore: package pitch viewer desktop app"
git tag v1.0.0
git push origin main --tags
```

Para bugfix:

```text
v1.0.1
```

Para nuevas features compatibles:

```text
v1.1.0
```

Para cambios incompatibles:

```text
v2.0.0
```

---

## 13. Problemas comunes

### No se detecta micrófono

Ejecuta:

```bash
python tools/diagnose_environment.py
```

Revisa que `sounddevice` vea dispositivos de entrada.

### Torchcrepe full es muy lento

Sin CUDA, `Torchcrepe full` no es recomendable para tiempo real. Úsalo offline o usa `YIN CMND`/`Autocorrelación FFT`/`Torchcrepe tiny`.

### Demucs no carga MP3

Ejecuta:

```bash
python tools/install_project_dependencies.py --separation
python tools/diagnose_separation_dependencies.py
```

La ruta estable usa `imageio-ffmpeg` y convierte a WAV temporal antes de Demucs.

### Demucs falla con TorchCodec

La app no debería depender de TorchCodec en el flujo estable. Usa:

```text
tools/run_demucs_soundfile.py
```

Si vuelve a aparecer TorchCodec en el error, probablemente se está ejecutando `python -m demucs` directamente y no el runner de PitchViewer.

### El build falla porque falta PyInstaller

```bash
python tools/install_project_dependencies.py --build
```

### El ejecutable abre, pero faltan features IA

El build base no empaqueta IA pesada. Instala dependencias opcionales y usa `--profile full`, o ejecuta esas funciones desde el entorno Python.
