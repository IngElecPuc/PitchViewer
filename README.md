# Pitch Viewer v0.7.1

Aplicación de escritorio en Tkinter para monitoreo de afinación vocal.

## Ejecución

```bash
cd "E:\\Felpipe\\Proyectos propios\\PitchViewer"
pip install -r requirements.txt
python .\main.py
```

## Dependencias base

```txt
numpy
sounddevice
```

## Dependencias opcionales para Torchcrepe

```bash
pip install -r optional-requirements-torchcrepe.txt
```

## Cambios de v0.7.1

- Se mantiene la etapa 7 visual: rangos vocales, seguimiento dinámico, bloques de nota alcanzada, grosor configurable de línea, zonas válidas por tolerancia y modo claro/oscuro.
- Se agrega transporte visual con botones:
  - `▶` inicia o reanuda captura en vivo.
  - `⏸` pausa la captura sin borrar el historial visual.
  - `⏹` detiene la captura.
  - `⏺` graba en memoria para análisis offline.
  - `⏪` / `⏩` navegan media ventana en una grabación offline; en modo online ajustan la ventana temporal +1s / -1s.
  - `⏮` / `⏭` saltan al inicio/final de una grabación offline.
- La grabación offline se limita a la ventana temporal configurada, con máximo configurable hasta 60s mediante los botones `⏪`/`⏩`.
- Al detener una grabación iniciada con `⏺`, se analiza el audio en memoria con Torchcrepe full.
- Si no se detecta CUDA, Torchcrepe full queda deshabilitado como backend en vivo, pero sigue disponible para análisis offline.

## Notas sobre Torchcrepe full

Torchcrepe full puede funcionar correctamente en CPU para análisis offline, pero no suele alcanzar tiempo real. En equipos sin CUDA se recomienda usar en vivo:

- YIN CMND
- Autocorrelación FFT
- Torchcrepe tiny, si responde suficientemente rápido

Torchcrepe full queda reservado para análisis offline y, más adelante, para producción de archivos karaoke.
