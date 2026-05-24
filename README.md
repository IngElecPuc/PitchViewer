# PitchViewer - Etapa 6

Monitor de afinación vocal de escritorio con Tkinter.

Esta etapa agrega una arquitectura de backends de detección de pitch. La app mantiene el backend portable de autocorrelación FFT y agrega un segundo backend portable tipo YIN/CMND. También deja preparados backends opcionales de Torchcrepe.

## Ejecutar

Desde la carpeta `PitchViewer`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python .\main.py
```

También puede ejecutarse desde la carpeta padre:

```bash
python -m PitchViewer.main
```

## Dependencias base

```txt
numpy>=1.24
sounddevice>=0.4.6
```

## Backends incluidos

- **Autocorrelación FFT**: backend por defecto, portable y rápido.
- **YIN CMND**: backend portable con NumPy, más conservador para voz monofónica sostenida.
- **Torchcrepe tiny/full**: backends opcionales. Requieren instalar `torch` y `torchcrepe`; no vienen en `requirements.txt` para no romper entornos Windows con Python reciente.

Para intentar Torchcrepe:

```bash
pip install -r optional-requirements-torchcrepe.txt
```

Luego selecciona el backend desde:

```text
Audio > Backend de detección
```

## Configuración persistente

La selección de backend queda guardada en `settings.json` junto con el resto de configuración.

En Windows:

```text
%APPDATA%\PitchViewer\settings.json
```

## Nota práctica

Para canto en tiempo real, prueba primero estos dos backends:

1. Autocorrelación FFT.
2. YIN CMND.

Si YIN produce menos saltos pero responde más lento, ajusta los controles de estabilidad desde:

```text
Audio > Estabilidad de pitch...
```
