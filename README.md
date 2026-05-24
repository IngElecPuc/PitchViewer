# PitchViewer v0.8.1

Monitor de afinación vocal de escritorio en Python/Tkinter.

Esta versión corrige el diseño de backends offline: `Record` ya no fuerza siempre `Torchcrepe full`. Ahora hay dos preferencias separadas:

- `Backend en vivo`: usado por `▶ Play`.
- `Backend offline / record`: usado por `⏺ Record` + `⏹ Stop`.

## Ejecución

```bash
cd "E:\Felpipe\Proyectos propios\PitchViewer"
pip install -r requirements.txt
python .\main.py
```

Para activar Torchcrepe:

```bash
pip install -r optional-requirements-torchcrepe.txt
```

## Cambios v0.8.1

- Menú `Audio > Backend en vivo`.
- Menú `Audio > Backend offline / record`.
- `Torchcrepe full` sigue deshabilitado como backend vivo si no hay CUDA.
- `Torchcrepe full` sigue disponible como backend offline aunque no haya CUDA.
- El modo offline puede usar libremente:
  - Autocorrelación FFT;
  - YIN CMND;
  - Torchcrepe tiny;
  - Torchcrepe full.
- La selección offline se guarda en `settings.json` como `offline_detector_backend`.
- `⏺ Record` + `⏹ Stop` usa el backend offline seleccionado, no el backend vivo.

## Transporte

- `▶`: inicia o reanuda captura viva con `Backend en vivo`.
- `⏸`: pausa sin borrar historial.
- `⏹`: detiene captura; si venías de `⏺`, lanza análisis offline.
- `⏺`: graba en memoria dentro de la ventana temporal vigente.
- `⏪` / `⏩`:
  - offline: retrocede/avanza media ventana;
  - online: aumenta/reduce la ventana temporal en 1 segundo.
- `⏮` / `⏭`:
  - offline: inicio/final de la grabación.

## Regla de diseño

`Play` y `Record` no significan “usar el mismo backend en dos modos”. Son flujos distintos:

- `▶ Play` usa `backend_live`.
- `⏺ Record` + `⏹ Stop` usa `backend_offline`.

Esto deja preparada la transición a karaoke: juego en tiempo real con backend vivo, producción/análisis con backend offline.
