# PitchViewer v0.8.2

Monitor de afinación vocal de escritorio en Python/Tkinter.

Esta versión corrige detalles visuales de la interfaz sin cambiar la regla de backends separada introducida en v0.8.1.

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

## Cambios v0.8.2

- Botones de transporte rediseñados con glifos más grandes y modo texto, evitando el recuadro interno de emoji.
- Se retira el botón visible `Pausar vista` de la barra superior para reducir ruido visual.
- La acción sigue disponible como `Vista > Congelar/reanudar vista`.
- El panel instructivo del canvas aparece al iniciar, dura 5 segundos y luego desaparece.
- El panel instructivo puede activarse/desactivarse desde `Vista > Mostrar panel instructivo`.
- La calibración ahora deja una marca gráfica temporal en el canvas:
  - diagnóstico en curso;
  - rango recomendado;
  - rango aplicado;
  - preset aplicado.

## Backends

- `▶ Play` usa `Audio > Backend en vivo`.
- `⏺ Record` + `⏹ Stop` usa `Audio > Backend offline / record`.
- `Torchcrepe full` queda deshabilitado para tiempo real si no hay CUDA.
- `Torchcrepe full` sigue disponible offline aunque corra en CPU.

## Transporte

- `▶`: inicia o reanuda captura viva con `Backend en vivo`.
- `⏸`: pausa captura sin borrar historial.
- `⏹`: detiene captura; si venías de `⏺`, lanza análisis offline.
- `⏺`: graba en memoria dentro de la ventana temporal vigente.
- `⏪` / `⏩`:
  - offline: retrocede/avanza media ventana;
  - online: aumenta/reduce la ventana temporal en 1 segundo.
- `⏮` / `⏭`:
  - offline: inicio/final de la grabación.

## Congelar vista

`Vista > Congelar/reanudar vista` no pausa el audio. Solo congela la posición visual de la ventana temporal para poder inspeccionar el dibujo mientras la captura puede seguir entrando. Es distinto de `⏸`, que pausa la captura.
