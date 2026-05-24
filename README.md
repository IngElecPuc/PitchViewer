# Pitch Viewer - Etapa 7

Aplicación Tkinter para seguimiento visual de pitch vocal en tiempo real.

## Ejecutar

Desde la carpeta `PitchViewer`:

```bash
pip install -r requirements.txt
python .\main.py
```

También funciona como módulo si se ejecuta desde la carpeta padre:

```bash
python -m PitchViewer.main
```

## Cambios de la etapa 7

- Rangos vocales extendidos en `Vista > Rango visible`:
  - Bajo ±½ octava.
  - Barítono ±½ octava.
  - Tenor ±½ octava.
  - Contralto ±½ octava.
  - Mezzo ±½ octava.
  - Soprano ±½ octava.
- Seguimiento dinámico de voz:
  - `Vista > Seguimiento dinámico de voz`.
  - Desplaza visualmente la ventana de notas para mantener la voz cerca del centro.
- Bloques de nota alcanzada:
  - `Vista > Mostrar bloques alcanzados`.
  - Los bloques aparecen solo cuando el pitch está dentro de la tolerancia en cents y dentro de la escala activa.
  - Esto deja preparadas las capas gráficas para karaoke: target vs. nota efectivamente alcanzada.
- Línea de pitch con color único.
- Grosor de línea ajustable:
  - `Vista > Grosor de línea de pitch...`.
- Zonas inválidas más explícitas:
  - La región válida de una nota es la banda de tolerancia.
  - Fuera de esa banda, la voz aparece como desafinada aunque esté cerca de una nota.
  - La tolerancia máxima se limita a 49 cents para evitar que las bandas de notas contiguas se toquen.
- Pausa visual sin detener audio:
  - Botón `Pausar vista` o `Vista > Pausar/reanudar visualización`.
- Zoom vertical con rueda del mouse.
- Arrastre vertical del rango con botón medio o botón derecho.
- Leyenda visual sobre el canvas.
- Modo claro/oscuro básico.

## Backends

La etapa 7 conserva los backends de la etapa 6:

- Autocorrelación FFT.
- YIN CMND.
- Torchcrepe tiny/full como opción experimental, si se instalan dependencias opcionales.

Torchcrepe sigue siendo opcional y puede fallar según versión de Python, PyTorch y Windows. Se deja para una instancia separada de reparación.
