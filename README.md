# Pitch Viewer v0.9.3

Monitor de afinación vocal en Python/Tkinter con modo karaoke producción y separación IA opcional.

## Ejecutar

```bash
cd "E:\\Felpipe\\Proyectos propios\\PitchViewer"
pip install -r requirements.txt
python .\main.py
```

## Dependencias opcionales

Backends Torchcrepe:

```bash
pip install -r optional-requirements-torchcrepe.txt
```

Separación IA con Demucs:

```bash
pip install -r optional-requirements-separation.txt
```

En Windows sin CUDA, Demucs y Torchcrepe full pueden correr en CPU, pero pueden tardar bastante. En Ubuntu con CUDA, la app detecta CUDA al iniciar y reutiliza ese estado cacheado para Torchcrepe y Demucs.

## Cambios v0.9.3

- Nuevo panel escondible de **Separación IA**.
- Nuevo menú `Separación IA`.
- Integración opcional con Demucs vía `python -m demucs`.
- Separación de canción/mezcla en stems.
- Barra de progreso/estado para separación.
- Barras de ganancia por pista detectada:
  - voz;
  - batería;
  - bajo;
  - otros instrumentos;
  - instrumental, en modo 2 stems.
- Exportación de mezcla WAV con ganancias ajustadas.
- Carga directa de `vocals.wav` como pista de karaoke producción.
- Estado CUDA centralizado al inicio de la app en `runtime.py`.

## Flujo sugerido para separación + karaoke

1. `Separación IA > Mostrar/ocultar panel separación IA`.
2. `Abrir mezcla...`.
3. Elegir modelo y salida:
   - `htdemucs` + `4stems` por defecto;
   - `2stems` si solo quieres `vocals` y `no_vocals`.
4. `Separar con Demucs`.
5. Ajustar ganancias si quieres exportar una mezcla.
6. `Usar voz en karaoke`.
7. `Karaoke > Analizar pista vocal`.
8. Guardar `.pvk`.

## Commit sugerido

```bash
git add .
git commit -m "feat: add AI source separation panel"
git tag v0.9.3
```
