# Pitch Viewer - Etapa 5

Aplicación de escritorio en Tkinter para monitorear afinación vocal en tiempo real.

La etapa 5 agrega persistencia de configuración. La app guarda y recarga automáticamente las preferencias del usuario en un archivo `settings.json`.

## Ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pitch_viewer.py
```

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pitch_viewer.py
```

Alternativa:

```bash
python -m pitch_viewer.main
```

## Persistencia de configuración

La aplicación guarda automáticamente:

- dispositivo de entrada seleccionado;
- idioma de notas: español o inglés;
- escala y tonalidad;
- ventana temporal;
- rango vertical visible;
- A4 de referencia;
- tolerancia en cents;
- visibilidad de notas fuera de escala, bandas y centros de nota;
- umbrales de detección;
- parámetros de estabilidad;
- tamaño/posición de la ventana.

En Windows el archivo queda en:

```text
%APPDATA%\PitchViewer\settings.json
```

En Linux/macOS queda normalmente en:

```text
~/.config/PitchViewer/settings.json
```

También se puede abrir la carpeta desde:

```text
Archivo > Abrir carpeta de configuración
```

## Menú Archivo

La etapa 5 agrega acciones nuevas:

```text
Guardar configuración
Recargar configuración
Restaurar configuración predeterminada...
Abrir carpeta de configuración
```

Aunque la app autoguarda, `Guardar configuración` permite forzar una escritura inmediata.

## Estructura

```text
pitch_viewer/
    main.py
    app.py
    constants.py
    models.py
    audio/
        __init__.py
    config/
        settings.py
    detection/
        autocorrelation.py
    music/
        notes.py
        scales.py
    ui/
        dialogs.py
run_pitch_viewer.py
requirements.txt
```

## Qué se conserva de la etapa 4

- Código modularizado.
- Captura de entrada con `sounddevice`.
- Detector portable por autocorrelación FFT.
- Estabilización con mediana, suavizado y guardia de octava.
- Menús de audio, vista, escala y afinación.
- Bandas de tolerancia, centros de nota y evaluación de afinación.
- Exportación CSV del historial.

## Qué cambia en la etapa 5

- Persistencia automática de settings.
- Carga de settings al abrir la app.
- Restauración de defaults desde menú.
- Recarga manual de configuración desde disco.
- Apertura de la carpeta de configuración desde la app.
- Guardado de fuente de entrada por nombre e índice, con fallback si el dispositivo ya no existe.
