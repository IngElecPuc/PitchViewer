# -*- coding: utf-8 -*-

"""Build idempotente de PitchViewer con PyInstaller.

Uso:
    python tools/build_app.py
    python tools/build_app.py --profile base
    python tools/build_app.py --profile full
    python tools/build_app.py --no-archive

El script detecta Windows/Linux/macOS y genera una carpeta onedir en dist/,
además de un archivo comprimido reproducible en dist/releases/.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import py_compile
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
RELEASES_DIR = DIST_DIR / "releases"
APP_MODULE = PROJECT_ROOT / "main.py"
PACKAGE_NAME = PROJECT_ROOT.name


@dataclass(frozen=True)
class BuildTarget:
    os_name: str
    platform_slug: str
    executable_name: str
    archive_format: str
    archive_suffix: str
    icon_path: Path | None


def print_step(message: str) -> None:
    print(f"\n== {message} ==")


def run(cmd: list[str], *, cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(str(part) for part in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def load_version() -> str:
    namespace: dict[str, object] = {}
    version_file = PROJECT_ROOT / "version.py"
    if not version_file.exists():
        return "0.0.0"
    exec(version_file.read_text(encoding="utf-8"), namespace)
    return str(namespace.get("APP_VERSION", "0.0.0"))


def detect_target() -> BuildTarget:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "x64" if machine in {"amd64", "x86_64"} else machine.replace(" ", "_")

    if system == "windows":
        return BuildTarget(
            os_name="Windows",
            platform_slug=f"windows-{arch}",
            executable_name="PitchViewer.exe",
            archive_format="zip",
            archive_suffix=".zip",
            icon_path=PROJECT_ROOT / "assets" / "icon.ico",
        )

    if system == "linux":
        return BuildTarget(
            os_name="Linux",
            platform_slug=f"ubuntu-{arch}",
            executable_name="PitchViewer",
            archive_format="gztar",
            archive_suffix=".tar.gz",
            icon_path=PROJECT_ROOT / "assets" / "icon.png",
        )

    if system == "darwin":
        return BuildTarget(
            os_name="macOS",
            platform_slug=f"macos-{arch}",
            executable_name="PitchViewer",
            archive_format="gztar",
            archive_suffix=".tar.gz",
            icon_path=PROJECT_ROOT / "assets" / "icon.icns",
        )

    raise SystemExit(f"Sistema operativo no soportado para build: {platform.system()}")


def validate_project_root() -> None:
    required = ["main.py", "app.py", "requirements.txt", "version.py"]
    missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
    if missing:
        raise SystemExit(
            "No parece ser la raíz del proyecto PitchViewer. Faltan: " + ", ".join(missing)
        )


def ensure_build_dependencies() -> None:
    if module_exists("PyInstaller"):
        return
    raise SystemExit(
        "Falta PyInstaller. Instala dependencias de build con:\n"
        "    python tools/install_project_dependencies.py --build\n"
        "o directamente:\n"
        "    pip install -r optional-requirements-build.txt"
    )


def compile_sources() -> None:
    print_step("Compilando módulos Python")
    failures: list[tuple[Path, Exception]] = []
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        if any(part in {".venv", "venv", "build", "dist", "__pycache__"} for part in path.parts):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:  # pragma: no cover - diagnóstico de build
            failures.append((path, exc))

    if failures:
        for path, exc in failures:
            print(f"[ERROR] {path.relative_to(PROJECT_ROOT)}: {exc}")
        raise SystemExit(1)

    print("OK")


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_for_target(target_dir: Path, archive_path: Path) -> None:
    print_step("Limpiando build anterior")
    remove_path(BUILD_DIR)
    remove_path(DIST_DIR / "PitchViewer")
    remove_path(target_dir)
    remove_path(archive_path)
    spec_path = PROJECT_ROOT / "PitchViewer.spec"
    remove_path(spec_path)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    print("OK")


def create_pyinstaller_entry() -> Path:
    """Crea un entrypoint temporal para que PyInstaller analice el paquete completo.

    No usamos main.py directamente porque main.py soporta ejecución directa,
    pero PyInstaller analiza mejor una importación explícita del paquete.
    """
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    entry_path = BUILD_DIR / "pyinstaller_entry.py"
    parent = PROJECT_ROOT.parent
    entry_path.write_text(
        "# -*- coding: utf-8 -*-\n"
        "from __future__ import annotations\n"
        "import sys\n"
        f"sys.path.insert(0, {str(parent)!r})\n"
        f"from {PACKAGE_NAME}.main import main\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n",
        encoding="utf-8",
    )
    return entry_path


def pyinstaller_command(target: BuildTarget, profile: str, entry_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        "PitchViewer",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(BUILD_DIR),
        "--collect-all",
        "sounddevice",
        "--hidden-import",
        "cffi",
        "--paths",
        str(PROJECT_ROOT.parent),
        "--hidden-import",
        f"{PACKAGE_NAME}.app",
    ]

    if target.icon_path is not None and target.icon_path.exists():
        cmd.extend(["--icon", str(target.icon_path)])

    if profile == "full":
        optional_modules = [
            "torch",
            "torchcrepe",
            "demucs",
            "soundfile",
            "imageio_ffmpeg",
        ]
        for module_name in optional_modules:
            if module_exists(module_name):
                cmd.extend(["--collect-all", module_name])

    cmd.append(str(entry_path))
    return cmd


def run_pyinstaller(target: BuildTarget, profile: str) -> Path:
    print_step(f"Ejecutando PyInstaller ({target.os_name}, perfil {profile})")
    entry_path = create_pyinstaller_entry()
    run(pyinstaller_command(target, profile, entry_path))

    generated = DIST_DIR / "PitchViewer"
    if not generated.exists():
        raise SystemExit(f"PyInstaller no generó la carpeta esperada: {generated}")

    return generated


def copy_distribution_files(target_dir: Path) -> None:
    print_step("Copiando documentación y scripts auxiliares")

    docs = [
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "requirements.txt",
        "optional-requirements-build.txt",
        "optional-requirements-torchcrepe.txt",
        "optional-requirements-separation.txt",
    ]

    for name in docs:
        src = PROJECT_ROOT / name
        if src.exists():
            shutil.copy2(src, target_dir / name)

    tools_dst = target_dir / "tools"
    tools_dst.mkdir(exist_ok=True)
    for name in [
        "diagnose_environment.py",
        "diagnose_separation_dependencies.py",
        "diagnose_demucs_track.py",
        "install_project_dependencies.py",
        "install_separation_dependencies.py",
    ]:
        src = PROJECT_ROOT / "tools" / name
        if src.exists():
            shutil.copy2(src, tools_dst / name)

    print("OK")


def rename_output(generated: Path, target_dir: Path) -> None:
    print_step("Normalizando carpeta de salida")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    generated.rename(target_dir)
    print(target_dir)


def create_archive(target_dir: Path, archive_path: Path, archive_format: str) -> None:
    print_step("Creando archivo comprimido")
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    base_name = archive_path
    if archive_path.suffix == ".zip":
        base_name = archive_path.with_suffix("")
    elif archive_path.name.endswith(".tar.gz"):
        base_name = archive_path.with_name(archive_path.name[:-7])
    else:
        base_name = archive_path.with_suffix("")

    created = shutil.make_archive(
        base_name=str(base_name),
        format=archive_format,
        root_dir=str(target_dir.parent),
        base_dir=target_dir.name,
    )
    created_path = Path(created)
    if created_path != archive_path:
        remove_path(archive_path)
        created_path.rename(archive_path)
    print(archive_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build idempotente de PitchViewer")
    parser.add_argument(
        "--profile",
        choices=["base", "full"],
        default="base",
        help="base no intenta empaquetar IA pesada; full incluye módulos opcionales instalados",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="genera carpeta onedir pero no crea zip/tar.gz",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="omite py_compile previo al build",
    )
    args = parser.parse_args()

    validate_project_root()
    ensure_build_dependencies()

    app_version = load_version()
    target = detect_target()
    target_dir = DIST_DIR / f"PitchViewer-{target.platform_slug}"
    archive_path = RELEASES_DIR / f"PitchViewer-v{app_version}-{target.platform_slug}{target.archive_suffix}"

    print(f"Proyecto : {PROJECT_ROOT}")
    print(f"Versión  : {app_version}")
    print(f"Sistema  : {target.os_name} ({target.platform_slug})")
    print(f"Perfil   : {args.profile}")

    if not args.skip_compile:
        compile_sources()

    clean_for_target(target_dir, archive_path)
    generated = run_pyinstaller(target, args.profile)
    rename_output(generated, target_dir)
    copy_distribution_files(target_dir)

    if not args.no_archive:
        create_archive(target_dir, archive_path, target.archive_format)

    print_step("Build terminado")
    print(f"Carpeta : {target_dir}")
    if not args.no_archive:
        print(f"Release : {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
