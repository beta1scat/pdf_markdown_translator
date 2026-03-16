from __future__ import annotations

from pathlib import Path
import sys
import tomllib


DEFAULT_VERSION = "0.1.2"


def _get_pyproject_path() -> Path:
    if getattr(sys, "frozen", False):
        bundle_dir = Path(
            getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
        )
        return bundle_dir / "pyproject.toml"
    return Path(__file__).resolve().parent.parent / "pyproject.toml"


def load_version() -> str:
    pyproject_path = _get_pyproject_path()
    if not pyproject_path.is_file():
        return DEFAULT_VERSION

    try:
        with pyproject_path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        return DEFAULT_VERSION

    project = data.get("project")
    if not isinstance(project, dict):
        return DEFAULT_VERSION

    version = project.get("version")
    if not isinstance(version, str):
        return DEFAULT_VERSION

    version = version.strip()
    return version or DEFAULT_VERSION


__version__ = load_version()
