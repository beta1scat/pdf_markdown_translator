from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore[assignment]


DEFAULT_VERSION = "0.1.5"


def _get_version_from_git() -> str | None:
    # Do not call git if packaged as frozen executable (PyInstaller)
    if getattr(sys, "frozen", False):
        return None
    try:
        repo_dir = Path(__file__).resolve().parent.parent
        output = subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode("utf-8").strip()
        if output:
            return output.lstrip("v")
    except Exception:
        pass
    return None


def _get_pyproject_path() -> Path:
    if getattr(sys, "frozen", False):
        bundle_dir = Path(
            getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
        )
        return bundle_dir / "pyproject.toml"
    return Path(__file__).resolve().parent.parent / "pyproject.toml"


def _get_version_from_pyproject() -> str | None:
    pyproject_path = _get_pyproject_path()
    if not pyproject_path.is_file():
        return None

    if tomllib is not None:
        try:
            with pyproject_path.open("rb") as file:
                data = tomllib.load(file)
            project = data.get("project")
            if isinstance(project, dict):
                version = project.get("version")
                if isinstance(version, str) and version.strip():
                    return version.strip()
        except (OSError, tomllib.TOMLDecodeError):
            pass

    try:
        content = pyproject_path.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1).strip()
    except OSError:
        pass

    return None


def load_version() -> str:
    # 1. Dynamically read git tag if running from git repo
    git_version = _get_version_from_git()
    if git_version:
        return git_version

    # 2. Read from pyproject.toml
    pyproject_version = _get_version_from_pyproject()
    if pyproject_version:
        return pyproject_version

    # 3. Fallback default
    return DEFAULT_VERSION


__version__ = load_version()

