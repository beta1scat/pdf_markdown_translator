from __future__ import annotations

import sys
from pathlib import Path


def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def append_runtime_log(message: str) -> None:
    from datetime import datetime

    log_path = get_app_base_dir() / "app.log"
    timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamped}\n")
    except OSError:
        pass

