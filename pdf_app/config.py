from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from .paths import get_app_base_dir


DEFAULT_API_URL = "https://f3c5z1l8l5o9ffa4.aistudio-app.com/layout-parsing"
DEFAULT_API_TOKEN = ""
DEFAULT_NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NVIDIA_MODEL = "openai/gpt-oss-120b"
CONFIG_FILE_PATH = get_app_base_dir() / "app_config.json"


@dataclass(frozen=True)
class AppConfig:
    api_url: str = DEFAULT_API_URL
    api_token: str = DEFAULT_API_TOKEN
    request_timeout_seconds: int = 300
    nvidia_api_url: str = DEFAULT_NVIDIA_API_URL
    nvidia_api_key: str = ""
    nvidia_model: str = DEFAULT_NVIDIA_MODEL
    max_requests_per_minute: int = 35
    translation_concurrency: int = 5


def _coerce_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _read_config_file() -> dict[str, object]:
    if not CONFIG_FILE_PATH.is_file():
        return {}
    try:
        data = json.loads(CONFIG_FILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_config() -> AppConfig:
    file_config = _read_config_file()
    return AppConfig(
        api_url=str(file_config.get("api_url", os.getenv("PDF_MD_API_URL", DEFAULT_API_URL))).strip(),
        api_token=str(file_config.get("api_token", os.getenv("PDF_MD_API_TOKEN", DEFAULT_API_TOKEN))).strip(),
        request_timeout_seconds=_coerce_int(
            file_config.get("request_timeout_seconds", os.getenv("PDF_MD_TIMEOUT_SECONDS", "300")),
            300,
        ),
        nvidia_api_url=str(file_config.get("nvidia_api_url", os.getenv("NVIDIA_API_URL", DEFAULT_NVIDIA_API_URL))).strip(),
        nvidia_api_key=str(file_config.get("nvidia_api_key", os.getenv("NVIDIA_API_KEY", ""))).strip(),
        nvidia_model=str(file_config.get("nvidia_model", os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL))).strip(),
        max_requests_per_minute=_coerce_int(
            file_config.get(
                "max_requests_per_minute",
                os.getenv("NVIDIA_MAX_REQUESTS_PER_MINUTE", "35"),
            ),
            35,
        ),
        translation_concurrency=_coerce_int(
            file_config.get(
                "translation_concurrency",
                os.getenv("TRANSLATION_CONCURRENCY", "3"),
            ),
            3,
        ),
    )


def save_config(config: AppConfig) -> None:
    CONFIG_FILE_PATH.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
