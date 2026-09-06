from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from .paths import get_app_base_dir


DEFAULT_API_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_API_TOKEN = ""
DEFAULT_PDF_MODEL = "PaddleOCR-VL-1.6"

PROVIDER_ZHIPU = "Zhipu AI (智谱)"
PROVIDER_GOOGLE = "Google Gemini"
PROVIDER_NVIDIA = "NVIDIA NIM"
PROVIDER_CUSTOM = "Custom (自定义)"

PROVIDER_OPTIONS = [
    PROVIDER_ZHIPU,
    PROVIDER_GOOGLE,
    PROVIDER_NVIDIA,
    PROVIDER_CUSTOM,
]

PROVIDER_PRESETS: dict[str, dict[str, object]] = {
    PROVIDER_ZHIPU: {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "default_model": "glm-4.7-flash",
        "models": [
            "glm-4.7-flash",
            "glm-4.5-flash",
            "glm-4-flash",
            "glm-4-flashx",
            "glm-5.3-flash",
            "glm-4-plus",
            "glm-4-air",
            "glm-4-long",
        ],
        "rate_limit": 60,
        "concurrency": 4,
        "supports_model_refresh": False,
        "website_hint": "智谱未开放模型查询接口，可在 bigmodel.cn 官网查看可用免费模型并直接填入",
    },
    PROVIDER_GOOGLE: {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "default_model": "gemini-2.5-flash",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "rate_limit": 15,
        "concurrency": 2,
        "supports_model_refresh": False,
        "website_hint": "Google 未开放在线查询端点，可在 aistudio.google.com 查看可用免费模型并直接填入",
    },
    PROVIDER_NVIDIA: {
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "default_model": "moonshotai/kimi-k3",
        "models": [
            "moonshotai/kimi-k3",
            "meta/llama-3.3-70b-instruct",
            "deepseek-ai/deepseek-r1",
            "nvidia/llama-3.1-nemotron-70b-instruct",
        ],
        "rate_limit": 35,
        "concurrency": 3,
        "supports_model_refresh": True,
        "website_hint": "可点击右侧 Refresh 在线查询 NVIDIA 当前开放的模型列表",
    },
    PROVIDER_CUSTOM: {
        "url": "",
        "default_model": "",
        "models": [],
        "rate_limit": 30,
        "concurrency": 3,
        "supports_model_refresh": True,
        "website_hint": "自定义服务商支持点击 Refresh 尝试请求 /models 接口",
    },
}

DEFAULT_LLM_API_URL = str(PROVIDER_PRESETS[PROVIDER_ZHIPU]["url"])
DEFAULT_LLM_MODEL = str(PROVIDER_PRESETS[PROVIDER_ZHIPU]["default_model"])
CONFIG_FILE_PATH = get_app_base_dir() / "app_config.json"


def detect_provider(url: str = "", key: str = "") -> str:
    lower_url = (url or "").lower()
    lower_key = (key or "").lower()
    if "bigmodel.cn" in lower_url:
        return PROVIDER_ZHIPU
    if "googleapis.com" in lower_url:
        return PROVIDER_GOOGLE
    if "nvidia.com" in lower_url or lower_key.startswith("nvapi-"):
        return PROVIDER_NVIDIA
    return PROVIDER_ZHIPU


@dataclass(frozen=True)
class AppConfig:
    api_url: str = DEFAULT_API_URL
    api_token: str = DEFAULT_API_TOKEN
    pdf_model: str = DEFAULT_PDF_MODEL
    request_timeout_seconds: int = 300
    llm_provider: str = PROVIDER_ZHIPU
    llm_api_url: str = DEFAULT_LLM_API_URL
    llm_api_key: str = ""
    llm_model: str = DEFAULT_LLM_MODEL
    max_requests_per_minute: int = 60
    translation_concurrency: int = 4
    translation_chunk_size: int = 5000
    provider_keys: dict[str, str] = field(default_factory=dict)
    provider_models: dict[str, str] = field(default_factory=dict)


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

    raw_url = str(
        file_config.get(
            "llm_api_url",
            file_config.get("nvidia_api_url", os.getenv("LLM_API_URL", "")),
        )
    ).strip()

    raw_key = str(
        file_config.get(
            "llm_api_key",
            file_config.get("nvidia_api_key", os.getenv("LLM_API_KEY", "")),
        )
    ).strip()

    raw_model = str(
        file_config.get(
            "llm_model",
            file_config.get("nvidia_model", os.getenv("LLM_MODEL", "")),
        )
    ).strip()

    provider = str(
        file_config.get(
            "llm_provider",
            os.getenv("LLM_PROVIDER", ""),
        )
    ).strip()

    if not provider or provider not in PROVIDER_PRESETS:
        provider = detect_provider(raw_url, raw_key)

    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS[PROVIDER_ZHIPU])
    resolved_url = raw_url or str(preset["url"])
    resolved_model = raw_model or str(preset["default_model"])

    raw_provider_keys = file_config.get("provider_keys", {})
    provider_keys: dict[str, str] = (
        {str(k): str(v) for k, v in raw_provider_keys.items()}
        if isinstance(raw_provider_keys, dict)
        else {}
    )

    raw_provider_models = file_config.get("provider_models", {})
    provider_models: dict[str, str] = (
        {str(k): str(v) for k, v in raw_provider_models.items()}
        if isinstance(raw_provider_models, dict)
        else {}
    )

    # Sync raw_key and provider_keys
    if raw_key:
        provider_keys.setdefault(provider, raw_key)
    elif provider in provider_keys and provider_keys[provider]:
        raw_key = provider_keys[provider]

    if resolved_model and provider:
        provider_models.setdefault(provider, resolved_model)

    config = AppConfig(
        api_url=str(
            file_config.get("api_url", os.getenv("PDF_MD_API_URL", DEFAULT_API_URL))
        ).strip()
        or DEFAULT_API_URL,
        api_token=str(
            file_config.get("api_token", os.getenv("PDF_MD_API_TOKEN", DEFAULT_API_TOKEN))
        ).strip(),
        pdf_model=str(
            file_config.get("pdf_model", os.getenv("PDF_MD_MODEL", DEFAULT_PDF_MODEL))
        ).strip()
        or DEFAULT_PDF_MODEL,
        request_timeout_seconds=_coerce_int(
            file_config.get(
                "request_timeout_seconds", os.getenv("PDF_MD_TIMEOUT_SECONDS", "300")
            ),
            300,
        ),
        llm_provider=provider,
        llm_api_url=resolved_url,
        llm_api_key=raw_key,
        llm_model=resolved_model,
        max_requests_per_minute=_coerce_int(
            file_config.get(
                "max_requests_per_minute",
                os.getenv("LLM_MAX_REQUESTS_PER_MINUTE", str(preset["rate_limit"])),
            ),
            int(preset["rate_limit"]),
        ),
        translation_concurrency=_coerce_int(
            file_config.get(
                "translation_concurrency",
                os.getenv("TRANSLATION_CONCURRENCY", str(preset["concurrency"])),
            ),
            int(preset["concurrency"]),
        ),
        translation_chunk_size=max(
            500,
            _coerce_int(
                file_config.get(
                    "translation_chunk_size",
                    os.getenv("TRANSLATION_CHUNK_SIZE", "5000"),
                ),
                5000,
            ),
        ),
        provider_keys=provider_keys,
        provider_models=provider_models,
    )
    save_config(config)
    return config


def save_config(config: AppConfig) -> None:
    CONFIG_FILE_PATH.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
