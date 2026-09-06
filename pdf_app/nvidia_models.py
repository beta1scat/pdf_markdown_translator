from __future__ import annotations

import re

import requests

from .config import AppConfig


class NvidiaModelFetchError(RuntimeError):
    pass


LLM_APIS_DOC_URL = "https://docs.api.nvidia.com/nim/reference/llm-apis"


def fetch_nvidia_models(config: AppConfig) -> list[str]:
    models = _fetch_from_models_endpoint(config)
    if models:
        return models

    models = _fetch_from_docs_page(config.request_timeout_seconds)
    if models:
        return models

    raise NvidiaModelFetchError("Failed to fetch available NVIDIA models from official sources.")


def _fetch_from_models_endpoint(config: AppConfig) -> list[str]:
    if not config.llm_api_key:
        return []

    headers = {
        "Authorization": f"Bearer {config.llm_api_key}",
        "Accept": "application/json",
    }
    try:
        response = requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers=headers,
            timeout=(2.5, 5.0),
        )
        if response.status_code != 200:
            return []

        data = response.json()
        items = data.get("data", [])
        if not isinstance(items, list):
            return []

        models = sorted(
            {
                str(item.get("id", "")).strip()
                for item in items
                if isinstance(item, dict) and str(item.get("id", "")).strip()
            }
        )
        return models
    except (requests.RequestException, ValueError, OSError):
        return []


def _fetch_from_docs_page(timeout_seconds: int) -> list[str]:
    try:
        response = requests.get(LLM_APIS_DOC_URL, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise NvidiaModelFetchError("Unable to load the NVIDIA LLM API documentation page.") from exc

    text = response.text
    patterns = [
        r"Create chat completion \(([^)]+)\)",
        r"Creates a model response for the given chat conversation\.\s*</[^>]+>.*?/(?:reference|deploy)/([a-z0-9._/-]+)",
        r"https://docs\.api\.nvidia\.com/nim/reference/([a-z0-9._-]+-[a-z0-9._-]+)",
    ]

    candidates: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            model = str(match).strip().replace("_", "/")
            if "/" not in model and "-" in model:
                continue
            if model:
                candidates.add(model)

    normalized = sorted(_normalize_doc_model_name(model) for model in candidates if _normalize_doc_model_name(model))
    return sorted(set(normalized))


def _normalize_doc_model_name(model: str) -> str:
    value = model.strip().strip("/")
    value = value.replace("%2F", "/")
    value = value.replace("_", "-") if value.count("/") == 0 else value
    return value


def fetch_openai_compatible_models(
    models_url: str, api_key: str, timeout_seconds: int = 5
) -> list[str]:
    """Fetch model IDs dynamically from any OpenAI-compatible /models endpoint."""
    if not api_key:
        return []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    try:
        response = requests.get(models_url, headers=headers, timeout=(2.5, 4.0))
        if response.status_code != 200:
            return []
        data = response.json()
        items = data.get("data") or data.get("models") or []
        if not isinstance(items, list):
            return []
        models: list[str] = []
        for item in items:
            if isinstance(item, dict):
                m_id = str(item.get("id") or item.get("name") or "").strip()
                if m_id.startswith("models/"):
                    m_id = m_id[len("models/"):]
                if m_id:
                    models.append(m_id)
            elif isinstance(item, str) and item.strip():
                models.append(item.strip())
        return sorted(set(models))
    except (requests.RequestException, ValueError, OSError):
        return []


def fetch_all_provider_models(
    provider: str,
    api_url: str,
    api_key: str,
    timeout_seconds: int = 5,
) -> tuple[list[str], str]:
    """
    Returns (models, status_summary_message).
    Queries live API with fast timeout, falling back to comprehensive presets.
    """
    from .config import (
        PROVIDER_CUSTOM,
        PROVIDER_GOOGLE,
        PROVIDER_NVIDIA,
        PROVIDER_PRESETS,
        PROVIDER_ZHIPU,
    )

    preset_info = PROVIDER_PRESETS.get(provider, {})
    preset_models = list(preset_info.get("models", []))

    if provider == PROVIDER_NVIDIA:
        dummy_config = AppConfig(
            llm_provider=provider,
            llm_api_url=api_url,
            llm_api_key=api_key,
            request_timeout_seconds=timeout_seconds,
        )
        try:
            models = fetch_nvidia_models(dummy_config)
            if models:
                return models, f"已成功从 NVIDIA 官方在线拉取到 {len(models)} 个可用模型。"
        except Exception:
            pass
        return (
            preset_models,
            f"未能在线获取 NVIDIA 模型列表（请检查 Key 或网络）；已载入 {len(preset_models)} 个官方推荐预设模型。",
        )

    if provider == PROVIDER_GOOGLE:
        models_url = (
            api_url.replace("/chat/completions", "/models")
            if "/chat/completions" in api_url
            else "https://generativelanguage.googleapis.com/v1beta/openai/models"
        )
        online_models = fetch_openai_compatible_models(
            models_url, api_key, timeout_seconds=timeout_seconds
        )
        if online_models:
            flash_m = [m for m in online_models if "flash" in m.lower()]
            other_m = [m for m in online_models if m not in flash_m]
            ordered = flash_m + other_m
            return ordered, f"已成功通过 Google AI Studio API 在线拉取到 {len(ordered)} 个最新模型。"
        return (
            preset_models,
            f"未能在线连接 Google API（连接超时或网络受限）；已载入 2026 最新官方模型库（共 {len(preset_models)} 个，首选 gemini-2.5-flash）。\n提示：您也可直接在输入框中手动键入任意模型名称。",
        )

    if provider == PROVIDER_ZHIPU:
        # Zhipu open platform does not provide a public /models REST endpoint.
        # Immediately return the official 2026 model catalog without blocking.
        return (
            preset_models,
            "智谱开放平台官方未开放动态查询接口 (/models)。\n"
            f"已自动载入智谱 2026 最新官方模型矩阵（共 {len(preset_models)} 个模型，首选最新免费轻量模型 glm-4.7-flash）。\n"
            "提示：您也可以在下拉输入框中直接输入任意新发布的模型名称！",
        )

    # Custom provider
    if api_url:
        models_url = (
            api_url.replace("/chat/completions", "/models")
            if "/chat/completions" in api_url
            else api_url.rstrip("/") + "/models"
        )
        online_models = fetch_openai_compatible_models(
            models_url, api_key, timeout_seconds=timeout_seconds
        )
        if online_models:
            return (
                online_models,
                f"已成功从自定义 API 端点在线拉取到 {len(online_models)} 个模型。",
            )

    return (
        preset_models or [],
        "未能获取到模型列表，请直接在模型输入框中手动键入模型名称。",
    )
