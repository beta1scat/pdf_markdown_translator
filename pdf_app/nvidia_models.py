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
    if not config.nvidia_api_key:
        return []

    headers = {
        "Authorization": f"Bearer {config.nvidia_api_key}",
        "Accept": "application/json",
    }
    try:
        response = requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers=headers,
            timeout=config.request_timeout_seconds,
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
    except (requests.RequestException, ValueError):
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
