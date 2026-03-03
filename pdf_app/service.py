from __future__ import annotations

import base64
import html
import re
import time
from pathlib import Path
from typing import Callable

import requests

from .config import AppConfig
from .markdown_translator import NvidiaMarkdownTranslator, TranslationError
from .models import ConversionResult, ConversionStats, TimingStats


class PdfConversionError(RuntimeError):
    pass


class MarkdownTranslationError(RuntimeError):
    pass


PhaseCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


def _encode_pdf(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("ascii")


def call_layout_api(file_path: Path, config: AppConfig) -> dict:
    headers = {
        "Authorization": f"token {config.api_token}",
        "Content-Type": "application/json",
    }
    required_payload = {
        "file": _encode_pdf(file_path),
        "fileType": 0,
    }
    optional_payload = {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": True,
        "useChartRecognition": False,
        "useSealRecognition": True,
        "useOcrForImageBlock": False,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "ocr",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    }
    payload = {**required_payload, **optional_payload}

    response = requests.post(
        config.api_url,
        json=payload,
        headers=headers,
        timeout=config.request_timeout_seconds,
    )
    if response.status_code != 200:
        raise PdfConversionError(
            f"Layout API request failed with status {response.status_code}: {response.text[:400]}"
        )

    data = response.json()
    result = data.get("result")
    if not isinstance(result, dict):
        raise PdfConversionError("Layout API response does not contain a valid result payload.")
    return result


def merge_markdown(result: dict) -> str:
    parts: list[str] = []
    for page in result.get("layoutParsingResults", []):
        markdown = page.get("markdown", {})
        text = markdown.get("text", "")
        if text:
            parts.append(text)
    return center_images_markdown("\n\n".join(parts))


def center_images_markdown(markdown_text: str) -> str:
    centered = re.sub(
        r'(?im)^[ \t]*<div\b[^>]*>\s*(<img\b[^>]*?/?>)\s*</div>[ \t]*$',
        lambda match: f'<div align="center">{match.group(1)}</div>',
        markdown_text,
    )
    centered = re.sub(
        r'(?im)^[ \t]*(<img\b[^>]*?/?>)[ \t]*$',
        lambda match: f'<div align="center">{match.group(1)}</div>',
        centered,
    )
    centered = re.sub(
        r'(?im)^[ \t]*!\[([^\]]*)\]\(([^)]+)\)[ \t]*$',
        lambda match: (
            f'<div align="center"><img src="{match.group(2)}" alt="{html.escape(match.group(1), quote=True)}" /></div>'
        ),
        centered,
    )
    return centered


def _resolve_output_image_path(output_dir: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    candidate = (output_dir / normalized).resolve()
    output_root = output_dir.resolve()

    try:
        candidate.relative_to(output_root)
    except ValueError as exc:
        raise PdfConversionError(
            f"Image path escapes the output directory: {relative_path}"
        ) from exc

    return candidate


def save_images(result: dict, output_dir: Path, timeout_seconds: int) -> int:
    image_count = 0
    for page_index, page in enumerate(result.get("layoutParsingResults", [])):
        markdown = page.get("markdown", {})
        images = markdown.get("images", {})
        for relative_path, image_url in images.items():
            image_path = _resolve_output_image_path(output_dir, str(relative_path))
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_response = requests.get(image_url, timeout=timeout_seconds)
            image_response.raise_for_status()
            image_path.write_bytes(image_response.content)
            image_count += 1

        output_images = page.get("outputImages", {})
        for image_name, image_url in output_images.items():
            safe_name = str(image_name).replace("\\", "_").replace("/", "_")
            image_path = output_dir / f"{safe_name}_{page_index}.jpg"
            image_response = requests.get(image_url, timeout=timeout_seconds)
            image_response.raise_for_status()
            image_path.write_bytes(image_response.content)
            image_count += 1
    return image_count


def convert_pdf_to_markdown(
    pdf_path: str | Path,
    output_base_dir: str | Path,
    config: AppConfig,
    translate_markdown: bool = True,
    phase_callback: PhaseCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    total_start = time.perf_counter()
    conversion_seconds = 0.0
    translation_seconds = 0.0
    source_path = Path(pdf_path).expanduser().resolve()
    base_output_path = Path(output_base_dir).expanduser().resolve()

    if not source_path.is_file():
        raise PdfConversionError(f"PDF file does not exist: {source_path}")
    if source_path.suffix.lower() != ".pdf":
        raise PdfConversionError(f"Only PDF files are supported: {source_path}")

    base_output_path.mkdir(parents=True, exist_ok=True)

    document_output_dir = base_output_path / source_path.stem
    document_output_dir.mkdir(parents=True, exist_ok=True)

    if phase_callback is not None:
        phase_callback("converting")
    conversion_start = time.perf_counter()
    result = call_layout_api(source_path, config)
    merged_markdown = merge_markdown(result)
    markdown_path = document_output_dir / f"{source_path.stem}_full.md"
    markdown_path.write_text(merged_markdown, encoding="utf-8")
    image_count = save_images(result, document_output_dir, config.request_timeout_seconds)
    conversion_seconds = time.perf_counter() - conversion_start

    translated_markdown_path: Path | None = None
    if translate_markdown:
        if phase_callback is not None:
            phase_callback("translating")
        translation_start = time.perf_counter()
        translator = NvidiaMarkdownTranslator(config, progress_callback=progress_callback)
        translated_markdown = translator.translate_markdown(merged_markdown)
        translated_markdown = center_images_markdown(translated_markdown)
        translated_markdown_path = document_output_dir / f"{source_path.stem}_full_zh.md"
        translated_markdown_path.write_text(translated_markdown, encoding="utf-8")
        translation_seconds = time.perf_counter() - translation_start

    page_count = len(result.get("layoutParsingResults", []))
    total_seconds = time.perf_counter() - total_start

    return ConversionResult(
        output_dir=document_output_dir,
        markdown_path=markdown_path,
        translated_markdown_path=translated_markdown_path,
        stats=ConversionStats(page_count=page_count, image_count=image_count),
        timings=TimingStats(
            conversion_seconds=conversion_seconds,
            translation_seconds=translation_seconds,
            total_seconds=total_seconds,
        ),
    )


def translate_markdown_file(
    markdown_path: str | Path,
    output_base_dir: str | Path,
    config: AppConfig,
    phase_callback: PhaseCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    total_start = time.perf_counter()
    source_path = Path(markdown_path).expanduser().resolve()
    base_output_path = Path(output_base_dir).expanduser().resolve()

    if not source_path.is_file():
        raise MarkdownTranslationError(f"Markdown file does not exist: {source_path}")
    if source_path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise MarkdownTranslationError(f"Only Markdown or text files are supported: {source_path}")
    if not config.nvidia_api_key:
        raise MarkdownTranslationError("NVIDIA_API_KEY is not configured.")

    base_output_path.mkdir(parents=True, exist_ok=True)

    original_markdown = center_images_markdown(source_path.read_text(encoding="utf-8"))
    translator = NvidiaMarkdownTranslator(config, progress_callback=progress_callback)
    try:
        if phase_callback is not None:
            phase_callback("translating")
        translation_start = time.perf_counter()
        translated_markdown = translator.translate_markdown(original_markdown)
        translated_markdown = center_images_markdown(translated_markdown)
        translation_seconds = time.perf_counter() - translation_start
    except TranslationError as exc:
        raise MarkdownTranslationError(str(exc)) from exc

    output_markdown_path = base_output_path / f"{source_path.stem}_zh.md"
    output_markdown_path.write_text(translated_markdown, encoding="utf-8")

    return ConversionResult(
        output_dir=base_output_path,
        markdown_path=source_path,
        translated_markdown_path=output_markdown_path,
        stats=ConversionStats(page_count=0, image_count=0),
        timings=TimingStats(
            conversion_seconds=0.0,
            translation_seconds=translation_seconds,
            total_seconds=time.perf_counter() - total_start,
        ),
    )
