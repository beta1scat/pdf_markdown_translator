from __future__ import annotations

import base64
import html
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import markdown as markdown_lib
import requests

from .config import AppConfig
from .markdown_translator import NvidiaMarkdownTranslator, TranslationError
from .models import ConversionResult, ConversionStats, TimingStats
from .paths import get_app_base_dir


class PdfConversionError(RuntimeError):
    pass


class MarkdownTranslationError(RuntimeError):
    pass


PhaseCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
LAYOUT_API_MAX_ATTEMPTS = 3
LAYOUT_API_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
LAYOUT_API_CONNECT_TIMEOUT_CAP_SECONDS = 20
IMAGE_DOWNLOAD_MAX_ATTEMPTS = 4
IMAGE_DOWNLOAD_BASE_DELAY_SECONDS = 1.0
IMAGE_DOWNLOAD_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


def _append_runtime_log(message: str) -> None:
    log_path = get_app_base_dir() / "app.log"
    timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamped}\n")
    except OSError:
        pass


def _encode_pdf(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("ascii")


def _call_layout_api_v2(
    file_path: Path,
    config: AppConfig,
    phase_callback: PhaseCallback | None = None,
) -> dict:
    headers: dict[str, str] = {}
    if config.api_token:
        headers["Authorization"] = f"bearer {config.api_token}"

    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }

    data = {
        "model": config.pdf_model or "PaddleOCR-VL-1.6",
        "optionalPayload": json.dumps(optional_payload),
    }

    _append_runtime_log(
        f"Submitting PDF job to {config.api_url} (model: {data['model']})..."
    )
    connect_timeout = max(
        5, min(config.request_timeout_seconds, LAYOUT_API_CONNECT_TIMEOUT_CAP_SECONDS)
    )

    try:
        with file_path.open("rb") as f:
            files = {"file": f}
            job_response = requests.post(
                config.api_url,
                headers=headers,
                data=data,
                files=files,
                timeout=(connect_timeout, config.request_timeout_seconds),
            )
    except requests.exceptions.RequestException as exc:
        raise PdfConversionError(f"Failed to submit OCR job: {exc}") from exc

    if job_response.status_code != 200:
        raise PdfConversionError(
            f"Layout API job submission failed with status {job_response.status_code}: {job_response.text[:400]}"
        )

    try:
        job_data = job_response.json()
        job_id = job_data.get("data", {}).get("jobId")
    except Exception as exc:
        raise PdfConversionError(
            f"Invalid JSON response from Layout API: {job_response.text[:300]}"
        ) from exc

    if not job_id:
        raise PdfConversionError(
            f"Layout API did not return a jobId: {job_response.text[:300]}"
        )

    _append_runtime_log(
        f"Job submitted successfully. Job ID: {job_id}. Polling for results..."
    )

    job_detail_url = f"{config.api_url.rstrip('/')}/{job_id}"
    poll_interval = 3
    start_poll_time = time.time()
    jsonl_url: str = ""

    while True:
        if time.time() - start_poll_time > config.request_timeout_seconds:
            raise PdfConversionError(
                f"Layout API job {job_id} timed out after {config.request_timeout_seconds} seconds."
            )

        try:
            poll_resp = requests.get(
                job_detail_url,
                headers=headers,
                timeout=(connect_timeout, 30),
            )
            if poll_resp.status_code == 200:
                poll_data = poll_resp.json().get("data", {})
                state = poll_data.get("state")
                if state == "pending":
                    _append_runtime_log(f"Job {job_id} is pending in queue...")
                elif state == "running":
                    progress = poll_data.get("extractProgress", {})
                    total_pages = progress.get("totalPages")
                    extracted_pages = progress.get("extractedPages")
                    if total_pages is not None and extracted_pages is not None:
                        log_msg = f"Job {job_id} is running: {extracted_pages}/{total_pages} pages extracted."
                        _append_runtime_log(log_msg)
                        if phase_callback:
                            phase_callback(f"converting ({extracted_pages}/{total_pages} pages)")
                    else:
                        _append_runtime_log(f"Job {job_id} is running...")
                elif state == "done":
                    progress = poll_data.get("extractProgress", {})
                    extracted = progress.get("extractedPages", "all")
                    _append_runtime_log(
                        f"Job {job_id} completed successfully (extracted {extracted} pages)."
                    )
                    result_url_info = poll_data.get("resultUrl", {})
                    jsonl_url = result_url_info.get("jsonUrl") or ""
                    break
                elif state == "failed":
                    error_msg = poll_data.get("errorMsg", "Unknown failure")
                    raise PdfConversionError(f"Layout API job failed: {error_msg}")
            else:
                _append_runtime_log(
                    f"Job polling returned status {poll_resp.status_code}, retrying..."
                )
        except requests.exceptions.RequestException as exc:
            _append_runtime_log(f"Job poll network warning: {exc}, will retry...")

        time.sleep(poll_interval)

    if not jsonl_url:
        raise PdfConversionError(
            "Layout API job completed but no jsonUrl was provided in the result."
        )

    _append_runtime_log(f"Downloading result JSONL from {jsonl_url}...")
    try:
        jsonl_response = requests.get(
            jsonl_url, timeout=(connect_timeout, config.request_timeout_seconds)
        )
        jsonl_response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise PdfConversionError(
            f"Failed to download Layout API JSONL results: {exc}"
        ) from exc

    combined_layout_results: list[dict] = []
    for line in jsonl_response.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            line_json = json.loads(line)
            res = line_json.get("result", {})
            combined_layout_results.extend(res.get("layoutParsingResults", []))
        except json.JSONDecodeError:
            continue

    return {"layoutParsingResults": combined_layout_results}


def _call_layout_api_v1(file_path: Path, config: AppConfig) -> dict:
    headers = {
        "Content-Type": "application/json",
    }
    if config.api_token:
        headers["Authorization"] = f"token {config.api_token}"
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
    connect_timeout = max(
        5, min(config.request_timeout_seconds, LAYOUT_API_CONNECT_TIMEOUT_CAP_SECONDS)
    )
    last_error: str | None = None

    for attempt in range(1, LAYOUT_API_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                config.api_url,
                json=payload,
                headers=headers,
                timeout=(connect_timeout, config.request_timeout_seconds),
            )
        except requests.exceptions.ConnectTimeout as exc:
            last_error = (
                "Unable to connect to the Layout API before the connection timeout expired. "
                "The service may be sleeping, overloaded, or temporarily unreachable. "
                f"Attempt {attempt}/{LAYOUT_API_MAX_ATTEMPTS}."
            )
            if attempt < LAYOUT_API_MAX_ATTEMPTS:
                time.sleep(attempt * 2)
                continue
            raise PdfConversionError(last_error) from exc
        except requests.exceptions.ReadTimeout as exc:
            last_error = (
                "The Layout API accepted the connection but did not finish responding before the timeout expired. "
                "Try increasing the timeout or retrying later. "
                f"Attempt {attempt}/{LAYOUT_API_MAX_ATTEMPTS}."
            )
            if attempt < LAYOUT_API_MAX_ATTEMPTS:
                time.sleep(attempt * 2)
                continue
            raise PdfConversionError(last_error) from exc
        except requests.exceptions.ConnectionError as exc:
            last_error = (
                "Unable to reach the Layout API. Please check the API URL, your network connection, proxy/VPN settings, "
                f"and whether the remote service is online. Attempt {attempt}/{LAYOUT_API_MAX_ATTEMPTS}."
            )
            if attempt < LAYOUT_API_MAX_ATTEMPTS:
                time.sleep(attempt * 2)
                continue
            raise PdfConversionError(last_error) from exc
        except requests.exceptions.RequestException as exc:
            raise PdfConversionError(f"Layout API request failed: {exc}") from exc

        if response.status_code == 200:
            data = response.json()
            result = data.get("result")
            if not isinstance(result, dict):
                raise PdfConversionError(
                    "Layout API response does not contain a valid result payload."
                )
            return result

        if (
            response.status_code in LAYOUT_API_RETRYABLE_STATUS_CODES
            and attempt < LAYOUT_API_MAX_ATTEMPTS
        ):
            last_error = (
                f"Layout API request failed with status {response.status_code}. "
                f"Attempt {attempt}/{LAYOUT_API_MAX_ATTEMPTS}."
            )
            time.sleep(attempt * 2)
            continue

        raise PdfConversionError(
            f"Layout API request failed with status {response.status_code}: {response.text[:400]}"
        )

    raise PdfConversionError(last_error or "Layout API request failed.")


def call_layout_api(
    file_path: Path,
    config: AppConfig,
    phase_callback: PhaseCallback | None = None,
) -> dict:
    if "jobs" in config.api_url.lower() or "api/v2" in config.api_url.lower():
        return _call_layout_api_v2(file_path, config, phase_callback=phase_callback)
    return _call_layout_api_v1(file_path, config)



def merge_markdown(result: dict) -> str:
    parts: list[str] = []
    for page in result.get("layoutParsingResults", []):
        markdown = page.get("markdown", {})
        text = markdown.get("text", "")
        if text:
            parts.append(text)
    return center_images_markdown("\n\n".join(parts))


def merge_trailing_hyphenated_words(markdown_text: str) -> str:
    newline = "\r\n" if "\r\n" in markdown_text else "\n"
    lines = markdown_text.splitlines()
    if len(lines) < 2:
        return markdown_text

    merged_lines = lines[:]

    for index, line in enumerate(merged_lines):
        hyphen_match = re.search(r"([A-Za-z]{2,})-[ \t]*$", line)
        if not hyphen_match:
            continue

        next_index = index + 1
        while next_index < len(merged_lines) and _is_dehyphenation_bridge_line(
            merged_lines[next_index].strip()
        ):
            next_index += 1

        if next_index >= len(merged_lines):
            continue

        continuation_match = re.match(
            r"^(\s*)([A-Za-z]{2,})(\b.*)$", merged_lines[next_index]
        )
        if not continuation_match:
            continue

        merged_lines[index] = (
            f"{line[: hyphen_match.start(1)]}{hyphen_match.group(1)}{continuation_match.group(2)}"
        )
        remainder = continuation_match.group(3)
        merged_lines[next_index] = f"{continuation_match.group(1)}{remainder.lstrip()}"

    trailing_newline = newline if markdown_text.endswith(("\n", "\r")) else ""
    return newline.join(merged_lines) + trailing_newline


def _is_dehyphenation_bridge_line(stripped_line: str) -> bool:
    if not stripped_line:
        return True
    lowered = stripped_line.lower()
    if stripped_line.startswith("!["):
        return True
    if lowered.startswith("<img"):
        return True
    if lowered.startswith("<figure") or lowered.startswith("</figure"):
        return True
    if lowered.startswith("<div") and ("<img" in lowered or "figure" in lowered):
        return True
    if re.match(r"^(figure|fig\.)\s*\d+\b", stripped_line, flags=re.IGNORECASE):
        return True
    return False


def center_images_markdown(markdown_text: str) -> str:
    centered = re.sub(
        r"(?im)^[ \t]*<div\b[^>]*>\s*(<img\b[^>]*?/?>)\s*</div>[ \t]*$",
        lambda match: f'<div align="center">{match.group(1)}</div>',
        markdown_text,
    )
    centered = re.sub(
        r"(?im)^[ \t]*(<img\b[^>]*?/?>)[ \t]*$",
        lambda match: f'<div align="center">{match.group(1)}</div>',
        centered,
    )
    centered = re.sub(
        r"(?im)^[ \t]*!\[([^\]]*)\]\(([^)]+)\)[ \t]*$",
        lambda match: (
            f'<div align="center"><img src="{match.group(2)}" alt="{html.escape(match.group(1), quote=True)}" /></div>'
        ),
        centered,
    )
    return centered


def _protect_math_expressions(markdown_text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    counter = 0

    def replace_block(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"MATH_BLOCK_PLACEHOLDER_{counter}"
        placeholders[token] = f'<div class="math-block">{match.group(0)}</div>'
        counter += 1
        return token

    def replace_inline(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"MATH_INLINE_PLACEHOLDER_{counter}"
        placeholders[token] = match.group(0)
        counter += 1
        return token

    protected = re.sub(
        r"(?s)(?<!\\)\$\$(.+?)(?<!\\)\$\$",
        replace_block,
        markdown_text,
    )
    protected = re.sub(
        r"(?s)(?<!\\)\\\[(.+?)(?<!\\)\\\]",
        replace_block,
        protected,
    )
    protected = re.sub(
        r"(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$",
        replace_inline,
        protected,
    )
    protected = re.sub(
        r"(?<!\\)\\\((.+?)(?<!\\)\\\)",
        replace_inline,
        protected,
    )

    return protected, placeholders


def _restore_math_expressions(html_text: str, placeholders: dict[str, str]) -> str:
    restored = html_text
    for token, value in placeholders.items():
        restored = restored.replace(token, value)
    return restored


def _render_markdown_html(markdown_text: str, title: str) -> str:
    protected_markdown, math_placeholders = _protect_math_expressions(markdown_text)
    body = markdown_lib.markdown(
        protected_markdown,
        extensions=["extra", "tables", "fenced_code", "toc"],
        output_format="html5",
    )
    body = _restore_math_expressions(body, math_placeholders)
    escaped_title = html.escape(title, quote=True)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f"  <title>{escaped_title}</title>\n"
        "  <style>\n"
        "    :root { color-scheme: light; }\n"
        "    body { margin: 0; font-family: Segoe UI, Helvetica, Arial, sans-serif; background: #f6f7fb; color: #1f2937; }\n"
        "    main { max-width: 960px; margin: 0 auto; padding: 40px 24px 64px; }\n"
        "    article { background: #ffffff; border-radius: 16px; padding: 32px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }\n"
        "    img { max-width: 100%; height: auto; }\n"
        "    pre { overflow-x: auto; background: #111827; color: #f9fafb; padding: 16px; border-radius: 12px; }\n"
        "    code { font-family: Consolas, Monaco, monospace; }\n"
        "    table { border-collapse: collapse; width: 100%; margin: 16px 0; }\n"
        "    th, td { border: 1px solid #d1d5db; padding: 8px 12px; vertical-align: top; }\n"
        "    th { background: #f3f4f6; }\n"
        "    blockquote { margin: 16px 0; padding: 12px 16px; border-left: 4px solid #94a3b8; background: #f8fafc; }\n"
        "    .math-block { overflow-x: auto; margin: 16px 0; }\n"
        "  </style>\n"
        "  <script>\n"
        "    window.MathJax = {\n"
        "      tex: {\n"
        "        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],\n"
        "        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]\n"
        "      },\n"
        "      svg: { fontCache: 'global' }\n"
        "    };\n"
        "  </script>\n"
        '  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>\n'
        "</head>\n"
        "<body>\n"
        "  <main>\n"
        "    <article>\n"
        f"{body}\n"
        "    </article>\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def _write_markdown_html(markdown_path: Path, markdown_text: str, title: str) -> Path:
    html_path = markdown_path.with_suffix(".html")
    html_path.write_text(_render_markdown_html(markdown_text, title), encoding="utf-8")
    return html_path


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


def _download_image_with_retry(image_url: str, timeout_seconds: int) -> bytes:
    last_error: Exception | None = None

    for attempt in range(1, IMAGE_DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(image_url, timeout=timeout_seconds)
            if response.status_code in IMAGE_DOWNLOAD_RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {response.status_code} for image download",
                    response=response,
                )
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else None
            if status_code not in IMAGE_DOWNLOAD_RETRYABLE_STATUS_CODES:
                raise PdfConversionError(
                    "Image download failed with a non-retryable status: "
                    f"{status_code} for {image_url}"
                ) from exc
            last_error = exc
            if attempt >= IMAGE_DOWNLOAD_MAX_ATTEMPTS:
                break
            delay_seconds = IMAGE_DOWNLOAD_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            _append_runtime_log(
                f"Retrying image download after HTTP {status_code} "
                f"(attempt {attempt + 1}/{IMAGE_DOWNLOAD_MAX_ATTEMPTS}, wait {delay_seconds:.1f}s): {image_url}"
            )
            time.sleep(delay_seconds)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt >= IMAGE_DOWNLOAD_MAX_ATTEMPTS:
                break
            delay_seconds = IMAGE_DOWNLOAD_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            _append_runtime_log(
                f"Retrying image download after {type(exc).__name__} "
                f"(attempt {attempt + 1}/{IMAGE_DOWNLOAD_MAX_ATTEMPTS}, wait {delay_seconds:.1f}s): {image_url}"
            )
            time.sleep(delay_seconds)

    raise PdfConversionError(
        "Failed to download image from the Layout API after multiple retries: "
        f"{image_url}. Last error: {last_error}"
    ) from last_error


def save_images(result: dict, output_dir: Path, timeout_seconds: int) -> int:
    image_count = 0
    layout_output_dir = output_dir / "layout"
    for page_index, page in enumerate(result.get("layoutParsingResults", [])):
        markdown = page.get("markdown", {})
        images = markdown.get("images", {})
        for relative_path, image_url in images.items():
            image_path = _resolve_output_image_path(output_dir, str(relative_path))
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(
                _download_image_with_retry(str(image_url), timeout_seconds)
            )
            image_count += 1

        output_images = page.get("outputImages", {})
        for image_name, image_url in output_images.items():
            safe_name = str(image_name).replace("\\", "_").replace("/", "_")
            layout_output_dir.mkdir(parents=True, exist_ok=True)
            image_path = layout_output_dir / f"{safe_name}_{page_index}.jpg"
            image_path.write_bytes(
                _download_image_with_retry(str(image_url), timeout_seconds)
            )
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
    result = call_layout_api(source_path, config, phase_callback=phase_callback)
    merged_markdown = merge_trailing_hyphenated_words(merge_markdown(result))
    markdown_path = document_output_dir / f"{source_path.stem}_full.md"
    markdown_path.write_text(merged_markdown, encoding="utf-8")
    html_path = _write_markdown_html(
        markdown_path, merged_markdown, f"{source_path.stem} Markdown"
    )
    image_count = save_images(
        result, document_output_dir, config.request_timeout_seconds
    )
    conversion_seconds = time.perf_counter() - conversion_start

    translated_markdown_path: Path | None = None
    translated_html_path: Path | None = None
    if translate_markdown:
        if phase_callback is not None:
            phase_callback("translating")
        translation_start = time.perf_counter()
        translator = NvidiaMarkdownTranslator(
            config, progress_callback=progress_callback
        )
        translated_markdown = translator.translate_markdown(merged_markdown)
        translated_markdown = center_images_markdown(translated_markdown)
        translated_markdown_path = (
            document_output_dir / f"{source_path.stem}_full_zh.md"
        )
        translated_markdown_path.write_text(translated_markdown, encoding="utf-8")
        translated_html_path = _write_markdown_html(
            translated_markdown_path,
            translated_markdown,
            f"{source_path.stem} Markdown Chinese",
        )
        translation_seconds = time.perf_counter() - translation_start

    page_count = len(result.get("layoutParsingResults", []))
    total_seconds = time.perf_counter() - total_start

    return ConversionResult(
        output_dir=document_output_dir,
        markdown_path=markdown_path,
        html_path=html_path,
        translated_markdown_path=translated_markdown_path,
        translated_html_path=translated_html_path,
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
        raise MarkdownTranslationError(
            f"Only Markdown or text files are supported: {source_path}"
        )
    if not config.nvidia_api_key:
        raise MarkdownTranslationError("NVIDIA_API_KEY is not configured.")

    base_output_path.mkdir(parents=True, exist_ok=True)

    original_markdown = merge_trailing_hyphenated_words(
        center_images_markdown(source_path.read_text(encoding="utf-8"))
    )
    source_html_path = _write_markdown_html(
        source_path, original_markdown, f"{source_path.stem} Markdown"
    )
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
    translated_html_path = _write_markdown_html(
        output_markdown_path,
        translated_markdown,
        f"{source_path.stem} Markdown Chinese",
    )

    return ConversionResult(
        output_dir=base_output_path,
        markdown_path=source_path,
        html_path=source_html_path,
        translated_markdown_path=output_markdown_path,
        translated_html_path=translated_html_path,
        stats=ConversionStats(page_count=0, image_count=0),
        timings=TimingStats(
            conversion_seconds=0.0,
            translation_seconds=translation_seconds,
            total_seconds=time.perf_counter() - total_start,
        ),
    )
