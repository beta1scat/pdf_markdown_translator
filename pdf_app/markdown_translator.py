from __future__ import annotations

from difflib import SequenceMatcher
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Callable

import requests

from .config import AppConfig
from .paths import get_app_base_dir


class TranslationError(RuntimeError):
    pass


CHUNK_TARGET_LIMIT = 5000
NVIDIA_API_MAX_ATTEMPTS = 4
NVIDIA_API_BASE_DELAY_SECONDS = 1.0
NVIDIA_API_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
ProgressCallback = Callable[[int, int], None]


PLACEHOLDER_SPLIT_REGEX = re.compile(
    r"(<<<(?:MULTILINE_CODE_\d+|LATEX_BLOCK_\d+|CODE_\d+|LATEX_INLINE_\d+|LINK_\d+|HTML_\d+|TABLE_\d+|IMAGE_\d+|ALGORITHM_\d+|REFERENCES_\d+)>>>)"
)
PLACEHOLDER_TEST_REGEX = re.compile(
    r"^<<<(?:MULTILINE_CODE_\d+|LATEX_BLOCK_\d+|CODE_\d+|LATEX_INLINE_\d+|LINK_\d+|HTML_\d+|TABLE_\d+|IMAGE_\d+|ALGORITHM_\d+|REFERENCES_\d+)>>>$"
)


@dataclass
class ProtectedMarkdown:
    text: str
    placeholders: dict[str, str]


def _append_runtime_log(message: str) -> None:
    log_path = get_app_base_dir() / "app.log"
    timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamped}\n")
    except OSError:
        pass


class RateLimiter:
    def __init__(self, max_requests: int, period_seconds: float) -> None:
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.request_times: deque[float] = deque()
        self._lock = Lock()

    def wait_for_slot(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while (
                    self.request_times
                    and now - self.request_times[0] >= self.period_seconds
                ):
                    self.request_times.popleft()

                if len(self.request_times) < self.max_requests:
                    self.request_times.append(now)
                    return

                wait_seconds = self.period_seconds - (now - self.request_times[0])
            time.sleep(max(wait_seconds, 0.1))


class NvidiaMarkdownTranslator:
    def __init__(
        self,
        config: AppConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.max_requests_per_minute, 60.0)
        self.progress_callback = progress_callback

    def translate_markdown(self, markdown_text: str) -> str:
        if not self.config.nvidia_api_key:
            raise TranslationError("NVIDIA_API_KEY is not configured.")

        protected = self._protect_markdown(markdown_text)
        chunks = self._split_into_chunks(protected.text)
        translated_chunks = [""] * len(chunks)
        total_chars = len(protected.text)
        translated_chars = 0
        progress_lock = Lock()

        def update_progress(chunk_length: int) -> None:
            nonlocal translated_chars
            with progress_lock:
                translated_chars += chunk_length
                self._report_progress(translated_chars, total_chars)

        def translate_chunk(index: int, chunk: str) -> tuple[int, str]:
            if self._should_skip_translation(chunk):
                update_progress(len(chunk))
                return index, chunk
            translated = self._call_nvidia_api(chunk)
            if self._should_retry_for_untranslated_output(chunk, translated):
                _append_runtime_log(
                    f"Retrying untranslated-looking translation chunk {index + 1}/{len(chunks)}"
                )
                translated = self._call_nvidia_api_with_system_prompt(
                    chunk,
                    (
                        "Translate all natural-language prose in the user's Markdown into Simplified Chinese. "
                        "Do not leave full English sentences or paragraphs unchanged unless they are code, URLs, placeholders, or proper names that must remain in English. "
                        "Preserve Markdown syntax, placeholders, citations, formulas, indentation, paragraph breaks, and line structure exactly. "
                        "Keep model names, paper names, dataset names, author names, and bracketed references intact when appropriate, but translate the surrounding prose. "
                        "Return only the translated Markdown content."
                    ),
                )
            update_progress(len(chunk))
            return index, translated

        max_workers = max(1, self.config.translation_concurrency)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(translate_chunk, index, chunk)
                for index, chunk in enumerate(chunks)
            ]
            for future in as_completed(futures):
                index, translated_chunk = future.result()
                translated_chunks[index] = translated_chunk

        translated = "".join(translated_chunks)
        restored = self._restore_markdown(translated, protected.placeholders)
        return self._force_translate_headings(restored)

    def _report_progress(self, translated_chars: int, total_chars: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(translated_chars, total_chars)

    def _split_into_chunks(self, text: str) -> list[str]:
        lines = text.splitlines(keepends=True)
        chunks: list[str] = []
        current_lines: list[str] = []
        current_length = 0

        def flush() -> None:
            nonlocal current_lines, current_length
            if current_lines:
                chunks.append("".join(current_lines))
                current_lines = []
                current_length = 0

        for line in lines:
            line_length = len(line)

            if current_lines and current_length + line_length > CHUNK_TARGET_LIMIT:
                flush()

            current_lines.append(line)
            current_length += line_length

            is_blank_boundary = not line.strip()
            if is_blank_boundary and current_length >= CHUNK_TARGET_LIMIT:
                flush()

        flush()

        if not chunks and text:
            chunks.append(text)
        return chunks

    def _call_nvidia_api(self, text: str) -> str:
        return self._call_nvidia_api_with_system_prompt(
            text,
            (
                "Translate the user's Markdown content into Simplified Chinese. "
                "Preserve Markdown syntax, placeholders, code, URLs, formulas, indentation, paragraph breaks, and line structure exactly. "
                "Do not modify, translate, remove, add spaces to, or reformat placeholders like <<<...>>>. "
                "Return only the translated content."
            ),
        )

    def _call_nvidia_api_with_system_prompt(self, text: str, system_prompt: str) -> str:
        leading_match = re.match(r"^\s*", text)
        trailing_match = re.search(r"\s*$", text)
        leading_whitespace = leading_match.group(0) if leading_match else ""
        trailing_whitespace = trailing_match.group(0) if trailing_match else ""
        headers = {
            "Authorization": f"Bearer {self.config.nvidia_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.nvidia_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False,
        }
        if self.config.nvidia_model.startswith("openai/gpt-oss-"):
            payload["reasoning_effort"] = "low"
        last_error: Exception | None = None

        for attempt in range(1, NVIDIA_API_MAX_ATTEMPTS + 1):
            try:
                self.rate_limiter.wait_for_slot()
                response = requests.post(
                    self.config.nvidia_api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout_seconds,
                )
                if response.status_code in NVIDIA_API_RETRYABLE_STATUS_CODES:
                    raise requests.exceptions.HTTPError(
                        f"HTTP {response.status_code} for NVIDIA translation request",
                        response=response,
                    )
                if response.status_code != 200:
                    raise TranslationError(
                        f"NVIDIA API request failed with status {response.status_code}: {response.text[:400]}"
                    )

                data = response.json()
                content = self._extract_content(data)

                if not isinstance(content, str) or not content.strip():
                    raise TranslationError(
                        "NVIDIA API returned empty translated content. "
                        f"Response preview: {self._preview_response(data)}"
                    )
                normalized_content = html.unescape(content)
                if leading_whitespace and not normalized_content.startswith(
                    leading_whitespace
                ):
                    normalized_content = leading_whitespace + normalized_content.lstrip(
                        " \t\r\n"
                    )
                if trailing_whitespace and not normalized_content.endswith(
                    trailing_whitespace
                ):
                    normalized_content = (
                        normalized_content.rstrip(" \t\r\n") + trailing_whitespace
                    )
                return normalized_content
            except requests.exceptions.HTTPError as exc:
                response = exc.response
                status_code = response.status_code if response is not None else None
                if status_code not in NVIDIA_API_RETRYABLE_STATUS_CODES:
                    raise TranslationError(
                        "NVIDIA API request failed with a non-retryable status: "
                        f"{status_code}"
                    ) from exc
                last_error = exc
                if attempt >= NVIDIA_API_MAX_ATTEMPTS:
                    break
                delay_seconds = NVIDIA_API_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                _append_runtime_log(
                    f"Retrying NVIDIA translation after HTTP {status_code} "
                    f"(attempt {attempt + 1}/{NVIDIA_API_MAX_ATTEMPTS}, wait {delay_seconds:.1f}s)"
                )
                time.sleep(delay_seconds)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.Timeout,
            ) as exc:
                last_error = exc
                if attempt >= NVIDIA_API_MAX_ATTEMPTS:
                    break
                delay_seconds = NVIDIA_API_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                _append_runtime_log(
                    f"Retrying NVIDIA translation after {type(exc).__name__} "
                    f"(attempt {attempt + 1}/{NVIDIA_API_MAX_ATTEMPTS}, wait {delay_seconds:.1f}s)"
                )
                time.sleep(delay_seconds)
            except requests.exceptions.RequestException as exc:
                raise TranslationError(f"NVIDIA API request failed: {exc}") from exc

        raise TranslationError(
            "NVIDIA API request failed after multiple retries. "
            f"Last error: {last_error}"
        ) from last_error

    def _extract_content(self, data: object) -> str:
        if not isinstance(data, dict):
            raise TranslationError(
                f"Unexpected NVIDIA API response type: {type(data).__name__}"
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TranslationError(
                "Unexpected NVIDIA API response format: missing choices. "
                f"Response preview: {self._preview_response(data)}"
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise TranslationError(
                "Unexpected NVIDIA API response format: invalid choice item. "
                f"Response preview: {self._preview_response(data)}"
            )

        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            normalized = self._normalize_content(content)
            if normalized is not None:
                return normalized
            if self._has_reasoning_without_content(message):
                raise TranslationError(
                    "NVIDIA API returned reasoning text but no final content. "
                    "This commonly happens with heading-only follow-up requests on some models. "
                    f"Response preview: {self._preview_response(data)}"
                )

        text = first_choice.get("text")
        if isinstance(text, str) and text.strip():
            return text

        delta = first_choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            normalized = self._normalize_content(content)
            if normalized is not None:
                return normalized

        raise TranslationError(
            "Unexpected NVIDIA API response format: no usable content field. "
            f"Response preview: {self._preview_response(data)}"
        )

    def _normalize_content(self, content: object) -> str | None:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            joined = "".join(text_parts)
            return joined if joined else None

        return None

    def _has_reasoning_without_content(self, message: dict[str, object]) -> bool:
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return True
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return True
        return False

    def _preview_response(self, data: object, limit: int = 800) -> str:
        try:
            serialized = json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(data)
        return serialized[:limit]

    def _protect_markdown(self, text: str) -> ProtectedMarkdown:
        placeholders: dict[str, str] = {}
        counters = {
            "MULTILINE_CODE": 100,
            "LATEX_BLOCK": 100,
            "CODE": 100,
            "LATEX_INLINE": 100,
            "LINK": 100,
            "HTML": 100,
            "TABLE": 100,
            "IMAGE": 100,
            "ALGORITHM": 100,
            "REFERENCES": 100,
        }

        def make_placeholder(kind: str, content: str) -> str:
            placeholder = f"<<<{kind}_{counters[kind]}>>>"
            counters[kind] += 1
            placeholders[placeholder] = content
            return placeholder

        protected_text = re.sub(
            r"```[\s\S]*?```",
            lambda match: make_placeholder("MULTILINE_CODE", match.group(0)),
            text,
        )
        protected_text = re.sub(
            r"<table\b[\s\S]*?</table>",
            lambda match: make_placeholder("TABLE", match.group(0)),
            protected_text,
            flags=re.IGNORECASE,
        )
        protected_text = self._normalize_div_blocks(protected_text)
        protected_text = re.sub(
            r"<figure\b[\s\S]*?</figure>",
            lambda match: make_placeholder("IMAGE", match.group(0)),
            protected_text,
            flags=re.IGNORECASE,
        )
        protected_text = re.sub(
            r"<img\b[^>]*?/?>",
            lambda match: make_placeholder("IMAGE", match.group(0)),
            protected_text,
            flags=re.IGNORECASE,
        )
        protected_text = re.sub(
            r"<algorithm\b[\s\S]*?</algorithm>",
            lambda match: make_placeholder("ALGORITHM", match.group(0)),
            protected_text,
            flags=re.IGNORECASE,
        )
        protected_text = re.sub(
            r"<algorithmic\b[\s\S]*?</algorithmic>",
            lambda match: make_placeholder("ALGORITHM", match.group(0)),
            protected_text,
            flags=re.IGNORECASE,
        )
        protected_text = re.sub(
            r"```(?:algorithm|pseudo|pseudocode)[\s\S]*?```",
            lambda match: make_placeholder("ALGORITHM", match.group(0)),
            protected_text,
            flags=re.IGNORECASE,
        )
        protected_text = re.sub(
            r"\$\$[\s\S]*?\$\$",
            lambda match: make_placeholder("LATEX_BLOCK", match.group(0)),
            protected_text,
        )

        protected_text = self._protect_reference_sections(
            protected_text, make_placeholder
        )

        processed_lines: list[str] = []
        for line in protected_text.splitlines(keepends=True):
            current = line
            line_ending = ""
            if current.endswith("\r\n"):
                current = current[:-2]
                line_ending = "\r\n"
            elif current.endswith("\n") or current.endswith("\r"):
                line_ending = current[-1]
                current = current[:-1]
            stripped = current.strip()

            if self._is_non_body_line(stripped):
                processed_lines.append(make_placeholder("TABLE", current) + line_ending)
                continue

            current = re.sub(
                r"`([^`]+?)`",
                lambda match: make_placeholder("CODE", match.group(0)),
                current,
            )
            current = re.sub(
                r"\$([^\$]+?)\$",
                lambda match: match.group(0)
                if re.fullmatch(r"[\s\d,.]+", match.group(1))
                else make_placeholder("LATEX_INLINE", match.group(0)),
                current,
            )
            current = re.sub(
                r"<!--[\s\S]*?-->",
                lambda match: make_placeholder("HTML", match.group(0)),
                current,
            )
            current = re.sub(
                r"<([a-zA-Z][a-zA-Z0-9-]*)(?:\s+[^>]*)?/?>|</([a-zA-Z][a-zA-Z0-9-]*)>",
                lambda match: make_placeholder("HTML", match.group(0)),
                current,
            )
            current = re.sub(
                r"(!\[.*?\]\(.*?\))",
                lambda match: make_placeholder("IMAGE", match.group(0)),
                current,
            )
            current = re.sub(
                r"(?<!!)(\[.*?\]\(.*?\))",
                lambda match: make_placeholder("LINK", match.group(0)),
                current,
            )
            processed_lines.append(current + line_ending)

        return ProtectedMarkdown(
            text="".join(processed_lines), placeholders=placeholders
        )

    def _protect_reference_sections(
        self,
        text: str,
        make_placeholder: Callable[[str, str], str],
    ) -> str:
        lines = text.splitlines(keepends=True)
        if not lines:
            return text

        protected_lines: list[str] = []
        inside_references = False

        for line in lines:
            stripped = line.strip()
            heading_level = self._get_heading_level(stripped)

            if inside_references and heading_level is not None:
                inside_references = False

            if (
                not inside_references
                and heading_level is not None
                and self._is_reference_heading(stripped)
            ):
                inside_references = True

            if inside_references:
                protected_lines.append(make_placeholder("REFERENCES", line))
            else:
                protected_lines.append(line)

        return "".join(protected_lines)

    def _is_reference_heading(self, stripped: str) -> bool:
        normalized = re.sub(r"^#{1,6}\s*", "", stripped).strip().lower()
        return normalized in {"references", "bibliography"}

    def _get_heading_level(self, stripped: str) -> int | None:
        match = re.match(r"^(#{1,6})\s+", stripped)
        if match:
            return len(match.group(1))
        return None

    def _is_non_body_line(self, stripped: str) -> bool:
        if not stripped:
            return False
        lowered = stripped.lower()
        if lowered.startswith("<table") or lowered.startswith("</table"):
            return True
        if lowered.startswith("<tr") or lowered.startswith("</tr"):
            return True
        if lowered.startswith("<td") or lowered.startswith("</td"):
            return True
        if lowered.startswith("<th") or lowered.startswith("</th"):
            return True
        if lowered.startswith("<figure") or lowered.startswith("</figure"):
            return True
        if lowered.startswith("<img"):
            return True
        if stripped.startswith("|") and stripped.endswith("|"):
            return True
        if re.fullmatch(r"\|?[\s:\-]+(?:\|[\s:\-]+)+\|?", stripped):
            return True
        if re.fullmatch(r"[-+| ]{3,}", stripped) and "|" in stripped:
            return True
        return False

    def _normalize_div_blocks(self, text: str) -> str:
        # Layout OCR often wraps normal paragraphs in div containers.
        unwrapped = text
        div_block_pattern = re.compile(
            r"<div\b[^>]*>([\s\S]*?)</div>", flags=re.IGNORECASE
        )
        while True:
            updated = div_block_pattern.sub(lambda match: match.group(1), unwrapped)
            if updated == unwrapped:
                break
            unwrapped = updated
        return re.sub(r"</?div\b[^>]*>", "", unwrapped, flags=re.IGNORECASE)

    def _restore_markdown(self, text: str, placeholders: dict[str, str]) -> str:
        restored = text
        for placeholder, content in sorted(
            placeholders.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            restored = restored.replace(placeholder, content)
        return restored

    def _force_translate_headings(self, text: str) -> str:
        lines = text.splitlines(keepends=True)
        if not lines:
            return text

        heading_map: dict[str, str] = {}
        heading_pattern = re.compile(r"^(#{1,6})(\s+)(.+?)(\s*)$")

        for line in lines:
            line_body = line.rstrip("\r\n")
            match = heading_pattern.match(line_body)
            if not match:
                continue
            heading_text = match.group(3)
            if not re.search(r"[A-Za-z]", heading_text):
                continue
            if heading_text not in heading_map:
                heading_map[heading_text] = self._translate_heading_text(heading_text)

        if not heading_map:
            return text

        translated_lines: list[str] = []
        for line in lines:
            line_ending = ""
            line_body = line
            if line_body.endswith("\r\n"):
                line_body = line_body[:-2]
                line_ending = "\r\n"
            elif line_body.endswith("\n") or line_body.endswith("\r"):
                line_ending = line_body[-1]
                line_body = line_body[:-1]

            match = heading_pattern.match(line_body)
            if not match:
                translated_lines.append(line)
                continue

            heading_text = match.group(3)
            translated_heading = heading_map.get(heading_text)
            if translated_heading is None:
                translated_lines.append(line)
                continue

            translated_lines.append(
                f"{match.group(1)}{match.group(2)}{translated_heading}{match.group(4)}{line_ending}"
            )
        return "".join(translated_lines)

    def _translate_heading_text(self, heading_text: str) -> str:
        try:
            translated = self._call_nvidia_api_with_system_prompt(
                heading_text,
                (
                    "Translate the heading text into Simplified Chinese. "
                    "Keep numbering, punctuation, abbreviations, and inline Markdown syntax intact. "
                    "Return only the translated heading text."
                ),
            )
        except TranslationError:
            return heading_text
        return translated.strip() or heading_text

    def _should_skip_translation(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if PLACEHOLDER_TEST_REGEX.fullmatch(stripped):
            return True
        if re.fullmatch(r"https?://\S+", stripped):
            return True
        if not re.search(r"[A-Za-z]", text):
            return True
        if re.fullmatch(
            r"(?:\s*" + PLACEHOLDER_SPLIT_REGEX.pattern + r"\s*)+", stripped
        ):
            return True
        return False

    def _should_retry_for_untranslated_output(
        self, source_text: str, translated_text: str
    ) -> bool:
        source_english_words = len(re.findall(r"\b[A-Za-z]{2,}\b", source_text))
        if source_english_words < 20:
            return False

        source_plain = self._normalize_for_translation_similarity(source_text)
        translated_plain = self._normalize_for_translation_similarity(translated_text)
        if not source_plain or not translated_plain:
            return False

        similarity = SequenceMatcher(None, source_plain, translated_plain).ratio()
        translated_english_words = len(re.findall(r"\b[A-Za-z]{2,}\b", translated_text))
        translated_chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", translated_text))

        return (
            similarity >= 0.85
            and translated_english_words >= max(12, int(source_english_words * 0.7))
            and translated_chinese_chars < max(8, source_english_words // 5)
        )

    def _normalize_for_translation_similarity(self, text: str) -> str:
        normalized = PLACEHOLDER_SPLIT_REGEX.sub(" ", text)
        normalized = re.sub(r"https?://\S+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip().lower()
