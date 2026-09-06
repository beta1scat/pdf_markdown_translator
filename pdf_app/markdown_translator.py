from __future__ import annotations

import html
import json
import re
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable

import requests

from .cache import TranslationCache, get_translation_cache
from .config import AppConfig
from .paths import append_runtime_log


class TranslationError(RuntimeError):
    pass


class TranslationCancelledError(TranslationError):
    pass


CHUNK_TARGET_LIMIT = 5000
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]
LogCallback = Callable[[str], None]

LLM_ERROR_EXPLANATIONS: dict[int, dict[str, str]] = {
    400: {
        "title": "Bad Request (请求格式/参数不合法)",
        "scenario_and_reason": "请求格式或参数不合法。通常由于传入了当前模型不支持的参数，或 messages 消息体格式不符合服务端规范。",
        "solution": "请检查模型名称是否正确，或尝试在 Settings 中切换通用对话大模型。",
    },
    401: {
        "title": "Unauthorized (身份认证失败 / API Key 无效)",
        "scenario_and_reason": "API Key 填写错误、前后带有多余空格、Key 已过期或已被平台撤销。",
        "solution": "请检查 Settings 中的 LLM API Key，确认从对应平台（智谱 open.bigmodel.cn / Google AI Studio aistudio.google.com / NVIDIA build.nvidia.com）获取有效 Key。",
    },
    403: {
        "title": "Forbidden (拒绝访问 / 权限受限或内容违规)",
        "scenario_and_reason": "账号无权访问该接口、或待翻译文本触发了云端内容安全审查 (Safety Guardrail 阻断)。",
        "solution": "请检查账号配额与状态，或检查待翻译文本中是否包含敏感违规内容。",
    },
    404: {
        "title": "Not Found (模型不存在或接口地址有误)",
        "scenario_and_reason": "请求的模型名称不存在，或 API URL 路径不正确。部分平台特殊模型需要企业授权或已迁移。",
        "solution": "请在 Settings 中核对 API URL 和模型名称，或点击 Refresh 选择可用模型。",
    },
    410: {
        "title": "Gone (模型已下线 / 生命周期结束)",
        "scenario_and_reason": "该模型版本已被官方下线淘汰 (EOL)，云端服务实例已停机注销。",
        "solution": "请在 Settings 中切换到更新代际的模型版本。",
    },
    422: {
        "title": "Unprocessable Entity (语义无法处理)",
        "scenario_and_reason": "单次请求输入的文本长度超过了模型的最大上下文窗口限制。",
        "solution": "请缩减单次分块大小或缩短文本长度后重试。",
    },
    429: {
        "title": "Too Many Requests (触发频次/并发限流)",
        "scenario_and_reason": "请求并发数或每分钟请求频次 (RPM/TPM) 超出了服务商平台的配额限制。",
        "solution": "程序会自动进行退避重试；也可在 Settings 中将 Translation Concurrency（并发数）调低至 1~2。",
    },
    500: {
        "title": "Internal Server Error (云端内部故障)",
        "scenario_and_reason": "LLM 云端推理服务器发生内部错误（如模型服务崩溃、显存溢出、计算异常等）。",
        "solution": "此为云端临时故障，程序会自动排队重试，若持续报错可尝试更换其他模型或服务商。",
    },
    502: {
        "title": "Bad Gateway (网关错误)",
        "scenario_and_reason": "前置反向代理网关无法连通后方的算力集群或模型服务实例。",
        "solution": "属于云端网络或节点暂时波动，请稍后重试，或调大超时时间。",
    },
    503: {
        "title": "Service Unavailable (服务暂时不可用 / 队列打满)",
        "scenario_and_reason": "该模型的算力集群当前排队已满，或服务正在扩缩容调度中。",
        "solution": "服务当前高负载，请稍等数秒后由程序自动重试，或切换至其他空闲模型。",
    },
    504: {
        "title": "Gateway Timeout (网关超时)",
        "scenario_and_reason": "网关在超时期限内未等到后端推理服务返回响应（长文本处理时常见）。",
        "solution": "请在 Settings 中调大 Request Timeout 超时时间，或稍后重试。",
    },
}

NVIDIA_ERROR_EXPLANATIONS = LLM_ERROR_EXPLANATIONS


def format_llm_api_error(
    status_code: int, response_text: str, model_name: str = ""
) -> str:
    raw_preview = response_text.strip()
    if len(raw_preview) > 300:
        raw_preview = raw_preview[:300] + "..."

    if status_code == 404 and "not found for account" in response_text.lower():
        title = "Not Found / Permission Denied (当前账号无此模型访问权限)"
        reason = (
            f"模型 '{model_name}' 在当前平台上属于受限/企业专属模型。"
            "云端未向您的账号开放该模型的实例访问授权。"
        )
        solution = "请在 Settings 中更换为当前账号授权可用的公开模型（如 moonshotai/kimi-k3 或 glm-4-flash）。"
    elif status_code in LLM_ERROR_EXPLANATIONS:
        info = LLM_ERROR_EXPLANATIONS[status_code]
        title = info["title"]
        reason = info["scenario_and_reason"]
        solution = info["solution"]
    else:
        title = f"HTTP {status_code} Error"
        reason = f"LLM API 返回了未预期的 HTTP {status_code} 状态码。"
        solution = "请检查网络连接、API 状态或稍后重试。"

    return (
        f"LLM API 请求失败 (HTTP {status_code} - {title}):\n"
        f"  • 【官方原始响应】: {raw_preview}\n"
        f"  • 【官方触发场景与原因】: {reason}\n"
        f"  • 【推荐解决办法】: {solution}"
    )


format_nvidia_api_error = format_llm_api_error


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


class RateLimiter:
    def __init__(self, max_requests: int, period_seconds: float) -> None:
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.request_times: deque[float] = deque()
        self._lock = Lock()

    def wait_for_slot(self, cancel_check: CancelCheck | None = None) -> None:
        while True:
            if cancel_check is not None and cancel_check():
                raise TranslationCancelledError("Translation cancelled by user.")
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

            sleep_end = time.monotonic() + max(wait_seconds, 0.1)
            while time.monotonic() < sleep_end:
                if cancel_check is not None and cancel_check():
                    raise TranslationCancelledError("Translation cancelled by user.")
                time.sleep(min(0.05, max(0.01, sleep_end - time.monotonic())))


class NvidiaMarkdownTranslator:
    _active_instances: set[NvidiaMarkdownTranslator] = set()
    _instances_lock = Lock()

    @classmethod
    def abort_all_active(cls) -> None:
        with cls._instances_lock:
            for inst in list(cls._active_instances):
                inst.abort()

    def __init__(
        self,
        config: AppConfig,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
        log_callback: LogCallback | None = None,
        cache: TranslationCache | None = None,
    ) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.max_requests_per_minute, 60.0)
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.log_callback = log_callback
        self.cache: TranslationCache = cache if cache is not None else get_translation_cache()
        self._aborted = Event()
        self.session: requests.Session | None = requests.Session()

    def abort(self) -> None:
        """Immediately aborts in-flight network requests and signals all worker threads to terminate."""
        self._aborted.set()
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass

    def _log(self, message: str) -> None:
        if self._aborted.is_set():
            return
        if self.log_callback is not None:
            try:
                self.log_callback(message)
            except Exception:
                pass
        else:
            append_runtime_log(message)

    def translate_markdown(
        self,
        markdown_text: str,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        self._aborted.clear()
        if self.session is None:
            self.session = requests.Session()

        def is_cancelled_or_aborted() -> bool:
            if self._aborted.is_set():
                return True
            if cancel_check is not None and cancel_check():
                return True
            if self.cancel_check is not None and self.cancel_check():
                return True
            return False

        if is_cancelled_or_aborted():
            raise TranslationCancelledError("Translation cancelled by user.")

        if not self.config.llm_api_key:
            raise TranslationError("LLM_API_KEY is not configured. Please configure it in Settings.")

        with self._instances_lock:
            self._active_instances.add(self)

        try:
            protected = self._protect_markdown(markdown_text)
            chunks = self._split_into_chunks(protected.text)
            total_chunks = len(chunks)
            translated_chunks = [""] * total_chunks
            total_chars = len(protected.text)
            translated_chars = 0
            progress_lock = Lock()
            max_workers = max(1, self.config.translation_concurrency)

            self._log(
                f"[模型推理] 待翻译文本准备就绪，已划分为 {total_chunks} 个请求分块 "
                f"(总计 {total_chars:,} 字符，并发线程数: {max_workers}，目标模型: {self.config.llm_model})"
            )

            def update_progress(chunk_length: int) -> None:
                if is_cancelled_or_aborted():
                    return
                nonlocal translated_chars
                with progress_lock:
                    translated_chars += chunk_length
                    self._report_progress(translated_chars, total_chars)

            def translate_chunk(index: int, chunk: str) -> tuple[int, str]:
                if is_cancelled_or_aborted():
                    raise TranslationCancelledError("Translation cancelled by user.")
                chunk_prefix = f"分块 [{index + 1}/{total_chunks}]"
                if is_cancelled_or_aborted():
                    raise TranslationCancelledError("Translation cancelled by user.")
                if self._should_skip_translation(chunk):
                    self._log(
                        f"[模型通信] {chunk_prefix} 仅包含结构标记与代码占位符，已跳过调用直接透传"
                    )
                    update_progress(len(chunk))
                    return index, chunk

                # Check breakpoint translation cache
                cache_key = self.cache.make_cache_key(self.config.llm_model, "chunk", chunk)
                cached = self.cache.get(cache_key)
                if cached is not None:
                    self._log(
                        f"[模型通信] {chunk_prefix} 命中本地断点缓存 (已跳过远程 API 调用，直接复用)"
                    )
                    update_progress(len(chunk))
                    return index, cached

                chunk_chars = len(chunk)
                self._log(
                    f"[模型通信] {chunk_prefix} 正在组装请求载荷 ({chunk_chars:,} 字符)，向推理端点发起调用，等待模型响应..."
                )
                chunk_start = time.perf_counter()
                translated = self._call_llm_api(
                    chunk, chunk_desc=chunk_prefix, cancel_check=is_cancelled_or_aborted
                )
                elapsed = time.perf_counter() - chunk_start
                self._log(
                    f"[模型通信] {chunk_prefix} 成功接收推理响应 (耗时 {elapsed:.2f}s, HTTP 200)，已完成语义解析与格式归一化"
                )

                # Persist translated chunk to cache immediately
                self.cache.set(cache_key, self.config.llm_model, chunk, translated)

                if is_cancelled_or_aborted():
                    raise TranslationCancelledError("Translation cancelled by user.")
                update_progress(len(chunk))
                return index, translated

            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures: list[concurrent.futures.Future[tuple[int, str]]] = []
            try:
                for index, chunk in enumerate(chunks):
                    if is_cancelled_or_aborted():
                        raise TranslationCancelledError("Translation cancelled by user.")
                    futures.append(executor.submit(translate_chunk, index, chunk))

                pending: set[concurrent.futures.Future[tuple[int, str]]] = set(futures)
                while pending:
                    if is_cancelled_or_aborted():
                        self.abort()
                        for f in pending:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise TranslationCancelledError("Translation cancelled by user.")

                    done, pending = concurrent.futures.wait(
                        pending, timeout=0.08, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        index, translated_chunk = future.result()
                        translated_chunks[index] = translated_chunk

            except BaseException as exc:
                self.abort()
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                if isinstance(exc, TranslationCancelledError):
                    raise
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            if is_cancelled_or_aborted():
                raise TranslationCancelledError("Translation cancelled by user.")

            translated = "".join(translated_chunks)
            restored = self._restore_markdown(
                translated, protected.placeholders, source_text=protected.text
            )
            final_result = self._force_translate_headings(restored, cancel_check=is_cancelled_or_aborted)
            self._log(
                f"[模型推理] 文档翻译处理完毕，已合并还原 {total_chunks} 个分块并完成占位符复原 (总字符数: {len(final_result):,})"
            )
            return final_result
        finally:
            with self._instances_lock:
                self._active_instances.discard(self)

    def _report_progress(self, translated_chars: int, total_chars: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(translated_chars, total_chars)

    def _split_into_chunks(self, text: str) -> list[str]:
        target_limit = getattr(self.config, "translation_chunk_size", CHUNK_TARGET_LIMIT)
        target_limit = max(500, int(target_limit))
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

            if current_lines and current_length + line_length > target_limit:
                flush()

            current_lines.append(line)
            current_length += line_length

            is_blank_boundary = not line.strip()
            if is_blank_boundary and current_length >= target_limit:
                flush()

        flush()

        if not chunks and text:
            chunks.append(text)
        return chunks

    def _call_llm_api(
        self,
        text: str,
        chunk_desc: str = "",
        cancel_check: CancelCheck | None = None,
    ) -> str:
        return self._call_llm_api_with_system_prompt(
            text,
            (
                "Translate the user's Markdown content into Simplified Chinese. "
                "Output the translated Chinese text directly without any thinking process, reasoning steps, or conversational preamble. "
                "Preserve Markdown syntax, placeholders, code, URLs, formulas, indentation, paragraph breaks, and line structure exactly. "
                "CRITICAL REQUIREMENTS:\n"
                "1. Do not modify, translate, remove, add spaces to, or reformat placeholders like <<<...>>> (e.g. <<<IMAGE_...>>>, <<<TABLE_...>>>, <<<LATEX_...>>>).\n"
                "2. Under NO circumstances should you omit, skip, or delete any image placeholder, table placeholder, or figure/table caption (e.g. 'Fig. 1: ...', 'Table 1: ...'), even if they appear in the middle of a paragraph or sentence. You MUST translate the caption and retain the placeholder in place.\n"
                "3. Return only the translated content."
            ),
            cancel_check=cancel_check,
            chunk_desc=chunk_desc,
        )

    def _call_llm_api_with_system_prompt(
        self,
        text: str,
        system_prompt: str,
        cancel_check: CancelCheck | None = None,
        chunk_desc: str = "",
    ) -> str:
        def is_cancelled() -> bool:
            if self._aborted.is_set():
                return True
            if cancel_check is not None and cancel_check():
                return True
            if self.cancel_check is not None and self.cancel_check():
                return True
            return False

        leading_match = re.match(r"^\s*", text)
        trailing_match = re.search(r"\s*$", text)
        leading_whitespace = leading_match.group(0) if leading_match else ""
        trailing_whitespace = trailing_match.group(0) if trailing_match else ""
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload: dict[str, object] = {
            "model": self.config.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 8192,
            "temperature": 0.1,
            "stream": False,
        }

        # For models supporting thinking/reasoning mode (e.g. Zhipu GLM-4.7-Flash / GLM-4.5-Flash),
        # disable thinking mode so tokens are not wasted on reasoning_content during translation.
        if "bigmodel.cn" in self.config.llm_api_url or self.config.llm_provider == "Zhipu AI (智谱)":
            payload["thinking"] = {"type": "disabled"}

        max_attempts = 8
        response: requests.Response | None = None
        desc_prefix = f"{chunk_desc} " if chunk_desc else ""

        for attempt in range(1, max_attempts + 1):
            if is_cancelled():
                raise TranslationCancelledError("Translation cancelled by user.")

            self.rate_limiter.wait_for_slot(cancel_check=is_cancelled)

            if is_cancelled():
                raise TranslationCancelledError("Translation cancelled by user.")

            if self.session is None:
                raise TranslationCancelledError("Translation cancelled by user.")

            try:
                response = self.session.post(
                    self.config.llm_api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout_seconds,
                )
            except requests.exceptions.RequestException as exc:
                if is_cancelled():
                    raise TranslationCancelledError("Translation cancelled by user.") from exc
                if attempt == max_attempts:
                    raise TranslationError(
                        f"LLM API network error after {max_attempts} attempts: {exc}"
                    ) from exc
                wait_time = attempt * 2
                self._log(
                    f"[模型通信] {desc_prefix}网络连接异常 ({exc})，将在 {wait_time}s 后执行第 {attempt + 1}/{max_attempts} 次重试..."
                )
                sleep_end = time.monotonic() + wait_time
                while time.monotonic() < sleep_end:
                    if is_cancelled():
                        raise TranslationCancelledError("Translation cancelled by user.")
                    time.sleep(min(0.05, max(0.01, sleep_end - time.monotonic())))
                continue

            if response.status_code == 200:
                break

            if (
                response.status_code in {429, 500, 502, 503, 504}
                and attempt < max_attempts
            ):
                retry_after = response.headers.get("Retry-After")
                retry_match = re.search(
                    r"retry after\s+([0-9]+(?:\.[0-9]+)?)\s*s",
                    response.text,
                    re.IGNORECASE,
                )
                try:
                    if retry_after:
                        wait_time = float(retry_after)
                    elif retry_match:
                        wait_time = float(retry_match.group(1)) + 1.0
                    else:
                        wait_time = float(2 ** attempt)
                except ValueError:
                    wait_time = float(2 ** attempt)
                wait_time = max(2.0, min(wait_time, 45.0))
                self._log(
                    f"[模型通信] {desc_prefix}触发服务商限流/服务暂时繁忙 (HTTP {response.status_code})，将在 {wait_time:.1f}s 后执行第 {attempt + 1}/{max_attempts} 次退避重试..."
                )
                sleep_end = time.monotonic() + wait_time
                while time.monotonic() < sleep_end:
                    if is_cancelled():
                        raise TranslationCancelledError("Translation cancelled by user.")
                    time.sleep(min(0.05, max(0.01, sleep_end - time.monotonic())))
                continue

            if response.status_code == 400 and "thinking" in payload:
                # If the chosen model doesn't support the thinking parameter, strip it and retry immediately
                self._log(
                    f"[模型通信] {desc_prefix}当前模型不支持 thinking 控制参数，已自动切换标准模式重试..."
                )
                del payload["thinking"]
                continue

            error_detail = format_llm_api_error(
                response.status_code, response.text, self.config.llm_model
            )
            raise TranslationError(error_detail)

        if response is None or response.status_code != 200:
            status = response.status_code if response else -1
            resp_txt = response.text if response else "No response from server"
            error_detail = format_llm_api_error(
                status, resp_txt, self.config.llm_model
            )
            raise TranslationError(error_detail)

        data = response.json()
        content = self._extract_content(data)

        if not isinstance(content, str) or not content.strip():
            raise TranslationError(
                "LLM API returned empty translated content. "
                f"Response preview: {self._preview_response(data)}"
            )
        normalized_content = html.unescape(content)
        if leading_whitespace and not normalized_content.startswith(leading_whitespace):
            normalized_content = leading_whitespace + normalized_content.lstrip(
                " \t\r\n"
            )
        if trailing_whitespace and not normalized_content.endswith(trailing_whitespace):
            normalized_content = (
                normalized_content.rstrip(" \t\r\n") + trailing_whitespace
            )
        return normalized_content

    # Backward compatibility aliases
    _call_nvidia_api = _call_llm_api
    _call_nvidia_api_with_system_prompt = _call_llm_api_with_system_prompt

    def _extract_content(self, data: object) -> str:
        if not isinstance(data, dict):
            raise TranslationError(
                f"Unexpected LLM API response type: {type(data).__name__}"
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TranslationError(
                "Unexpected LLM API response format: missing choices. "
                f"Response preview: {self._preview_response(data)}"
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise TranslationError(
                "Unexpected LLM API response format: invalid choice item. "
                f"Response preview: {self._preview_response(data)}"
            )

        finish_reason = first_choice.get("finish_reason")
        if finish_reason == "length":
            self._log(
                "[模型通信] ⚠️ 提示：检测到模型触发单次最大输出长度截断 (finish_reason: length)。"
                "生成文本已达模型单次输出 Token 极限，若发现该分块翻译末尾内容不全，建议在 Settings 中调小 Translation Chunk Size (推荐 4,000~6,000 字符)。"
            )

        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            normalized = self._normalize_content(content)
            if normalized is not None and normalized.strip():
                return normalized
            if finish_reason == "length":
                raise TranslationError(
                    "大模型输出长度超限 (finish_reason: length)：模型在思考或生成阶段耗尽了单次最大输出 Token 配额导致翻译内容被截断为空。"
                    "说明：尽管大模型输入上下文窗口很大，但单次最大生成输出（Output Tokens，通常为 8192 tokens）有限。"
                    "建议在 Settings 中将 Translation Chunk Size 调小（推荐 4,000~6,000 字符），避免单次分块过大。"
                    f"Response preview: {self._preview_response(data)}"
                )
            if self._has_reasoning_without_content(message):
                raise TranslationError(
                    "大模型仅输出了思考过程 (reasoning_content)，未生成最终翻译内容。"
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
            "Unexpected LLM API response format: no usable content field. "
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
        counter = 100

        def make_placeholder(kind: str, content: str) -> str:
            nonlocal counter
            placeholder = f"<<<{kind}_{counter}>>>"
            counter += 1
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
        ref_buffer: list[str] = []

        for line in lines:
            stripped = line.strip()
            heading_level = self._get_heading_level(stripped)

            if inside_references and heading_level is not None:
                inside_references = False
                if ref_buffer:
                    protected_lines.append(make_placeholder("REFERENCES", "".join(ref_buffer)))
                    ref_buffer = []

            if (
                not inside_references
                and heading_level is not None
                and self._is_reference_heading(stripped)
            ):
                inside_references = True
                protected_lines.append(line)
                continue

            if inside_references:
                ref_buffer.append(line)
            else:
                protected_lines.append(line)

        if inside_references and ref_buffer:
            protected_lines.append(make_placeholder("REFERENCES", "".join(ref_buffer)))

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

    TAG_SYNONYMS: dict[str, list[str]] = {
        "REFERENCES": ["REFERENCES", "REFERENCE", "REF", "BIBLIOGRAPHY", "参考文献", "参考资料", "参考"],
        "IMAGE": ["IMAGE", "IMG", "FIGURE", "FIG", "图片", "图像", "插图", "图"],
        "TABLE": ["TABLE", "TAB", "表格", "表"],
        "CODE": ["CODE", "INLINE_CODE", "代码", "行内代码"],
        "MULTILINE_CODE": ["MULTILINE_CODE", "CODE_BLOCK", "多行代码", "代码块", "程序代码", "代码"],
        "HTML": ["HTML", "TAG", "网页", "标签", "超文本"],
        "LINK": ["LINK", "URL", "链接", "网址"],
        "LATEX_INLINE": ["LATEX_INLINE", "INLINE_MATH", "MATH_INLINE", "行内公式", "数学公式", "公式", "数学"],
        "LATEX_BLOCK": ["LATEX_BLOCK", "BLOCK_MATH", "MATH_BLOCK", "公式块", "块级公式", "数学公式", "公式", "数学"],
        "ALGORITHM": ["ALGORITHM", "ALGO", "PSEUDOCODE", "算法", "伪代码"],
    }

    def _restore_markdown(
        self, text: str, placeholders: dict[str, str], source_text: str = ""
    ) -> str:
        restored = text
        restored_keys: set[str] = set()

        # 1. 严格字面值精准替换 (Fast path)
        for placeholder, content in sorted(
            placeholders.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if placeholder in restored:
                restored = restored.replace(placeholder, content)
                restored_keys.add(placeholder)

        BRACKET_OPEN = r"[<《【\[{«(〈〔]{1,4}"
        BRACKET_CLOSE = r"[>》】\]}»)〉〕]{1,4}"

        # 2. 双语及格式容错正则替换 (Bilingual & Formatting Fuzzy Path)
        # 针对大模型偶发性吞掉括号、带入空格、使用全角括号【】/《》/［］，
        # 以及大模型擅自将英文标签词翻译为中文（如 <<<REFERENCES_100>>> 变成 <<<参考文献_100>>>、<<<IMAGE_101>>> 变成 <<<图片 101>>>）等情况
        for placeholder, content in sorted(
            placeholders.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if placeholder in restored_keys:
                continue
            tag_match = re.search(r"([A-Z_]+)_(\d+)", placeholder)
            if not tag_match:
                continue
            kind = tag_match.group(1)
            idx = tag_match.group(2)
            synonyms = self.TAG_SYNONYMS.get(kind, [kind])
            kind_pat = "(?:" + "|".join(re.escape(s) for s in synonyms) + ")"
            fuzzy_pattern = re.compile(
                BRACKET_OPEN + r"\s*" + kind_pat + r"[\s_#\-]*" + idx + r"\s*" + BRACKET_CLOSE,
                flags=re.IGNORECASE,
            )
            if fuzzy_pattern.search(restored):
                restored = fuzzy_pattern.sub(lambda _: content, restored)
                restored_keys.add(placeholder)
                self._log(f"[格式保护] 检测到大模型轻微变形/翻译的占位符 ({kind}_{idx})，已通过容错匹配精准复原")

        # 2.5 极致编号唯一性容错替换 (ID-Only Fallback Path)
        # 针对模型省略标签词，仅保留编号（如 <<<100>>> 或 【100】）等极端情况
        for placeholder, content in sorted(
            placeholders.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if placeholder in restored_keys:
                continue
            tag_match = re.search(r"_(\d+)", placeholder)
            if not tag_match:
                continue
            idx = tag_match.group(1)
            id_pattern = re.compile(
                BRACKET_OPEN + r"\s*(?:[A-Za-z\u4e00-\u9fa5_#\-]+[\s_#\-]*)?" + idx + r"\s*" + BRACKET_CLOSE,
                flags=re.IGNORECASE,
            )
            if id_pattern.search(restored):
                restored = id_pattern.sub(lambda _: content, restored)
                restored_keys.add(placeholder)
                self._log(f"[格式保护] 检测到大模型标签词缺失但保留编号的占位符 (ID: {idx})，已通过全局唯一标识精准复原")

        # 3. 遗漏占位符安全兜底恢复 (Orphan placeholder safeguard & recovery)
        # 针对大模型在翻译长文本时擅自将插图/表格/代码块/参考文献等关键占位符彻底删减丢弃的情况进行拦截兜底
        critical_types = ("IMAGE", "TABLE", "ALGORITHM", "CODE_BLOCK", "REFERENCES")
        missing = [
            (ph, content)
            for ph, content in sorted(
                placeholders.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            if ph not in restored_keys and any(ct in ph for ct in critical_types)
        ]

        if missing:
            paragraphs = [p for p in restored.split("\n\n") if p.strip()]
            for ph, content in missing:
                tag_match = re.search(r"([A-Z_]+_\d+)", ph)
                tag = tag_match.group(1) if tag_match else ph

                # 计算在原文中的相对位置以恢复到最匹配的段落位置
                insert_idx = len(paragraphs)
                if source_text and ph in source_text:
                    source_pos = source_text.find(ph)
                    ratio = source_pos / max(1, len(source_text))
                    insert_idx = int(round(ratio * len(paragraphs)))
                    insert_idx = max(0, min(len(paragraphs), insert_idx))

                paragraphs.insert(insert_idx, content)
                self._log(
                    f"[格式保护] ⚠️ 检测到大模型在翻译回复中彻底遗漏了关键占位符 ({tag})，程序已自动安全兜底，将其完整补回译文对应段落！"
                )

            restored = "\n\n".join(paragraphs)

        return restored

    def _force_translate_headings(
        self,
        text: str,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        lines = text.splitlines(keepends=True)
        if not lines:
            return text

        heading_map: dict[str, str] = {}
        heading_pattern = re.compile(r"^(#{1,6})(\s+)(.+?)(\s*)$")

        heading_texts_to_translate: list[str] = []
        for line in lines:
            if cancel_check is not None and cancel_check():
                raise TranslationCancelledError("Translation cancelled by user.")
            line_body = line.rstrip("\r\n")
            match = heading_pattern.match(line_body)
            if not match:
                continue
            heading_text = match.group(3)
            if not re.search(r"[A-Za-z]", heading_text):
                continue
            if heading_text not in heading_map:
                heading_map[heading_text] = heading_text
                heading_texts_to_translate.append(heading_text)

        if not heading_texts_to_translate:
            return text

        uncached_headings: list[str] = []
        for h_text in heading_texts_to_translate:
            cache_key = self.cache.make_cache_key(self.config.llm_model, "heading", h_text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                heading_map[h_text] = cached
            else:
                uncached_headings.append(h_text)

        total_headings = len(heading_texts_to_translate)
        cached_count = total_headings - len(uncached_headings)

        if not uncached_headings:
            self._log(
                f"[排版校准] 检测到 {total_headings} 处待校准章节标题，已全部命中本地缓存 (0 次远程调用，直接复用)"
            )
        else:
            self._log(
                f"[排版校准] 检测到 {total_headings} 处待校准章节标题 "
                f"({cached_count} 处命中本地缓存，剩余 {len(uncached_headings)} 处未缓存)，"
                "正在通过 1 次单批次请求向模型发起批量校准..."
            )
            h_start = time.perf_counter()
            batch_results = self._batch_translate_headings(uncached_headings, cancel_check=cancel_check)
            heading_map.update(batch_results)
            h_elapsed = time.perf_counter() - h_start
            self._log(f"[排版校准] 章节标题批量校准完成 (耗时 {h_elapsed:.2f}s, 仅消耗 1 次 API 请求)")

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

    def _batch_translate_headings(
        self,
        headings: list[str],
        cancel_check: CancelCheck | None = None,
    ) -> dict[str, str]:
        if not headings:
            return {}

        system_prompt = (
            "You are a professional academic translator. "
            "Translate the following Markdown section headings into Simplified Chinese. "
            "Preserve numbering, punctuation, abbreviations, and formatting exactly. "
            "Output the translated headings as a valid JSON array of strings, in the exact same order and length as the input list. "
            "Return only the JSON array without any other markdown explanation or thinking text."
        )
        user_payload = json.dumps(headings, ensure_ascii=False)

        try:
            raw_response = self._call_llm_api_with_system_prompt(
                user_payload,
                system_prompt,
                cancel_check=cancel_check,
                chunk_desc="[标题批量校准]",
            )
            cleaned = raw_response.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            json_match = re.search(r"\[[\s\S]*\]", cleaned)
            if json_match:
                cleaned = json_match.group(0)

            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) == len(headings):
                results: dict[str, str] = {}
                for orig, trans in zip(headings, parsed):
                    trans_str = str(trans).strip() or orig
                    results[orig] = trans_str
                    cache_key = self.cache.make_cache_key(self.config.llm_model, "heading", orig)
                    self.cache.set(cache_key, self.config.llm_model, orig, trans_str)
                return results
            else:
                self._log(
                    f"[排版校准] 批量校准返回列表长度不匹配 (预期 {len(headings)}, 收到 {len(parsed) if isinstance(parsed, list) else 0})，自动降级为逐条校准..."
                )
        except Exception as exc:
            self._log(f"[排版校准] 批量校准请求或解析异常 ({exc})，自动降级为逐条校准...")

        # Fallback to individual translation
        results = {}
        for h_text in headings:
            if cancel_check is not None and cancel_check():
                raise TranslationCancelledError("Translation cancelled by user.")
            results[h_text] = self._translate_heading_text(h_text, cancel_check=cancel_check)
        return results

    def _translate_heading_text(
        self,
        heading_text: str,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        check_cancelled = cancel_check or self.cancel_check
        cache_key = self.cache.make_cache_key(self.config.llm_model, "heading", heading_text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            translated = self._call_llm_api_with_system_prompt(
                heading_text,
                (
                    "Translate the heading text into Simplified Chinese. "
                    "Output the translated text directly without any thinking process, reasoning steps, or explanation. "
                    "Keep numbering, punctuation, abbreviations, and inline Markdown syntax intact. "
                    "Return only the translated heading text."
                ),
                cancel_check=check_cancelled,
            )
        except TranslationCancelledError:
            raise
        except TranslationError:
            return heading_text

        result = translated.strip() or heading_text
        if result and result != heading_text:
            self.cache.set(cache_key, self.config.llm_model, heading_text, result)
        return result

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


LlmMarkdownTranslator = NvidiaMarkdownTranslator
