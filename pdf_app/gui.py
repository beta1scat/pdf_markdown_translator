from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from typing import cast

from . import __version__
from .config import (
    AppConfig,
    PROVIDER_CUSTOM,
    PROVIDER_GOOGLE,
    PROVIDER_NVIDIA,
    PROVIDER_OPTIONS,
    PROVIDER_PRESETS,
    PROVIDER_ZHIPU,
    load_config,
    save_config,
)
from .models import BatchConversionResult, FileConversionItem
from .nvidia_models import (
    NvidiaModelFetchError,
    fetch_all_provider_models,
    fetch_nvidia_models,
)
from .cache import get_translation_cache
from .paths import get_app_base_dir
from .markdown_translator import NvidiaMarkdownTranslator
from .service import (
    ConversionCancelledError,
    MarkdownTranslationError,
    PdfConversionError,
    convert_pdf_to_markdown,
    translate_markdown_file,
)


class PdfToMarkdownApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"PDF to Markdown v{__version__}")
        self.root.geometry("760x480")
        self.root.minsize(700, 400)

        self.config = load_config()
        self.is_running = False
        self._is_cancelled = False
        self.log_file_path = get_app_base_dir() / "app.log"

        self.run_button: ttk.Button | None = None
        self.stop_button: ttk.Button | None = None

        self.selected_files: list[Path] = []
        self.input_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str((Path.cwd() / "output").resolve()))
        self.translate_var = tk.BooleanVar(value=True)
        self.input_mode_var = tk.StringVar(value="pdf")
        self.status_var = tk.StringVar(value="Ready.")
        self.settings_window: tk.Toplevel | None = None
        self._last_progress_log_chars = -1

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=0)
        container.rowconfigure(6, weight=1)

        ttk.Label(container, text="Input Type").grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )
        input_mode_frame = ttk.Frame(container)
        input_mode_frame.grid(row=0, column=1, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Radiobutton(
            input_mode_frame,
            text="PDF",
            value="pdf",
            variable=self.input_mode_var,
            command=self._on_mode_change,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            input_mode_frame,
            text="Markdown",
            value="markdown",
            variable=self.input_mode_var,
            command=self._on_mode_change,
        ).pack(side=tk.LEFT, padx=(16, 0))

        ttk.Label(container, text="Input Files").grid(
            row=1, column=0, sticky="w", pady=(0, 12)
        )
        ttk.Entry(container, textvariable=self.input_path_var).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(12, 12),
            pady=(0, 12),
        )
        ttk.Button(container, text="Browse", command=self.select_input_file).grid(
            row=1,
            column=2,
            sticky="ew",
            pady=(0, 12),
        )

        ttk.Label(container, text="Output Dir").grid(
            row=2, column=0, sticky="w", pady=(0, 12)
        )
        ttk.Entry(container, textvariable=self.output_dir_var).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(12, 12),
            pady=(0, 12),
        )
        ttk.Button(container, text="Browse", command=self.select_output_dir).grid(
            row=2,
            column=2,
            sticky="ew",
            pady=(0, 12),
        )

        ttk.Checkbutton(
            container,
            text="Translate Markdown to Simplified Chinese",
            variable=self.translate_var,
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(container, text="Status").grid(
            row=4, column=0, sticky="nw", pady=(0, 8)
        )
        ttk.Label(
            container,
            textvariable=self.status_var,
            wraplength=560,
            justify=tk.LEFT,
        ).grid(row=4, column=1, columnspan=2, sticky="w", pady=(0, 8))

        actions = ttk.Frame(container)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        actions.columnconfigure(0, weight=1)

        button_group = ttk.Frame(actions)
        button_group.grid(row=0, column=0, sticky="w")
        self.run_button = ttk.Button(
            button_group, text="Run", command=self.start_conversion
        )
        self.run_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_button = ttk.Button(
            button_group, text="Stop", command=self.stop_conversion, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)

        ttk.Button(actions, text="Help", command=self.show_help).grid(
            row=0, column=1, sticky="e", padx=(0, 8)
        )
        ttk.Button(actions, text="Settings", command=self.open_settings).grid(
            row=0, column=2, sticky="e"
        )

        self.log_text = tk.Text(container, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=6, column=0, columnspan=3, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            container, orient=tk.VERTICAL, command=self.log_text.yview
        )
        scrollbar.grid(row=6, column=3, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        ttk.Label(container, text=f"Version v{__version__}").grid(
            row=7,
            column=0,
            columnspan=3,
            sticky="e",
            pady=(8, 0),
        )

        self._append_log("App initialized.")
        self._append_log(f"App version: {__version__}")
        self._append_log(f"PDF API URL: {self.config.api_url}")
        self._append_log(f"PDF Model: {self.config.pdf_model}")
        self._append_log(f"LLM Provider: {self.config.llm_provider}")
        self._append_log(f"LLM API URL: {self.config.llm_api_url}")
        self._append_log(f"LLM Model: {self.config.llm_model}")
        self._append_log(
            f"LLM rate limit: {self.config.max_requests_per_minute} requests/minute."
        )
        self._on_mode_change()

    def _toggle_secret_visibility(
        self,
        entry: ttk.Entry,
        visible_var: tk.BooleanVar,
        button: ttk.Button,
    ) -> None:
        is_visible = not visible_var.get()
        visible_var.set(is_visible)
        entry.configure(show="" if is_visible else "*")
        button.configure(text="Hide" if is_visible else "Show")

    def show_help(self) -> None:
        help_text = (
            "Input modes:\n"
            "1. PDF: convert PDF to Markdown, save images, and optionally translate the Markdown.\n"
            "2. Markdown: translate an existing Markdown or text file directly.\n\n"
            "Basic steps:\n"
            "1. Select the input type.\n"
            "2. Choose the input file(s).\n"
            "3. Choose the output directory.\n"
            "4. Open Settings, choose an LLM Provider (Zhipu AI, Google Gemini, NVIDIA NIM, or Custom), and fill in your API key.\n"
            "5. Click Run.\n\n"
            "Output rules:\n"
            "- PDF mode writes files into OUTPUT_DIR/<pdf_name>/.\n"
            "- Markdown mode writes <file_name>_zh.md directly into the output directory.\n\n"
            "Notes:\n"
            "- PDF translation requires both the Paddle/PDF API and the LLM API.\n"
            "- Markdown translation requires only the LLM API.\n"
            "- The log area and app.log record progress, timings, and output paths.\n"
            "- Settings are saved locally in app_config.json."
        )
        messagebox.showinfo("Help", help_text, parent=self.root)

    def open_settings(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            self.settings_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        window.title("Settings")
        window.geometry("740x605")
        window.resizable(False, False)
        window.transient(self.root)
        self.settings_window = window

        api_url_var = tk.StringVar(value=self.config.api_url)
        api_token_var = tk.StringVar(value=self.config.api_token)
        pdf_model_var = tk.StringVar(value=self.config.pdf_model)
        timeout_var = tk.StringVar(value=str(self.config.request_timeout_seconds))
        provider_var = tk.StringVar(value=self.config.llm_provider)
        llm_url_var = tk.StringVar(value=self.config.llm_api_url)
        llm_key_var = tk.StringVar(value=self.config.llm_api_key)
        llm_model_var = tk.StringVar(value=self.config.llm_model)
        rate_limit_var = tk.StringVar(value=str(self.config.max_requests_per_minute))
        concurrency_var = tk.StringVar(value=str(self.config.translation_concurrency))
        chunk_size_var = tk.StringVar(value=str(self.config.translation_chunk_size))

        current_provider_keys: dict[str, str] = dict(self.config.provider_keys)
        current_provider_models: dict[str, str] = dict(self.config.provider_models)

        active_provider = self.config.llm_provider.strip()
        if active_provider and self.config.llm_api_key:
            current_provider_keys.setdefault(active_provider, self.config.llm_api_key)
        if self.config.llm_model and active_provider:
            current_provider_models.setdefault(active_provider, self.config.llm_model)

        previous_provider_holder = [active_provider]

        current_preset = PROVIDER_PRESETS.get(self.config.llm_provider, {})
        preset_models = list(current_preset.get("models", []))
        if self.config.llm_model and self.config.llm_model not in preset_models:
            preset_models.insert(0, self.config.llm_model)
        model_values = preset_models

        container = ttk.Frame(window, padding=16)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)

        fields = [
            ("Paddle/PDF API URL", api_url_var, False),
            ("Paddle/PDF API Token", api_token_var, True),
            ("Paddle/PDF Model", pdf_model_var, False),
            ("Request Timeout (s)", timeout_var, False),
            ("LLM Provider (服务商)", provider_var, False),
            ("LLM API URL", llm_url_var, False),
            ("LLM API Key", llm_key_var, True),
            ("LLM Model", llm_model_var, False),
            ("LLM Rate Limit / min", rate_limit_var, False),
            ("Translation Concurrency", concurrency_var, False),
            ("Translation Chunk Size (分块字符数)", chunk_size_var, False),
        ]

        model_combobox_holder: list[ttk.Combobox] = []
        refresh_button_holder: list[ttk.Button] = []

        def update_provider_ui_state(chosen: str) -> None:
            preset = PROVIDER_PRESETS.get(chosen, {})
            supports_refresh = bool(preset.get("supports_model_refresh", False))
            if refresh_button_holder:
                if supports_refresh:
                    refresh_button_holder[0].grid()
                else:
                    refresh_button_holder[0].grid_remove()
            website_hint = str(preset.get("website_hint", ""))
            if website_hint:
                hint_label.configure(text=f"💡 提示：{website_hint}")
                hint_label.grid()
            else:
                hint_label.configure(text="")
                hint_label.grid_remove()

        def on_provider_selected(event: object = None) -> None:
            old_provider = previous_provider_holder[0]
            new_provider = provider_var.get().strip()

            # 1. 记忆旧服务商已填入的 Key 与 Model
            old_key = llm_key_var.get().strip()
            old_model = llm_model_var.get().strip()
            if old_provider:
                current_provider_keys[old_provider] = old_key
                if old_model:
                    current_provider_models[old_provider] = old_model

            previous_provider_holder[0] = new_provider

            preset = PROVIDER_PRESETS.get(new_provider)
            if not preset:
                return
            if new_provider != PROVIDER_CUSTOM:
                llm_url_var.set(str(preset["url"]))
                rate_limit_var.set(str(preset["rate_limit"]))
                concurrency_var.set(str(preset["concurrency"]))

            # 2. 还原新服务商已保存的模型（若无则使用默认模型）
            target_model = current_provider_models.get(new_provider) or str(preset["default_model"])
            llm_model_var.set(target_model)

            # 3. 还原新服务商已保存的 API Key
            target_key = current_provider_keys.get(new_provider, "")
            llm_key_var.set(target_key)

            # 4. 更新模型下拉选择列表
            models_list = list(preset.get("models", []))
            if target_model and target_model not in models_list:
                models_list.insert(0, target_model)
            if model_combobox_holder:
                model_combobox_holder[0]["values"] = models_list

            # 5. 更新刷新按钮与官网指引提示
            update_provider_ui_state(new_provider)

        for row_index, (label, variable, secret) in enumerate(fields):
            ttk.Label(container, text=label).grid(
                row=row_index, column=0, sticky="w", pady=(0, 10)
            )
            if label == "LLM Provider (服务商)":
                entry = ttk.Combobox(
                    container,
                    textvariable=variable,
                    values=PROVIDER_OPTIONS,
                    state="readonly",
                )
                entry.bind("<<ComboboxSelected>>", on_provider_selected)
            elif label == "LLM Model":
                entry = ttk.Combobox(
                    container,
                    textvariable=variable,
                    values=model_values,
                )
                model_combobox_holder.append(entry)

                def on_model_selected(event: object = None) -> None:
                    p = provider_var.get().strip()
                    m = llm_model_var.get().strip()
                    if m and p:
                        current_provider_models[p] = m

                entry.bind("<<ComboboxSelected>>", on_model_selected)
            else:
                entry = ttk.Entry(
                    container,
                    textvariable=variable,
                    show="*" if secret else "",
                )
            entry.grid(
                row=row_index, column=1, sticky="ew", padx=(12, 12), pady=(0, 10)
            )
            if secret:
                visible_var = tk.BooleanVar(value=False)
                toggle_button = ttk.Button(container, text="Show")
                toggle_button.configure(
                    command=lambda e=entry,
                    v=visible_var,
                    b=toggle_button: self._toggle_secret_visibility(e, v, b)
                )
                toggle_button.grid(row=row_index, column=2, sticky="e", pady=(0, 10))
            elif label == "LLM Model":
                model_combobox = cast(ttk.Combobox, entry)
                refresh_button = ttk.Button(container, text="Refresh")
                refresh_button.configure(
                    command=lambda b=refresh_button, c=model_combobox: self._refresh_models(
                        provider_var.get(),
                        llm_url_var.get(),
                        llm_key_var.get(),
                        b,
                        c,
                        llm_model_var,
                        window,
                    )
                )
                refresh_button.grid(row=row_index, column=2, sticky="e", pady=(0, 10))
                refresh_button_holder.append(refresh_button)

        hint_label = ttk.Label(
            container,
            text="",
            foreground="#475569",
            font=("", 9),
            wraplength=520,
            justify="left",
        )
        hint_label.grid(
            row=len(fields), column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        update_provider_ui_state(provider_var.get().strip())

        # 断点续传与缓存管理
        cache = get_translation_cache()
        cached_count, cached_bytes = cache.get_stats()
        cache_frame = ttk.Frame(container)
        cache_frame.grid(
            row=len(fields) + 1, column=0, columnspan=3, sticky="ew", pady=(8, 8)
        )
        cache_frame.columnconfigure(1, weight=1)

        ttk.Label(cache_frame, text="断点翻译缓存:").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        cache_info_var = tk.StringVar(
            value=f"已缓存 {cached_count} 个分块 (约 {cached_bytes / 1024:.1f} KB)"
        )
        ttk.Label(cache_frame, textvariable=cache_info_var).grid(
            row=0, column=1, sticky="w"
        )

        def on_clear_cache() -> None:
            count, _ = cache.get_stats()
            if count == 0:
                messagebox.showinfo("缓存管理", "本地暂无翻译缓存数据", parent=window)
                return
            if messagebox.askyesno(
                "清除缓存",
                f"确定清空本地已有的 {count} 条分块翻译缓存？\n清空后再次翻译相同内容将需要重新调用 API。",
                parent=window,
            ):
                cache.clear()
                cache_info_var.set("已缓存 0 个分块 (约 0.0 KB)")
                messagebox.showinfo("缓存管理", "本地翻译缓存已清空", parent=window)

        ttk.Button(
            cache_frame, text="清空缓存 (Clear Cache)", command=on_clear_cache
        ).grid(row=0, column=2, sticky="e")

        button_row = ttk.Frame(container)
        button_row.grid(
            row=len(fields) + 2, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        button_row.columnconfigure(0, weight=1)
        ttk.Button(
            button_row,
            text="Save",
            command=lambda: self._save_settings(
                window,
                api_url_var.get(),
                api_token_var.get(),
                pdf_model_var.get(),
                timeout_var.get(),
                provider_var.get(),
                llm_url_var.get(),
                llm_key_var.get(),
                llm_model_var.get(),
                rate_limit_var.get(),
                concurrency_var.get(),
                chunk_size_var.get(),
                current_provider_keys,
                current_provider_models,
            ),
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(button_row, text="Cancel", command=self._close_settings_window).grid(
            row=0, column=1, sticky="e"
        )


        window.protocol("WM_DELETE_WINDOW", self._close_settings_window)

    def _close_settings_window(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            try:
                self.root.focus_set()
            except Exception:
                pass
            win = self.settings_window
            self.settings_window = None
            try:
                win.destroy()
            except Exception:
                pass
        else:
            self.settings_window = None

    def _refresh_models(
        self,
        provider: str,
        llm_api_url: str,
        llm_api_key: str,
        refresh_button: ttk.Button,
        combobox: ttk.Combobox,
        model_var: tk.StringVar,
        window: tk.Toplevel,
    ) -> None:
        provider_clean = provider.strip()
        url_clean = llm_api_url.strip()
        key_clean = llm_api_key.strip()

        refresh_button.configure(state=tk.DISABLED, text="Loading...")
        self._append_log(f"[{provider_clean}] Fetching models...")

        def worker() -> None:
            models, status_message = fetch_all_provider_models(
                provider_clean, url_clean, key_clean, timeout_seconds=5
            )

            def update_ui() -> None:
                if not window.winfo_exists():
                    return
                refresh_button.configure(state=tk.NORMAL, text="Refresh")

                current_typed = model_var.get().strip()
                display_models = list(models)
                if current_typed and current_typed not in display_models:
                    display_models.insert(0, current_typed)

                if display_models:
                    combobox.configure(values=display_models)
                    if not model_var.get().strip():
                        model_var.set(display_models[0])

                self._append_log(f"[{provider_clean}] {status_message}")
                messagebox.showinfo("Models Refreshed", status_message, parent=window)

            try:
                window.after(0, update_ui)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    # Alias for backward compatibility
    _refresh_nvidia_models = _refresh_models

    def _save_settings(
        self,
        window: tk.Toplevel,
        api_url: str,
        api_token: str,
        pdf_model: str,
        timeout_seconds: str,
        llm_provider: str,
        llm_api_url: str,
        llm_api_key: str,
        llm_model: str,
        rate_limit: str,
        translation_concurrency: str,
        translation_chunk_size: str,
        provider_keys: dict[str, str] | None = None,
        provider_models: dict[str, str] | None = None,
    ) -> None:
        api_url = api_url.strip()
        api_token = api_token.strip()
        pdf_model = pdf_model.strip() or "PaddleOCR-VL-1.6"
        llm_provider = llm_provider.strip() or PROVIDER_ZHIPU
        llm_api_url = llm_api_url.strip()
        llm_api_key = llm_api_key.strip()
        llm_model = llm_model.strip()

        if not api_url:
            messagebox.showerror(
                "Invalid Settings", "Paddle/PDF API URL is required.", parent=window
            )
            return
        if not llm_api_url:
            messagebox.showerror(
                "Invalid Settings", "LLM API URL is required.", parent=window
            )
            return
        if not llm_model:
            messagebox.showerror(
                "Invalid Settings", "LLM model is required.", parent=window
            )
            return

        try:
            timeout_value = int(timeout_seconds)
            rate_limit_value = int(rate_limit)
            concurrency_value = int(translation_concurrency)
            chunk_size_value = int(translation_chunk_size)
        except ValueError:
            messagebox.showerror(
                "Invalid Settings",
                "Request timeout, rate limit, translation concurrency, and chunk size must be integers.",
                parent=window,
            )
            return

        if timeout_value <= 0 or rate_limit_value <= 0 or concurrency_value <= 0 or chunk_size_value < 500:
            messagebox.showerror(
                "Invalid Settings",
                "Request timeout, rate limit, and translation concurrency must be greater than 0. Chunk size must be >= 500.",
                parent=window,
            )
            return

        p_keys = dict(provider_keys) if provider_keys is not None else dict(self.config.provider_keys)
        p_models = dict(provider_models) if provider_models is not None else dict(self.config.provider_models)

        p_keys[llm_provider] = llm_api_key
        if llm_model:
            p_models[llm_provider] = llm_model

        self.config = AppConfig(
            api_url=api_url,
            api_token=api_token,
            pdf_model=pdf_model,
            request_timeout_seconds=timeout_value,
            llm_provider=llm_provider,
            llm_api_url=llm_api_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            max_requests_per_minute=rate_limit_value,
            translation_concurrency=concurrency_value,
            translation_chunk_size=chunk_size_value,
            provider_keys=p_keys,
            provider_models=p_models,
        )
        save_config(self.config)
        self._append_log("Settings saved.")
        self._append_log(f"PDF API URL: {self.config.api_url}")
        self._append_log(f"PDF Model: {self.config.pdf_model}")
        self._append_log(f"LLM Provider: {self.config.llm_provider}")
        self._append_log(f"LLM API URL: {self.config.llm_api_url}")
        self._append_log(f"LLM Model: {self.config.llm_model}")
        self._append_log(
            f"LLM rate limit: {self.config.max_requests_per_minute} requests/minute."
        )
        self._append_log(
            f"Translation concurrency: {self.config.translation_concurrency} workers."
        )
        self._append_log(
            f"Translation chunk size: {self.config.translation_chunk_size} chars."
        )
        configured_providers = [k for k, v in p_keys.items() if v]
        if configured_providers:
            self._append_log(
                f"Configured provider keys saved: {', '.join(configured_providers)}"
            )
        self._close_settings_window()
        self.status_var.set("Settings saved successfully.")

    def _on_mode_change(self) -> None:
        self.selected_files = []
        self.input_path_var.set("")
        if self.input_mode_var.get() == "markdown":
            self.translate_var.set(True)
            self.status_var.set("Ready to translate Markdown file(s) directly.")
        else:
            self.status_var.set("Ready.")

    def _update_phase_status(self, phase: str) -> None:
        phase_map = {
            "converting": "Converting PDF to Markdown...",
            "translating": "Translating Markdown...",
            "completed": "Completed.",
            "failed": "Failed.",
        }
        if phase.startswith("converting ("):
            message = f"Converting PDF to Markdown {phase[len('converting '):]}..."
        else:
            message = phase_map.get(phase, phase)
        self.root.after(0, self.status_var.set, message)
        self.root.after(0, self._append_log, f"Phase: {message}")

    def _update_translation_progress(
        self, translated_chars: int, total_chars: int
    ) -> None:
        if total_chars <= 0:
            return
        if (
            translated_chars == self._last_progress_log_chars
            and translated_chars != total_chars
        ):
            return

        self._last_progress_log_chars = translated_chars
        self.root.after(
            0,
            self._append_log,
            f"Translation progress: {translated_chars}/{total_chars} chars",
        )

    def select_input_file(self) -> None:
        if self.input_mode_var.get() == "markdown":
            filetypes = [
                ("Markdown files", "*.md *.markdown"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
            title = "Select Markdown file(s)"
        else:
            filetypes = [("PDF files", "*.pdf"), ("All files", "*.*")]
            title = "Select PDF file(s)"

        selected = filedialog.askopenfilenames(title=title, filetypes=filetypes)
        if selected:
            paths = [Path(p).resolve() for p in selected]
            self.selected_files = paths
            if len(paths) == 1:
                self.input_path_var.set(str(paths[0]))
            else:
                preview = "; ".join(p.name for p in paths[:3])
                if len(paths) > 3:
                    preview += f"... (+{len(paths) - 3} more)"
                self.input_path_var.set(f"[{len(paths)} files] {preview}")

    def _resolve_input_files(self) -> list[Path]:
        raw_input = self.input_path_var.get().strip()
        if not raw_input:
            return []

        if self.selected_files:
            if len(self.selected_files) == 1 and str(self.selected_files[0]) == raw_input:
                return self.selected_files
            if raw_input.startswith(f"[{len(self.selected_files)} files]"):
                return self.selected_files

        parts = [p.strip() for p in re.split(r"[;\n\r]+", raw_input) if p.strip()]
        resolved: list[Path] = []
        for part in parts:
            cleaned = part.strip("\"'")
            if cleaned:
                resolved.append(Path(cleaned).expanduser().resolve())
        return resolved

    def select_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_dir_var.set(directory)

    def start_conversion(self) -> None:
        if self.is_running:
            return

        files = self._resolve_input_files()
        output_dir = self.output_dir_var.get().strip()
        translate_markdown = self.translate_var.get()
        input_mode = self.input_mode_var.get()

        if not files:
            messagebox.showerror(
                "Missing Input File", "Please select at least one input file.", parent=self.root
            )
            return
        if not output_dir:
            messagebox.showerror(
                "Missing Output Directory", "Please select an output directory.", parent=self.root
            )
            return
        if (
            translate_markdown or input_mode == "markdown"
        ) and not self.config.llm_api_key:
            messagebox.showerror(
                "Missing LLM API Key",
                "Please configure your LLM API Key in Settings before translating.",
                parent=self.root,
            )
            return

        NvidiaMarkdownTranslator.abort_all_active()
        self.is_running = True
        self._is_cancelled = False
        if self.run_button:
            self.run_button.configure(state=tk.DISABLED)
        if self.stop_button:
            self.stop_button.configure(state=tk.NORMAL)

        self._last_progress_log_chars = -1
        self.status_var.set(f"Starting batch of {len(files)} file(s)...")
        self._append_log(f"Input mode: {input_mode}")
        self._append_log(f"Files to process: {len(files)}")
        for i, f in enumerate(files, 1):
            self._append_log(f"  [{i}/{len(files)}] {f}")
        self._append_log(f"Output directory: {output_dir}")
        self._append_log(f"Translate Markdown: {'yes' if translate_markdown else 'no'}")

        worker = threading.Thread(
            target=self._run_batch_conversion,
            args=(input_mode, files, output_dir, translate_markdown),
            daemon=True,
        )
        worker.start()

    def stop_conversion(self) -> None:
        if not self.is_running or self._is_cancelled:
            return
        self._is_cancelled = True
        self.status_var.set("Stopping...")
        if self.stop_button:
            self.stop_button.configure(state=tk.DISABLED)
        self._append_log("Stop requested by user. Cancelling process...")
        NvidiaMarkdownTranslator.abort_all_active()

    def _run_batch_conversion(
        self,
        input_mode: str,
        files: list[Path],
        output_dir: str,
        translate_markdown: bool,
    ) -> None:
        total_start = time.perf_counter()
        total_files = len(files)
        items: list[FileConversionItem] = []

        for index, file_path in enumerate(files, 1):
            if self._is_cancelled:
                self.root.after(0, self._append_log, "Batch conversion stopped by user.")
                break

            file_prefix = f"[{index}/{total_files}]"
            self.root.after(
                0,
                self._append_log,
                f"\n=== {file_prefix} Processing: {file_path.name} ===",
            )
            self.root.after(
                0,
                self.status_var.set,
                f"{file_prefix} Processing {file_path.name}...",
            )

            last_logged_phase = [""]

            def file_phase_callback(
                phase: str, pfx: str = file_prefix, fname: str = file_path.name
            ) -> None:
                if self._is_cancelled:
                    return
                if phase.startswith("converting ("):
                    status_msg = f"{pfx} Converting {fname} {phase[len('converting '):]}"
                    log_msg = f"Phase: Converting PDF {phase[len('converting '):]}"
                elif phase == "converting":
                    status_msg = f"{pfx} Converting {fname} to Markdown..."
                    log_msg = "Phase: Converting PDF to Markdown..."
                elif phase == "translating":
                    status_msg = f"{pfx} Translating {fname}..."
                    log_msg = "Phase: Translating Markdown..."
                elif phase == "completed":
                    status_msg = f"{pfx} Completed {fname}."
                    log_msg = "Phase: Completed."
                elif phase == "failed":
                    status_msg = f"{pfx} Failed {fname}."
                    log_msg = "Phase: Failed."
                else:
                    status_msg = f"{pfx} {phase}"
                    log_msg = f"Phase: {phase}"

                self.root.after(0, self.status_var.set, status_msg)
                if log_msg != last_logged_phase[0]:
                    last_logged_phase[0] = log_msg
                    self.root.after(0, self._append_log, log_msg)

            def file_log_callback(msg: str) -> None:
                if self._is_cancelled:
                    return
                self.root.after(0, self._append_log, msg)

            try:
                if input_mode == "markdown":
                    result = translate_markdown_file(
                        file_path,
                        output_dir,
                        self.config,
                        phase_callback=file_phase_callback,
                        progress_callback=self._update_translation_progress,
                        cancel_check=lambda: self._is_cancelled,
                        log_callback=file_log_callback,
                    )
                else:
                    result = convert_pdf_to_markdown(
                        file_path,
                        output_dir,
                        self.config,
                        translate_markdown=translate_markdown,
                        phase_callback=file_phase_callback,
                        progress_callback=self._update_translation_progress,
                        cancel_check=lambda: self._is_cancelled,
                        log_callback=file_log_callback,
                    )
                items.append(
                    FileConversionItem(
                        file_path=file_path, result=result, is_success=True
                    )
                )
                self.root.after(
                    0,
                    self._append_log,
                    f"✓ {file_prefix} Successfully processed {file_path.name} in {result.timings.total_seconds:.2f}s",
                )
            except ConversionCancelledError as exc:
                items.append(
                    FileConversionItem(
                        file_path=file_path, error="Cancelled by user", is_success=False
                    )
                )
                self.root.after(
                    0,
                    self._append_log,
                    f"⚠ {file_prefix} Stopped {file_path.name}: Cancelled by user.",
                )
                break
            except (PdfConversionError, MarkdownTranslationError) as exc:
                items.append(
                    FileConversionItem(
                        file_path=file_path, error=str(exc), is_success=False
                    )
                )
                self.root.after(
                    0,
                    self._append_log,
                    f"✗ {file_prefix} Failed {file_path.name}:\n{exc}",
                )
            except Exception as exc:
                stack = traceback.format_exc()
                err_text = f"{exc}\n{stack}"
                items.append(
                    FileConversionItem(
                        file_path=file_path, error=err_text, is_success=False
                    )
                )
                self.root.after(
                    0,
                    self._append_log,
                    f"✗ {file_prefix} Error on {file_path.name}: {err_text}",
                )

        total_time = time.perf_counter() - total_start
        batch_result = BatchConversionResult(items=items, total_time_seconds=total_time)
        self.root.after(0, self._on_batch_complete, batch_result)

    def _on_batch_complete(self, batch_result: BatchConversionResult) -> None:
        self.is_running = False
        NvidiaMarkdownTranslator.abort_all_active()
        if self.run_button:
            self.run_button.configure(state=tk.NORMAL)
        if self.stop_button:
            self.stop_button.configure(state=tk.DISABLED)

        success_items = [
            item
            for item in batch_result.items
            if item.is_success and item.result is not None
        ]
        failed_items = [item for item in batch_result.items if not item.is_success]

        total_count = len(batch_result.items)
        success_count = len(success_items)
        failed_count = len(failed_items)
        total_pages = sum(
            item.result.stats.page_count for item in success_items if item.result
        )
        total_images = sum(
            item.result.stats.image_count for item in success_items if item.result
        )

        self._append_log("\n================ BATCH SUMMARY ================")
        self._append_log(f"Total files: {total_count}")
        self._append_log(f"Successful: {success_count}")
        self._append_log(f"Failed: {failed_count}")
        self._append_log(f"Total Pages: {total_pages}, Total Images: {total_images}")
        self._append_log(f"Total Elapsed Time: {batch_result.total_time_seconds:.2f}s")
        if self._is_cancelled:
            self._append_log("Status: Stopped by user.")
        self._append_log("===============================================\n")

        if self._is_cancelled:
            self.status_var.set("Stopped by user.")
        elif failed_count == 0:
            self.status_var.set("All files completed successfully.")
        elif success_count == 0:
            self.status_var.set("All files failed.")
        else:
            self.status_var.set(
                f"Completed with {success_count} succeeded, {failed_count} failed."
            )

        self.root.update_idletasks()
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after_idle(self.root.attributes, "-topmost", False)
        except Exception:
            pass

        if total_count == 1:
            item = batch_result.items[0]
            if self._is_cancelled:
                messagebox.showinfo(
                    "Conversion Stopped",
                    f"Conversion of {item.file_path.name} was stopped by user.",
                    parent=self.root,
                )
            elif item.is_success and item.result is not None:
                res = item.result
                translated_label = res.translated_markdown_path or "Not generated"
                translated_html_label = res.translated_html_path or "Not generated"
                html_label = res.html_path or "Not generated"
                messagebox.showinfo(
                    "Completed",
                    (
                        f"File: {item.file_path.name}\n\n"
                        f"Markdown saved to:\n{res.markdown_path}\n\n"
                        f"HTML saved to:\n{html_label}\n\n"
                        f"Translated Markdown:\n{translated_label}\n\n"
                        f"Translated HTML:\n{translated_html_label}\n\n"
                        f"Pages: {res.stats.page_count}\nImages: {res.stats.image_count}\n"
                        f"Conversion time: {res.timings.conversion_seconds:.2f}s\n"
                        f"Translation time: {res.timings.translation_seconds:.2f}s\n"
                        f"Total time: {res.timings.total_seconds:.2f}s"
                    ),
                    parent=self.root,
                )
            else:
                brief_err = (item.error or "Unknown error").splitlines()[0]
                if len(brief_err) > 120:
                    brief_err = brief_err[:120] + "..."
                messagebox.showerror(
                    "Conversion Failed",
                    f"文件处理失败: {item.file_path.name}\n\n{brief_err}\n\n详细错误原因及解决建议已输出在下方日志窗口中。",
                    parent=self.root,
                )
        else:
            summary_lines = [
                f"Batch Finished in {batch_result.total_time_seconds:.2f}s\n",
                f"Total files: {total_count}",
                f"Succeeded: {success_count}",
                f"Failed: {failed_count}",
                f"Total Pages: {total_pages} | Total Images: {total_images}\n",
            ]
            if self._is_cancelled:
                summary_lines.insert(0, "[STOPPED BY USER]\n")

            if failed_items:
                summary_lines.append("Failed/Cancelled Files:")
                for f_item in failed_items:
                    err_brief = (f_item.error or "Unknown error").splitlines()[0][:100]
                    summary_lines.append(f"• {f_item.file_path.name}: {err_brief}")

            summary_text = "\n".join(summary_lines)
            if self._is_cancelled:
                messagebox.showinfo("Batch Stopped", summary_text, parent=self.root)
            elif failed_count == 0:
                messagebox.showinfo("Batch Completed", summary_text, parent=self.root)
            else:
                messagebox.showwarning("Batch Completed with Errors", summary_text, parent=self.root)

    def _append_log(self, message: str) -> None:
        timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{timestamped}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        try:
            with self.log_file_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{timestamped}\n")
        except OSError:
            pass


def run() -> None:
    root = tk.Tk()
    PdfToMarkdownApp(root)
    root.mainloop()

