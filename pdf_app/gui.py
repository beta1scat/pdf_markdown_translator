from __future__ import annotations

from datetime import datetime
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import AppConfig, load_config, save_config
from .nvidia_models import NvidiaModelFetchError, fetch_nvidia_models
from .paths import get_app_base_dir
from .service import (
    MarkdownTranslationError,
    PdfConversionError,
    convert_pdf_to_markdown,
    translate_markdown_file,
)


class PdfToMarkdownApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PDF to Markdown")
        self.root.geometry("760x480")
        self.root.minsize(700, 400)

        self.config = load_config()
        self.is_running = False
        self.log_file_path = get_app_base_dir() / "app.log"

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

        ttk.Label(container, text="Input Type").grid(row=0, column=0, sticky="w", pady=(0, 12))
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

        ttk.Label(container, text="Input File").grid(row=1, column=0, sticky="w", pady=(0, 12))
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

        ttk.Label(container, text="Output Dir").grid(row=2, column=0, sticky="w", pady=(0, 12))
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

        ttk.Label(container, text="Status").grid(row=4, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            container,
            textvariable=self.status_var,
            wraplength=560,
            justify=tk.LEFT,
        ).grid(row=4, column=1, columnspan=2, sticky="w", pady=(0, 8))

        actions = ttk.Frame(container)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="Run", command=self.start_conversion).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Help", command=self.show_help).grid(row=0, column=1, sticky="e", padx=(0, 8))
        ttk.Button(actions, text="Settings", command=self.open_settings).grid(row=0, column=2, sticky="e")

        self.log_text = tk.Text(container, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=6, column=0, columnspan=3, sticky="nsew")

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=6, column=3, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self._append_log("App initialized.")
        self._append_log(f"PDF API URL: {self.config.api_url}")
        self._append_log(f"NVIDIA API URL: {self.config.nvidia_api_url}")
        self._append_log(
            f"NVIDIA rate limit: {self.config.max_requests_per_minute} requests/minute."
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
            "2. Choose the input file.\n"
            "3. Choose the output directory.\n"
            "4. Open Settings and fill in the Paddle/PDF API and NVIDIA API configuration.\n"
            "5. Click Run.\n\n"
            "Output rules:\n"
            "- PDF mode writes files into OUTPUT_DIR/<pdf_name>/.\n"
            "- Markdown mode writes <file_name>_zh.md directly into the output directory.\n\n"
            "Notes:\n"
            "- PDF translation requires both the Paddle/PDF API and the NVIDIA API.\n"
            "- Markdown translation requires only the NVIDIA API.\n"
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
        window.geometry("720x360")
        window.resizable(False, False)
        window.transient(self.root)
        window.grab_set()
        self.settings_window = window

        api_url_var = tk.StringVar(value=self.config.api_url)
        api_token_var = tk.StringVar(value=self.config.api_token)
        timeout_var = tk.StringVar(value=str(self.config.request_timeout_seconds))
        nvidia_url_var = tk.StringVar(value=self.config.nvidia_api_url)
        nvidia_key_var = tk.StringVar(value=self.config.nvidia_api_key)
        nvidia_model_var = tk.StringVar(value=self.config.nvidia_model)
        rate_limit_var = tk.StringVar(value=str(self.config.max_requests_per_minute))
        concurrency_var = tk.StringVar(value=str(self.config.translation_concurrency))
        model_values = [self.config.nvidia_model] if self.config.nvidia_model else []

        container = ttk.Frame(window, padding=16)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)

        fields = [
            ("Paddle/PDF API URL", api_url_var, False),
            ("Paddle/PDF API Token", api_token_var, True),
            ("Request Timeout (s)", timeout_var, False),
            ("NVIDIA API URL", nvidia_url_var, False),
            ("NVIDIA API Key", nvidia_key_var, True),
            ("NVIDIA Model", nvidia_model_var, False),
            ("NVIDIA Rate Limit / min", rate_limit_var, False),
            ("Translation Concurrency", concurrency_var, False),
        ]

        for row_index, (label, variable, secret) in enumerate(fields):
            ttk.Label(container, text=label).grid(row=row_index, column=0, sticky="w", pady=(0, 12))
            if label == "NVIDIA Model":
                entry = ttk.Combobox(
                    container,
                    textvariable=variable,
                    values=model_values,
                )
            else:
                entry = ttk.Entry(
                    container,
                    textvariable=variable,
                    show="*" if secret else "",
                )
            entry.grid(row=row_index, column=1, sticky="ew", padx=(12, 12), pady=(0, 12))
            if secret:
                visible_var = tk.BooleanVar(value=False)
                toggle_button = ttk.Button(container, text="Show")
                toggle_button.configure(
                    command=lambda e=entry, v=visible_var, b=toggle_button: self._toggle_secret_visibility(e, v, b)
                )
                toggle_button.grid(row=row_index, column=2, sticky="e", pady=(0, 12))
            elif label == "NVIDIA Model":
                model_combobox = entry
                refresh_button = ttk.Button(
                    container,
                    text="Refresh",
                    command=lambda c=model_combobox: self._refresh_nvidia_models(
                        nvidia_url_var.get(),
                        nvidia_key_var.get(),
                        timeout_var.get(),
                        rate_limit_var.get(),
                        c,
                        nvidia_model_var,
                        window,
                    ),
                )
                refresh_button.grid(row=row_index, column=2, sticky="e", pady=(0, 12))

        button_row = ttk.Frame(container)
        button_row.grid(row=len(fields), column=0, columnspan=2, sticky="ew", pady=(8, 0))
        button_row.columnconfigure(0, weight=1)
        ttk.Button(
            button_row,
            text="Save",
            command=lambda: self._save_settings(
                window,
                api_url_var.get(),
                api_token_var.get(),
                timeout_var.get(),
                nvidia_url_var.get(),
                nvidia_key_var.get(),
                nvidia_model_var.get(),
                rate_limit_var.get(),
                concurrency_var.get(),
            ),
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(button_row, text="Cancel", command=self._close_settings_window).grid(row=0, column=1, sticky="e")
        window.protocol("WM_DELETE_WINDOW", self._close_settings_window)

    def _close_settings_window(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.destroy()
        self.settings_window = None

    def _refresh_nvidia_models(
        self,
        nvidia_api_url: str,
        nvidia_api_key: str,
        timeout_seconds: str,
        rate_limit: str,
        combobox: ttk.Combobox,
        model_var: tk.StringVar,
        window: tk.Toplevel,
    ) -> None:
        try:
            timeout_value = int(timeout_seconds)
            rate_limit_value = int(rate_limit)
        except ValueError:
            messagebox.showerror(
                "Invalid Settings",
                "Request timeout and rate limit must be integers before refreshing models.",
                parent=window,
            )
            return

        temp_config = AppConfig(
            api_url=self.config.api_url,
            api_token=self.config.api_token,
            request_timeout_seconds=timeout_value,
            nvidia_api_url=nvidia_api_url.strip() or self.config.nvidia_api_url,
            nvidia_api_key=nvidia_api_key.strip(),
            nvidia_model=model_var.get().strip() or self.config.nvidia_model,
            max_requests_per_minute=rate_limit_value,
            translation_concurrency=self.config.translation_concurrency,
        )

        try:
            models = fetch_nvidia_models(temp_config)
        except NvidiaModelFetchError as exc:
            messagebox.showerror("Refresh Failed", str(exc), parent=window)
            return

        combobox.configure(values=models)
        if models and model_var.get().strip() not in models:
            model_var.set(models[0])
        self._append_log(f"Loaded {len(models)} NVIDIA models from official sources.")
        messagebox.showinfo("Models Refreshed", f"Loaded {len(models)} models.", parent=window)

    def _save_settings(
        self,
        window: tk.Toplevel,
        api_url: str,
        api_token: str,
        timeout_seconds: str,
        nvidia_api_url: str,
        nvidia_api_key: str,
        nvidia_model: str,
        rate_limit: str,
        translation_concurrency: str,
    ) -> None:
        api_url = api_url.strip()
        api_token = api_token.strip()
        nvidia_api_url = nvidia_api_url.strip()
        nvidia_api_key = nvidia_api_key.strip()
        nvidia_model = nvidia_model.strip()

        if not api_url:
            messagebox.showerror("Invalid Settings", "Paddle/PDF API URL is required.", parent=window)
            return
        if not api_token:
            messagebox.showerror("Invalid Settings", "Paddle/PDF API token is required.", parent=window)
            return
        if not nvidia_api_url:
            messagebox.showerror("Invalid Settings", "NVIDIA API URL is required.", parent=window)
            return
        if not nvidia_model:
            messagebox.showerror("Invalid Settings", "NVIDIA model is required.", parent=window)
            return

        try:
            timeout_value = int(timeout_seconds)
            rate_limit_value = int(rate_limit)
            concurrency_value = int(translation_concurrency)
        except ValueError:
            messagebox.showerror(
                "Invalid Settings",
                "Request timeout, rate limit, and translation concurrency must be integers.",
                parent=window,
            )
            return

        if timeout_value <= 0 or rate_limit_value <= 0 or concurrency_value <= 0:
            messagebox.showerror(
                "Invalid Settings",
                "Request timeout, rate limit, and translation concurrency must be greater than 0.",
                parent=window,
            )
            return

        self.config = AppConfig(
            api_url=api_url,
            api_token=api_token,
            request_timeout_seconds=timeout_value,
            nvidia_api_url=nvidia_api_url,
            nvidia_api_key=nvidia_api_key,
            nvidia_model=nvidia_model,
            max_requests_per_minute=rate_limit_value,
            translation_concurrency=concurrency_value,
        )
        save_config(self.config)
        self._append_log("Settings saved.")
        self._append_log(f"PDF API URL: {self.config.api_url}")
        self._append_log(f"NVIDIA API URL: {self.config.nvidia_api_url}")
        self._append_log(f"NVIDIA model: {self.config.nvidia_model}")
        self._append_log(
            f"NVIDIA rate limit: {self.config.max_requests_per_minute} requests/minute."
        )
        self._append_log(
            f"Translation concurrency: {self.config.translation_concurrency}"
        )
        self._close_settings_window()
        messagebox.showinfo("Settings Saved", "Configuration saved successfully.")

    def _on_mode_change(self) -> None:
        if self.input_mode_var.get() == "markdown":
            self.translate_var.set(True)
            self.status_var.set("Ready to translate a Markdown file directly.")
        else:
            self.status_var.set("Ready.")

    def _update_phase_status(self, phase: str) -> None:
        phase_map = {
            "converting": "Converting PDF to Markdown...",
            "translating": "Translating Markdown...",
            "completed": "Completed.",
            "failed": "Failed.",
        }
        message = phase_map.get(phase, phase)
        self.root.after(0, self.status_var.set, message)
        self.root.after(0, self._append_log, f"Phase: {message}")

    def _update_translation_progress(self, translated_chars: int, total_chars: int) -> None:
        if total_chars <= 0:
            return
        if translated_chars == self._last_progress_log_chars and translated_chars != total_chars:
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
            title = "Select a Markdown file"
        else:
            filetypes = [("PDF files", "*.pdf")]
            title = "Select a PDF file"

        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if file_path:
            self.input_path_var.set(file_path)

    def select_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_dir_var.set(directory)

    def start_conversion(self) -> None:
        if self.is_running:
            return

        input_path = self.input_path_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        translate_markdown = self.translate_var.get()
        input_mode = self.input_mode_var.get()

        if not input_path:
            messagebox.showerror("Missing Input File", "Please select an input file.")
            return
        if not output_dir:
            messagebox.showerror("Missing Output Directory", "Please select an output directory.")
            return
        if (translate_markdown or input_mode == "markdown") and not self.config.nvidia_api_key:
            messagebox.showerror("Missing NVIDIA API Key", "Please configure the NVIDIA API key in Settings.")
            return

        self.is_running = True
        self._last_progress_log_chars = -1
        self.status_var.set(
            "Translating Markdown..." if input_mode == "markdown" else "Converting PDF to Markdown..."
        )
        self._append_log(f"Input mode: {input_mode}")
        self._append_log(f"Input file: {input_path}")
        self._append_log(f"Output directory: {output_dir}")
        self._append_log(f"Translate Markdown: {'yes' if translate_markdown else 'no'}")

        worker = threading.Thread(
            target=self._run_conversion,
            args=(input_mode, input_path, output_dir, translate_markdown),
            daemon=True,
        )
        worker.start()

    def _run_conversion(
        self,
        input_mode: str,
        input_path: str,
        output_dir: str,
        translate_markdown: bool,
    ) -> None:
        try:
            if input_mode == "markdown":
                result = translate_markdown_file(
                    input_path,
                    output_dir,
                    self.config,
                    phase_callback=self._update_phase_status,
                    progress_callback=self._update_translation_progress,
                )
            else:
                result = convert_pdf_to_markdown(
                    input_path,
                    output_dir,
                    self.config,
                    translate_markdown=translate_markdown,
                    phase_callback=self._update_phase_status,
                    progress_callback=self._update_translation_progress,
                )
        except (PdfConversionError, MarkdownTranslationError) as exc:
            self.root.after(0, self._on_failure, str(exc))
            return
        except Exception as exc:
            stack = traceback.format_exc()
            self.root.after(0, self._on_failure, f"{exc}\n\n{stack}")
            return

        self.root.after(
            0,
            self._on_success,
            result.markdown_path,
            result.html_path,
            result.translated_markdown_path,
            result.translated_html_path,
            result.output_dir,
            result.stats.page_count,
            result.stats.image_count,
            result.timings.conversion_seconds,
            result.timings.translation_seconds,
            result.timings.total_seconds,
        )

    def _on_success(
        self,
        markdown_path: Path,
        html_path: Path | None,
        translated_markdown_path: Path | None,
        translated_html_path: Path | None,
        output_dir: Path,
        page_count: int,
        image_count: int,
        conversion_seconds: float,
        translation_seconds: float,
        total_seconds: float,
    ) -> None:
        self.is_running = False
        self.status_var.set("Completed.")
        self._append_log(f"Source Markdown: {markdown_path}")
        if html_path is not None:
            self._append_log(f"Source HTML saved to: {html_path}")
        if translated_markdown_path is not None:
            self._append_log(f"Translated Markdown saved to: {translated_markdown_path}")
        if translated_html_path is not None:
            self._append_log(f"Translated HTML saved to: {translated_html_path}")
        self._append_log(f"Output folder: {output_dir}")
        self._append_log(f"Pages: {page_count}, Images: {image_count}")
        self._append_log(f"Conversion time: {conversion_seconds:.2f}s")
        self._append_log(f"Translation time: {translation_seconds:.2f}s")
        self._append_log(f"Total time: {total_seconds:.2f}s")

        translated_label = translated_markdown_path or "Not generated"
        translated_html_label = translated_html_path or "Not generated"
        html_label = html_path or "Not generated"
        messagebox.showinfo(
            "Completed",
            (
                f"Markdown saved to:\n{markdown_path}\n\n"
                f"HTML saved to:\n{html_label}\n\n"
                f"Translated Markdown:\n{translated_label}\n\n"
                f"Translated HTML:\n{translated_html_label}\n\n"
                f"Pages: {page_count}\nImages: {image_count}\n"
                f"Conversion time: {conversion_seconds:.2f}s\n"
                f"Translation time: {translation_seconds:.2f}s\n"
                f"Total time: {total_seconds:.2f}s"
            ),
        )

    def _on_failure(self, error_message: str) -> None:
        self.is_running = False
        self.status_var.set("Failed.")
        self._append_log(f"Error: {error_message}")
        messagebox.showerror("Conversion Failed", error_message)

    def _append_log(self, message: str) -> None:
        timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{timestamped}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        with self.log_file_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamped}\n")


def run() -> None:
    root = tk.Tk()
    PdfToMarkdownApp(root)
    root.mainloop()
