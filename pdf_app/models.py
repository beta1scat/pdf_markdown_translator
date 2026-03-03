from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConversionStats:
    page_count: int
    image_count: int


@dataclass(frozen=True)
class TimingStats:
    conversion_seconds: float = 0.0
    translation_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class ConversionResult:
    output_dir: Path
    markdown_path: Path
    translated_markdown_path: Path | None
    stats: ConversionStats
    timings: TimingStats
