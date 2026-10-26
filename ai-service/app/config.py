"""Runtime configuration, read from environment variables with safe defaults.

The service refuses to start with an invalid configuration (see ``__post_init__``),
so a mistyped threshold fails loudly instead of silently degrading detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = SERVICE_ROOT / "models" / "bisindo_yolo11_int8.onnx"

# Origins allowed to call the API from a browser: the local dev server and the
# production frontend. Override with a comma-separated ALLOWED_ORIGINS value; "*"
# disables the check (not recommended for a public deployment).
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://dewantara.cloud",
    "https://www.dewantara.cloud",
)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_origins(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip()) if raw else default


@dataclass(frozen=True)
class Settings:
    """Service settings. Every field can be overridden through an environment variable."""

    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    )
    input_size: int = field(default_factory=lambda: _env_int("INPUT_SIZE", 480))
    confidence_threshold: float = field(
        default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.40)
    )
    iou_threshold: float = field(default_factory=lambda: _env_float("IOU_THRESHOLD", 0.45))

    # Request limits. Values are deliberately generous for a 480 px webcam frame.
    max_frame_bytes: int = field(
        default_factory=lambda: _env_int("MAX_FRAME_BYTES", 4 * 1024 * 1024)
    )
    max_image_pixels: int = field(default_factory=lambda: _env_int("MAX_IMAGE_PIXELS", 8_000_000))
    max_concurrent_inferences: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_INFERENCES", 1)
    )
    idle_timeout_seconds: float = field(
        default_factory=lambda: _env_float("IDLE_TIMEOUT_SECONDS", 120.0)
    )

    allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: _env_origins("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1 (exclusive)")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("IOU_THRESHOLD must be between 0 and 1")
        if self.input_size <= 0 or self.input_size % 32:
            raise ValueError("INPUT_SIZE must be a positive multiple of 32")
        if self.max_frame_bytes <= 0 or self.max_image_pixels <= 0:
            raise ValueError("MAX_FRAME_BYTES and MAX_IMAGE_PIXELS must be positive")
        if self.max_concurrent_inferences <= 0:
            raise ValueError("MAX_CONCURRENT_INFERENCES must be at least 1")
        if self.idle_timeout_seconds <= 0:
            raise ValueError("IDLE_TIMEOUT_SECONDS must be positive")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
