from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np


class JobStatus(StrEnum):
    QUEUED = "대기"
    DETECTING = "로고 탐지"
    RESTORING = "복원"
    ENCODING = "인코딩"
    COMPLETED = "완료"
    FAILED = "실패"
    CANCELLED = "취소"


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


@dataclass(frozen=True, slots=True)
class LogoDetection:
    """Detected logo in display-oriented frame coordinates."""

    x: int
    y: int
    width: int
    height: int
    mask: np.ndarray
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True, slots=True)
class JobResult:
    input_path: Path
    output_path: Path | None
    status: JobStatus
    message: str = ""


class ProcessingError(RuntimeError):
    """A user-facing media processing failure."""


class LogoNotFoundError(ProcessingError):
    """Raised when a Gemini logo cannot be identified safely."""


class ProcessingCancelled(ProcessingError):
    """Raised after a cancellation request."""
