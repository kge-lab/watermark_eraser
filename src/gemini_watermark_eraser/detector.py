from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .models import LogoDetection, LogoNotFoundError, MediaInfo, ProcessingCancelled, ProcessingError

CancelCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class _Candidate:
    score: float
    x: int
    y: int
    size: int
    per_frame_scores: np.ndarray


def _star_mask(size: int) -> np.ndarray:
    """Create an anti-aliased four-point Gemini-style silhouette."""
    scale = 4
    canvas_size = size * scale
    center = canvas_size / 2
    outer = canvas_size * 0.47
    inner = canvas_size * 0.16
    points: list[tuple[int, int]] = []
    for index in range(16):
        angle = -np.pi / 2 + index * np.pi / 8
        if index % 4 == 0:
            radius = outer
        elif index % 2 == 0:
            radius = inner * 1.18
        else:
            radius = inner
        points.append(
            (int(round(center + np.cos(angle) * radius)), int(round(center + np.sin(angle) * radius)))
        )
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.asarray(points, dtype=np.int32)], 255, lineType=cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=max(0.7, scale * 0.55))
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _template_for(size: int) -> tuple[np.ndarray, np.ndarray]:
    mask = _star_mask(size)
    background = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, size * 0.20))
    template = mask - background
    template -= template.mean()
    norm = float(np.linalg.norm(template))
    if norm:
        template /= norm
    return mask, template.astype(np.float32)


def _sample_frames(media: MediaInfo, count: int, cancelled: CancelCheck) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(media.path))
    if not capture.isOpened():
        raise ProcessingError("로고 탐지를 위해 영상을 열 수 없습니다.")
    indices = np.linspace(0, max(0, media.frame_count - 1), num=count, dtype=np.int64)
    frames: list[np.ndarray] = []
    try:
        for index in indices:
            if cancelled():
                raise ProcessingCancelled("작업이 취소되었습니다.")
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        capture.release()
    if len(frames) < 4:
        raise ProcessingError("로고를 분석할 프레임이 부족합니다.")
    return frames


def detect_logo(
    media: MediaInfo,
    *,
    cancelled: CancelCheck = lambda: False,
    minimum_confidence: float = 0.16,
) -> LogoDetection:
    sample_count = min(24, max(12, int(round(media.duration * 2))))
    frames = _sample_frames(media, sample_count, cancelled)
    height, width = frames[0].shape[:2]

    x0, y0 = int(width * 0.64), int(height * 0.54)
    search_width, search_height = width - x0, height - y0
    shortest = min(width, height)
    sizes = sorted({max(20, int(round(shortest * ratio))) for ratio in np.linspace(0.065, 0.115, 7)})

    gray_regions: list[np.ndarray] = []
    for frame in frames:
        gray = cv2.cvtColor(frame[y0:, x0:], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_regions.append(gray)

    best: _Candidate | None = None
    for size in sizes:
        if cancelled():
            raise ProcessingCancelled("작업이 취소되었습니다.")
        if size >= search_width or size >= search_height:
            continue
        _, template = _template_for(size)
        maps: list[np.ndarray] = []
        for gray in gray_regions:
            local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(2.0, size * 0.20))
            high_pass = gray - local_background
            maps.append(cv2.matchTemplate(high_pass, template, cv2.TM_CCOEFF_NORMED))
        score_stack = np.stack(maps)
        aggregate = np.median(score_stack, axis=0)

        map_height, map_width = aggregate.shape
        yy, xx = np.mgrid[:map_height, :map_width]
        expected_x = width * 0.915 - x0 - size / 2
        expected_y = height * 0.86 - y0 - size / 2
        sigma_x = max(1.0, width * 0.12)
        sigma_y = max(1.0, height * 0.14)
        prior = np.exp(-0.5 * (((xx - expected_x) / sigma_x) ** 2 + ((yy - expected_y) / sigma_y) ** 2))
        ranked = aggregate + 0.07 * prior.astype(np.float32)
        _, _, _, location = cv2.minMaxLoc(ranked)
        candidate_x, candidate_y = location
        raw_scores = score_stack[:, candidate_y, candidate_x]
        stable_score = float(np.median(raw_scores))
        consistency = float(np.mean(raw_scores > max(0.08, stable_score * 0.55)))
        scale_ratio = size / shortest
        scale_prior = float(np.exp(-0.5 * ((scale_ratio - 0.09) / 0.008) ** 2))
        position_prior = float(prior[candidate_y, candidate_x])
        stable_component = stable_score * (0.70 + 0.30 * consistency)
        combined = stable_component * (0.55 + 0.45 * position_prior) + 0.06 * scale_prior
        candidate = _Candidate(combined, x0 + candidate_x, y0 + candidate_y, size, raw_scores)
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None or best.score < minimum_confidence:
        score = 0.0 if best is None else best.score
        raise LogoNotFoundError(f"제미나이 로고를 확실히 찾지 못했습니다 (신뢰도 {score:.2f}).")

    base_mask, _ = _template_for(best.size)
    hard = (base_mask > 0.035).astype(np.uint8)
    # The visible Gemini mark has a faint translucent halo outside its bright core.
    # Cover it deliberately; a narrow mask is perceived as a blurred logo residue.
    dilation = max(2, int(round(best.size * 0.035)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
    hard = cv2.dilate(hard, kernel)
    soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigmaX=max(0.8, best.size * 0.016))
    # Raise the feather slightly so low-alpha logo pixels are replaced instead
    # of being mixed back into the repaired texture as a faint ghost.
    soft = np.clip(soft * 1.16, 0.0, 1.0)
    return LogoDetection(best.x, best.y, best.size, best.size, soft, best.score)


def detection_debug_frame(path: Path, detection: LogoDetection) -> np.ndarray:
    """Small helper used by tests and diagnostics, not exposed in the UI."""
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ProcessingError("진단 프레임을 읽을 수 없습니다.")
    x, y, width, height = detection.bbox
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return overlay
