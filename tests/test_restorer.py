from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from gemini_watermark_eraser.detector import _star_mask
from gemini_watermark_eraser.models import LogoDetection
from gemini_watermark_eraser.restorer import FramePatch, TemporalLogoRestorer


def _clean_frame(index: int, width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    shifted = xx + index * 3
    red = (shifted * 3 + yy) % 255
    green = (shifted + yy * 2) % 255
    blue = ((shifted // 8 % 2) * 150 + (yy // 8 % 2) * 60) % 255
    return np.stack([blue, green, red], axis=2).astype(np.uint8)


def _watermarked(frame: np.ndarray, detection: LogoDetection) -> np.ndarray:
    result = frame.copy().astype(np.float32)
    x, y, width, height = detection.bbox
    alpha = (detection.mask * 0.65)[..., None]
    result[y : y + height, x : x + width] = (
        result[y : y + height, x : x + width] * (1.0 - alpha) + 255.0 * alpha
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def test_temporal_restoration_is_local_and_reduces_logo_error() -> None:
    width, height, size = 320, 180, 24
    x, y = 270, 135
    mask = _star_mask(size)
    detection = LogoDetection(x, y, size, size, mask, 1.0)
    restorer = TemporalLogoRestorer(width, height, detection)

    clean_frames = [_clean_frame(index, width, height) for index in range(36)]
    marked_frames = [_watermarked(frame, detection) for frame in clean_frames]
    prepared = [restorer.prepare_frame(index, frame) for index, frame in enumerate(marked_frames)]
    past: deque[FramePatch] = deque(maxlen=restorer.temporal_radius)
    outputs: list[np.ndarray] = []
    for index, item in enumerate(prepared):
        output = restorer.restore(item, past=past, future=prepared[index + 1 :])
        past.append(restorer.make_patch(item))
        outputs.append(output)

    hard = np.zeros((height, width), dtype=bool)
    hard[y : y + size, x : x + size] = mask > 0.02
    before = np.mean(np.abs(marked_frames[20].astype(np.float32) - clean_frames[20])[hard])
    after = np.mean(np.abs(outputs[20].astype(np.float32) - clean_frames[20])[hard])
    assert after < before * 0.75
    clean_gray = cv2.cvtColor(clean_frames[20], cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(outputs[20], cv2.COLOR_BGR2GRAY)
    clean_detail = np.mean(np.abs(cv2.Laplacian(clean_gray, cv2.CV_32F))[hard])
    output_detail = np.mean(np.abs(cv2.Laplacian(output_gray, cv2.CV_32F))[hard])
    assert output_detail >= clean_detail * 0.65
    outside_bbox = np.ones((height, width), dtype=bool)
    outside_bbox[y : y + size, x : x + size] = False
    assert np.array_equal(outputs[20][outside_bbox], marked_frames[20][outside_bbox])


def test_mask_is_soft_and_does_not_fill_bounding_box_corners() -> None:
    mask = _star_mask(64)
    assert mask[32, 32] > 0.95
    assert mask[0, 0] < 0.01
    assert mask[:, 32].max() > 0.95


def test_fallback_keeps_visible_texture_under_translucent_logo() -> None:
    width, height, size = 320, 180, 24
    x, y = 270, 135
    detection = LogoDetection(x, y, size, size, _star_mask(size), 1.0)
    restorer = TemporalLogoRestorer(width, height, detection)
    yy, xx = np.mgrid[:height, :width]
    clean = np.stack(
        [
            ((xx * 9 + yy * 3) % 255),
            ((xx * 5 + yy * 11) % 255),
            ((xx * 13 + yy * 7) % 255),
        ],
        axis=2,
    ).astype(np.uint8)
    marked = _watermarked(clean, detection)
    roi = restorer.extract(marked)
    base = restorer._fallback(roi)
    detailed = restorer._restore_visible_detail(roi, base)
    interior = cv2.erode(restorer.hard_mask, np.ones((5, 5), np.uint8)) > 0
    change = np.mean(np.abs(detailed.astype(np.float32) - base.astype(np.float32))[interior])
    assert change > 0.1
