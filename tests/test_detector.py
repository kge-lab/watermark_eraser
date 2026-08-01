from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import gemini_watermark_eraser.detector as detector_module
from gemini_watermark_eraser.detector import detect_logo
from gemini_watermark_eraser.media import probe_media
from gemini_watermark_eraser.models import LogoNotFoundError, MediaInfo


@pytest.mark.parametrize("name", ["sample_1.mp4", "sample_2.mp4", "sample_3.mp4"])
def test_detects_logo_in_supplied_samples(name: str) -> None:
    media = probe_media(Path("sample") / name)
    detection = detect_logo(media)
    center_x = detection.x + detection.width / 2
    center_y = detection.y + detection.height / 2
    assert center_x == pytest.approx(media.width * 0.915, abs=22)
    assert center_y == pytest.approx(media.height * 0.86, abs=22)
    assert 0.06 <= detection.width / min(media.width, media.height) <= 0.12
    assert detection.confidence >= 0.16
    assert detection.mask.shape == (detection.height, detection.width)
    assert detection.mask.max() == pytest.approx(1.0, abs=0.02)


def test_rejects_video_without_a_fixed_sparkle(monkeypatch: pytest.MonkeyPatch) -> None:
    width, height = 640, 360
    yy, xx = np.mgrid[:height, :width]
    frames = []
    for index in range(12):
        gray = ((xx + index * 17) * 0.35 + yy * 0.25) % 255
        frame = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        frames.append(frame)
    media = MediaInfo(Path("negative.mp4"), width, height, 24.0, 120, 5.0)
    monkeypatch.setattr(detector_module, "_sample_frames", lambda *_args, **_kwargs: frames)
    with pytest.raises(LogoNotFoundError):
        detect_logo(media)
