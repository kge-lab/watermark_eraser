from __future__ import annotations

from base64 import b64decode
from functools import lru_cache

import cv2
import numpy as np

from .models import LogoDetection


# GeminiWatermarkTool's current-profile 96 px background capture, used under
# its MIT license.  Keeping the calibrated capture (instead of a hand-drawn
# star) matters here: reverse alpha blending is highly sensitive to the exact
# antialiased edge profile.  See THIRD_PARTY_NOTICES.md.
_GWT_V2_96_PNG = (
    "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"
)


@lru_cache(maxsize=1)
def _canonical_v2_alpha() -> np.ndarray:
    encoded = np.frombuffer(b64decode(_GWT_V2_96_PNG), dtype=np.uint8)
    capture = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if capture is None or capture.shape[:2] != (96, 96):
        raise RuntimeError("내장된 제미나이 알파 프로필을 읽을 수 없습니다.")
    alpha = np.max(capture, axis=2).astype(np.float32) / 255.0
    alpha.setflags(write=False)
    return alpha


def profile_sizes(frame_width: int, frame_height: int) -> tuple[int, ...]:
    """Return calibrated profile sizes, best-known size first."""
    dimensions = (frame_width, frame_height)
    if dimensions in ((1280, 720), (720, 1280)):
        return (48,)
    if dimensions in ((1920, 1080), (1080, 1920)):
        return (96,)
    return ()


def alpha_profile_candidates(
    roi_shape: tuple[int, int],
    *,
    roi_x0: int,
    roi_y0: int,
    frame_width: int,
    frame_height: int,
    detection: LogoDetection,
) -> tuple[np.ndarray, ...]:
    """Build centered, calibrated physical-alpha maps in ROI coordinates."""
    roi_height, roi_width = roi_shape
    center_x = detection.x + detection.width / 2.0 - roi_x0
    center_y = detection.y + detection.height / 2.0 - roi_y0
    canonical = _canonical_v2_alpha()
    candidates: list[np.ndarray] = []
    for size in profile_sizes(frame_width, frame_height):
        x0 = int(round(center_x - size / 2.0))
        y0 = int(round(center_y - size / 2.0))
        x1, y1 = x0 + size, y0 + size
        if x0 < 0 or y0 < 0 or x1 > roi_width or y1 > roi_height:
            continue
        resized = cv2.resize(canonical, (size, size), interpolation=cv2.INTER_AREA)
        # The captured PNG contains a faint square compression floor around
        # the star.  It is not part of the physical watermark and deblending
        # it would turn that floor into a visible rectangle.
        resized[resized < 0.02] = 0.0
        profile = np.zeros((roi_height, roi_width), dtype=np.float32)
        profile[y0:y1, x0:x1] = resized
        profile.setflags(write=False)
        candidates.append(profile)
    return tuple(candidates)
