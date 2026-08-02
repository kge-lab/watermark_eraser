from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2

from gemini_watermark_eraser.detector import detect_logo
from gemini_watermark_eraser.media import probe_media
from gemini_watermark_eraser.restorer import FramePatch, PreparedFrame, TemporalLogoRestorer


def test_sample_three_keeps_every_frame_on_the_legacy_fallback() -> None:
    source = Path(__file__).resolve().parents[1] / "sample" / "sample_3.mp4"
    media = probe_media(source)
    detection = detect_logo(media)
    restorer = TemporalLogoRestorer(media.width, media.height, detection)
    capture = cv2.VideoCapture(str(source))
    assert capture.isOpened()

    future: deque[PreparedFrame] = deque()
    past: deque[FramePatch] = deque(maxlen=restorer.temporal_radius)
    next_index = 0

    def read_next() -> bool:
        nonlocal next_index
        ok, frame = capture.read()
        if not ok or frame is None:
            return False
        future.append(restorer.prepare_frame(next_index, frame))
        next_index += 1
        return True

    try:
        for _ in range(restorer.temporal_radius + 1):
            if not read_next():
                break

        processed = 0
        while future:
            prepared = future.popleft()
            read_next()
            restorer.restore(prepared, past=past, future=future)
            past.append(restorer.make_patch(prepared))
            processed += 1
    finally:
        capture.release()

    # OpenCV reports this sample as 240 frames on Windows/arm64 and 241 on
    # Intel macOS because the container's duration metadata rounds differently.
    # The restoration contract is to process every frame that the decoder
    # actually yields, not to trust a platform-specific container estimate.
    assert processed == next_index
    assert processed >= 240
    assert restorer._dynamic_state.accepted_frames == 0
    assert restorer._dynamic_disabled
