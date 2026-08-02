from __future__ import annotations

import numpy as np

from gemini_watermark_eraser.alpha_profiles import alpha_profile_candidates, profile_sizes
from gemini_watermark_eraser.models import LogoDetection


def test_720p_profiles_are_centered_on_the_detected_logo() -> None:
    detection = LogoDetection(1127, 567, 65, 65, np.ones((65, 65), np.float32), 0.8)
    candidates = alpha_profile_candidates(
        (227, 227),
        roi_x0=1046,
        roi_y0=486,
        frame_width=1280,
        frame_height=720,
        detection=detection,
    )

    assert profile_sizes(1280, 720) == (48,)
    assert profile_sizes(720, 1280) == (48,)
    assert profile_sizes(1920, 1080) == (96,)
    assert len(candidates) == 1
    for profile in candidates:
        assert profile.shape == (227, 227)
        assert profile.dtype == np.float32
        assert 0.30 <= float(profile.max()) <= 0.38
        assert float(profile.min()) == 0.0
        assert not np.any((profile > 0.0) & (profile < 0.02))
        assert not profile.flags.writeable
        yy, xx = np.nonzero(profile > 0.02)
        np.testing.assert_allclose(
            np.mean(xx), detection.x + detection.width / 2 - 1046, atol=0.75
        )
        np.testing.assert_allclose(
            np.mean(yy), detection.y + detection.height / 2 - 486, atol=0.75
        )


def test_unvalidated_resolution_does_not_offer_a_deblend_profile() -> None:
    detection = LogoDetection(10, 10, 24, 24, np.ones((24, 24), np.float32), 1.0)
    assert profile_sizes(640, 360) == ()
    assert profile_sizes(1024, 720) == ()
    assert alpha_profile_candidates(
        (80, 80),
        roi_x0=0,
        roi_y0=0,
        frame_width=640,
        frame_height=360,
        detection=detection,
    ) == ()
