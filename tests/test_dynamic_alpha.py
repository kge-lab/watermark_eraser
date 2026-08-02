from __future__ import annotations

import numpy as np
import pytest

from gemini_watermark_eraser.dynamic_alpha import (
    DynamicAlphaConfig,
    DynamicAlphaState,
    restore_dynamic_alpha,
    reverse_alpha_deblend,
    select_alpha_scale,
)


def _marked(
    clean: np.ndarray,
    watermark: np.ndarray,
    profile: np.ndarray,
    scale: float,
) -> np.ndarray:
    alpha = profile[..., None] * scale
    return clean * (1.0 - alpha) + watermark * alpha


def test_reverse_alpha_deblend_recovers_clean_pixels_and_preserves_dtype() -> None:
    clean = np.array(
        [[[35, 80, 120], [60, 100, 140]], [[90, 120, 150], [110, 140, 170]]],
        dtype=np.float32,
    )
    watermark = np.array([245.0, 250.0, 255.0], dtype=np.float32)
    alpha = np.array([[0.0, 0.12], [0.30, 0.48]], dtype=np.float32)
    observed = clean * (1.0 - alpha[..., None]) + watermark * alpha[..., None]
    original = observed.copy()

    result = reverse_alpha_deblend(observed, watermark, alpha)

    assert result.candidate.dtype == observed.dtype
    assert np.allclose(result.candidate, clean, atol=2e-5)
    assert np.array_equal(result.candidate[0, 0], observed[0, 0])
    assert result.clipping_fraction == pytest.approx(0.0)
    assert np.array_equal(observed, original)


def test_scale_selection_is_robust_and_honours_exact_reference_confidence() -> None:
    yy, xx = np.mgrid[:8, :9]
    profile = ((xx + yy + 1) / 16.0).astype(np.float32)
    clean_gray = 40.0 + xx * 5.0 + yy * 3.0
    clean = np.stack([clean_gray, clean_gray + 15.0, clean_gray + 30.0], axis=2).astype(np.float32)
    watermark = np.array([240.0, 248.0, 252.0], dtype=np.float32)
    observed = _marked(clean, watermark, profile, 0.46)
    observed[:2, :2] = 0.0
    confidence = np.ones(profile.shape, dtype=np.float32)
    confidence[:2, :2] = 0.0

    selection = select_alpha_scale(
        observed,
        watermark,
        profile,
        clean_reference=clean,
        reference_confidence=confidence,
    )

    assert selection.used_clean_reference
    assert selection.reference_samples > 100
    assert selection.raw_scale == pytest.approx(0.46, abs=1e-4)
    assert selection.scale == pytest.approx(0.46, abs=1e-4)


def test_restore_returns_strict_quality_decision_for_good_reference() -> None:
    profile = np.linspace(0.05, 1.0, 42, dtype=np.float32).reshape(6, 7)
    clean = np.full((6, 7, 3), (60.0, 100.0, 145.0), dtype=np.float32)
    watermark = np.array([245.0, 250.0, 255.0], dtype=np.float32)
    observed = _marked(clean, watermark, profile, 0.42)

    result = restore_dynamic_alpha(
        observed,
        watermark,
        profile,
        clean_reference=clean,
    )

    assert result.quality.accepted, result.quality.reason
    assert result.selection.scale == pytest.approx(0.42, abs=1e-5)
    assert np.allclose(result.candidate, clean, atol=3e-5)
    assert result.quality.reference_improvement == pytest.approx(1.0, abs=1e-5)
    assert result.quality.reference_tail_ratio == pytest.approx(0.0, abs=1e-5)
    assert result.quality.reference_worsened_fraction == pytest.approx(0.0)


def test_temporal_state_rate_limits_scale_and_only_records_accepted_results() -> None:
    profile = np.ones((5, 5), dtype=np.float32) * 0.5
    clean = np.full((5, 5, 3), 80.0, dtype=np.float32)
    watermark = np.array([240.0, 245.0, 250.0], dtype=np.float32)
    state = DynamicAlphaState()
    config = DynamicAlphaConfig(maximum_scale_step=0.05)

    first = restore_dynamic_alpha(
        _marked(clean, watermark, profile, 0.40),
        watermark,
        profile,
        clean_reference=clean,
        state=state,
        config=config,
    )
    second = restore_dynamic_alpha(
        _marked(clean, watermark, profile, 0.70),
        watermark,
        profile,
        clean_reference=clean,
        state=state,
        config=config,
    )

    assert first.quality.accepted
    assert second.selection.rate_limited
    assert second.selection.scale == pytest.approx(0.45, abs=1e-5)
    assert state.accepted_frames == 2
    assert state.last_scale == pytest.approx(0.45, abs=1e-5)


def test_clipping_and_noise_amplification_reject_unsafe_deblend() -> None:
    observed = np.zeros((4, 4, 3), dtype=np.uint8)
    profile = np.ones((4, 4), dtype=np.float32)
    state = DynamicAlphaState(last_scale=0.8)
    config = DynamicAlphaConfig(maximum_scale_step=1.0)

    result = restore_dynamic_alpha(
        observed,
        255.0,
        profile,
        proposed_scale=0.8,
        state=state,
        config=config,
    )

    assert not result.quality.accepted
    assert "excessive_clipping" in result.quality.reasons
    assert "excessive_noise_gain" in result.quality.reasons
    assert result.quality.clipping_fraction == pytest.approx(1.0)
    assert np.all(np.isfinite(result.candidate))
    assert state.last_scale == pytest.approx(0.8)
    assert state.rejected_frames == 1


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (np.ones((2, 3), dtype=np.float32), "shape"),
        (np.array([[0.0, -0.1], [0.2, 1.0]], dtype=np.float32), r"\[0, 1\]"),
    ],
)
def test_invalid_profiles_are_rejected(profile: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        restore_dynamic_alpha(np.zeros((2, 2, 3), dtype=np.uint8), 255.0, profile)
