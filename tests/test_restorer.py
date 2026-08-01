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


def _frame_mask(width: int, height: int, detection: LogoDetection) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, mask_width, mask_height = detection.bbox
    mask[y : y + mask_height, x : x + mask_width] = detection.mask > 0.02
    return mask


def _boundary_band(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel) > 0
    eroded = cv2.erode(mask, kernel) > 0
    return dilated & ~eroded


def _sobel_magnitude(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return cv2.magnitude(horizontal, vertical)


def _laplacian_magnitude(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.abs(cv2.Laplacian(gray, cv2.CV_32F))


def _allowed_edit_support(width: int, height: int, detection: LogoDetection) -> np.ndarray:
    detector_support = _frame_mask(width, height, detection)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(detector_support, kernel) > 0


def _restore_sequence(restorer: TemporalLogoRestorer, frames: list[np.ndarray]) -> list[np.ndarray]:
    prepared = [restorer.prepare_frame(index, frame) for index, frame in enumerate(frames)]
    past: deque[FramePatch] = deque(maxlen=restorer.temporal_radius)
    outputs: list[np.ndarray] = []
    for index, item in enumerate(prepared):
        outputs.append(restorer.restore(item, past=past, future=prepared[index + 1 :]))
        past.append(restorer.make_patch(item))
    return outputs


def _shift_mask(mask: np.ndarray, horizontal: int, vertical: int) -> np.ndarray:
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    source_y0 = max(0, -vertical)
    source_y1 = min(height, height - vertical)
    source_x0 = max(0, -horizontal)
    source_x1 = min(width, width - horizontal)
    shifted[
        source_y0 + vertical : source_y1 + vertical,
        source_x0 + horizontal : source_x1 + horizontal,
    ] = mask[source_y0:source_y1, source_x0:source_x1]
    return shifted


def _ambiguous_roof_frames(
    count: int,
    width: int,
    height: int,
    detection: LogoDetection,
) -> list[np.ndarray]:
    """Build two near-tied repeating roof motifs with deterministic micro variation."""
    support = _frame_mask(width, height, detection)
    context_ring = (cv2.dilate(support, np.ones((9, 9), dtype=np.uint8)) > 0) & ~(support > 0)
    motif_offsets = ((-26, 13), (16, -26))
    motif_supports = [_shift_mask(support, *offset) > 0 for offset in motif_offsets]
    ring_y, ring_x = np.nonzero(context_ring)

    yy, xx = np.mgrid[:height, :width]
    rng = np.random.default_rng(444)
    roof_noise = cv2.GaussianBlur(
        rng.normal(0.0, 1.0, (height, width)).astype(np.float32),
        (0, 0),
        sigmaX=0.55,
    ) * 6.0
    phase = (xx + yy * 2) % 12
    ridge = np.where(phase < 2, 58.0, np.where(phase < 4, 18.0, 0.0))
    motif_strength = 48.0

    frames: list[np.ndarray] = []
    for index in range(count):
        brightness_drift = ((index % 5) - 2) * 0.18
        micro_variation = 0.28 * np.sin(xx * 0.37 + yy * 0.19 + index * 0.41)
        base = 70.0 + xx * 0.10 + yy * 0.04 + ridge + roof_noise
        base = base + brightness_drift + micro_variation
        frame = np.stack([base * 0.93, base + 8.0, base], axis=2)

        # The same roof phase appears at both locations, but their subtle
        # colour subpatterns differ enough to make donor switching visible.
        frame[motif_supports[0]] += np.array(
            [motif_strength, -0.35 * motif_strength, 0.65 * motif_strength]
        )
        frame[motif_supports[1]] += np.array(
            [-motif_strength, 0.35 * motif_strength, -0.65 * motif_strength]
        )

        # Alternate the closer contextual match every two frames. A stable
        # repair should not alternate its filled core with this near tie.
        horizontal, vertical = motif_offsets[(index // 2) % len(motif_offsets)]
        frame[ring_y, ring_x] = frame[ring_y + vertical, ring_x + horizontal]
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def test_temporal_restoration_is_local_and_reduces_logo_error() -> None:
    width, height, size = 320, 180, 24
    x, y = 270, 135
    mask = _star_mask(size)
    detection = LogoDetection(x, y, size, size, mask, 1.0)
    restorer = TemporalLogoRestorer(width, height, detection)

    clean_frames = [_clean_frame(index, width, height) for index in range(36)]
    marked_frames = [_watermarked(frame, detection) for frame in clean_frames]
    outputs = _restore_sequence(restorer, marked_frames)

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

    boundary = _boundary_band(_frame_mask(width, height, detection))
    clean_boundary_sobel = float(np.mean(_sobel_magnitude(clean_frames[20])[boundary]))
    output_boundary_sobel = float(np.mean(_sobel_magnitude(outputs[20])[boundary]))
    sobel_ratio = output_boundary_sobel / clean_boundary_sobel
    assert 0.65 <= sobel_ratio <= 1.25, f"boundary Sobel ratio {sobel_ratio:.3f} indicates blur or an outline"

    clean_boundary_laplacian = float(np.mean(_laplacian_magnitude(clean_frames[20])[boundary]))
    output_boundary_laplacian = float(np.mean(_laplacian_magnitude(outputs[20])[boundary]))
    laplacian_ratio = output_boundary_laplacian / clean_boundary_laplacian
    assert 0.60 <= laplacian_ratio <= 1.30, (
        f"boundary Laplacian ratio {laplacian_ratio:.3f} indicates blur or ringing"
    )

    outside_allowed_support = ~_allowed_edit_support(width, height, detection)
    assert np.array_equal(outputs[20][outside_allowed_support], marked_frames[20][outside_allowed_support])


def test_mask_is_soft_and_does_not_fill_bounding_box_corners() -> None:
    mask = _star_mask(64)
    assert mask[32, 32] > 0.95
    assert mask[0, 0] < 0.01
    assert mask[:, 32].max() > 0.95


def test_detail_restoration_does_not_reinject_the_source_logo_edge() -> None:
    width, height, size = 256, 160, 48
    x, y = 176, 88
    detection = LogoDetection(x, y, size, size, _star_mask(size), 1.0)
    restorer = TemporalLogoRestorer(width, height, detection)
    clean = np.full((height, width, 3), (70, 110, 150), dtype=np.uint8)
    marked = _watermarked(clean, detection)
    output = restorer.restore(restorer.prepare_frame(0, marked), past=[], future=[])
    interior = cv2.erode(_frame_mask(width, height, detection), np.ones((5, 5), np.uint8)) > 0
    error = np.mean(np.abs(output.astype(np.float32) - clean.astype(np.float32)), axis=2)

    # The clean background is flat, so every high-frequency residual here came
    # from the watermark rather than from useful source texture.
    assert float(np.mean(error[interior])) <= 0.5
    assert float(np.percentile(error[interior], 95)) <= 1.0


def test_restore_does_not_blend_a_completed_repair_twice(monkeypatch) -> None:
    width, height, size = 256, 160, 48
    x, y = 176, 88
    detection = LogoDetection(x, y, size, size, _star_mask(size), 1.0)
    restorer = TemporalLogoRestorer(width, height, detection)
    clean = np.full((height, width, 3), (70, 110, 150), dtype=np.uint8)
    marked = _watermarked(clean, detection)
    prepared = restorer.prepare_frame(0, marked)
    completed_repair = restorer.extract(clean)

    monkeypatch.setattr(restorer, "_fuse", lambda _current, _candidates: completed_repair.copy())
    output = restorer.restore(prepared, past=[], future=[])
    output_roi = restorer.extract(output)
    assert np.array_equal(output_roi, completed_repair)


def test_dark_to_bright_scene_transition_does_not_leave_a_dark_or_bright_outline() -> None:
    width, height, size = 320, 180, 24
    x, y = 270, 135
    detection = LogoDetection(x, y, size, size, _star_mask(size), 1.0)
    restorer = TemporalLogoRestorer(width, height, detection)

    frame_count = 26
    shift_per_frame = 3
    rng = np.random.default_rng(20260801)
    dark_texture = rng.integers(
        25,
        145,
        size=(height, width + frame_count * shift_per_frame, 3),
        dtype=np.uint8,
    )
    dark_texture = cv2.GaussianBlur(dark_texture, (0, 0), sigmaX=0.8)
    dark_marked = [
        _watermarked(
            dark_texture[:, index * shift_per_frame : index * shift_per_frame + width].copy(),
            detection,
        )
        for index in range(frame_count)
    ]
    _restore_sequence(restorer, dark_marked)

    bright_clean = np.full((height, width, 3), (205, 215, 225), dtype=np.uint8)
    bright_marked = _watermarked(bright_clean, detection)
    bright_output = restorer.restore(
        restorer.prepare_frame(frame_count, bright_marked),
        past=[],
        future=[],
    )

    support = _frame_mask(width, height, detection)
    boundary = _boundary_band(support)
    core = cv2.erode(support, np.ones((3, 3), dtype=np.uint8)) > 0
    absolute_error = np.mean(
        np.abs(bright_output.astype(np.float32) - bright_clean.astype(np.float32)),
        axis=2,
    )
    signed_luma_error = cv2.cvtColor(bright_output, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(
        bright_clean,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    assert float(np.percentile(absolute_error[core], 95)) <= 1.0
    assert float(np.percentile(absolute_error[boundary], 95)) <= 2.0
    assert abs(float(np.mean(signed_luma_error[boundary]))) <= 0.5
    assert float(np.mean(_sobel_magnitude(bright_output)[boundary])) <= 3.0


def test_structured_roof_fallback_preserves_detail_and_temporal_stability() -> None:
    width, height, size = 320, 180, 40
    x, y = 255, 112
    detection = LogoDetection(x, y, size, size, _star_mask(size), 1.0)
    restorer = TemporalLogoRestorer(width, height, detection)
    clean_frames = _ambiguous_roof_frames(24, width, height, detection)
    marked_frames = [_watermarked(frame, detection) for frame in clean_frames]
    outputs = [
        restorer.restore(restorer.prepare_frame(index, frame), past=[], future=[])
        for index, frame in enumerate(marked_frames)
    ]

    support = _frame_mask(width, height, detection)
    core = cv2.erode(support, np.ones((5, 5), dtype=np.uint8)) > 0
    boundary = _boundary_band(support)
    outer_ring = cv2.dilate(support, np.ones((19, 19), dtype=np.uint8)) > 0
    inner_ring = cv2.dilate(support, np.ones((7, 7), dtype=np.uint8)) > 0
    surrounding_ring = outer_ring & ~inner_ring

    core_sobel_ratios: list[float] = []
    boundary_sobel_ratios: list[float] = []
    core_laplacian_ratios: list[float] = []
    for clean, marked, output in zip(clean_frames, marked_frames, outputs, strict=True):
        clean_sobel = _sobel_magnitude(clean)
        output_sobel = _sobel_magnitude(output)
        clean_laplacian = _laplacian_magnitude(clean)
        output_laplacian = _laplacian_magnitude(output)
        core_sobel_ratios.append(float(np.mean(output_sobel[core]) / np.mean(clean_sobel[core])))
        boundary_sobel_ratios.append(float(np.mean(output_sobel[boundary]) / np.mean(clean_sobel[boundary])))
        core_laplacian_ratios.append(float(np.mean(output_laplacian[core]) / np.mean(clean_laplacian[core])))

        before_error = np.mean(np.abs(marked.astype(np.float32) - clean.astype(np.float32)), axis=2)[core]
        after_error = np.mean(np.abs(output.astype(np.float32) - clean.astype(np.float32)), axis=2)[core]
        assert float(np.mean(after_error)) <= float(np.mean(before_error)) * 0.35

    assert 0.80 <= min(core_sobel_ratios) <= max(core_sobel_ratios) <= 1.25
    assert 0.75 <= min(boundary_sobel_ratios) <= max(boundary_sobel_ratios) <= 1.30
    assert 0.70 <= min(core_laplacian_ratios) <= max(core_laplacian_ratios) <= 1.40

    output_gray = [cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32) for output in outputs]
    preference_flip_excesses: list[float] = []
    for frame_index, (previous, current) in enumerate(zip(output_gray, output_gray[1:]), start=1):
        frame_jump = np.abs(current - previous)
        core_jump = float(np.mean(frame_jump[core]))
        ring_jump = float(np.mean(frame_jump[surrounding_ring]))
        jump_excess = core_jump - ring_jump
        assert jump_excess <= 2.75
        if frame_index % 2 == 0:
            preference_flip_excesses.append(jump_excess)

    assert float(np.mean(preference_flip_excesses)) <= 2.0
    assert float(np.percentile(preference_flip_excesses, 95)) <= 2.5
