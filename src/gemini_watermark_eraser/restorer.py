from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import cv2
import numpy as np

from .alpha_profiles import alpha_profile_candidates
from .dynamic_alpha import DynamicAlphaConfig, DynamicAlphaState, restore_dynamic_alpha
from .models import LogoDetection

TEMPORAL_OFFSETS = (4, 12, 24, 48, 72)
# If a clip offers no safe dynamic-alpha frame in roughly its first four
# seconds at the common 24 fps, keep the cheaper legacy path for the rest.
DYNAMIC_INITIAL_TRIAL_FRAMES = 96


@dataclass(slots=True)
class FramePatch:
    index: int
    original: np.ndarray
    original_gray: np.ndarray


@dataclass(slots=True)
class PreparedFrame:
    index: int
    frame: np.ndarray
    original: np.ndarray
    original_gray: np.ndarray


@dataclass(frozen=True, slots=True)
class _WarpedCandidate:
    pixels: np.ndarray
    valid: np.ndarray
    weight: float


class TemporalLogoRestorer:
    """Restore a small fixed logo using aligned neighbouring frame patches."""

    def __init__(self, frame_width: int, frame_height: int, detection: LogoDetection) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detection = detection

        x, y, width, height = detection.bbox
        margin_x = int(round(width * 1.25))
        margin_y = int(round(height * 1.25))
        self.roi_x0 = max(0, x - margin_x)
        self.roi_y0 = max(0, y - margin_y)
        self.roi_x1 = min(frame_width, x + width + margin_x)
        self.roi_y1 = min(frame_height, y + height + margin_y)

        roi_height = self.roi_y1 - self.roi_y0
        roi_width = self.roi_x1 - self.roi_x0
        self.soft_mask = np.zeros((roi_height, roi_width), dtype=np.float32)
        local_x = x - self.roi_x0
        local_y = y - self.roi_y0
        mask_height = min(height, roi_height - local_y)
        mask_width = min(width, roi_width - local_x)
        self.soft_mask[local_y : local_y + mask_height, local_x : local_x + mask_width] = detection.mask[
            :mask_height, :mask_width
        ]
        self.hard_mask = (self.soft_mask > 0.025).astype(np.uint8) * 255
        self._flow_mask = (self.hard_mask > 0).astype(np.uint8)
        # The detector already supplies a deliberately widened, anti-aliased
        # logo support. Keep that smooth profile for the sole final composite;
        # a binary hard edge creates the very outline this repair must avoid.
        feather_radius = 2
        fill_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (feather_radius * 2 + 1, feather_radius * 2 + 1),
        )
        self._fill_mask = cv2.dilate(self.hard_mask, fill_kernel)
        outside_distance = cv2.distanceTransform(
            (self.hard_mask == 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        feather_progress = np.clip(outside_distance / float(feather_radius), 0.0, 1.0)
        feather_progress = feather_progress * feather_progress * (3.0 - 2.0 * feather_progress)
        outside_feather = 1.0 - feather_progress
        self._composite_alpha = np.where(self.hard_mask > 0, 1.0, outside_feather).astype(np.float32)
        self._composite_alpha[self._fill_mask == 0] = 0.0
        inside_distance = cv2.distanceTransform(
            (self.hard_mask > 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        self._temporal_transition = np.clip(inside_distance / 3.0, 0.0, 1.0)
        self._detail_taper = np.clip((inside_distance - 1.0) / 2.0, 0.0, 1.0)
        self._fill_y, self._fill_x = np.nonzero(self._fill_mask)
        self._grid_y, self._grid_x = np.mgrid[:roi_height, :roi_width].astype(np.float32)
        self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.last_temporal_coverage = 0.0
        self._exemplar_locked_offset: tuple[int, int] | None = None
        self._exemplar_locked_structured: bool | None = None
        self._exemplar_locked_cohort: tuple[tuple[int, int], ...] = ()
        mask_y, mask_x = np.nonzero(self.hard_mask)
        exemplar_pad = max(5, int(round(min(width, height) * 0.12)))
        self._template_x0 = max(0, int(mask_x.min()) - exemplar_pad)
        self._template_y0 = max(0, int(mask_y.min()) - exemplar_pad)
        self._template_x1 = min(roi_width, int(mask_x.max()) + exemplar_pad + 1)
        self._template_y1 = min(roi_height, int(mask_y.max()) + exemplar_pad + 1)
        local_hard = self.hard_mask[
            self._template_y0 : self._template_y1,
            self._template_x0 : self._template_x1,
        ]
        local_fill = self._fill_mask[
            self._template_y0 : self._template_y1,
            self._template_x0 : self._template_x1,
        ]
        ring_width = max(7, int(round(min(width, height) * 0.20)))
        if ring_width % 2 == 0:
            ring_width += 1
        self._exemplar_local_hard = local_hard
        self._exemplar_local_fill = local_fill
        self._exemplar_ring = cv2.dilate(local_hard, np.ones((ring_width, ring_width), np.uint8)) - local_hard

        # Dynamic reverse-alpha recovery is deliberately optional.  It is
        # enabled only for a calibrated resolution and a confident detector;
        # every rejected frame remains byte-for-byte on the legacy temporal
        # restoration path.
        profiles = (
            alpha_profile_candidates(
                self.roi_shape,
                roi_x0=self.roi_x0,
                roi_y0=self.roi_y0,
                frame_width=frame_width,
                frame_height=frame_height,
                detection=detection,
            )
            if detection.confidence >= 0.35
            else ()
        )
        self._dynamic_profile = profiles[0] if profiles else None
        self._dynamic_state = DynamicAlphaState()
        self._dynamic_config = DynamicAlphaConfig(
            default_scale=0.92,
            minimum_scale=0.85,
            maximum_scale=1.10,
            maximum_scale_step=0.05,
            minimum_profile_value=0.02,
            maximum_reference_worsened_fraction=0.45,
            maximum_reference_tail_ratio=1.20,
        )
        self._dynamic_reference_confidence: np.ndarray | None = None
        self._dynamic_detail_weight: np.ndarray | None = None
        self._dynamic_bounds: tuple[int, int, int, int] | None = None
        self._dynamic_disabled = False
        if self._dynamic_profile is not None:
            profile_support = (self._dynamic_profile > 0.055).astype(np.uint8)
            distance = cv2.distanceTransform(profile_support, cv2.DIST_L2, 5)
            self._dynamic_reference_confidence = (
                (distance >= 2.2) & (self._dynamic_profile > 0.07)
            ).astype(np.float32)
            inner_weight = np.clip((distance - 1.8) / 2.6, 0.0, 1.0)
            opacity_weight = np.clip((self._dynamic_profile - 0.06) / 0.12, 0.0, 1.0)
            self._dynamic_detail_weight = (inner_weight * opacity_weight * 0.65).astype(np.float32)
            profile_y, profile_x = np.nonzero(self._dynamic_profile > 0.0)
            if len(profile_y) > 0:
                crop_pad = 6
                self._dynamic_bounds = (
                    max(0, int(profile_y.min()) - crop_pad),
                    min(roi_height, int(profile_y.max()) + crop_pad + 1),
                    max(0, int(profile_x.min()) - crop_pad),
                    min(roi_width, int(profile_x.max()) + crop_pad + 1),
                )

    @property
    def temporal_radius(self) -> int:
        return TEMPORAL_OFFSETS[-1]

    @property
    def roi_shape(self) -> tuple[int, int]:
        return self.soft_mask.shape

    def extract(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.roi_y0 : self.roi_y1, self.roi_x0 : self.roi_x1].copy()

    def prepare_frame(self, index: int, frame: np.ndarray) -> PreparedFrame:
        original = self.extract(frame)
        return PreparedFrame(index, frame, original, self._flow_gray(original, True))

    def make_patch(self, prepared: PreparedFrame) -> FramePatch:
        return FramePatch(prepared.index, prepared.original, prepared.original_gray)

    def _flow_gray(self, image: np.ndarray, source_has_logo: bool) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if source_has_logo:
            gray = cv2.inpaint(gray, self.hard_mask, 4, cv2.INPAINT_TELEA)
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8)

    def _warp(
        self,
        current_gray: np.ndarray,
        source: np.ndarray,
        source_gray: np.ndarray,
        *,
        frame_distance: int,
        source_has_logo: bool,
    ) -> _WarpedCandidate | None:
        flow = self._flow.calc(current_gray, source_gray, None)
        height, width = self.soft_mask.shape
        map_x = self._grid_x + flow[..., 0]
        map_y = self._grid_y + flow[..., 1]
        bounds = (map_x >= 0) & (map_x <= width - 1.001) & (map_y >= 0) & (map_y <= height - 1.001)
        warped = cv2.remap(source, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)

        if source_has_logo:
            source_logo = cv2.remap(
                self._flow_mask,
                map_x,
                map_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=1,
            )
            bounds &= source_logo == 0

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        comparison_ring = cv2.dilate(self.hard_mask, np.ones((15, 15), np.uint8)) > 0
        comparison_ring &= self.hard_mask == 0
        comparison_ring &= bounds
        if int(comparison_ring.sum()) < 32:
            return None
        error = float(np.median(np.abs(warped_gray.astype(np.float32) - current_gray)[comparison_ring]))
        if error > 54.0:
            return None
        distance_weight = np.exp(-abs(frame_distance) / 18.0)
        weight = float(distance_weight / (1.0 + (error / 18.0) ** 2))
        return _WarpedCandidate(warped.astype(np.float32), bounds, weight)

    def _fallback(self, current: np.ndarray, *, allow_exemplar: bool = True) -> np.ndarray:
        if allow_exemplar:
            exemplar = self._exemplar_fill(current)
            if exemplar is not None:
                return exemplar
        radius = max(3, int(round(min(self.detection.width, self.detection.height) * 0.09)))
        return cv2.inpaint(current, self._fill_mask, radius, cv2.INPAINT_TELEA)

    def _composite_repair(self, current: np.ndarray, replacement: np.ndarray) -> np.ndarray:
        """Composite source and clean replacement exactly once."""
        alpha = self._composite_alpha[..., None]
        return np.clip(
            current.astype(np.float32) * (1.0 - alpha) + replacement.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

    def _enhance_with_dynamic_alpha(self, current: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """Recover safe interior texture without exposing the logo boundary.

        A full reverse-alpha result is too sensitive to tiny profile or codec
        mismatches and can create a dark star-shaped hole.  Use it only as a
        source of bounded high-frequency detail, while the proven temporal /
        exemplar result keeps all low-frequency colour and boundary pixels.
        """
        profile = self._dynamic_profile
        confidence = self._dynamic_reference_confidence
        detail_weight = self._dynamic_detail_weight
        bounds = self._dynamic_bounds
        if (
            self._dynamic_disabled
            or profile is None
            or confidence is None
            or detail_weight is None
            or bounds is None
        ):
            return baseline

        y0, y1, x0, x1 = bounds
        current_crop = current[y0:y1, x0:x1]
        baseline_crop = baseline[y0:y1, x0:x1]
        profile_crop = profile[y0:y1, x0:x1]
        confidence_crop = confidence[y0:y1, x0:x1]
        detail_weight_crop = detail_weight[y0:y1, x0:x1]

        def reject(scale: float | None = None) -> np.ndarray:
            rejection_scale = (
                scale
                if scale is not None
                else self._dynamic_state.last_scale
                if self._dynamic_state.last_scale is not None
                else self._dynamic_config.default_scale
            )
            self._dynamic_state.record(float(rejection_scale), accepted=False)
            if (
                self._dynamic_state.accepted_frames == 0
                and self._dynamic_state.rejected_frames >= DYNAMIC_INITIAL_TRIAL_FRAMES
            ):
                self._dynamic_disabled = True
            return baseline

        trial_state = DynamicAlphaState(last_scale=self._dynamic_state.last_scale)
        try:
            result = restore_dynamic_alpha(
                current_crop,
                np.asarray((245.0, 250.0, 255.0), dtype=np.float32),
                profile_crop,
                clean_reference=baseline_crop,
                reference_confidence=confidence_crop,
                state=trial_state,
                config=self._dynamic_config,
            )
        except (ValueError, FloatingPointError, cv2.error):
            return reject()
        scale_was_clamped = not np.isclose(
            result.selection.raw_scale,
            result.selection.bounded_scale,
            rtol=0.0,
            atol=1e-6,
        )
        if scale_was_clamped:
            return reject(result.selection.scale)
        if not result.quality.accepted:
            return reject(result.selection.scale)

        candidate = result.candidate.astype(np.float32)
        baseline_float = baseline_crop.astype(np.float32)
        candidate_highpass = candidate - cv2.GaussianBlur(candidate, (0, 0), sigmaX=0.9)
        baseline_highpass = baseline_float - cv2.GaussianBlur(baseline_float, (0, 0), sigmaX=0.9)
        detail_delta = np.clip(candidate_highpass - baseline_highpass, -8.0, 8.0)
        quality_gain = float(np.clip(result.quality.score / 0.55, 0.0, 1.0))
        enhanced = baseline_float + detail_delta * (detail_weight_crop * quality_gain)[..., None]
        if not np.all(np.isfinite(enhanced)):
            return reject(result.selection.scale)
        enhanced_uint8 = np.rint(np.clip(enhanced, 0.0, 255.0)).astype(np.uint8)
        active = detail_weight_crop > 0.0
        delta = enhanced_uint8.astype(np.float32) - baseline_float
        if np.any(np.abs(np.mean(delta[active], axis=0)) > 0.35):
            return reject(result.selection.scale)
        newly_clipped = np.any(
            ((enhanced_uint8 <= 0) | (enhanced_uint8 >= 255))
            & ((baseline_crop > 0) & (baseline_crop < 255)),
            axis=2,
        )
        if float(np.mean(newly_clipped[active])) > 0.001:
            return reject(result.selection.scale)

        baseline_gray = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        enhanced_gray = cv2.cvtColor(enhanced_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        baseline_detail = baseline_gray - cv2.GaussianBlur(baseline_gray, (0, 0), sigmaX=0.9)
        enhanced_detail = enhanced_gray - cv2.GaussianBlur(enhanced_gray, (0, 0), sigmaX=0.9)
        baseline_energy = max(float(np.mean(np.abs(baseline_detail[active]))), 0.25)
        detail_ratio = float(np.mean(np.abs(enhanced_detail[active]))) / baseline_energy
        baseline_horizontal = cv2.Sobel(baseline_gray, cv2.CV_32F, 1, 0, ksize=3)
        baseline_vertical = cv2.Sobel(baseline_gray, cv2.CV_32F, 0, 1, ksize=3)
        enhanced_horizontal = cv2.Sobel(enhanced_gray, cv2.CV_32F, 1, 0, ksize=3)
        enhanced_vertical = cv2.Sobel(enhanced_gray, cv2.CV_32F, 0, 1, ksize=3)
        baseline_sobel = max(
            float(np.mean(cv2.magnitude(baseline_horizontal, baseline_vertical)[active])),
            0.5,
        )
        sobel_ratio = float(
            np.mean(cv2.magnitude(enhanced_horizontal, enhanced_vertical)[active])
        ) / baseline_sobel
        detail_improved = detail_ratio >= 1.002 and sobel_ratio >= 0.98
        if not detail_improved or detail_ratio > 1.25 or sobel_ratio > 1.25:
            return reject(result.selection.scale)

        self._dynamic_state.record(result.selection.scale, accepted=True)
        output = baseline.copy()
        output[y0:y1, x0:x1] = enhanced_uint8
        return output

    @staticmethod
    def _gradient_structure(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        """Return structure-tensor orientation coherence and dominant angle."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        selected = mask > 0
        if int(np.sum(selected)) < 16:
            return 0.0, 0.0
        xx = float(np.sum(horizontal[selected] ** 2))
        yy = float(np.sum(vertical[selected] ** 2))
        xy = float(np.sum(horizontal[selected] * vertical[selected]))
        energy = xx + yy
        if energy <= 1e-6:
            return 0.0, 0.0
        coherence = float(np.hypot(xx - yy, 2.0 * xy) / energy)
        angle = float(0.5 * np.arctan2(2.0 * xy, xx - yy))
        return coherence, angle

    @staticmethod
    def _match_ring_color(
        source: np.ndarray,
        target: np.ndarray,
        ring: np.ndarray,
    ) -> np.ndarray:
        """Match clean source low-frequency colour to the clean target ring."""
        selected = ring > 0
        source_values = source[selected].astype(np.float32)
        target_values = target[selected].astype(np.float32)
        if len(source_values) < 16:
            return source

        source_low, source_high = np.percentile(source_values, (20, 80), axis=0)
        target_low, target_high = np.percentile(target_values, (20, 80), axis=0)
        source_span = source_high - source_low
        target_span = target_high - target_low
        gain = np.ones(3, dtype=np.float32)
        stable = source_span >= 6.0
        gain[stable] = np.clip(target_span[stable] / source_span[stable], 0.85, 1.15)
        source_median = np.median(source_values, axis=0)
        target_median = np.median(target_values, axis=0)
        bias = np.clip(target_median - source_median * gain, -24.0, 24.0).astype(np.float32)
        return np.clip(source.astype(np.float32) * gain + bias, 0.0, 255.0)

    def _exemplar_fill(self, current: np.ndarray) -> np.ndarray | None:
        template = current[
            self._template_y0 : self._template_y1,
            self._template_x0 : self._template_x1,
        ]
        if template.shape[:2] != self._exemplar_local_hard.shape:
            return None
        scores = cv2.matchTemplate(
            current,
            template,
            cv2.TM_SQDIFF_NORMED,
            mask=self._exemplar_ring,
        )
        finite = np.nan_to_num(scores, nan=np.inf, posinf=np.inf, neginf=np.inf)
        overlap = cv2.matchTemplate(
            (self.hard_mask > 0).astype(np.float32),
            (self._exemplar_local_fill > 0).astype(np.float32),
            cv2.TM_CCORR,
        )
        finite[overlap > 0.5] = np.inf
        flat = finite.ravel()
        valid_indices = np.flatnonzero(np.isfinite(flat))
        candidate_count = min(160, len(valid_indices))
        if candidate_count == 0:
            return None
        selected_valid = np.argpartition(flat[valid_indices], candidate_count - 1)[:candidate_count]
        indices = valid_indices[selected_valid]

        ranked_candidates: list[tuple[float, float, int, int]] = []
        target_x, target_y = self._template_x0, self._template_y0
        fill_pixels = self._exemplar_local_fill > 0
        scale = max(1.0, float(self.detection.width))
        for index in indices:
            source_y, source_x = np.unravel_index(int(index), scores.shape)
            raw_score = float(finite[source_y, source_x])
            distance = np.hypot(source_x - target_x, source_y - target_y) / scale
            ranked_score = raw_score + 0.025 * distance
            ranked_candidates.append((ranked_score, raw_score, int(source_x), int(source_y)))

        if not ranked_candidates:
            return None
        ranked_candidates.sort(key=lambda item: item[0])
        best = ranked_candidates[0]
        best_score, best_raw_score, source_x, source_y = best
        if best_raw_score > 0.36:
            return None

        best_clean_patch = current[
            source_y : source_y + template.shape[0],
            source_x : source_x + template.shape[1],
        ]
        best_coherence, best_angle = self._gradient_structure(best_clean_patch, self._exemplar_local_fill)
        proposed_structured = best_coherence >= 0.55

        def candidate_at(offset: tuple[int, int]) -> tuple[float, float, int, int] | None:
            candidate_x = target_x + offset[0]
            candidate_y = target_y + offset[1]
            if not (0 <= candidate_x < scores.shape[1] and 0 <= candidate_y < scores.shape[0]):
                return None
            raw_score = float(finite[candidate_y, candidate_x])
            if not np.isfinite(raw_score) or raw_score > 0.36:
                return None
            distance = np.hypot(candidate_x - target_x, candidate_y - target_y) / scale
            ranked_score = raw_score + 0.025 * distance
            if ranked_score > 0.46:
                return None
            return ranked_score, raw_score, int(candidate_x), int(candidate_y)

        def candidate_structure(item: tuple[float, float, int, int]) -> tuple[float, float]:
            patch = current[
                item[3] : item[3] + template.shape[0],
                item[2] : item[2] + template.shape[1],
            ]
            return self._gradient_structure(patch, self._exemplar_local_fill)

        previous_offset = self._exemplar_locked_offset
        previous_structured = self._exemplar_locked_structured
        locked = candidate_at(previous_offset) if previous_offset is not None else None
        locked_coherence = 0.0
        locked_angle = 0.0
        locked_valid = locked is not None
        if locked is not None:
            locked_coherence, locked_angle = candidate_structure(locked)
            if previous_structured and locked_coherence < 0.55:
                locked_valid = False

        keep_locked_anchor = bool(
            locked_valid
            and locked is not None
            and best_score + 0.035 >= locked[0]
        )
        if keep_locked_anchor:
            anchor = locked
            structured_target = bool(previous_structured)
            anchor_angle = locked_angle
        else:
            anchor = best
            structured_target = proposed_structured
            anchor_angle = best_angle

        new_offset = (anchor[2] - target_x, anchor[3] - target_y)
        anchor_switched = previous_offset is not None and new_offset != previous_offset
        mode_switched = previous_structured is not None and structured_target != previous_structured
        if anchor_switched or mode_switched:
            self._exemplar_locked_cohort = ()
        self._exemplar_locked_offset = new_offset
        self._exemplar_locked_structured = structured_target

        anchor_score, _, source_x, source_y = anchor
        cluster_radius = max(3.0, min(float(self.detection.width) * 0.12, 8.0))
        selected = [anchor]
        if structured_target:
            normal_x = float(np.cos(anchor_angle))
            normal_y = float(np.sin(anchor_angle))

            def phase_valid(item: tuple[float, float, int, int]) -> bool:
                if item[0] > anchor_score + 0.045:
                    return False
                dx = item[2] - source_x
                dy = item[3] - source_y
                if np.hypot(dx, dy) > cluster_radius:
                    return False
                coherence, angle = candidate_structure(item)
                normal_offset = abs(dx * normal_x + dy * normal_y)
                angle_alignment = abs(float(np.cos(angle - anchor_angle)))
                return coherence >= 0.55 and normal_offset <= 0.45 and angle_alignment >= 0.985

            locked_cohort: list[tuple[float, float, int, int]] = []
            cohort_valid = len(self._exemplar_locked_cohort) >= 2
            if cohort_valid:
                for offset in self._exemplar_locked_cohort:
                    item = candidate_at(offset)
                    if item is None or not phase_valid(item):
                        cohort_valid = False
                        break
                    locked_cohort.append(item)
            if cohort_valid:
                selected = locked_cohort
            else:
                selected = []
                seen_positions: set[tuple[int, int]] = set()
                for item in (anchor, *ranked_candidates):
                    position = (item[2], item[3])
                    if position in seen_positions:
                        continue
                    seen_positions.add(position)
                    if phase_valid(item):
                        selected.append(item)
                    if len(selected) == 3:
                        break
                if len(selected) < 2:
                    self._exemplar_locked_cohort = ()
                    return None
                cohort_offsets = tuple((item[2] - target_x, item[3] - target_y) for item in selected)
                self._exemplar_locked_cohort = cohort_offsets
        else:
            self._exemplar_locked_cohort = ()
            selected.extend(
                item
                for item in ranked_candidates
                if (item[2], item[3]) != (source_x, source_y)
                and item[0] <= anchor_score + 0.045
                and np.hypot(item[2] - source_x, item[3] - source_y) <= cluster_radius
            )
            selected = selected[:5]
        if not selected:
            selected = [anchor]
        clean_color_ring = self._exemplar_ring > 0
        for _, _, x, y in selected:
            source_logo = self.hard_mask[
                y : y + template.shape[0],
                x : x + template.shape[1],
            ] > 0
            clean_color_ring &= ~source_logo
        patches = np.stack(
            [
                current[y : y + template.shape[0], x : x + template.shape[1]].astype(np.float32)
                for _, _, x, y in selected
            ]
        )
        source = np.median(patches, axis=0)
        # Preserve fine texture without copying the full content of one patch.
        # Only a bounded high-frequency residual from the best match is added
        # to the robust median, so leaves and roof lines stay crisp without
        # introducing a visible cloned object.
        best_patch = patches[0]
        best_smooth = cv2.GaussianBlur(best_patch, (0, 0), sigmaX=0.75)
        detail = np.clip(best_patch - best_smooth, -10.0, 10.0)
        detail_gain = 0.45 if structured_target else 0.25
        source = np.clip(source + detail * detail_gain, 0, 255)
        source = self._match_ring_color(
            source,
            template,
            clean_color_ring.astype(np.uint8),
        )
        result = current.copy()
        target = result[
            self._template_y0 : self._template_y1,
            self._template_x0 : self._template_x1,
        ]
        # This is a clean candidate patch. Copy it through the complete fill
        # support without mixing the logo-bearing current pixels back in; the
        # sole source-to-repair blend happens later in restore().
        target[fill_pixels] = np.clip(source[fill_pixels], 0, 255).astype(np.uint8)
        return result

    def _finish_repair(
        self,
        current: np.ndarray,
        fallback: np.ndarray,
        temporal: np.ndarray,
        temporal_confidence: np.ndarray,
    ) -> np.ndarray:
        """Blend clean temporal pixels over fallback, then composite once."""
        temporal_weight = np.clip(temporal_confidence, 0.0, 1.0)
        fallback_weight = 1.0 - temporal_weight
        replacement = (
            fallback.astype(np.float32) * fallback_weight[..., None]
            + temporal.astype(np.float32) * temporal_weight[..., None]
        )
        baseline = self._composite_repair(current, np.clip(replacement, 0, 255).astype(np.uint8))
        return self._enhance_with_dynamic_alpha(current, baseline)

    def _fuse(self, current: np.ndarray, candidates: list[_WarpedCandidate]) -> np.ndarray:
        if not candidates:
            self.last_temporal_coverage = 0.0
            fallback = self._fallback(current)
            return self._finish_repair(
                current,
                fallback,
                fallback,
                np.zeros(self.roi_shape, dtype=np.float32),
            )

        values = np.stack([candidate.pixels[self._fill_y, self._fill_x] for candidate in candidates], axis=0)
        valid = np.stack([candidate.valid[self._fill_y, self._fill_x] for candidate in candidates], axis=0)
        weights = np.asarray([candidate.weight for candidate in candidates], dtype=np.float32)[:, None]
        masked_values = np.where(valid[..., None], values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(masked_values, axis=0)
        deviation = np.max(np.abs(values - np.nan_to_num(median, nan=0.0)[None, ...]), axis=2)
        robust = valid & (deviation < 58.0)
        effective_weights = robust.astype(np.float32) * weights
        support = robust.sum(axis=0)
        weight_sum = effective_weights.sum(axis=0)
        accepted = (weight_sum > 0.09) & (support >= 2)
        core = self.hard_mask[self._fill_y, self._fill_x] > 0
        self.last_temporal_coverage = float(np.mean(accepted[core])) if np.any(core) else 0.0
        fused = np.sum(values * effective_weights[..., None], axis=0) / np.maximum(
            weight_sum[..., None],
            1e-6,
        )
        # The robust average prevents flicker and cloned-object artifacts. Add
        # back only a small, bounded high-frequency residual from the strongest
        # real candidate to counteract alignment softness.
        color_error = np.mean(np.abs(values - np.nan_to_num(median, nan=0.0)[None, ...]), axis=2)
        selection_score = color_error / np.maximum(weights, 0.04)
        selection_score[~robust] = np.inf
        best_index = np.argmin(selection_score, axis=0)
        pixel_index = np.arange(values.shape[1])
        candidate_details = np.stack(
            [
                candidate.pixels
                - cv2.GaussianBlur(candidate.pixels, (0, 0), sigmaX=0.75)
                for candidate in candidates
            ],
            axis=0,
        )[:, self._fill_y, self._fill_x]
        detail = np.clip(candidate_details[best_index, pixel_index], -10.0, 10.0)
        detail_strength = np.clip((support.astype(np.float32) - 1.0) / 3.0, 0.0, 1.0)[..., None]
        detail_taper = self._detail_taper[self._fill_y, self._fill_x][..., None]
        fused = np.clip(fused + detail * detail_strength * detail_taper * 0.55, 0, 255)
        fallback = self._fallback(current, allow_exemplar=self.last_temporal_coverage < 0.28)
        # Confidence changes smoothly across temporal support boundaries. The
        # clean temporal estimate is blended only with another clean fill, so
        # low confidence can never reveal pixels from the marked current frame.
        confidence = np.clip(weight_sum / 0.28, 0.0, 1.0)
        confidence[~accepted] = 0.0
        raw_confidence_map = np.zeros(self.roi_shape, dtype=np.float32)
        raw_confidence_map[self._fill_y, self._fill_x] = confidence
        confidence_map = cv2.GaussianBlur(raw_confidence_map, (0, 0), sigmaX=0.85)
        accepted_map = np.zeros(self.roi_shape, dtype=np.uint8)
        accepted_map[self._fill_y[accepted], self._fill_x[accepted]] = 255
        accepted_interior = cv2.erode(accepted_map, np.ones((3, 3), np.uint8)) > 0
        confidence_map = np.maximum(confidence_map, raw_confidence_map * accepted_interior)
        confidence_map *= self._temporal_transition
        confidence_map[self._fill_mask == 0] = 0.0
        confidence_map = np.clip(confidence_map, 0.0, 1.0)

        temporal = fallback.astype(np.float32)
        temporal[self._fill_y[accepted], self._fill_x[accepted]] = fused[accepted]
        return self._finish_repair(current, fallback, temporal, confidence_map)

    def restore(
        self,
        prepared: PreparedFrame,
        *,
        past: Iterable[FramePatch],
        future: Iterable[PreparedFrame],
    ) -> np.ndarray:
        current = prepared.original
        current_gray = prepared.original_gray
        current_index = prepared.index
        past_list = list(past)
        candidates: list[_WarpedCandidate] = []

        wanted = set(TEMPORAL_OFFSETS)
        for patch in reversed(past_list):
            distance = current_index - patch.index
            if distance in wanted:
                warped = self._warp(
                    current_gray,
                    patch.original,
                    patch.original_gray,
                    frame_distance=distance,
                    source_has_logo=True,
                )
                if warped is not None:
                    candidates.append(warped)

        for future_frame in future:
            distance = future_frame.index - current_index
            if distance in wanted:
                warped = self._warp(
                    current_gray,
                    future_frame.original,
                    future_frame.original_gray,
                    frame_distance=distance,
                    source_has_logo=True,
                )
                if warped is not None:
                    candidates.append(warped)

        restored = self._fuse(current, candidates)
        output = prepared.frame.copy()
        output[self.roi_y0 : self.roi_y1, self.roi_x0 : self.roi_x1] = restored
        return output
