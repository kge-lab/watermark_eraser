from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import cv2
import numpy as np

from .models import LogoDetection

TEMPORAL_OFFSETS = (4, 12, 24, 48, 72)


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
        self._mask_y, self._mask_x = np.nonzero(self.hard_mask)
        self._grid_y, self._grid_x = np.mgrid[:roi_height, :roi_width].astype(np.float32)
        self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.last_temporal_coverage = 0.0
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
        ring_width = max(7, int(round(min(width, height) * 0.20)))
        if ring_width % 2 == 0:
            ring_width += 1
        self._exemplar_local_hard = local_hard
        self._exemplar_ring = cv2.dilate(local_hard, np.ones((ring_width, ring_width), np.uint8)) - local_hard

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
        first = cv2.inpaint(current, self.hard_mask, radius, cv2.INPAINT_TELEA)
        # A second pass at half scale reduces directional streaks on detailed textures.
        half_size = (max(1, current.shape[1] // 2), max(1, current.shape[0] // 2))
        small = cv2.resize(first, half_size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(self.hard_mask, half_size, interpolation=cv2.INTER_NEAREST)
        small = cv2.inpaint(small, small_mask, max(2, radius // 2), cv2.INPAINT_TELEA)
        smooth = cv2.resize(small, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_CUBIC)
        inner = cv2.erode(self.hard_mask, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0
        inner = cv2.GaussianBlur(inner, (0, 0), sigmaX=1.2)[..., None]
        return np.clip(first * (1.0 - inner) + smooth * inner, 0, 255).astype(np.uint8)

    def _restore_visible_detail(self, current: np.ndarray, restored: np.ndarray) -> np.ndarray:
        """Keep fine source texture that remains visible beneath the translucent mark."""
        interior = cv2.erode(self.hard_mask, np.ones((5, 5), np.uint8))
        if not np.any(interior):
            return restored
        source = current.astype(np.float32)
        low_frequency = cv2.GaussianBlur(source, (0, 0), sigmaX=1.15)
        detail = np.clip(source - low_frequency, -24.0, 24.0)
        repaired_low = cv2.GaussianBlur(restored.astype(np.float32), (0, 0), sigmaX=0.85)
        repaired_detail = np.clip(restored.astype(np.float32) - repaired_low, -18.0, 18.0)
        alpha = cv2.GaussianBlur(interior.astype(np.float32) / 255.0, (0, 0), sigmaX=1.0)
        alpha = (alpha * 0.72)[..., None]
        return np.clip(restored.astype(np.float32) + detail * alpha + repaired_detail * alpha * 0.70, 0, 255).astype(
            np.uint8
        )

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
        flat = finite.ravel()
        candidate_count = min(160, flat.size)
        if candidate_count == 0:
            return None
        indices = np.argpartition(flat, candidate_count - 1)[:candidate_count]

        ranked_candidates: list[tuple[float, float, int, int]] = []
        target_x, target_y = self._template_x0, self._template_y0
        logo_pixels = self._exemplar_local_hard > 0
        scale = max(1.0, float(self.detection.width))
        for index in indices:
            source_y, source_x = np.unravel_index(int(index), scores.shape)
            source_mask = self.hard_mask[
                source_y : source_y + template.shape[0],
                source_x : source_x + template.shape[1],
            ]
            if source_mask.shape != self._exemplar_local_hard.shape or np.any(source_mask[logo_pixels]):
                continue
            raw_score = float(finite[source_y, source_x])
            distance = np.hypot(source_x - target_x, source_y - target_y) / scale
            ranked_score = raw_score + 0.025 * distance
            ranked_candidates.append((ranked_score, raw_score, int(source_x), int(source_y)))

        if not ranked_candidates:
            return None
        ranked_candidates.sort(key=lambda item: item[0])
        best_score, _, source_x, source_y = ranked_candidates[0]
        if best_score > 0.36:
            return None
        cluster_radius = max(3.0, min(float(self.detection.width) * 0.12, 8.0))
        selected = [
            item
            for item in ranked_candidates
            if item[0] <= best_score + 0.045
            and np.hypot(item[2] - source_x, item[3] - source_y) <= cluster_radius
        ][:5]
        if not selected:
            selected = [ranked_candidates[0]]
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
        source = np.clip(source + detail * 0.45, 0, 255)
        result = current.copy()
        target = result[
            self._template_y0 : self._template_y1,
            self._template_x0 : self._template_x1,
        ]
        alpha = cv2.GaussianBlur(
            logo_pixels.astype(np.float32),
            (0, 0),
            sigmaX=max(0.8, self.detection.width * 0.018),
        )[..., None]
        target[:] = np.clip(target.astype(np.float32) * (1.0 - alpha) + source.astype(np.float32) * alpha, 0, 255)
        return result

    def _fuse(self, current: np.ndarray, candidates: list[_WarpedCandidate]) -> np.ndarray:
        if not candidates:
            self.last_temporal_coverage = 0.0
            return self._restore_visible_detail(current, self._fallback(current))

        values = np.stack([candidate.pixels[self._mask_y, self._mask_x] for candidate in candidates], axis=0)
        valid = np.stack([candidate.valid[self._mask_y, self._mask_x] for candidate in candidates], axis=0)
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
        self.last_temporal_coverage = float(np.mean(accepted))
        fallback = self._fallback(current, allow_exemplar=self.last_temporal_coverage < 0.28)
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
        )[:, self._mask_y, self._mask_x]
        detail = np.clip(candidate_details[best_index, pixel_index], -10.0, 10.0)
        detail_strength = np.clip((support.astype(np.float32) - 1.0) / 3.0, 0.0, 1.0)[..., None]
        fused = np.clip(fused + detail * detail_strength * 0.70, 0, 255)
        result = fallback.astype(np.float32)
        result[self._mask_y[accepted], self._mask_x[accepted]] = fused[accepted]
        return self._restore_visible_detail(current, np.clip(result, 0, 255).astype(np.uint8))

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
        alpha = self.soft_mask[..., None]
        blended = np.clip(current.astype(np.float32) * (1.0 - alpha) + restored.astype(np.float32) * alpha, 0, 255)
        output = prepared.frame.copy()
        output[self.roi_y0 : self.roi_y1, self.roi_x0 : self.roi_x1] = blended.astype(np.uint8)
        return output
