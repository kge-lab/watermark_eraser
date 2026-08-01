from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class DynamicAlphaConfig:
    """Numerical and quality limits for conservative alpha deblending.

    ``profile * scale`` is interpreted as the physical watermark opacity.  A
    detector mask can describe the profile's shape, but its values are not
    assumed to be measured opacity values.
    """

    value_range: tuple[float, float] = (0.0, 255.0)
    default_scale: float = 0.50
    minimum_scale: float = 0.0
    maximum_scale: float = 1.25
    maximum_alpha: float = 0.92
    denominator_floor: float = 0.08
    maximum_scale_step: float = 0.08
    minimum_profile_value: float = 1e-4
    minimum_regression_samples: int = 12
    regression_signal_fraction: float = 0.002
    robust_iterations: int = 8
    huber_delta: float = 1.5
    clipping_penalty_weight: float = 2.0
    noise_penalty_weight: float = 0.05
    maximum_clipping_fraction: float = 0.025
    maximum_alpha_clipping_fraction: float = 0.01
    maximum_noise_gain_p95: float = 3.5
    maximum_total_penalty: float = 0.35
    minimum_reference_improvement: float = 0.02
    maximum_reference_error_ratio: float = 0.98
    maximum_reference_tail_ratio: float = 1.10
    maximum_reference_worsened_fraction: float = 0.35
    reference_tail_quantile: float = 0.90
    reference_worsening_tolerance: float = 0.5

    def __post_init__(self) -> None:
        low, high = self.value_range
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("value_range must contain two finite increasing values")
        if not 0.0 <= self.minimum_scale <= self.maximum_scale:
            raise ValueError("scale limits must be ordered and non-negative")
        if not self.minimum_scale <= self.default_scale <= self.maximum_scale:
            raise ValueError("default_scale must be within the scale limits")
        if not 0.0 < self.denominator_floor < 1.0:
            raise ValueError("denominator_floor must be between zero and one")
        if not 0.0 < self.maximum_alpha <= 1.0 - self.denominator_floor + 1e-12:
            raise ValueError("maximum_alpha must leave at least denominator_floor")
        if self.maximum_scale_step < 0.0 or not np.isfinite(self.maximum_scale_step):
            raise ValueError("maximum_scale_step must be finite and non-negative")
        if not 0.0 <= self.minimum_profile_value < 1.0:
            raise ValueError("minimum_profile_value must be in [0, 1)")
        if self.minimum_regression_samples < 1:
            raise ValueError("minimum_regression_samples must be positive")
        if self.regression_signal_fraction < 0.0:
            raise ValueError("regression_signal_fraction must be non-negative")
        if self.robust_iterations < 1 or self.huber_delta <= 0.0:
            raise ValueError("robust regression settings must be positive")
        if self.clipping_penalty_weight < 0.0 or self.noise_penalty_weight < 0.0:
            raise ValueError("penalty weights must be non-negative")
        for name, value in (
            ("maximum_clipping_fraction", self.maximum_clipping_fraction),
            ("maximum_alpha_clipping_fraction", self.maximum_alpha_clipping_fraction),
            ("maximum_reference_worsened_fraction", self.maximum_reference_worsened_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.maximum_noise_gain_p95 < 1.0 or self.maximum_total_penalty < 0.0:
            raise ValueError("noise and total-penalty limits are invalid")
        if self.minimum_reference_improvement < 0.0:
            raise ValueError("minimum_reference_improvement must be non-negative")
        if self.maximum_reference_error_ratio < 0.0 or self.maximum_reference_tail_ratio < 0.0:
            raise ValueError("reference error ratios must be non-negative")
        if not 0.0 < self.reference_tail_quantile < 1.0:
            raise ValueError("reference_tail_quantile must be between zero and one")
        if self.reference_worsening_tolerance < 0.0:
            raise ValueError("reference_worsening_tolerance must be non-negative")


@dataclass(slots=True)
class DynamicAlphaState:
    """State carried between frames for bounded opacity changes.

    Only accepted decisions advance ``last_scale``.  A rejected or corrupted
    frame therefore cannot move the temporal anchor used by later frames.
    """

    last_scale: float | None = None
    accepted_frames: int = 0
    rejected_frames: int = 0

    def record(self, scale: float, *, accepted: bool) -> None:
        if not np.isfinite(scale):
            raise ValueError("scale must be finite")
        if accepted:
            self.last_scale = float(scale)
            self.accepted_frames += 1
        else:
            self.rejected_frames += 1

    def reset(self) -> None:
        self.last_scale = None
        self.accepted_frames = 0
        self.rejected_frames = 0


@dataclass(frozen=True, slots=True)
class ScaleSelection:
    """A robust scale estimate and the limits applied to it."""

    raw_scale: float
    bounded_scale: float
    scale: float
    used_clean_reference: bool
    rate_limited: bool
    reference_samples: int


@dataclass(frozen=True, slots=True)
class DeblendResult:
    """A reverse-alpha candidate and its intrinsic numerical risk metrics."""

    candidate: np.ndarray
    alpha: np.ndarray
    clipping_fraction: float
    alpha_clipping_fraction: float
    noise_gain_p95: float
    clipping_penalty: float
    noise_penalty: float
    total_penalty: float


@dataclass(frozen=True, slots=True)
class QualityDecision:
    """Whether a deblend is safe enough for a caller to use as a candidate."""

    accepted: bool
    reasons: tuple[str, ...]
    score: float
    reference_error_before: float | None
    reference_error_after: float | None
    reference_improvement: float | None
    reference_tail_ratio: float | None
    reference_worsened_fraction: float | None
    clipping_fraction: float
    alpha_clipping_fraction: float
    noise_gain_p95: float
    clipping_penalty: float
    noise_penalty: float
    total_penalty: float

    @property
    def reason(self) -> str:
        return "accepted" if self.accepted else ", ".join(self.reasons)


@dataclass(frozen=True, slots=True)
class DynamicAlphaResult:
    """Complete dynamic-alpha result.

    ``candidate`` is always the deblended clean-image proposal, even when the
    quality decision rejects it.  Callers should keep their existing temporal
    or inpaint fallback unless ``quality.accepted`` is true.
    """

    candidate: np.ndarray
    alpha: np.ndarray
    selection: ScaleSelection
    quality: QualityDecision


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(image)
    if result.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape (H, W) or (H, W, C)")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if result.ndim == 3 and result.shape[2] == 0:
        raise ValueError(f"{name} must have at least one channel")
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    converted = result.astype(np.float32, copy=False)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _spatial_map(value: np.ndarray, image: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    spatial_shape = image.shape[:2]
    if result.shape == spatial_shape:
        pass
    elif result.shape == spatial_shape + (1,):
        result = result[..., 0]
    else:
        raise ValueError(f"{name} must have shape {spatial_shape} or {spatial_shape + (1,)}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _coerce_profile(profile: np.ndarray, image: np.ndarray, config: DynamicAlphaConfig) -> np.ndarray:
    result = _spatial_map(profile, image, "profile")
    if np.any(result < 0.0) or np.any(result > 1.0 + 1e-6):
        raise ValueError("profile values must be in [0, 1]")
    result = np.clip(result, 0.0, 1.0).astype(np.float32, copy=True)
    result[result <= config.minimum_profile_value] = 0.0
    return result


def _broadcast_watermark(watermark: np.ndarray | float, image: np.ndarray) -> np.ndarray:
    result = np.asarray(watermark, dtype=np.float32)
    if image.ndim == 3 and result.shape == image.shape[:2]:
        result = result[..., None]
    try:
        result = np.broadcast_to(result, image.shape)
    except ValueError as error:
        raise ValueError("watermark is not broadcastable to the observed image") from error
    if not np.all(np.isfinite(result)):
        raise ValueError("watermark must contain only finite values")
    return result


def _broadcast_spatial(spatial: np.ndarray, image: np.ndarray) -> np.ndarray:
    return spatial if image.ndim == 2 else spatial[..., None]


def _reference_confidence(
    confidence: np.ndarray | None,
    image: np.ndarray,
) -> np.ndarray:
    if confidence is None:
        return np.ones(image.shape[:2], dtype=np.float32)
    result = _spatial_map(confidence, image, "reference_confidence")
    if np.any(result < 0.0) or np.any(result > 1.0 + 1e-6):
        raise ValueError("reference_confidence values must be in [0, 1]")
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=True)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    flat_values = np.asarray(values, dtype=np.float64).ravel()
    flat_weights = np.asarray(weights, dtype=np.float64).ravel()
    valid = np.isfinite(flat_values) & np.isfinite(flat_weights) & (flat_weights > 0.0)
    if not np.any(valid):
        return 0.0
    flat_values = flat_values[valid]
    flat_weights = flat_weights[valid]
    order = np.argsort(flat_values)
    flat_values = flat_values[order]
    flat_weights = flat_weights[order]
    cumulative = np.cumsum(flat_weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(flat_values) - 1)
    return float(flat_values[index])


def _robust_scale(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    config: DynamicAlphaConfig,
) -> float:
    ratios = y / x
    estimate = _weighted_quantile(ratios, weights * np.abs(x), 0.5)
    data_span = config.value_range[1] - config.value_range[0]
    residual_floor = data_span * 1e-6
    for _ in range(config.robust_iterations):
        residual = y - estimate * x
        centre = _weighted_quantile(residual, weights, 0.5)
        mad = _weighted_quantile(np.abs(residual - centre), weights, 0.5)
        sigma = max(1.4826 * mad, residual_floor)
        cutoff = config.huber_delta * sigma
        robust_weight = np.ones_like(residual, dtype=np.float64)
        large = np.abs(residual - centre) > cutoff
        robust_weight[large] = cutoff / np.abs(residual[large] - centre)
        combined_weight = weights * robust_weight
        denominator = float(np.sum(combined_weight * x * x))
        if denominator <= 1e-18:
            break
        updated = float(np.sum(combined_weight * x * y) / denominator)
        if not np.isfinite(updated):
            break
        if abs(updated - estimate) <= 1e-7 * max(1.0, abs(estimate)):
            estimate = updated
            break
        estimate = updated
    return float(estimate)


def select_alpha_scale(
    observed: np.ndarray,
    watermark: np.ndarray | float,
    profile: np.ndarray,
    *,
    clean_reference: np.ndarray | None = None,
    reference_confidence: np.ndarray | None = None,
    proposed_scale: float | None = None,
    state: DynamicAlphaState | None = None,
    config: DynamicAlphaConfig | None = None,
) -> ScaleSelection:
    """Select a robust scalar opacity and apply temporal rate limiting.

    With a clean reference this solves ``observed - reference =
    scale * profile * (watermark - reference)`` using Huber IRLS.  Confidence
    values are exact fitting weights; zero means the reference pixel is not
    trusted.  Without enough reference evidence, ``proposed_scale``, the last
    accepted state scale, or ``config.default_scale`` is used in that order.
    """

    settings = config or DynamicAlphaConfig()
    observed_array = _validate_image(observed, "observed")
    observed_float = observed_array.astype(np.float32, copy=False)
    watermark_float = _broadcast_watermark(watermark, observed_array)
    profile_float = _coerce_profile(profile, observed_array, settings)
    if reference_confidence is not None and clean_reference is None:
        raise ValueError("reference_confidence requires clean_reference")
    confidence = _reference_confidence(reference_confidence, observed_array)

    if proposed_scale is not None and not np.isfinite(proposed_scale):
        raise ValueError("proposed_scale must be finite")
    if state is not None and state.last_scale is not None and not np.isfinite(state.last_scale):
        raise ValueError("state.last_scale must be finite")
    fallback = (
        float(proposed_scale)
        if proposed_scale is not None
        else float(state.last_scale)
        if state is not None and state.last_scale is not None
        else settings.default_scale
    )

    raw_scale = fallback
    used_reference = False
    reference_samples = 0
    if clean_reference is not None:
        reference_array = _validate_image(clean_reference, "clean_reference")
        if reference_array.shape != observed_array.shape:
            raise ValueError("clean_reference must have the same shape as observed")
        reference_float = reference_array.astype(np.float32, copy=False)
        profile_channels = _broadcast_spatial(profile_float, observed_array)
        confidence_channels = _broadcast_spatial(confidence, observed_array)
        if observed_array.ndim == 3:
            profile_channels = np.broadcast_to(profile_channels, observed_array.shape)
            confidence_channels = np.broadcast_to(confidence_channels, observed_array.shape)
        x = profile_channels * (watermark_float - reference_float)
        y = observed_float - reference_float
        signal_floor = (
            (settings.value_range[1] - settings.value_range[0])
            * settings.regression_signal_fraction
        )
        valid = (
            (profile_channels > settings.minimum_profile_value)
            & (confidence_channels > 0.0)
            & (np.abs(x) > signal_floor)
        )
        reference_samples = int(np.count_nonzero(valid))
        if reference_samples >= settings.minimum_regression_samples:
            raw_scale = _robust_scale(
                x[valid].astype(np.float64),
                y[valid].astype(np.float64),
                confidence_channels[valid].astype(np.float64),
                settings,
            )
            used_reference = np.isfinite(raw_scale)
            if not used_reference:
                raw_scale = fallback

    bounded_scale = float(np.clip(raw_scale, settings.minimum_scale, settings.maximum_scale))
    selected_scale = bounded_scale
    rate_limited = False
    if state is not None and state.last_scale is not None:
        previous = float(state.last_scale)
        lower = max(settings.minimum_scale, previous - settings.maximum_scale_step)
        upper = min(settings.maximum_scale, previous + settings.maximum_scale_step)
        selected_scale = float(np.clip(bounded_scale, lower, upper))
        rate_limited = not np.isclose(selected_scale, bounded_scale, rtol=0.0, atol=1e-12)
    return ScaleSelection(
        raw_scale=float(raw_scale),
        bounded_scale=bounded_scale,
        scale=selected_scale,
        used_clean_reference=used_reference,
        rate_limited=rate_limited,
        reference_samples=reference_samples,
    )


def _cast_like(array: np.ndarray, template: np.ndarray) -> np.ndarray:
    if np.issubdtype(template.dtype, np.integer):
        return np.rint(array).astype(template.dtype)
    return array.astype(template.dtype)


def reverse_alpha_deblend(
    observed: np.ndarray,
    watermark: np.ndarray | float,
    alpha: np.ndarray,
    *,
    config: DynamicAlphaConfig | None = None,
) -> DeblendResult:
    """Reverse ``observed = clean * (1-alpha) + watermark * alpha`` safely."""

    settings = config or DynamicAlphaConfig()
    observed_array = _validate_image(observed, "observed")
    observed_float = observed_array.astype(np.float32, copy=False)
    watermark_float = _broadcast_watermark(watermark, observed_array)
    requested_alpha = _spatial_map(alpha, observed_array, "alpha")
    if np.any(requested_alpha < 0.0):
        raise ValueError("alpha must be non-negative")
    active_requested = requested_alpha > settings.minimum_profile_value
    effective_alpha = np.clip(requested_alpha, 0.0, settings.maximum_alpha).astype(np.float32)
    effective_alpha[~active_requested] = 0.0
    if np.any(active_requested):
        alpha_clipping_fraction = float(
            np.mean(requested_alpha[active_requested] > settings.maximum_alpha + 1e-7)
        )
    else:
        alpha_clipping_fraction = 0.0

    alpha_channels = _broadcast_spatial(effective_alpha, observed_array)
    denominator = np.maximum(1.0 - alpha_channels, settings.denominator_floor)
    raw_candidate = (observed_float - alpha_channels * watermark_float) / denominator
    low, high = settings.value_range
    if observed_array.ndim == 3:
        clipped_pixel = np.any((raw_candidate < low) | (raw_candidate > high), axis=2)
    else:
        clipped_pixel = (raw_candidate < low) | (raw_candidate > high)
    alpha_weight = effective_alpha.astype(np.float64)
    weight_sum = float(np.sum(alpha_weight))
    clipping_fraction = (
        float(np.sum(alpha_weight * clipped_pixel) / weight_sum) if weight_sum > 0.0 else 0.0
    )

    gain = 1.0 / np.maximum(1.0 - effective_alpha, settings.denominator_floor)
    noise_gain_p95 = _weighted_quantile(gain, alpha_weight, 0.95) if weight_sum > 0.0 else 1.0
    mean_squared_gain = (
        float(np.sum(alpha_weight * (gain - 1.0) ** 2) / weight_sum)
        if weight_sum > 0.0
        else 0.0
    )
    clipping_penalty = settings.clipping_penalty_weight * clipping_fraction
    noise_penalty = settings.noise_penalty_weight * mean_squared_gain
    total_penalty = clipping_penalty + noise_penalty
    clipped = np.clip(raw_candidate, low, high)
    candidate = _cast_like(clipped, observed_array)
    return DeblendResult(
        candidate=candidate,
        alpha=effective_alpha.copy(),
        clipping_fraction=clipping_fraction,
        alpha_clipping_fraction=alpha_clipping_fraction,
        noise_gain_p95=noise_gain_p95,
        clipping_penalty=clipping_penalty,
        noise_penalty=noise_penalty,
        total_penalty=total_penalty,
    )


def _weighted_winsorised_mean(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    ceiling = _weighted_quantile(values, weights, quantile)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return 0.0
    return float(np.sum(weights * np.minimum(values, ceiling)) / weight_sum)


def assess_deblend_quality(
    observed: np.ndarray,
    deblend: DeblendResult,
    *,
    clean_reference: np.ndarray | None = None,
    reference_confidence: np.ndarray | None = None,
    reference_was_used: bool = False,
    config: DynamicAlphaConfig | None = None,
) -> QualityDecision:
    """Combine numerical safety and optional reference agreement into a decision."""

    settings = config or DynamicAlphaConfig()
    observed_array = _validate_image(observed, "observed")
    if deblend.candidate.shape != observed_array.shape:
        raise ValueError("deblend candidate must have the same shape as observed")
    alpha = _spatial_map(deblend.alpha, observed_array, "deblend.alpha")
    if reference_confidence is not None and clean_reference is None:
        raise ValueError("reference_confidence requires clean_reference")

    reasons: list[str] = []
    if not np.any(alpha > settings.minimum_profile_value):
        reasons.append("empty_profile")
    if deblend.clipping_fraction > settings.maximum_clipping_fraction:
        reasons.append("excessive_clipping")
    if deblend.alpha_clipping_fraction > settings.maximum_alpha_clipping_fraction:
        reasons.append("alpha_saturated")
    if deblend.noise_gain_p95 > settings.maximum_noise_gain_p95:
        reasons.append("excessive_noise_gain")
    if deblend.total_penalty > settings.maximum_total_penalty:
        reasons.append("excessive_penalty")

    error_before: float | None = None
    error_after: float | None = None
    improvement: float | None = None
    tail_ratio: float | None = None
    worsened_fraction: float | None = None
    reference_factor = 1.0
    if clean_reference is not None:
        reference_array = _validate_image(clean_reference, "clean_reference")
        if reference_array.shape != observed_array.shape:
            raise ValueError("clean_reference must have the same shape as observed")
        confidence = _reference_confidence(reference_confidence, observed_array)
        quality_weights = confidence.astype(np.float64) * alpha.astype(np.float64)
        if not reference_was_used or float(np.sum(quality_weights)) <= 0.0:
            reasons.append("insufficient_reference")
        else:
            observed_float = observed_array.astype(np.float32, copy=False)
            candidate_float = deblend.candidate.astype(np.float32, copy=False)
            reference_float = reference_array.astype(np.float32, copy=False)
            if observed_array.ndim == 3:
                before_map = np.mean(np.abs(observed_float - reference_float), axis=2)
                after_map = np.mean(np.abs(candidate_float - reference_float), axis=2)
            else:
                before_map = np.abs(observed_float - reference_float)
                after_map = np.abs(candidate_float - reference_float)
            data_span = settings.value_range[1] - settings.value_range[0]
            error_before = _weighted_winsorised_mean(
                before_map / data_span,
                quality_weights,
                settings.reference_tail_quantile,
            )
            error_after = _weighted_winsorised_mean(
                after_map / data_span,
                quality_weights,
                settings.reference_tail_quantile,
            )
            epsilon = 1e-12
            if error_before <= epsilon:
                improvement = 0.0 if error_after <= epsilon else -float("inf")
                error_ratio = 0.0 if error_after <= epsilon else float("inf")
            else:
                improvement = (error_before - error_after) / error_before
                error_ratio = error_after / error_before

            tail_before = _weighted_quantile(
                before_map,
                quality_weights,
                settings.reference_tail_quantile,
            )
            tail_after = _weighted_quantile(
                after_map,
                quality_weights,
                settings.reference_tail_quantile,
            )
            if tail_before <= 1e-12:
                tail_ratio = 0.0 if tail_after <= 1e-12 else float("inf")
            else:
                tail_ratio = tail_after / tail_before
            worsened = after_map > before_map + settings.reference_worsening_tolerance
            worsened_fraction = float(
                np.sum(quality_weights * worsened) / np.sum(quality_weights)
            )

            if improvement < settings.minimum_reference_improvement:
                reasons.append("no_reference_improvement")
            if error_ratio > settings.maximum_reference_error_ratio:
                reasons.append("reference_error_regressed")
            if tail_ratio > settings.maximum_reference_tail_ratio:
                reasons.append("reference_tail_regressed")
            if worsened_fraction > settings.maximum_reference_worsened_fraction:
                reasons.append("too_many_worsened_pixels")
            reference_factor = float(np.clip(improvement, 0.0, 1.0))

    safety_score = float(np.clip(1.0 - deblend.total_penalty, 0.0, 1.0))
    score = safety_score if clean_reference is None else safety_score * reference_factor
    return QualityDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        score=score,
        reference_error_before=error_before,
        reference_error_after=error_after,
        reference_improvement=improvement,
        reference_tail_ratio=tail_ratio,
        reference_worsened_fraction=worsened_fraction,
        clipping_fraction=deblend.clipping_fraction,
        alpha_clipping_fraction=deblend.alpha_clipping_fraction,
        noise_gain_p95=deblend.noise_gain_p95,
        clipping_penalty=deblend.clipping_penalty,
        noise_penalty=deblend.noise_penalty,
        total_penalty=deblend.total_penalty,
    )


def restore_dynamic_alpha(
    observed: np.ndarray,
    watermark: np.ndarray | float,
    profile: np.ndarray,
    *,
    clean_reference: np.ndarray | None = None,
    reference_confidence: np.ndarray | None = None,
    proposed_scale: float | None = None,
    state: DynamicAlphaState | None = None,
    config: DynamicAlphaConfig | None = None,
) -> DynamicAlphaResult:
    """Estimate opacity, reverse the alpha blend, and make a quality decision."""

    settings = config or DynamicAlphaConfig()
    observed_array = _validate_image(observed, "observed")
    profile_float = _coerce_profile(profile, observed_array, settings)
    selection = select_alpha_scale(
        observed_array,
        watermark,
        profile_float,
        clean_reference=clean_reference,
        reference_confidence=reference_confidence,
        proposed_scale=proposed_scale,
        state=state,
        config=settings,
    )
    requested_alpha = profile_float * selection.scale
    deblend = reverse_alpha_deblend(
        observed_array,
        watermark,
        requested_alpha,
        config=settings,
    )
    quality = assess_deblend_quality(
        observed_array,
        deblend,
        clean_reference=clean_reference,
        reference_confidence=reference_confidence,
        reference_was_used=selection.used_clean_reference,
        config=settings,
    )
    if state is not None:
        state.record(selection.scale, accepted=quality.accepted)
    return DynamicAlphaResult(
        candidate=deblend.candidate,
        alpha=deblend.alpha,
        selection=selection,
        quality=quality,
    )


__all__ = [
    "DeblendResult",
    "DynamicAlphaConfig",
    "DynamicAlphaResult",
    "DynamicAlphaState",
    "QualityDecision",
    "ScaleSelection",
    "assess_deblend_quality",
    "restore_dynamic_alpha",
    "reverse_alpha_deblend",
    "select_alpha_scale",
]
