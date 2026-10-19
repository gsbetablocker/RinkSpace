"""Native pixel noise propagated through an unchanged material reader.

Transition support and paint width are not inputs to the noise variance.
The caller supplies authenticated, unmasked same-material plateau pixels and
the exact original reader. Bootstrap exposures are numerical noise draws,
never new independent source observations or replacements for the source UV.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np


SCHEMA = "rinkspace.native-pixel-block-localization-noise.v1"


class LocalizationNoiseFailure(ValueError):
    def __init__(self, reason, report=None):
        super().__init__(reason)
        self.report = dict(reason=reason, passed=False, **(report or {}))


@dataclass(frozen=True)
class NativeProfileSample:
    values: np.ndarray
    valid: np.ndarray
    contributor_counts: np.ndarray
    native_sampling_uv: np.ndarray


def sample_native_profile(image, sample_uv, exclusion_mask, *,
                          median_across_tangent=True, quantize_lattice32=True):
    """Replay remap and median using only valid native bilinear contributors."""
    image = np.asarray(image)
    xy = np.asarray(sample_uv, np.float32)
    mask = np.asarray(exclusion_mask, bool)
    if xy.ndim != 3 or xy.shape[-1] != 2 or mask.shape != image.shape[:2]:
        raise LocalizationNoiseFailure("Invalid native profile sampling inventory")
    if quantize_lattice32:
        xy = np.rint(xy*32.)/32.
    floor = np.floor(np.where(np.isfinite(xy), xy, -100.)).astype(int)
    h, w = mask.shape
    valid = np.isfinite(xy).all(2)
    valid &= (floor >= 0).all(2) & (floor < [w-1, h-1]).all(2)
    ids = np.where(valid)
    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        valid[ids] &= ~mask[floor[ids][..., 1]+dy, floor[ids][..., 0]+dx]
    values = cv2.remap(image, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR)
    counts = valid.sum(0)
    if median_across_tangent:
        extension = (None,) if values.ndim == 3 else ()
        ordered = np.sort(np.where(valid[(...,)+extension], values, np.inf), axis=0)
        col = np.arange(len(counts))
        output = .5*(ordered[np.maximum(0, (counts-1)//2), col]
                     + ordered[np.maximum(0, counts//2), col])
        output[counts == 0] = np.nan
    else:
        if len(xy) != 1:
            raise LocalizationNoiseFailure("Nonmedian profile requires one native transverse section")
        # Preserve the reader's uint8 remap/rounding, then represent invalid
        # contributors with NaN in floating output.
        output = values[0].astype(float)
        output[~valid[0]] = np.nan
    return NativeProfileSample(output, counts > 0, counts, xy)


def plane_residual_blocks(image, exclusion_mask, centers_xy, *, block_shape=(8, 8),
                          permitted_native_mask=None):
    """Measure correlated native-pixel residual blocks in certified plateaus.

    A local plane removes illumination trend. No iid residual-variance
    correction is applied: correlated residuals require native ACF and block
    length checks, and the removed low-frequency component stays explicit.
    Source and optional material-domain masks cover every pixel in a block.
    """
    values = np.asarray(image, float)
    mask = np.asarray(exclusion_mask, bool)
    bh, bw = map(int, block_shape)
    if min(bh, bw) < 2 or bh*bw <= 3 or mask.shape != values.shape[:2]:
        raise LocalizationNoiseFailure("Invalid native noise block shape")
    yy, xx = np.indices((bh, bw))
    design = np.column_stack((np.ones(bh*bw), xx.ravel()-xx.mean(), yy.ravel()-yy.mean()))
    accepted, records, rejected = [], [], []
    for center in np.asarray(centers_xy, float).reshape(-1, 2):
        x, y = np.rint(center-[.5*(bw-1), .5*(bh-1)]).astype(int)
        if x < 0 or y < 0 or x+bw > mask.shape[1] or y+bh > mask.shape[0]:
            rejected.append(dict(center_xy=center.tolist(), reason="native_block_outside_sensor")); continue
        region = np.s_[y:y+bh, x:x+bw]
        if mask[region].any() or (permitted_native_mask is not None
                                 and not np.asarray(permitted_native_mask, bool)[region].all()):
            rejected.append(dict(center_xy=center.tolist(), reason="native_block_mask_or_material_domain")); continue
        patch = values[region]
        flat = patch.reshape(bh*bw, -1)
        residual = flat-design@np.linalg.lstsq(design, flat, rcond=None)[0]
        residual = residual.reshape(patch.shape)
        accepted.append(residual)
        records.append(dict(top_left_native_xy=[int(x), int(y)], shape=list(patch.shape),
                            native_value_sha256=hashlib.sha256(patch.tobytes()).hexdigest(),
                            residual_mean_square=float(np.mean(residual**2))))
    if not accepted:
        raise LocalizationNoiseFailure("No complete unmasked native noise plateau blocks", dict(rejected=rejected))
    blocks = np.asarray(accepted)
    numerical_scale = 128.*np.finfo(float).eps*max(1., float(np.max(abs(values))))
    if not np.isfinite(blocks).all() or np.max(abs(blocks)) <= numerical_scale:
        raise LocalizationNoiseFailure("Native plateau noise is not empirically identifiable")
    return blocks, dict(block_shape=[bh, bw], accepted_native_blocks=records,
                        rejected_native_blocks=rejected, detrending="one native-pixel plane per block and channel",
                        residual_degrees_of_freedom=bh*bw-3,
                        iid_residual_variance_correction_applied=False,
                        removed_low_frequency_trend_and_block_length_require_native_ACF_QA=True,
                        full_channel_and_within_block_pixel_correlation_retained=True,
                        source_readouts_added=0)


def moving_block_noise(image_shape, residual_blocks, rng):
    """Resample intact oriented native residual blocks with random grid phase.

    The single returned pixel field is shared by all queries in this draw:
    reusing a native pixel never creates an independent noise observation.
    """
    blocks = np.asarray(residual_blocks, float)
    if blocks.ndim not in (3, 4) or not len(blocks) or not np.isfinite(blocks).all():
        raise LocalizationNoiseFailure("Finite native residual blocks required")
    h, w = map(int, image_shape[:2]); bh, bw = blocks.shape[1:3]
    phase_y, phase_x = rng.integers([bh, bw])
    ny = (h+phase_y+bh-1)//bh; nx = (w+phase_x+bw-1)//bw
    chosen = blocks[rng.integers(len(blocks), size=(ny, nx))]
    axes = (0, 2, 1, 3) if blocks.ndim == 3 else (0, 2, 1, 3, 4)
    field = chosen.transpose(axes).reshape((ny*bh, nx*bw)+blocks.shape[3:])
    return field[phase_y:phase_y+h, phase_x:phase_x+w].astype(np.float32)


def bootstrap_localization_covariance(source_image, residual_blocks, readout_callback, *,
                                     repetitions=128, seed=0, clip_native_RGB=False):
    """Replay fixed material readers on correlated native-pixel noise draws.

    The callback includes the original remap/median/filter, both actual flanks,
    and the original midpoint map. Any failed/changed material branch remains
    explicit and invalidates this covariance result; no sigma inflation hides it.
    """
    image = np.asarray(source_image, np.float32)
    if not callable(readout_callback) or repetitions < 2:
        raise LocalizationNoiseFailure("A fixed native reader and repeated pixel draws are required")
    baseline = np.asarray(readout_callback(image), float)
    if baseline.ndim != 1 or not len(baseline) or not np.isfinite(baseline).all():
        raise LocalizationNoiseFailure("Original native material branch cannot be replayed")
    rng = np.random.default_rng(seed); outputs, failures = [], []
    for index in range(int(repetitions)):
        perturbed = image+moving_block_noise(image.shape, residual_blocks, rng)
        if clip_native_RGB:
            perturbed = np.clip(perturbed, 0., 255.)
        try:
            result = np.asarray(readout_callback(perturbed), float)
            if result.shape != baseline.shape or not np.isfinite(result).all():
                raise LocalizationNoiseFailure("Noise replay changed or lost native material membership")
            outputs.append(result)
        except (LocalizationNoiseFailure, ValueError) as error:
            failures.append(dict(draw=index, reason=str(error), details=getattr(error, "report", None)))
    metadata = dict(schema=SCHEMA, requested_draws=int(repetitions), completed_draws=len(outputs),
                    seed=int(seed), source_UV_ID_material_and_clock_modified=False,
                    actual_native_pixels_shared_between_queries=True,
                    native_pixel_blocks_before_remap_median_and_reader=True,
                    empirical_native_noise_block_sha256=hashlib.sha256(np.asarray(residual_blocks).tobytes()).hexdigest(),
                    block_shape=list(np.asarray(residual_blocks).shape[1:]),
                    source_width_or_transition_support_used_as_noise_variance=False,
                    world_or_camera_fit_residual_used=False, native_RGB_censoring_replayed=clip_native_RGB,
                    independent_source_observations_added=0, failures=failures)
    if failures:
        raise LocalizationNoiseFailure("Native material branch is unstable under measured pixel noise", metadata)
    outputs = np.asarray(outputs)
    average = outputs.mean(0)
    factor = (outputs-average).T/np.sqrt(len(outputs)-1)
    covariance = factor@factor.T
    if not np.isfinite(covariance).all() or not np.all(np.diag(covariance) > 0):
        raise LocalizationNoiseFailure("Reader position noise is not empirically identifiable", metadata)
    metadata.update(passed=True, conditional_reader_mean_shift=(average-baseline).tolist(),
                    localization_standard_deviation=np.sqrt(np.diag(covariance)).tolist(),
                    bootstrap_Monte_Carlo_precision_is_finite=True,
                    filtering_correlation_not_divided_by_nominal_sample_count=True)
    return dict(covariance=covariance, noise_factor=factor, baseline=baseline,
                bootstrap_readouts=outputs, metadata=metadata)


def midpoint_normal_jacobian(scalar_sides, source_sides_at_offsets, midpoint_from_sides,
                             recorded_normal, *, step=.05):
    """Differentiate the unchanged actual two-interface midpoint coordinate map."""
    sides = np.asarray(scalar_sides, float)
    normal = np.asarray(recorded_normal, float)
    if sides.shape != (2,) or normal.shape != (2,) or not np.isfinite(sides).all():
        raise LocalizationNoiseFailure("Two actual flank offsets and recorded native normal required")
    jacobian = []
    for side in range(2):
        plus, minus = sides.copy(), sides.copy(); plus[side] += step; minus[side] -= step
        a = midpoint_from_sides(source_sides_at_offsets(plus)); b = midpoint_from_sides(source_sides_at_offsets(minus))
        if a is None or b is None:
            raise LocalizationNoiseFailure("Original material midpoint chart is not finite")
        jacobian.append(float(normal@(np.asarray(a)-np.asarray(b))/(2*step)))
    return np.asarray(jacobian)
