import numpy as np
import pytest
from scipy.special import erf

from calibration.program_tracking.first_shot_localization_noise import (
    LocalizationNoiseFailure, bootstrap_localization_covariance,
    midpoint_normal_jacobian, sample_native_profile,
)
from calibration.program_tracking.native_ice_material_repair import whole_colour_bands


def _paint_midpoint_noise(width, blur):
    x = np.arange(240.) - 119.5
    signal = 90. + 70.*.5*(erf((x+width/2)/(np.sqrt(2)*blur))
                                - erf((x-width/2)/(np.sqrt(2)*blur)))
    image = np.tile(signal, (24, 1)).astype(np.float32)
    rng = np.random.default_rng(73)
    blocks = rng.normal(0., 1.4, (64, 8, 8))

    def reader(perturbed):
        # This is a complete broad material band, as used for C. The narrow
        # outline reader enumerates local valleys and can propose the same
        # wide flat band twice after pixel noise; that is not a noise estimate.
        candidates = whole_colour_bands(np.zeros(2), np.array([1., 0.]), x,
                                        perturbed[12], np.ones(len(x), bool),
                                        maximum_width=90.)
        same_material = [r for r in candidates if abs(r['uv'][0]) < 5.
                         and abs(r['width_px']-width) < 6.]
        if len(same_material) != 1:
            raise LocalizationNoiseFailure('The original two paint flanks were lost')
        return np.asarray([same_material[0]['uv'][0]])

    result = bootstrap_localization_covariance(image, blocks, reader,
                                               repetitions=192, seed=48)
    return np.sqrt(result['covariance'][0, 0])


def test_flat_paint_width_is_not_position_noise_but_blur_affects_gradient():
    narrow = _paint_midpoint_noise(20., 1.5)
    wide = _paint_midpoint_noise(60., 1.5)
    blurry = _paint_midpoint_noise(20., 4.)
    # The actual two-flank reader sees three times the flat paint width.
    # Its localization uncertainty must not become three times larger.
    assert .5 < wide/narrow < 1.7
    assert blurry > 1.6*narrow


def test_shared_native_pixels_preserve_common_mode_and_mask_all_contributors():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros_like(image, bool)
    xy = np.array([[[4.25, 5.25], [4.25, 5.25]]])
    sampled = sample_native_profile(image, xy, mask, median_across_tangent=False)
    assert sampled.valid.all()
    mask[6, 5] = True
    assert not sample_native_profile(image, xy, mask, median_across_tangent=False).valid.any()

    blocks = np.random.default_rng(5).normal(size=(16, 4, 4))
    def reader(v):
        value = float(v[5, 4])
        return np.array([value, value])
    covariance = bootstrap_localization_covariance(image, blocks, reader,
        repetitions=96, seed=0)['covariance']
    assert covariance[0, 1] == pytest.approx(covariance[0, 0])
    assert covariance[1, 1] == pytest.approx(covariance[0, 0])


def test_midpoint_uses_two_actual_interfaces_and_uint8_remap_is_supported():
    def source_sides(offsets):
        return np.c_[2*np.asarray(offsets), np.asarray(offsets)*.25]
    def midpoint(sides):
        return sides.mean(0)
    result = midpoint_normal_jacobian([-20., 20.], source_sides, midpoint,
                                      np.array([1., 0.]))
    np.testing.assert_allclose(result, [1., 1.], atol=1e-12)
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    sampled = sample_native_profile(image, np.array([[[4.2, 3.3]]]),
                                    np.zeros_like(image, bool), median_across_tangent=False)
    assert np.isfinite(sampled.values).all()
