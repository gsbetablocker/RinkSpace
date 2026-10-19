"""Numeric checks for per-frame Brown12 fixed-centre ray transfer."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "calibration" / "shot_tracking" / "pipeline"))
sys.path.insert(0, str(ROOT / "calibration" / "work"))

import camera  # noqa: E402
import ray_motion  # noqa: E402
import zoom_ray_motion  # noqa: E402
from physical_projection import safe_project  # noqa: E402


def make_camera(rvec, *, distortion=None, centre=(2., -3., 4.),
                fx=1250., fy=1220., cx=960., cy=540., skew=8.):
    v = np.zeros(25)
    v[:3] = centre
    v[3:6] = rvec
    v[6:8] = np.log([fx, fy])
    v[8:11] = [cx, cy, skew]
    if distortion is not None:
        v[11:23] = distortion
    return v


def test_shared_brown_matches_existing_ray_transfer_and_warmstart():
    d = np.array([-.08, .015, .001, -.0007, -.002, .004, -.001, .0001,
                  .0002, -.0001, .00015, -.0001])
    cameras = np.stack([
        make_camera([0., 0., 0.], distortion=d),
        make_camera([.018, -.025, .012], distortion=d, fx=1380., fy=1340., skew=11.),
        make_camera([-.026, .015, -.009], distortion=d, fx=1120., fy=1095., skew=4.),
    ])
    source = np.array([[180., 210.], [870., 450.], [1540., 820.],
                       [600., 700.], [1200., 320.], [1730., 620.]])
    start = np.array([0, 0, 1, 1, 2, 2])
    end = np.array([1, 2, 0, 2, 0, 1])
    old = ray_motion.predict_pairs(cameras, start, end, source, tolerance_px=1e-10)
    new = zoom_ray_motion.predict_pairs(cameras, start, end, source, tolerance_px=1e-10)
    np.testing.assert_array_equal(new["valid"], old["valid"])
    assert new["valid"].all()
    for key in ("predicted_uv", "normalized_from", "normalized_to", "inverse_error_px"):
        np.testing.assert_allclose(new[key], old[key], atol=1e-9, rtol=0)
    warm = zoom_ray_motion.predict_pairs(cameras, start, end, source,
                                         inverse_initial=new["normalized_from"],
                                         tolerance_px=1e-10)
    np.testing.assert_array_equal(warm["valid"], new["valid"])
    np.testing.assert_allclose(warm["predicted_uv"], new["predicted_uv"], atol=1e-9, rtol=0)
    assert max(new["inverse_error_px"]) <= 2e-10


def test_changing_brown_matches_world_rays_and_strict_target_projection():
    d0 = np.array([-.07, .018, .0008, -.001, -.003, .003, -.0007, .0001,
                   .0002, -.0001, .00015, -.0001])
    d1 = np.array([-.11, .028, -.0006, .0013, -.004, .008, -.0012, .0002,
                   -.0001, .0002, -.0002, .0001])
    c0 = make_camera([.01, -.02, .005], distortion=d0)
    c1 = make_camera([-.018, .025, -.013], distortion=d1,
                     fx=1430., fy=1370., cx=950., cy=548., skew=12.)
    cameras = np.stack([c0, c1])
    normalized = np.array([[-.32, -.20], [-.12, .16], [.10, -.08], [.38, .24]])
    rotation = cv2.Rodrigues(c0[3:6])[0]
    world = c0[:3] + 60. * (np.column_stack((normalized, np.ones(len(normalized)))) @ rotation)
    raw_from, _, from_valid = safe_project(c0, world)
    raw_to, _, to_valid = safe_project(c1, world)
    assert from_valid.all() and to_valid.all()
    predicted = zoom_ray_motion.predict_pairs(cameras, np.zeros(len(world), int),
                                              np.ones(len(world), int), raw_from,
                                              tolerance_px=1e-10)
    assert predicted["valid"].all()
    np.testing.assert_allclose(predicted["predicted_uv"], raw_to, atol=1e-8, rtol=0)
    assert max(predicted["inverse_error_px"]) <= 2e-10
    reverse = zoom_ray_motion.predict_pairs(cameras, np.ones(len(world), int),
                                            np.zeros(len(world), int), raw_to,
                                            tolerance_px=1e-10)
    assert reverse["valid"].all()
    np.testing.assert_allclose(reverse["predicted_uv"], raw_from, atol=1e-8, rtol=0)
    mixed_source = np.stack((raw_from, raw_to), axis=1).reshape(-1, 2)
    mixed_expected = np.stack((raw_to, raw_from), axis=1).reshape(-1, 2)
    mixed_from = np.tile([0, 1], len(world))
    mixed = zoom_ray_motion.predict_pairs(cameras, mixed_from, 1-mixed_from,
                                          mixed_source, tolerance_px=1e-10)
    assert mixed["valid"].all()
    np.testing.assert_allclose(mixed["predicted_uv"], mixed_expected, atol=1e-8, rtol=0)

    rays, ray_valid = camera.world_rays(c0, raw_from)
    assert ray_valid.all()
    target_from_independent_rays, _, valid = safe_project(c1, c0[:3] + 60.*rays)
    assert valid.all()
    np.testing.assert_allclose(predicted["predicted_uv"], target_from_independent_rays,
                               atol=.005, rtol=0)

    # A small Brown change must remain visible to finite-difference fitting.
    changed = cameras.copy()
    changed[1, 11] += 1e-6
    after = zoom_ray_motion.predict_pairs(changed, np.zeros(len(world), int),
                                          np.ones(len(world), int), raw_from,
                                          inverse_initial=predicted["normalized_from"],
                                          tolerance_px=1e-10)
    expected_after, _, valid = safe_project(changed[1], world)
    assert valid.all() and after["valid"].all()
    assert np.max(np.abs(after["predicted_uv"] - predicted["predicted_uv"])) > 1e-5
    np.testing.assert_allclose(after["predicted_uv"], expected_after, atol=1e-8, rtol=0)


def test_invalid_pixels_and_back_facing_rays_are_masked_and_centre_must_match():
    c0 = make_camera([0., 0., 0.])
    c1 = make_camera([0., 0., 0.], distortion=np.array([-.06, .01] + [0.]*10))
    c2 = make_camera([0., np.pi, 0.])
    cameras = np.stack([c0, c1, c2])
    result = zoom_ray_motion.predict_pairs(cameras, np.array([0, 0, 0]),
                                           np.array([1, 1, 2]),
                                           np.array([[960., 540.], [np.nan, 540.], [960., 540.]]))
    np.testing.assert_array_equal(result["valid"], [True, False, False])
    assert np.isnan(result["predicted_uv"][1:]).all()
    shifted = cameras.copy()
    shifted[1, 0] += 1e-4
    with pytest.raises(ValueError, match="fixed optical centre"):
        zoom_ray_motion.predict_pairs(shifted, [0], [1], [[960., 540.]])
