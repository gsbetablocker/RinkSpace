"""Geometry and unit checks for the one shared skate binding."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.skate_binding import (
    NEW_B_TO_OLD, SHAPE_NAMES, SkateBinding,
)
from calibration.program_tracking.body_tracking.skate_template import (
    SkateParameters, build_skate, profile_height,
)
from calibration.program_tracking.body_tracking.footwear_layout import OBSERVATION_LAYOUT


@pytest.fixture
def synthetic_binding():
    # Exercise generated geometry without needing a source surface/native DLL.
    binding = SkateBinding.__new__(SkateBinding)
    binding.layout = OBSERVATION_LAYOUT
    binding.ratio = .91
    binding.base_parameters = SkateParameters()
    binding.x = np.r_[np.zeros(12), np.log([1.04, .95, 1.08, .94, 1.06])]
    binding.R0 = np.broadcast_to(NEW_B_TO_OLD, (2, 3, 3)).copy()
    binding.t0 = np.array([[.01, -.03, .02], [-.02, -.04, .01]])
    binding._parameter_key = None
    binding._packet = None
    return binding


def section_margins(points, polygon):
    edge = np.roll(polygon, -1, axis=0) - polygon
    inward = np.c_[-edge[:, 1], edge[:, 0]]
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    return np.einsum("pij,ij->pi", points[:, None] - polygon, inward)


def test_proper_frame_and_single_metric_conversion(synthetic_binding):
    b = synthetic_binding
    np.testing.assert_allclose(NEW_B_TO_OLD.T @ NEW_B_TO_OLD, np.eye(3))
    assert np.linalg.det(NEW_B_TO_OLD) == 1.
    np.testing.assert_array_equal(NEW_B_TO_OLD @ [1., 0., 0.], [0., 0., 1.])
    p = b.packet()["parameters"]
    metric = replace(b.base_parameters, **{
        name: getattr(b.base_parameters, name) * np.exp(shift)
        for name, shift in zip(SHAPE_NAMES, b.x[12:])
    })
    for name in metric.__dataclass_fields__:
        actual = getattr(p, name)
        expected = getattr(metric, name)
        np.testing.assert_allclose(actual if name.endswith("_fraction") else actual*b.ratio, expected)
    generated = build_skate(metric)
    np.testing.assert_allclose(b.packet()["template"]["vertices"]*b.ratio,
                               generated["vertices"], atol=2e-14)


def test_shared_shape_rebuilds_installation_and_blade_adapter(synthetic_binding):
    b = synthetic_binding
    zero = np.zeros(17)
    initial = b.packet(zero)
    packet = b.packet(b.x)
    assert initial is not packet
    assert b.packet(b.x.copy()) is packet
    only_registration = b.x.copy()
    only_registration[:12] = .03
    assert b.packet(only_registration) is packet
    p, template = packet["parameters"], packet["template"]
    for patch in template["mounting_patches"]:
        plate = template["part_ranges"][patch["id"]+"_mount_plate"]
        vertices = template["vertices"][slice(*plate)]
        expected = getattr(p, patch["id"]+"_mount_height_m")
        np.testing.assert_allclose(vertices[:, 2].max(), expected, atol=1e-14)
        assert patch["top_gap_m"] == 0.
    for post in template["post_connections"]:
        vertices = template["vertices"][slice(*template["part_ranges"][post["id"]])]
        np.testing.assert_allclose(vertices[:, 2].max(), post["plate_bottom_z_m"], atol=1e-14)
        assert post["minimum_web_overlap_m"] > .00099/b.ratio
    blade = packet["adapter"][14:]
    np.testing.assert_allclose(blade[:, 2], profile_height(blade[:, 0], p), atol=1e-14)
    np.testing.assert_allclose(abs(blade[:, 1]), p.blade_width_m/2, atol=1e-14)
    for edge in packet["edges"]:
        np.testing.assert_allclose(edge[:, 2], profile_height(edge[:, 0], p), atol=1e-14)


def test_adapter_upper_points_lie_on_generated_shell(synthetic_binding):
    packet = synthetic_binding.packet()
    count = len(packet["xs"])
    outer = packet["upper"][:count*6].reshape(count, 6, 3)
    for point in packet["adapter"][:14]:
        section = int(np.argmin(abs(packet["xs"]-point[0])))
        np.testing.assert_allclose(packet["xs"][section], point[0], atol=1e-14)
        margins = section_margins(point[None, 1:], outer[section, :, 1:])[0]
        assert margins.min() >= -1e-13
        assert abs(margins.min()) < 1e-13


def test_inner_sections_are_inward_offsets_above_sole(synthetic_binding):
    b = synthetic_binding
    packet = b.packet()
    count = len(packet["xs"])
    outer = packet["upper"][:count*6].reshape(count, 6, 3)
    inner = packet["inner"][:count*6].reshape(count, 6, 3)
    lining = .003/b.ratio
    for i in range(count):
        margins = section_margins(inner[i, :, 1:], outer[i, :, 1:])
        assert margins.min() >= lining-1e-13
        np.testing.assert_allclose(margins.min(axis=1), lining, atol=1e-13)
        assert inner[i, :, 2].min() >= packet["floor"][i]+lining-1e-13
    assert np.ptp(packet["width"]) > .01/b.ratio


def test_bounded_upper_deformation_keeps_inner_inside_section_envelope(synthetic_binding):
    b = synthetic_binding
    packet = b.packet()
    alpha = np.array([[[.020, .015, .012], [.007, .004, .009]]])/b.ratio
    count = len(packet["xs"])
    outer = b.deform(packet["upper"], alpha)[0, :, :count*6].reshape(2, count, 6, 3)
    inner = b.deform(packet["inner"], alpha)[0, :, :count*6].reshape(2, count, 6, 3)
    for foot in range(2):
        assert np.all(np.diff(outer[foot, :, :, 0], axis=0) > 0.)
        for point in inner[foot].reshape(-1, 3):
            walls = np.array([[np.interp(point[0], outer[foot, :, j, 0],
                                        outer[foot, :, j, c])
                               for c in (1, 2)] for j in range(6)])
            assert section_margins(point[None, 1:], walls).min() >= -1e-12
