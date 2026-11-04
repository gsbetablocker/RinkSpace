"""Targeted checks for the newly proposed source-anchor mathematics only."""
import unittest
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from calibration.program_tracking.body_tracking.support_anchor_factors import (
    NativeRegionObservation, pixel_region_factor, lowest_runner_witness,
    contact_region_plane_lift,
)


def observation(**changes):
    base = NativeRegionObservation(
        source_id="synthetic-camera-source", shot_id="s05", pts=1062400,
        time_base=(1, 12800), foot=1, kind="support_contact_region",
        centre_uv=(100., 80.), region_shape_px2=((16., 3.), (3., 9.)),
        visibility="visible", contact_state="supported", role="fit",
        reliability=.8, objective_weight=.02, provenance="synthetic test, not video annotation")
    return replace(base, **changes)


def region(uv, o=None):
    o = o or observation()
    return pixel_region_factor(uv, o, source_id=o.source_id, shot_id=o.shot_id,
        pts=o.pts, time_base=o.time_base, predicted_foot=o.foot,
        predicted_kind=o.kind, robust_delta=2.)


def rotation(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle == 0:
        return np.eye(3)
    a = v/angle
    K = np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def arc(lo=-.3, hi=.3):
    return dict(id="synthetic_runner", semantic="runner", center=[0., .002, 2.],
        axis_cos=[0., 0., -1.], axis_sin=[1., 0., 0.], radius_m=2.,
        theta_start_rad=lo, theta_end_rad=hi)


class RegionTests(unittest.TestCase):
    def test_floating_point_precision_is_not_an_accuracy_gate(self):
        # In-plane translation changes the source pixel residual, whereas the
        # scalar distance to an ideal plane cannot see it at all.
        a, b = np.array([0., 0., .02]), np.array([2., -1., .02])
        self.assertEqual(a[2], b[2])
        self.assertEqual(region([100., 80.])["cost"], 0.)
        self.assertGreater(region([120., 70.])["cost"], 0.)

    def test_inside_source_region_has_zero_pressure(self):
        r = region([101., 81.])
        self.assertTrue(r["enabled"])
        assert_allclose(r["residual"], 0.)
        assert_allclose(r["jacobian_native_uv"], 0.)

    def test_anisotropic_region_robust_derivative(self):
        uv = np.array([112., 72.])
        h = 1e-5
        numeric = np.column_stack([(region(uv + np.eye(2)[j]*h)["residual"]
                                  - region(uv - np.eye(2)[j]*h)["residual"])/(2*h)
                                 for j in range(2)])
        assert_allclose(region(uv)["jacobian_native_uv"], numeric, rtol=1e-7, atol=1e-10)

    def test_exact_identity_is_required(self):
        o = observation()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            pixel_region_factor([100., 80.], o, source_id=o.source_id, shot_id=o.shot_id,
                pts=o.pts+1, time_base=o.time_base, predicted_foot=1,
                predicted_kind=o.kind, robust_delta=2.)
        with self.assertRaises(ValueError):
            observation(pts=1062400.)
        with self.assertRaises(ValueError):
            observation(kind="ankle_contact")

    def test_unknown_contact_does_not_erase_visible_boot_evidence(self):
        o = observation(contact_state="unknown")
        self.assertFalse(region([130., 80.], o)["enabled"])
        boot = replace(o, kind="boot_marker_region")
        self.assertTrue(region([130., 80.], boot)["enabled"])

    def test_review_occlusion_and_unknown_side_are_not_measured_zero(self):
        for o in (observation(role="review"), observation(visibility="occluded"),
                  observation(foot=None), observation(contact_state="airborne")):
            r = region([140., 80.], o)
            self.assertFalse(r["enabled"])
            self.assertFalse(r["measured_error_available"])
        self.assertGreater(region([140., 80.])["cost"], 0.)


class RockerTests(unittest.TestCase):
    def check_derivative(self, R, a):
        O = np.array([.2, -.1, .08])
        evaluate = lambda R_: lowest_runner_witness([a], R_, O, plane_normal_world=[0., 0., 1.])
        row = evaluate(R)
        h = 1e-6
        numeric = np.column_stack([(evaluate(R @ rotation(np.eye(3)[j]*h))["world_point"]
                                   - evaluate(R @ rotation(-np.eye(3)[j]*h))["world_point"])/(2*h)
                                  for j in range(3)])
        assert_allclose(row["jacobian_right_rotation"], numeric, atol=3e-9, rtol=2e-7)
        return row

    def test_interior_image_derivative_includes_moving_argmin(self):
        row = self.check_derivative(rotation([.025, .07, .12]), arc())
        self.assertTrue(row["interior"])
        self.assertGreater(np.linalg.norm(row["jacobian_right_rotation"]
                                         - row["frozen_material_point_jacobian"]), 1.)
        # Only the normal-height derivative benefits from the envelope theorem.
        assert_allclose(row["jacobian_right_rotation"][2],
                        row["frozen_material_point_jacobian"][2], atol=1e-12)

    def test_clamped_endpoint_has_no_argmin_motion(self):
        row = self.check_derivative(rotation([.02, .6, .05]), arc(-.05, .05))
        self.assertFalse(row["interior"])
        self.assertFalse(row["endpoint_transition"])
        assert_allclose(row["jacobian_right_rotation"], row["frozen_material_point_jacobian"])

    def test_stationary_endpoint_flags_derivative_branch_transition(self):
        # At identity the unconstrained minimum is exactly theta=0, which is
        # the lower endpoint. A positive/negative pitch selects different
        # interior/clamped derivative branches despite a continuous point.
        row = lowest_runner_witness([arc(0., .3)], np.eye(3), np.zeros(3),
                                    plane_normal_world=[0., 0., 1.])
        self.assertFalse(row["interior"])
        self.assertTrue(row["endpoint_transition"])
        self.assertTrue(row["differentiability_warning"])

    def test_same_point_different_curvature_tie_flags_distinct_derivatives(self):
        a, b = arc(), arc()
        a["id"], b["id"] = "curvature_a", "curvature_b"
        # The two arcs share the same interior minimum at [0, .002, 0], but
        # their pitch derivatives differ because their radii are different.
        b["radius_m"] = 4.
        b["center"] = [0., .002, 4.]
        row = lowest_runner_witness([a, b], np.eye(3), np.zeros(3),
                                    plane_normal_world=[0., 0., 1.])
        self.assertFalse(row["distinct_minima_tied"])
        self.assertFalse(row["endpoint_transition"])
        self.assertTrue(row["tied_derivatives_differ"])
        self.assertTrue(row["differentiability_warning"])

    def test_two_level_edges_do_not_claim_unique_contact(self):
        a, b = arc(), arc()
        a["id"], b["id"] = "left_edge", "right_edge"
        b["center"] = [0., -.002, 2.]
        row = lowest_runner_witness([a,b], np.eye(3), np.zeros(3), plane_normal_world=[0,0,1])
        self.assertTrue(row["distinct_minima_tied"])
        with self.assertRaisesRegex(ValueError, "runner"):
            lowest_runner_witness([dict(a, semantic="solid_guard")], np.eye(3),
                                  np.zeros(3), plane_normal_world=[0,0,1])


class PlaneLiftTests(unittest.TestCase):
    def lift(self, o, **changes):
        options = dict(camera_origin_world=[0., 0., 2.],
            ray_at_native_uv=lambda uv: np.array([(uv[0]-100.)/100., 1., -(uv[1]-60.)/100.]),
            plane_normal_world=[0.,0.,1.], plane_offset=0., plane_height_sigma=.04,
            extra_world_covariance=np.zeros((3,3)), pixel_difference_step=1e-3,
            minimum_incidence_cosine=1e-4)
        options.update(changes)
        return contact_region_plane_lift(o, **options)

    def test_known_plane_and_uncertainty(self):
        row = self.lift(observation())
        assert_allclose(row["position_world"], [0., 10., 0.], atol=1e-12)
        self.assertFalse(row["independent_evidence"])
        self.assertGreater(row["world_uncertainty_shape"][1,1],
                           row["world_uncertainty_shape"][0,0])
        self.assertGreater(row["world_uncertainty_shape"][2,2], 0.)

    def test_equivalent_scaled_plane_preserves_geometry_and_uncertainty(self):
        expected = self.lift(observation(), plane_offset=.4)
        for scale in (2., -3.):
            with self.subTest(scale=scale):
                actual = self.lift(observation(),
                    plane_normal_world=[0., 0., scale], plane_offset=.4*scale)
                for name in ("position_world", "jacobian_native_uv", "world_uncertainty_shape"):
                    assert_allclose(actual[name], expected[name], rtol=1e-11, atol=1e-11)
                self.assertEqual(actual["pixel_jacobian_conditioning"], "fixed_observed_row_camera")

    def test_ankle_or_boot_pixels_are_not_projected_onto_the_ice(self):
        for o in (observation(kind="boot_marker_region"), observation(contact_state="unknown")):
            with self.assertRaisesRegex(ValueError, "contact"):
                self.lift(o)

    def test_grazing_or_backward_ray_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Grazing"):
            self.lift(observation(centre_uv=(100.,60.)))
        with self.assertRaisesRegex(ValueError, "behind"):
            self.lift(observation(centre_uv=(100.,40.)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
