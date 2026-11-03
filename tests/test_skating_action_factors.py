"""Small synthetic checks only; no model weights, video, fitting, or GPU needed."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.skating_action_factors import (
    ActionEvidence, MassWeightedCOM, SourceTimeline, airborne_ballistic_residual,
    glide_lateral_residual, root_so3_diagnostics, validate_source_timeline,
)


def clock(pts=(283136, 283392, 283904, 284160)):
    return validate_source_timeline(pts, (1, 12800), source_id="synthetic-video")


def z_rotations(angles_deg):
    angles = np.deg2rad(np.asarray(angles_deg))
    vectors = np.zeros((len(angles), 3))
    vectors[:, 2] = angles
    return Rotation.from_rotvec(vectors).as_matrix()


def glide_args(c):
    n = len(c)
    return dict(
        rotation_world=np.repeat(np.eye(3)[None], n, axis=0),
        origin_velocity_world_mps=[12., 0., 0.], angular_velocity_world_radps=[0., 0., 0.],
        contact_point_local_m=[0.04, 0., -0.1], blade_tangent_local=[1., 0., 0.],
        ice_normal_world=[0., 0., 1.], phases="stable_glide", confidence=1., sigma_mps=0.2,
    )


def ballistic_data(c):
    t = c.elapsed_s
    gravity = np.array([0., 0., -9.81])
    C = np.array([1., 2., 3.]) + t[:, None] * [4., -2., 5.] + .5*t[:, None]**2*gravity
    offset = np.stack((.2*np.sin(7*t), .1*np.cos(5*t), .1*np.sin(11*t)), axis=1)
    # Different segment motions with an exact mass-weighted ballistic centre.
    segments = np.stack((C+offset, C-offset/3), axis=1)
    com = MassWeightedCOM(c, segments, [1., 3.], ("upper_group", "lower_group"), "synthetic-mass-v1")
    return C, com, gravity


class TimelineTests(unittest.TestCase):
    def test_nonuniform_pts_preserved_and_input_not_modified(self):
        pts = np.array([1000000000, 1000000256, 1000000640, 1000001024], dtype=np.int64)
        original = pts.copy()
        c = validate_source_timeline(pts, (1, 12800), source_id="test")
        assert_array_equal(c.pts, original)
        assert_allclose(c.elapsed_s, [0., .02, .05, .08])
        assert_array_equal(pts, original)
        pts[0] = 7
        self.assertEqual(c.pts[0], original[0])
        self.assertFalse(c.pts.flags.writeable)

    def test_invalid_clock_is_explicit_error(self):
        for pts in ([1, 1, 2], [2, 1], [1., 2.], [], [True, 2]):
            with self.subTest(pts=pts), self.assertRaises(ValueError):
                validate_source_timeline(pts, (1, 12800), source_id="x")
        for base in ((0, 1), (1, 0), (-1, 1), (1., 12800)):
            with self.subTest(base=base), self.assertRaises(ValueError):
                validate_source_timeline([1, 2], base, source_id="x")


class GlideTests(unittest.TestCase):
    def test_fast_forward_and_backward_glide_are_unpenalized(self):
        c = clock()
        for speed in (-12., 12.):
            args = glide_args(c)
            args["origin_velocity_world_mps"] = [speed, 0., 0.]
            result = glide_lateral_residual(c, **args)
            self.assertEqual(result.status, "active")
            assert_allclose(result.residual, 0., atol=1e-14)
            assert_allclose(result.diagnostics["along_blade_velocity_mps"], speed)
            assert_array_equal(result.pts, c.pts)

    def test_side_slip_has_signed_residual_and_sqrt_confidence(self):
        c = clock()
        args = glide_args(c)
        args.update(origin_velocity_world_mps=[8., .6, 0.], confidence=.25, sigma_mps=.2)
        result = glide_lateral_residual(c, **args)
        assert_allclose(result.residual[:, 0], 1.5)
        assert_allclose(result.diagnostics["lateral_velocity_mps"], .6)

    def test_moving_rocker_location_does_not_add_qdot(self):
        c = clock()
        args = glide_args(c)
        # A changing active rocker/candidate location is not a material particle.
        # Its large locus speed must NOT be added to body material velocity.
        args["contact_point_local_m"] = np.array([
            [-.1, 0., -.09], [-.02, .002, -.1], [.07, -.002, -.095], [.1, 0., -.09],
        ])
        result = glide_lateral_residual(c, **args)
        assert_allclose(result.diagnostics["material_velocity_world_mps"],
                        np.tile([12., 0., 0.], (len(c), 1)))
        assert_allclose(result.residual, 0.)

    def test_angular_lever_arm_contributes_at_material_contact(self):
        c = clock()
        args = glide_args(c)
        args.update(origin_velocity_world_mps=[0., 0., 0.],
                    angular_velocity_world_radps=[0., 0., 2.],
                    contact_point_local_m=[.1, 0., -.1])
        result = glide_lateral_residual(c, **args)
        assert_allclose(result.diagnostics["material_velocity_world_mps"],
                        np.tile([0., .2, 0.], (len(c), 1)))
        assert_allclose(result.residual, 1.)

    def test_world_rotation_and_plane_normal_are_used(self):
        c = clock()
        args = glide_args(c)
        args["rotation_world"] = z_rotations([90]*len(c))
        args["origin_velocity_world_mps"] = [0., 9., 0.]
        args["ice_normal_world"] = [0., 0., 7.]
        result = glide_lateral_residual(c, **args)
        assert_allclose(result.residual, 0., atol=1e-12)

    def test_pivot_push_flight_unknown_are_disabled(self):
        c = clock()
        args = glide_args(c)
        args.update(phases=["pivot", "push", "flight", "unknown"],
                    origin_velocity_world_mps=[1., 3., 0.])
        result = glide_lateral_residual(c, **args)
        self.assertEqual(result.status, "disabled")
        assert_array_equal(result.enabled, [False]*4)
        assert_allclose(result.residual, 0.)
        self.assertEqual(result.reasons, ("phase:pivot", "phase:push", "phase:flight", "phase:unknown"))

    def test_zero_confidence_and_vertical_tangent_are_explained(self):
        c = clock()
        args = glide_args(c)
        args["confidence"] = [0., 1., 1., 1.]
        args["blade_tangent_local"] = np.array([[1.,0.,0.], [0.,0.,1.], [1.,0.,0.], [1.,0.,0.]])
        result = glide_lateral_residual(c, **args)
        self.assertEqual(result.status, "partially_active")
        self.assertEqual(result.reasons[:2], ("zero_confidence", "blade_tangent_normal_to_ice"))

    def test_invalid_geometry_weight_and_scale_raise(self):
        c = clock()
        for key, value in (("sigma_mps", 0.), ("confidence", 1.1),
                           ("origin_velocity_world_mps", [np.nan, 0., 0.]),
                           ("ice_normal_world", [0.,0.,0.]), ("phases", "typo")):
            args = glide_args(c)
            args[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                glide_lateral_residual(c, **args)


class BallisticTests(unittest.TestCase):
    def test_nonuniform_parabolic_com_is_zero(self):
        c = SourceTimeline([10000, 10040, 10110, 10250, 10330], (1,1000), "flight")
        expected, com, gravity = ballistic_data(c)
        assert_allclose(com.position_world_m, expected, atol=1e-14)
        result = airborne_ballistic_residual(c, com, phases="flight", confidence=1.,
                                             gravity_world_mps2=gravity, sigma_mps2=2.)
        self.assertEqual(result.status, "active")
        assert_allclose(result.residual, 0., atol=1e-10)
        assert_array_equal(result.pts, c.pts[1:-1])
        assert_array_equal(result.diagnostics["stencil_pts"][0], c.pts[:3])

    def test_pelvis_array_and_one_segment_are_rejected(self):
        c = clock()
        _, com, gravity = ballistic_data(c)
        with self.assertRaisesRegex(ValueError, "pelvis/root"):
            airborne_ballistic_residual(c, com.segment_com_world_m[:, 0], phases="flight", confidence=1.,
                                        gravity_world_mps2=gravity, sigma_mps2=1.)
        with self.assertRaisesRegex(ValueError, "at least two"):
            MassWeightedCOM(c, com.position_world_m[:, None], [4.], ("pelvis",), "not-whole-body")

    def test_contact_boundary_and_unknown_do_not_get_ballistic_force(self):
        c = SourceTimeline([100, 120, 140, 180, 200], (1,1000), "flight")
        _, com, gravity = ballistic_data(c)
        result = airborne_ballistic_residual(
            c, com, phases=["push", "flight", "flight", "flight", "unknown"], confidence=1.,
            gravity_world_mps2=gravity, sigma_mps2=1.)
        assert_array_equal(result.enabled, [False, True, False])
        self.assertEqual(result.reasons[0], "stencil_crosses_contact_or_unknown")

    def test_mismatched_source_times_and_nonpositive_mass_fail(self):
        c = clock()
        _, com, gravity = ballistic_data(c)
        other = SourceTimeline(c.pts+1, c.time_base, c.source_id)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            airborne_ballistic_residual(other, com, phases="flight", confidence=1.,
                                        gravity_world_mps2=gravity, sigma_mps2=1.)
        with self.assertRaisesRegex(ValueError, "positive"):
            MassWeightedCOM(c, com.segment_com_world_m, [0.,3.], ("a","b"), "bad-mass")


class RotationTests(unittest.TestCase):
    def test_crossing_yaw_pi_nonuniform_clock_keeps_rate(self):
        c = SourceTimeline([100, 190, 210, 300], (1,1000), "rotation")
        R = z_rotations([170.,179.,-179.,-170.])
        original = R.copy()
        result = root_so3_diagnostics(c, R)
        assert_allclose(result.omega_world_radps[:, 2], np.deg2rad(100.), atol=1e-12)
        assert_allclose(result.omega_world_radps[:, :2], 0., atol=1e-12)
        assert_allclose(result.alpha_world_radps2, 0., atol=1e-10)
        assert_array_equal(result.source_pts, c.pts)
        assert_array_equal(result.velocity_interval_pts, [[100,190], [190,210], [210,300]])
        assert_array_equal(R, original)

    def test_world_frame_omega_and_two_sample_status(self):
        c = SourceTimeline([10, 30], (1,100), "rotation")
        base = Rotation.from_euler("y", 70, degrees=True).as_matrix()
        R = np.stack((base, Rotation.from_euler("z", .2).as_matrix() @ base))
        result = root_so3_diagnostics(c, R)
        assert_allclose(result.omega_world_radps[0], [0.,0.,1.], atol=1e-12)
        self.assertEqual(result.acceleration_status, "disabled:insufficient_samples")
        self.assertEqual(result.alpha_world_radps2.shape, (0,3))

    def test_near_pi_aliasing_and_invalid_rotations_are_explicit(self):
        c = SourceTimeline([10, 30], (1,100), "rotation")
        R = z_rotations([0., 180.])
        self.assertTrue(root_so3_diagnostics(c, R).near_pi_ambiguous[0])
        R[0,0,0] = 2.
        with self.assertRaisesRegex(ValueError, "SO\\(3\\)"):
            root_so3_diagnostics(c, R)


class EvidenceTests(unittest.TestCase):
    def test_protocol_label_never_infers_support_edge_or_confirmation(self):
        c = clock()
        event = ActionEvidence(c.source_id, int(c.pts[0]), int(c.pts[-1]),
                               "official_protocol", "3Lz", .9)
        result = event.validate(c)
        self.assertEqual(result.support_side, "unknown")
        self.assertEqual((result.left_edge, result.right_edge), ("unknown", "unknown"))
        self.assertFalse(result.human_confirmed)
        self.assertFalse(result.score_is_calibrated_probability)

    def test_protocol_cannot_claim_measured_edge(self):
        c = clock()
        event = ActionEvidence(c.source_id, int(c.pts[0]), int(c.pts[-1]),
                               "official_protocol", "3Lz", .9, left_edge="outer")
        with self.assertRaisesRegex(ValueError, "protocol labels"):
            event.validate(c)

    def test_model_inference_retains_unknown_and_source_identity(self):
        c = clock()
        event = ActionEvidence(c.source_id, int(c.pts[1]), int(c.pts[2]),
                               "model_inference", "jump_takeoff", .7,
                               evidence_pts=(int(c.pts[1]),))
        result = event.validate(c)
        self.assertEqual(result.origin, "model_inference")
        self.assertEqual(result.support_side, "unknown")
        self.assertFalse(result.human_confirmed)
        bad = ActionEvidence("different-video", int(c.pts[1]), int(c.pts[2]),
                             "video_observation", "contact", .8)
        with self.assertRaisesRegex(ValueError, "source_id"):
            bad.validate(c)

    def test_event_must_reference_real_pts(self):
        c = clock()
        event = ActionEvidence(c.source_id, int(c.pts[1])+1, int(c.pts[2]),
                               "video_observation", "contact", .8)
        with self.assertRaisesRegex(ValueError, "actual source samples"):
            event.validate(c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
