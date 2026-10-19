"""Small physical/semantic counterexamples; no real-shot fit is claimed."""
import unittest
import numpy as np

from rinkfit_start.camera import (
    Brown, CameraDomainError, CameraTrajectory, FixedHead, GaugeFixedSpline,
    Intrinsics, PhysicalCamera, Pose, ProjectionResult, ProjectionStatus,
    RSIntervalCertificate, SingularRSRoot, UniformCubicBasis, UniformCubicSpline,
    ZoomOptics, implicit_jacobian, project, rolling_shutter_project, row_time,
)
from rinkfit_start.geometry import (
    BoardSplineMaterial, BoardSurface, CircleArc, FiniteLine, GeometryDomainError,
    PhysicalPoint, ProfileQuery, ProfileStatus, RoundedRink, UnsupportedGeometryModel,
    find_profile_intersections, ordinary_curve_residual, physical_point_residual,
    profile_auxiliary, verified_order_inequalities,
)


class PhysicalModelTests(unittest.TestCase):
    def test_fixed_mechanism_reference_is_stationary_and_center_can_move(self):
        head = FixedHead(np.eye(3), [0, 1, 0], [1, 0, 0], [2, 3, -4],
                         [.2, -.1, .3], a=.12, pupil_linear=.08)
        centers = []
        for alpha, beta, z in ((0., 0., 0.), (.5, -.2, .3), (-.3, .4, -.1)):
            pose = head.pose(alpha, beta, z)
            h = np.cross(head.n, pose.R.T @ head.b)
            h /= np.linalg.norm(h)
            recovered_P = pose.C - pose.R.T @ (head.d + [0, 0, head.pupil(z)]) - head.a*h
            np.testing.assert_allclose(recovered_P, head.P, atol=1e-14)
            np.testing.assert_allclose(pose.R.T @ pose.R, np.eye(3), atol=1e-14)
            centers.append(pose.C)
        self.assertGreater(np.linalg.norm(centers[1]-centers[0]), .01)
        # At intersecting axes with a fixed pupil, the same physical reference
        # point always has camera coordinates -d despite rotation.
        intersecting = FixedHead(np.eye(3), [0, 1, 0], [1, 0, 0], head.P, head.d)
        for angle in (-.3, 0., .5):
            p = intersecting.pose(angle, -.2*angle, 0.)
            np.testing.assert_allclose(p.R @ (head.P-p.C), -head.d, atol=1e-14)

    def test_exact_spline_function_gauge_and_native_boundary_coverage(self):
        basis = UniformCubicBasis.for_native_frames(8, max_abs_readout=.02)
        controls = np.linspace(-.3, .5, basis.n_controls-1)
        angle = GaugeFixedSpline(basis, controls, reference_time=.047)
        self.assertAlmostEqual(angle(.047), 0., places=15)
        np.testing.assert_allclose(angle.jacobian_free(.047), 0., atol=1e-15)
        self.assertNotEqual(angle.controls[angle.dependent_index], 0.)
        constant = UniformCubicSpline(basis, np.ones(basis.n_controls)*2.5)
        for t in (basis.domain[0], -.01, 0., .14, .15, basis.domain[1]):
            self.assertAlmostEqual(constant(t), 2.5, places=13)
        with self.assertRaises(CameraDomainError):
            constant(basis.domain[0]-1e-6)
        with self.assertRaises(CameraDomainError):
            basis.require_coverage(0., .14, .2)

    def test_full_camera_analytic_time_velocity_matches_finite_difference(self):
        basis = UniformCubicBasis.for_native_frames(8, max_abs_readout=.02)
        free = np.linspace(-.01, .015, basis.n_controls-1)
        traj = CameraTrajectory(GaugeFixedSpline(basis, free, .04),
                                GaugeFixedSpline(basis, -.5*free, .04),
                                UniformCubicSpline(basis, np.log(900)+np.linspace(-.03, .02, basis.n_controls)))
        head = FixedHead(np.eye(3), [0, 1, 0], [1, 0, 0], [0, 0, -3],
                         [.03, -.02, .04], a=.01, pupil_linear=.02, pupil_quadratic=.003)
        optics = ZoomOptics(900, principal_linear=(1.2, -.8), k1=(.004, .001),
                            k2=(-.0001, .0001), p1=.0001, p2=-.0002)
        cam = PhysicalCamera(head, optics, traj)
        X, t, h = np.array([.2, .1, 0.]), .073, 1e-6
        fd = (cam.project(X,t+h).require_uv()-cam.project(X,t-h).require_uv())/(2*h)
        np.testing.assert_allclose(cam.time_derivative(X,t), fd, rtol=1e-7, atol=2e-7)
        self.assertEqual(cam.project(X, -1.).status, ProjectionStatus.TIME_DOMAIN_INVALID)

    def test_rs_self_consistency_and_implicit_jacobian_after_resolving_root(self):
        velocity = np.array([60., 90.])
        frame_time, height = .2, 1080
        certificate = RSIntervalCertificate((.17,.23),90.,"analytic affine projection: derivative exactly 90 px/s")
        def solve(theta):
            a,b,tau = theta
            def projection(t):
                return ProjectionResult(ProjectionStatus.VALID, [120+a+60*t,430+b+90*t], 1.)
            return rolling_shutter_project(projection,lambda t:velocity,frame_time=frame_time,
                                           readout_time=tau,image_height=height,certificate=certificate)
        theta=np.array([.3,-.2,.012])
        result=solve(theta)
        self.assertTrue(result.valid)
        root=result.roots[0]
        self.assertLess(abs(root.equation_error_px),1e-9)
        self.assertAlmostEqual(root.time,row_time(frame_time,theta[2],result.uv[1],height),places=14)
        J=implicit_jacobian(np.array([[1.,0.,0.],[0.,1.,0.]]),velocity,theta[2],height,
                            [0.,0.,result.uv[1]/(height-1)-.5])
        fd=np.empty((2,3))
        for j in range(3):
            step=np.zeros(3);step[j]=1e-6
            fd[:,j]=(solve(theta+step).require_uv()-solve(theta-step).require_uv())/(2e-6)
        np.testing.assert_allclose(J,fd,rtol=2e-7,atol=3e-8)
        self.assertTrue(solve(np.array([.3,-.2,-.012])).valid)

    def test_rs_no_root_multiple_root_singular_and_uncertified_are_distinct(self):
        def response(y):
            return ProjectionResult(ProjectionStatus.VALID,[50.,y],1.)
        no_root=rolling_shutter_project(lambda t:response(200.),lambda t:np.zeros(2),
                                        frame_time=.5,readout_time=1.,image_height=101,
                                        certificate=RSIntervalCertificate((0.,1.),0.,"constant projection"))
        self.assertEqual(no_root.status,ProjectionStatus.NO_ROOT)
        self.assertIsNone(no_root.uv)
        multiple=rolling_shutter_project(lambda t:response(100*t-(100*t-20)*(100*t-70)/100),
                                         lambda t:np.array([0.,190.-200*t]),frame_time=.5,
                                         readout_time=1.,image_height=101)
        self.assertEqual(multiple.status,ProjectionStatus.MULTIPLE_ROOTS)
        np.testing.assert_allclose([r.row for r in multiple.roots],[20.,70.],atol=1e-7)
        singular=rolling_shutter_project(lambda t:response(100*t-(100*t-30)**2/100),
                                         lambda t:np.array([0.,160.-200*t]),frame_time=.5,
                                         readout_time=1.,image_height=101)
        self.assertEqual(singular.status,ProjectionStatus.SINGULAR_ROOT)
        with self.assertRaises(SingularRSRoot):
            implicit_jacobian([0.,1.],[0.,100.],1.,image_height=101)
        uncertified=rolling_shutter_project(lambda t:response(50.+t),lambda t:np.array([0.,1.]),
                                            frame_time=.5,readout_time=.01,image_height=101)
        self.assertEqual(uncertified.status,ProjectionStatus.UNIQUENESS_UNCERTIFIED)
        self.assertIsNotNone(uncertified.uv)
        with self.assertRaises(CameraDomainError):
            uncertified.require_uv()

    def test_projection_domains_never_produce_valid_zero_pixels(self):
        pose=Pose(np.eye(3),np.zeros(3)); K=Intrinsics(900,900,959.5,539.5)
        behind=project([0,0,-1],pose,K,Brown())
        self.assertEqual(behind.status,ProjectionStatus.NON_POSITIVE_DEPTH)
        self.assertIsNone(behind.uv)
        uncertified=project([0,0,1],pose,K,Brown(k1=-1.,radius_max=2.))
        self.assertEqual(uncertified.status,ProjectionStatus.DISTORTION_UNCERTIFIED)
        outside=project([3,0,1],pose,K,Brown(radius_max=2.))
        self.assertEqual(outside.status,ProjectionStatus.DISTORTION_DOMAIN_INVALID)


class FiniteGeometryTests(unittest.TestCase):
    def test_finite_endpoint_retains_tangential_coverage_error(self):
        line=FiniteLine([0,0,0],[1,0,0])
        station=line.closest_station([2,.1,0])
        e=ordinary_curve_residual(lambda X:X[:2],line,station,[2,.1])
        self.assertEqual(station,1.)
        self.assertAlmostEqual(float(e@e),1.01)
        with self.assertRaises(GeometryDomainError):
            line.point(1.001)
        point=PhysicalPoint("confirmed_right_endpoint",[1,0,0])
        np.testing.assert_allclose(physical_point_residual(lambda X:X[:2],point,[2,.1]),e)
        arc=CircleArc([0,0],1.,0.,np.pi/2)
        with self.assertRaises(GeometryDomainError):
            arc.point(np.pi)

    def test_closed_rink_tangency_and_material_across_seam(self):
        boundary=RoundedRink(width=.52,radius_x=.12,radius_y=.09)
        np.testing.assert_allclose(boundary.point(0.),boundary.point(8.),atol=1e-15)
        for join in range(1,8):
            np.testing.assert_allclose(boundary.point(join-1e-8),boundary.point(join+1e-8),atol=2e-8)
            np.testing.assert_allclose(boundary.unit_tangent(join-1e-8),boundary.unit_tangent(join+1e-8),atol=3e-8)
        np.testing.assert_allclose(boundary.unit_tangent(8.-1e-8),boundary.unit_tangent(1e-8),atol=3e-8)
        board=BoardSurface(boundary,.2)
        logo=BoardSplineMaterial("shared_logo_outline",board,[[7.8,.1],[7.95,.12],[8.05,.14],[8.2,.1]],
                                 (7.7,8.3),"fixture: physical board interval spanning the seam")
        np.testing.assert_allclose(logo.point(.5-1e-7),logo.point(.5+1e-7),atol=1e-7)
        self.assertFalse(logo.controls_sh.flags.writeable)
        with self.assertRaises(UnsupportedGeometryModel):
            RoundedRink(exponent=1.9051596)

    def test_profile_zero_normal_error_does_not_hide_no_intersection(self):
        line=FiniteLine([0,0,0],[1,0,0])
        query=ProfileQuery([2,0],[1,0],[0,1],(-.5,.5),.4,2.,"fixture: actual vertical query through x=2")
        evaluation=profile_auxiliary(lambda X:X[:2],line,1.,0.,query)
        self.assertEqual(evaluation.budget_q,0.)
        self.assertEqual(evaluation.tangent_equality_px,-1.)
        np.testing.assert_allclose(evaluation.equality_xy,[-1.,0.])
        intersections=find_profile_intersections(lambda X:X[:2],line,query)
        self.assertEqual(intersections.status,ProfileStatus.NO_INTERSECTION)
        np.testing.assert_array_equal(verified_order_inequalities([.1,.1,.7],verified=True),[0.,-.6])
        with self.assertRaises(ValueError):
            verified_order_inequalities([.1,.7],verified=False)


if __name__ == "__main__":
    unittest.main()

# IMPLEMENTED TEST COVERAGE: mechanism invariance, analytic camera velocity,
# exact function gauge/boundary, actual implicit-root finite differences,
# root/status distinctions, finite endpoints, closed tangent seam and profile
# support failure.  NOT COVERED/NOT CLAIMED: real-shot fit, interval-proof
# generation, Ceres production replay, identifiability or final acceptance.
