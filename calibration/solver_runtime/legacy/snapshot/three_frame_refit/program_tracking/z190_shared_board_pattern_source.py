"""Shared, source-qualified print patterns on a finite zero-thickness board.

Nine shared metre coordinates describe Ad13's actual circular outer stroke
and Ad12's three fixed printed material points. The points have no imposed
symmetry or common upper height. Ad25/26 are exact horizontal half-turns of
the same unknowns. Source rectangles retain their original blur support;
pixel ellipse centres and silhouette extrema are never world point factors.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json

import cv2
import numpy as np
from scipy.optimize import brentq, minimize_scalar

try:
    from .projection import require_central_projection, world_rays
except ImportError:
    from projection import require_central_projection, world_rays


SHIELD_ROLES = ("upper_left_printed_outline_corner",
                "upper_right_printed_outline_corner", "bottom_printed_outline_tip")
PARAMETER_NAMES = ("Ad13_center_y_m", "Ad13_center_z_m", "Ad13_radius_m",
                   "Ad12_upper_left_y_m", "Ad12_upper_left_z_m",
                   "Ad12_upper_right_y_m", "Ad12_upper_right_z_m",
                   "Ad12_bottom_tip_y_m", "Ad12_bottom_tip_z_m")
PERIMETER_PARAMETER_NAMES = ("Ad13_center_course_station_m", "Ad13_center_z_m", "Ad13_radius_m",
    "Ad12_upper_left_station_m", "Ad12_upper_left_z_m", "Ad12_upper_right_station_m",
    "Ad12_upper_right_z_m", "Ad12_bottom_tip_station_m", "Ad12_bottom_tip_z_m")


def fixed_pivot_jacobian(camera_state11_jacobian, camera25, lever_xyz_m, scene_unit_m):
    """Chain exact existing optical-centre columns to fixed XYZ/R3/logf.

    No independent per-frame optical-centre translation is introduced. The
    shared nonzero lever remains fixed, while C=P+R.T@lever changes with R.
    Returns XYZ columns per metre and the four actual R3/log-native-f columns.
    """
    j = np.asarray(camera_state11_jacobian, float)
    c = np.asarray(camera25, float); lever = np.asarray(lever_xyz_m, float)
    unit = float(scene_unit_m)
    if (j.ndim != 2 or j.shape[1] != 11 or c.shape != (25,) or lever.shape != (3,) or
        not np.isfinite(unit) or unit <= 0 or not np.isfinite(j).all() or
        not np.isfinite(c).all() or not np.isfinite(lever).all()):
        raise ValueError("Actual state11 Jacobian, camera25, fixed metre lever and scale required")
    _, raw = cv2.Rodrigues(c[3:6]); dr = raw.reshape(3, 3, 3)
    centre_rotation = np.column_stack([d.T@lever for d in dr])/(100*unit)
    return j[:, :3]/(100*unit), np.c_[j[:, 3:6]+j[:, :3]@centre_rotation, j[:, 6]+j[:, 7]]


def _q9(value):
    q = np.asarray(value, float)
    if q.shape != (9,) or not np.isfinite(q).all():
        raise ValueError("Nine finite shared board-pattern metre coordinates required")
    return q


def _project(camera25, world_scene_units, with_jacobian=False):
    c = np.asarray(camera25, float)
    world = np.asarray(world_scene_units, float).reshape(-1, 3)
    result = require_central_projection(c, world, stage="shared_board_pattern",
        family="qualified_native_print_pattern", source_keys=(),
        with_jacobian=with_jacobian)
    if not with_jacobian:
        return result[0], None, None
    rotation = cv2.Rodrigues(c[3:6])[0]
    intrinsic = np.array([[np.exp(c[6]), 0., c[8]],
                          [0., np.exp(c[7]), c[9]], [0., 0., 1.]])
    _, raw = cv2.projectPoints(world, c[3:6], -rotation@c[:3], intrinsic, c[11:25])
    world_jac = (raw[:, 3:6]@rotation).reshape(-1, 2, 3)
    world_jac[:, 0] += c[10]/np.exp(c[7])*world_jac[:, 1]
    return result[0], result[2].reshape(-1, 2, 11), world_jac


class SharedBoardPatternGeometry:
    """Unknown shared patterns on the actual named left-short straight face.

    q9 is metric, while scene/camera25 use released scene units. Facet identity
    is an explicit caller authority; the plane is not chosen by projection.
    """
    def __init__(self, scene, scene_unit_m, *, facet_authority):
        if not facet_authority:
            raise ValueError("Actual source-to-left-short facet authority required")
        self.scene_unit_m = float(scene_unit_m)
        if not np.isfinite(self.scene_unit_m) or self.scene_unit_m <= 0:
            raise ValueError("Verified positive scene metre scale required")
        self.facet_authority = copy.deepcopy(facet_authority)
        bottom = np.asarray(scene["ice_boundary"]["points_xyz"], float)
        top = np.asarray(scene["board_surface"]["top_edge_xyz"], float)
        if bottom.ndim != 2 or bottom.shape[1] != 3 or not np.isfinite(bottom).all():
            raise ValueError("Actual finite board geometry required")
        x = float(bottom[:, 0].min())
        straight = bottom[np.isclose(bottom[:, 0], x, atol=1e-7, rtol=0)]
        if len(straight) < 2:
            raise ValueError("Named left-short straight plane absent")
        self.x_m = x*self.scene_unit_m
        self.y_domain_m = np.array([straight[:, 1].min(), straight[:, 1].max()])*self.scene_unit_m
        self.z_domain_m = np.array([np.median(bottom[:, 2]), np.max(top[:, 2])])*self.scene_unit_m
        if np.any(self.y_domain_m[1:] <= self.y_domain_m[:1]) or self.z_domain_m[1] <= self.z_domain_m[0]:
            raise ValueError("Positive finite board-face domain required")

    def guard(self, q9):
        q = _q9(q9); yc, zc, radius = q[:3]
        points = q[3:].reshape(3, 2)
        if radius <= 0:
            raise ValueError("Shared printed-circle radius must be positive")
        lows = np.r_[yc-radius, zc-radius]
        highs = np.r_[yc+radius, zc+radius]
        face_lows = np.r_[self.y_domain_m[0], self.z_domain_m[0]]
        face_highs = np.r_[self.y_domain_m[1], self.z_domain_m[1]]
        if np.any(lows < face_lows) or np.any(highs > face_highs):
            raise ValueError("Entire printed circle must be inside its finite board face")
        if np.any(points < face_lows) or np.any(points > face_highs):
            raise ValueError("Printed shield material point outside its finite board face")
        twice_area = np.linalg.det(np.stack((points[1]-points[0], points[2]-points[0])))
        if abs(twice_area) <= np.finfo(float).eps:
            raise ValueError("Three distinct, noncollinear printed shield points required")
        return True

    def circle(self, q9, theta, *, ad_id=13):
        q = _q9(q9); theta = np.asarray(theta, float).reshape(-1)
        if ad_id not in (13, 26) or not np.isfinite(theta).all():
            raise ValueError("Circle is the shared Ad13/26 material, at finite angles")
        cos, sin = np.cos(theta), np.sin(theta)
        world_m = np.column_stack((np.full(len(theta), self.x_m),
                                  q[0]+q[2]*cos, q[1]+q[2]*sin))
        jac_m = np.zeros((len(theta), 3, 9))
        jac_m[:, 1, 0] = 1.; jac_m[:, 2, 1] = 1.
        jac_m[:, 1, 2] = cos; jac_m[:, 2, 2] = sin
        if ad_id == 26:
            world_m[:, :2] *= -1.; jac_m[:, :2] *= -1.
        return world_m/self.scene_unit_m, jac_m/self.scene_unit_m

    def shield(self, q9, *, ad_id=12):
        q = _q9(q9)
        if ad_id not in (12, 25):
            raise ValueError("Shield points are the shared Ad12/25 printed material")
        points = q[3:].reshape(3, 2)
        world_m = np.column_stack((np.full(3, self.x_m), points))
        jac_m = np.zeros((3, 3, 9))
        for i in range(3):
            jac_m[i, 1, 3+2*i] = 1.; jac_m[i, 2, 4+2*i] = 1.
        if ad_id == 25:
            world_m[:, :2] *= -1.; jac_m[:, :2] *= -1.
        return world_m/self.scene_unit_m, jac_m/self.scene_unit_m

    def descriptor(self, q9, *, engineering_circle_samples=256):
        """The same solved entities for rendering; no unseen shield outline."""
        self.guard(q9)
        if engineering_circle_samples < 16:
            raise ValueError("Circle render sample count is an explicit engineering discretization")
        theta = np.linspace(0., 2*np.pi, engineering_circle_samples+1)
        circles = {str(ad): self.circle(q9, theta, ad_id=ad)[0].tolist() for ad in (13, 26)}
        shield = {str(ad): dict(zip(SHIELD_ROLES, self.shield(q9, ad_id=ad)[0].tolist()))
                  for ad in (12, 25)}
        return dict(schema="rinkspace.shared-native-board-pattern-candidate.v1",
            parameter_names=list(PARAMETER_NAMES), shared_parameters_m=_q9(q9).tolist(),
            facet="left_short_side", facet_authority=copy.deepcopy(self.facet_authority),
            plane_x_m=self.x_m, finite_y_domain_m=self.y_domain_m.tolist(),
            finite_z_domain_m=self.z_domain_m.tolist(),
            circle_outer_stroke_paths_xyz_scene_units=circles,
            shield_fixed_material_points_xyz_scene_units=shield,
            circle_samples_are_engineering_not_source_observations=True,
            shield_symmetry_or_equal_upper_heights_imposed=False,
            unobserved_shield_outline_created=False,
            mirror_rule="Exact horizontal 180-degree rotation of Ad13/12 into Ad26/25, unchanged Z",
            unknown_geometry_is_shared_estimate_not_world_measurement=True)

    def seed(self, camera_by_actual, circle_frames, shield_frames):
        """Ray-plane intersections of true source boundaries, used only as seed.

        Invert every visible ring locator onto the declared finite plane and
        fit a WORLD-plane circle. Never invert a pixel conic/ellipse centre.
        This initialization is neither a source measurement nor a new prior.
        """
        circle_yz = []; shield_yz = [[] for _ in SHIELD_ROLES]
        def intersect(frame, uv):
            camera = np.asarray(camera_by_actual[int(frame["source_actual_pts"])], float)
            rays, good = world_rays(camera, np.asarray(uv, float))
            distance = (self.x_m/self.scene_unit_m-camera[0])/rays[:, 0]
            xyz = camera[:3]+distance[:, None]*rays
            if not np.all(good & np.isfinite(xyz).all(axis=1) & (distance > 0)):
                raise ValueError("Pattern initialization ray cannot reach the named board face")
            return xyz[:, 1:]*self.scene_unit_m
        for frame in circle_frames:
            if frame["split"] != "fit": continue
            circle_yz.extend(intersect(frame, [r["uv"] for r in frame["source_curve_support"]]))
        yz = np.asarray(circle_yz, float)
        if len(yz) < 3: raise ValueError("Qualified development ring boundaries required for seed")
        lhs = np.column_stack((2*yz, np.ones(len(yz))))
        fit = np.linalg.lstsq(lhs, np.sum(yz**2, axis=1), rcond=None)[0]
        radius2 = float(fit[2]+fit[:2]@fit[:2])
        if radius2 <= 0: raise ValueError("Positive initial world-circle radius required")
        for frame in shield_frames:
            if frame["split"] != "fit": continue
            by_role = {r["physical_pattern_point_role"]: r for r in frame["source_pattern_points"]}
            values = intersect(frame, [by_role[role]["uv"] for role in SHIELD_ROLES])
            for i, value in enumerate(values): shield_yz[i].append(value)
        if any(not values for values in shield_yz): raise ValueError("Three development material identities required")
        q = np.r_[fit[:2], np.sqrt(radius2), np.asarray([np.mean(v, axis=0) for v in shield_yz]).ravel()]
        return q, dict(initialization_only=True, new_world_measurement_or_prior=False,
            source_held_not_used=True, pixel_ellipse_centre_not_used=True,
            ring_source_boundary_intersections=len(yz),
            shield_source_point_intersections=[len(v) for v in shield_yz])


class SharedPerimeterBoardPatternGeometry(SharedBoardPatternGeometry):
    """Opt-in fixed left-short -> lower-left zero-thickness board course.

    Nine unknowns are station/Z/r and three station/Z material points. The
    underlying board polyline is exactly the existing scene, including its
    rounded corner. The print circle is circular in the board's unfolded
    station-Z chart. There is no extra board thickness or per-frame facet
    reassignment, no unobserved shield template, and no scene-origin change.
    """
    def __init__(self, scene, scene_unit_m, *, facet_authority):
        super().__init__(scene, scene_unit_m, facet_authority=facet_authority)
        boundary = np.asarray(scene["ice_boundary"]["points_xyz"], float)
        if np.linalg.norm(boundary[-1]-boundary[0]) < 1e-9: boundary = boundary[:-1]
        left = np.flatnonzero(np.isclose(boundary[:, 0], boundary[:, 0].min(), rtol=0, atol=1e-7))
        lower = np.flatnonzero(np.isclose(boundary[:, 1], boundary[:, 1].min(), rtol=0, atol=1e-7))
        if len(left) < 2 or len(lower) < 2 or left[-1] >= lower[0]:
            raise ValueError("Actual continuous left-short then lower-left board course required")
        self.course_source_boundary_indices = [int(left[0]), int(lower[0])]
        self.course_xy_m = boundary[left[0]:lower[0]+1, :2]*self.scene_unit_m
        edges = np.diff(self.course_xy_m, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        if np.any(lengths <= 0): raise ValueError("Actual board course cannot contain zero-length edges")
        self.course_station_m = np.r_[0., np.cumsum(lengths)]
        self.course_tangents = edges/lengths[:, None]
        self.station_domain_m = np.array([0., self.course_station_m[-1]])

    def _arc(self, station):
        s = np.asarray(station, float).reshape(-1)
        if not np.isfinite(s).all() or np.any(s < 0) or np.any(s > self.station_domain_m[1]):
            raise ValueError("Pattern station outside the source-qualified finite board course")
        index = np.clip(np.searchsorted(self.course_station_m, s, side="right")-1,
                        0, len(self.course_tangents)-1)
        xy = self.course_xy_m[index]+(s-self.course_station_m[index])[:, None]*self.course_tangents[index]
        return xy, self.course_tangents[index]

    def guard(self, q9):
        q = _q9(q9); sc, zc, radius = q[:3]; points = q[3:].reshape(3, 2)
        if radius <= 0: raise ValueError("Shared unfolded printed-circle radius must be positive")
        if sc-radius < 0 or sc+radius > self.station_domain_m[1]:
            raise ValueError("Entire printed circle must remain on the fixed finite board course")
        if zc-radius < self.z_domain_m[0] or zc+radius > self.z_domain_m[1]:
            raise ValueError("Entire printed circle must be inside the existing board height")
        if (np.any(points[:, 0] < 0) or np.any(points[:, 0] > self.station_domain_m[1]) or
            np.any(points[:, 1] < self.z_domain_m[0]) or np.any(points[:, 1] > self.z_domain_m[1])):
            raise ValueError("Shared printed shield material point outside its finite board course")
        area = np.linalg.det(np.stack((points[1]-points[0], points[2]-points[0])))
        if abs(area) <= np.finfo(float).eps:
            raise ValueError("Three noncollinear printed shield material points required")
        return True

    def circle(self, q9, theta, *, ad_id=13):
        q = _q9(q9); theta = np.asarray(theta, float).reshape(-1)
        if ad_id not in (13, 26) or not np.isfinite(theta).all():
            raise ValueError("Circle is the shared Ad13/26 material, at finite angles")
        cos, sin = np.cos(theta), np.sin(theta)
        xy, tangent = self._arc(q[0]+q[2]*cos)
        world_m = np.column_stack((xy, q[1]+q[2]*sin))
        jac_m = np.zeros((len(theta), 3, 9))
        jac_m[:, :2, 0] = tangent; jac_m[:, 2, 1] = 1.
        jac_m[:, :2, 2] = tangent*cos[:, None]; jac_m[:, 2, 2] = sin
        if ad_id == 26: world_m[:, :2] *= -1.; jac_m[:, :2] *= -1.
        return world_m/self.scene_unit_m, jac_m/self.scene_unit_m

    def shield(self, q9, *, ad_id=12):
        q = _q9(q9)
        if ad_id not in (12, 25): raise ValueError("Shield is the shared Ad12/25 printed material")
        local = q[3:].reshape(3, 2); xy, tangent = self._arc(local[:, 0])
        world_m = np.column_stack((xy, local[:, 1])); jac_m = np.zeros((3, 3, 9))
        for i in range(3):
            jac_m[i, :2, 3+2*i] = tangent[i]; jac_m[i, 2, 4+2*i] = 1.
        if ad_id == 25: world_m[:, :2] *= -1.; jac_m[:, :2] *= -1.
        return world_m/self.scene_unit_m, jac_m/self.scene_unit_m

    def descriptor(self, q9, *, engineering_circle_samples=256):
        result = super().descriptor(q9, engineering_circle_samples=engineering_circle_samples)
        result.update(parameter_names=list(PERIMETER_PARAMETER_NAMES),
            facet="source-qualified continuous left_short_side and lower_left course",
            geometry_chart="Actual existing zero-thickness board perimeter station-Z",
            finite_station_domain_m=self.station_domain_m.tolist(),
            course_boundary_indices=self.course_source_boundary_indices,
            course_xy_m=self.course_xy_m.tolist(), course_station_m=self.course_station_m.tolist(),
            circle_is_circular_in_unfolded_station_Z_not_pixel_ellipse=True,
            course_chart_is_engineering_parameterization_not_new_material_measurement=True)
        result.pop("plane_x_m", None); result.pop("finite_y_domain_m", None)
        return result

    def _intersect_source_rays(self, camera25, uv):
        camera = np.asarray(camera25, float)
        rays, good = world_rays(camera, np.asarray(uv, float))
        centre = camera[:3]*self.scene_unit_m
        local = []; branches = []
        for ray, valid in zip(rays, good):
            hits = []
            if valid:
                for i, (start, end) in enumerate(zip(self.course_xy_m[:-1], self.course_xy_m[1:])):
                    edge = end-start; matrix = np.column_stack((ray[:2], -edge))
                    if abs(np.linalg.det(matrix)) <= 1e-12: continue
                    distance, fraction = np.linalg.solve(matrix, start-centre[:2])
                    if distance > 0 and -1e-10 <= fraction <= 1+1e-10:
                        fraction = np.clip(fraction, 0., 1.)
                        hits.append((float(distance), self.course_station_m[i]+fraction*np.linalg.norm(edge),
                                     centre[2]+distance*ray[2], i))
            if not hits: raise ValueError("True source ray has no positive hit on the declared board course")
            hit = min(hits)
            local.append(hit[1:3]); branches.append(hit[3])
        return np.asarray(local), branches

    def seed(self, camera_by_actual, circle_frames, shield_frames):
        circle = []; shield = [[] for _ in SHIELD_ROLES]; ledger = []
        for frame in circle_frames:
            if frame["split"] != "fit": continue
            camera = camera_by_actual[int(frame["source_actual_pts"])]
            points, branches = self._intersect_source_rays(camera, [r["uv"] for r in frame["source_curve_support"]])
            circle.extend(points); ledger.append(dict(ad=13, pts=frame["source_actual_pts"], course_segment_indices=branches))
        points = np.asarray(circle)
        if len(points) < 3: raise ValueError("Development ring source boundaries required")
        fit = np.linalg.lstsq(np.c_[2*points, np.ones(len(points))], np.sum(points**2, axis=1), rcond=None)[0]
        radius2 = float(fit[2]+fit[:2]@fit[:2])
        if radius2 <= 0: raise ValueError("Positive conditional chart-circle seed radius required")
        for frame in shield_frames:
            if frame["split"] != "fit": continue
            camera = camera_by_actual[int(frame["source_actual_pts"])]
            by_role = {r["physical_pattern_point_role"]: r for r in frame["source_pattern_points"]}
            points, branches = self._intersect_source_rays(camera, [by_role[r]["uv"] for r in SHIELD_ROLES])
            for i, point in enumerate(points): shield[i].append(point)
            ledger.append(dict(ad=12, pts=frame["source_actual_pts"], course_segment_indices=branches))
        if any(not p for p in shield): raise ValueError("Three development printed material points required")
        q = np.r_[fit[:2], np.sqrt(radius2), np.asarray([np.mean(p, axis=0) for p in shield]).ravel()]
        return q, dict(initialization_only=True, new_world_measurement_or_prior=False,
            source_held_not_used=True, pixel_ellipse_centre_not_used=True,
            native_source_rays_intersect_actual_finite_course=True,
            source_entities_not_relabelled_per_frame=True,
            conditional_initialization_intersection_ledger=ledger)


def admit_pattern_source(path, aliases, *, expected_ad_id, expected_observer,
                         expected_sha256=None):
    """Verify frozen true pixels/roles; leave every frame observation intact."""
    path = Path(path); data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("Frozen pattern source changed")
    source = json.loads(data.decode("utf8"))
    if (source.get("schema") != "rinkspace.native-advert-pattern-source.v1" or
        source.get("physical_ad_id") != expected_ad_id or
        source.get("observer_installation") != expected_observer or
        source.get("native_size_wh") != [1920, 1080] or
        source.get("source_pts_timebase") != [1, 15360]):
        raise ValueError("Qualified same-event pattern/observer/native identity required")
    try:
        from .z190_continuous_source_bundle import SOURCE_SHA
    except ImportError:
        from z190_continuous_source_bundle import SOURCE_SHA
    if source.get("source_video_sha256") != SOURCE_SHA:
        raise ValueError("Same-event source video authority mismatch")
    if (source.get("model_projection_used_to_generate_or_select_uv") is not False or
        source.get("frozen_before_any_fit_using_these_observations") is not True):
        raise ValueError("Pattern source must be frozen independently of projection/fit")
    frames = copy.deepcopy(source["frames"]); checked = {}; fit_actual = set(); held_actual = set()
    for frame in frames:
        native, actual = int(frame["source_native_pts"]), int(frame["source_actual_pts"])
        if int(aliases[native]) != actual or frame["split"] not in ("fit", "held"):
            raise ValueError("Pattern source actual alias/split mismatch")
        (fit_actual if frame["split"] == "fit" else held_actual).add(actual)
        if frame.get("model_projection_used_to_generate_or_select_uv") is not False or frame.get("frozen_before_fit") is not True:
            raise ValueError("Frame pattern pixels must precede fitting")
        sigma = np.asarray(frame["engineering_sigma_native_px"], float)
        if sigma.shape != (2,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("Original explicit positive native engineering sigma required")
        raw = Path(frame["raw_png"])
        if str(raw) not in checked:
            if hashlib.sha256(raw.read_bytes()).hexdigest() != frame["raw_png_sha256"]:
                raise ValueError("Native pattern PNG changed")
            pixels = cv2.imread(str(raw))
            if pixels is None or pixels.shape != (1080, 1920, 3) or hashlib.sha256(pixels.tobytes()).hexdigest() != frame["native_bgr_sha256"]:
                raise ValueError("Native pattern source pixels changed")
            checked[str(raw)] = frame["raw_png_sha256"]
        rows = frame["source_curve_support"] if expected_ad_id in (13, 26) else frame["source_pattern_points"]
        for row in rows:
            uv = np.asarray(row["uv"], float); bounds = np.asarray(row["native_support_bounds_xy_px"], float)
            if (uv.shape != (2,) or bounds.shape != (2, 2) or not np.isfinite(np.r_[uv, bounds.ravel()]).all() or
                np.any(bounds[:, 0] > bounds[:, 1]) or np.any(uv < bounds[:, 0]) or np.any(uv > bounds[:, 1]) or
                row.get("source_visible") is not True or row.get("source_support_manually_qualified") is not True):
                raise ValueError("True visible frozen pattern support required")
        if expected_ad_id in (12, 25) and set(r["physical_pattern_point_role"] for r in rows) != set(SHIELD_ROLES):
            raise ValueError("Exactly the three confirmed shield material roles required")
    user_actual = int(aliases[int(source["user_exact_native_pts"])])
    if fit_actual & held_actual or user_actual in fit_actual:
        raise ValueError("Held/user actual aliases cannot become pattern fit")
    return frames, dict(path=str(path), sha256=sha, fit_actual=sorted(fit_actual),
        held_actual=sorted(held_actual), original_pixels_sigma_roles_exact=True,
        raw_native_sources_verified=len(checked), ellipse_center_or_silhouette_extrema_used_as_worldpoint=False)


class SharedBoardPatternSourceFactor:
    """Frozen source intervals, same unknown world material across all frames.

    Circle boxes have one scalar: minimum sigma-scaled Euclidean excess of
    the true projected circle outside the box. The closest curve coordinate
    is profiled, never a new observed pixel/world correspondence. Its scalar
    derivative follows the envelope theorem. Shield fixed points keep two
    signed interval excesses. Camera J uses projection.to_state's 11 columns;
    the caller chains its fixed-pivot/nonzero-lever R/f parameterization.
    """
    def __init__(self, geometry, *, engineering_circle_samples=256):
        self.geometry = geometry
        if engineering_circle_samples < 64:
            raise ValueError("Explicit engineering curve search sampling >=64 required")
        self.samples = int(engineering_circle_samples)
        self.theta = np.linspace(0., 2*np.pi, self.samples, endpoint=False)

    def evaluate(self, q9, camera25, frame, with_jacobian=True):
        self.geometry.guard(q9)
        sigma = np.asarray(frame["engineering_sigma_native_px"], float)
        if (frame.get("split") not in ("fit", "held") or sigma.shape != (2,) or
            not np.isfinite(sigma).all() or np.any(sigma <= 0) or
            frame.get("model_projection_used_to_generate_or_select_uv") is not False or
            frame.get("frozen_before_fit") is not True):
            raise ValueError("Frozen native pattern role/positive original sigma/source-only authority required")
        key = "source_curve_support" if frame["physical_ad_id"] in (13, 26) else "source_pattern_points"
        for source in frame[key]:
            bounds = np.asarray(source["native_support_bounds_xy_px"], float)
            if (source.get("source_visible") is not True or source.get("source_support_manually_qualified") is not True or
                bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1])):
                raise ValueError("Each source support must be genuinely visible and independently qualified")
        if frame["physical_ad_id"] in (13, 26):
            return self._circle(q9, camera25, frame, with_jacobian)
        if frame["physical_ad_id"] in (12, 25):
            return self._shield(q9, camera25, frame, with_jacobian)
        raise ValueError("Only explicitly shared Ad12/13/25/26 pattern identities supported")

    def _circle(self, q9, camera25, frame, jac):
        ad = int(frame["physical_ad_id"]); sigma = np.asarray(frame["engineering_sigma_native_px"], float)
        world, _ = self.geometry.circle(q9, self.theta, ad_id=ad)
        coarse, _, _ = _project(camera25, world)
        residual = []; q_jac = []; camera_jac = []; reports = []
        width = 2*np.pi/self.samples
        for source in frame["source_curve_support"]:
            bounds = np.asarray(source["native_support_bounds_xy_px"], float)
            def point(theta, derivative=False):
                w, jw = self.geometry.circle(q9, [theta], ad_id=ad)
                uv, jc, jp = _project(camera25, w, derivative)
                excess = (uv[0]-np.clip(uv[0], bounds[:, 0], bounds[:, 1]))/sigma
                return excess, uv[0], jc, jp, jw
            excess_coarse = (coarse-np.clip(coarse, bounds[:, 0], bounds[:, 1]))/sigma
            costs = np.sum(excess_coarse**2, axis=1); best = int(np.argmin(costs))
            theta = float(self.theta[best]); cost = float(costs[best])
            if cost > 0:
                local = np.flatnonzero((costs <= np.roll(costs, 1)) & (costs <= np.roll(costs, -1)))
                order = local[np.argsort(costs[local])][:4]
                for i in order:
                    centre = float(self.theta[i])
                    def cost_at(t):
                        e = point(t)[0]
                        return float(e@e)
                    fit = minimize_scalar(cost_at,
                        bounds=(centre-width, centre+width), method="bounded",
                        options={"xatol": 1e-12})
                    # Bounded minimization has a sqrt(eps)*|theta| position
                    # tolerance. Refine the same profiled curve coordinate
                    # with its exact objective derivative for stable source FD.
                    def slope_at(t):
                        e, _, _, jp, jw = point(t, True)
                        d_world = -q9[2]*np.sin(t)*jw[0, :, 0]+q9[2]*np.cos(t)*jw[0, :, 1]
                        return float(2*(e/sigma)@(jp[0]@d_world))
                    low, high = centre-width, centre+width
                    if slope_at(low) < 0 < slope_at(high):
                        refined = brentq(slope_at, low, high, xtol=1e-14)
                        refined_cost = cost_at(refined)
                        if refined_cost <= fit.fun+1e-12:
                            fit.x, fit.fun = refined, refined_cost
                    if fit.fun < cost: theta, cost = float(fit.x), float(fit.fun)
            excess, uv, jc, jp, jw = point(theta, jac)
            r = float(np.linalg.norm(excess)); residual.append(r)
            if jac:
                active = ((uv < bounds[:, 0]) | (uv > bounds[:, 1])).astype(float)
                chain = np.zeros(2) if r == 0 else excess/r*active/sigma
                q_jac.append(chain@(jp[0]@jw[0])); camera_jac.append(chain@jc[0])
            reports.append(dict(source_id=source["id"], curve_theta_profiled=theta,
                closest_projected_uv_native_px=uv.tolist(), outside_support_native_px=(excess*sigma).tolist(),
                native_sigma_scaled_excess=r, profiled_angle_is_not_source_correspondence=True))
        return SimpleNamespace(residual=np.asarray(residual),
            shared_q9_jacobian=np.asarray(q_jac).reshape(-1, 9) if jac else None,
            camera_state11_jacobian=np.asarray(camera_jac).reshape(-1, 11) if jac else None,
            source_reports=reports, source_held=frame["split"] == "held",
            source_extrema_or_pixel_ellipse_center_used=False)

    def _shield(self, q9, camera25, frame, jac):
        sigma = np.asarray(frame["engineering_sigma_native_px"], float)
        world, jw = self.geometry.shield(q9, ad_id=int(frame["physical_ad_id"]))
        uv, jc, jp = _project(camera25, world, jac)
        by_role = {r["physical_pattern_point_role"]: r for r in frame["source_pattern_points"]}
        residual = []; q_jac = []; camera_jac = []; reports = []
        for i, role in enumerate(SHIELD_ROLES):
            source = by_role[role]; bounds = np.asarray(source["native_support_bounds_xy_px"], float)
            outside = uv[i]-np.clip(uv[i], bounds[:, 0], bounds[:, 1]); residual.extend(outside/sigma)
            if jac:
                chain = (((uv[i] < bounds[:, 0]) | (uv[i] > bounds[:, 1])).astype(float)/sigma)[:, None]
                q_jac.extend(chain*(jp[i]@jw[i])); camera_jac.extend(chain*jc[i])
            reports.append(dict(source_id=source["id"], physical_material_role=role,
                projected_uv_native_px=uv[i].tolist(), outside_support_native_px=outside.tolist()))
        return SimpleNamespace(residual=np.asarray(residual),
            shared_q9_jacobian=np.asarray(q_jac).reshape(-1, 9) if jac else None,
            camera_state11_jacobian=np.asarray(camera_jac).reshape(-1, 11) if jac else None,
            source_reports=reports, source_held=frame["split"] == "held",
            source_silhouette_extrema_used_as_fixed_point=False)
