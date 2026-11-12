"""Actual template mesh boundaries and source-derived pose proposals."""

from collections import defaultdict
import cv2
import numpy as np

from inputs import BOOT, STEEL, distribute, masked_distance
from projection import PoseCurve, exp, inverse, log
from track import unit
from calibration.program_tracking.body_tracking.source_visible_model import monotone_match


def triangles_and_edges(faces):
    occurrences = defaultdict(list)
    for j, face in enumerate(faces):
        for i, k in zip(face, np.roll(face, -1)):
            occurrences[tuple(sorted((i, k)))].append(j)
    edges = np.array(list(occurrences), dtype=int)
    adjacency = np.asarray([(j[0], j[1] if len(j) > 1 else -1) for j in occurrences.values()])
    return edges, adjacency


class SkateModel:
    def __init__(self, geometry):
        self.geometry = geometry
        sole_faces = geometry["lower_faces"][np.isin(geometry["parts"], ("outsole", "heel_block"))]
        sole_vertex, relabel = np.unique(sole_faces, return_inverse=True)
        self.boot = np.concatenate((geometry["upper"], geometry["lower"][:, sole_vertex]), axis=1)
        self.faces = np.concatenate((geometry["upper_faces"], relabel.reshape(-1, 3) + geometry["upper"].shape[1]))
        self.mesh_edges, self.adjacency = triangles_and_edges(self.faces)
        self.pivot = self.boot.mean(axis=1)
        self.steel = geometry["edges"][:, :, ::4].copy()
        steel_faces = geometry["lower_faces"][geometry["parts"] == "steel_runner"]
        steel_vertices = geometry["lower"][:, np.unique(steel_faces)]
        upper = self.steel.copy()
        # The web's upper border comes from the actual side face vertices.
        # Keep the two lateral positions distinct even for a nearly end-on view.
        for f in range(2):
            for side in range(2):
                near = steel_vertices[f, np.abs(steel_vertices[f, :, 1] - self.steel[f, side, 0, 1]) < .002]
                if len(near):
                    # World-to-local roundoff can split one generated station
                    # into multiple almost equal x values. Merge the ring
                    # before taking its highest steel vertex.
                    order, station = np.unique(np.round(near[:, 0], 8), return_inverse=True)
                    top = np.full(len(order), -np.inf)
                    np.maximum.at(top, station, near[:, 2])
                    upper[f, side, :, 2] = np.interp(self.steel[f, side, :, 0], order, top)
        self.runner = np.stack((self.steel, upper), axis=1)
        self.point_count = self.boot.shape[1]
        self.points = np.concatenate((self.boot, self.runner.reshape(2, -1, 3)), axis=1)

    def boundary(self, uv):
        """Exterior segments of the projected triangle mesh, including concavities."""
        triangles = uv[self.faces]
        v = triangles[:, 1] - triangles[:, 0]
        w = triangles[:, 2] - triangles[:, 0]
        front = v[:, 0] * w[:, 1] > v[:, 1] * w[:, 0]
        j, k = self.adjacency.T
        chosen = self.mesh_edges[(k < 0) | (front[j] != front[k])]
        if len(chosen) == 0:
            return chosen
        end = uv[chosen]
        middle = end.mean(axis=1)
        # A silhouette seam can lie under a different shell triangle.
        # Reject strict interior points of the mesh union.
        e = np.roll(triangles, -1, axis=1) - triangles
        d = middle[:, None, None] - triangles
        signed = e[None, :, :, 0] * d[..., 1] - e[None, :, :, 1] * d[..., 0]
        tol = .30 * np.linalg.norm(e, axis=-1)
        interior = (signed > tol).all(axis=-1) | (signed < -tol).all(axis=-1)
        active = np.linalg.norm(end[:, 1] - end[:, 0], axis=1) > .32
        return chosen[active & ~interior.any(axis=1)]

    def source_outline(self, uv, *, padding=2):
        """Raster mesh union for review only. Nothing fills its convex envelope."""
        lo = np.floor(uv.min(axis=0)).astype(int) - padding
        hi = np.ceil(uv.max(axis=0)).astype(int) + padding
        size = hi - lo
        if (size <= 0).any() or max(size) > 430:
            return None
        image = np.zeros(tuple(size[::-1]), np.uint8)
        fixed = np.rint((uv - lo) * 16.).astype(np.int32)
        for f in self.faces:
            cv2.fillConvexPoly(image, fixed[f], 1, shift=4)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)[:, 0] + lo


class RowNavigation:
    """Frozen native optical values at the shoe's source row.

    This approximation only proposes silhouette correspondences and start
    branches. The residual and saved reprojection use the implicit row solve.
    """
    def __init__(self, native, rows):
        camera = native.camera
        self.offset = native.tau[:, None] * (np.clip(rows, 0, 1079) / 1079 - .5)
        self.r = np.empty((*rows.shape, 3, 3))
        self.c = np.empty((*rows.shape, 3))
        self.intr = np.empty((*rows.shape, 4))
        self.d = np.empty((*rows.shape, 4))
        for ix, shot, local in camera._groups(np.arange(len(rows))):
            p = shot.p
            time = p["motion_times"][local, None] + self.offset[ix]
            state = shot.core.evaluate_motion_camera(
                p["rig"], p["controls"], float(p["first_control_time"]), float(p["control_dt"]),
                time.ravel())
            if np.any(state["status"]):
                raise ValueError("Native camera row outside its registered support")
            self.r[ix] = state["rotation"].reshape(-1, 2, 3, 3)
            self.c[ix] = state["centre"].reshape(-1, 2, 3) * camera.info["metres_per_scene_unit"]
            self.intr[ix] = np.stack([state[k] for k in ("fx", "fy", "cx", "cy")], -1).reshape(-1, 2, 4)
            self.d[ix] = np.stack([state[k] for k in ("k1", "k2", "p1", "p2")], -1).reshape(-1, 2, 4)

    def rays(self, uv, i, f):
        fx, fy, cx, cy = self.intr[i, f]
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        norm = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), k, self.d[i, f])[:, 0]
        return np.c_[norm, np.ones(len(norm))]

    def project(self, xyz, frames=None):
        frames = np.arange(len(self.r)) if frames is None else frames
        camera = (xyz - self.c[frames, :, None]) @ inverse(self.r[frames])
        norm = camera[..., :2] / np.maximum(camera[..., 2:], .01)
        x, y = norm[..., 0], norm[..., 1]
        r2 = x*x + y*y
        k1, k2, p1, p2 = [self.d[frames, :, i, None] for i in range(4)]
        radial = 1 + r2*(k1 + k2*r2)
        xn = radial*x + 2*p1*x*y + p2*(r2 + 2*x*x)
        yn = radial*y + p1*(r2 + 2*y*y) + 2*p2*x*y
        return np.stack((xn, yn), axis=-1) * self.intr[frames, :, None, :2] + self.intr[frames, :, None, 2:]


def closest(points, segments):
    """Find a geometric arc station, which is free to slide next iteration."""
    edge = segments[:, 1] - segments[:, 0]
    delta = points[:, None] - segments[None, :, 0]
    station = np.clip(np.sum(delta * edge, axis=-1) / np.maximum(np.sum(edge * edge, axis=-1), 1e-8), 0, 1)
    v = delta - station[..., None] * edge
    squared = np.sum(v*v, axis=-1)
    k = squared.argmin(axis=1)
    at = np.arange(len(k))
    return k, station[at, k], np.sqrt(squared[at, k])


def arc_factors(target, uv, local, *, sigma, cap, weight, ordered=False):
    if ordered:
        fractions = np.linspace(0., 1., 5)
        grid = (uv[:, :1] + fractions[None, :, None] * (uv[:, 1:] - uv[:, :1])).reshape(-1, 2)
        distance = np.linalg.norm(target[:, None] - grid[None], axis=-1)
        selected, _ = monotone_match(distance)
        k, offset = np.divmod(selected, len(fractions))
        station = fractions[offset]
        error = distance[np.arange(len(k)), selected]
    else:
        k, station, error = closest(target, uv)
    b = uv[k, 1] - uv[k, 0]
    normal = np.column_stack((-b[:, 1], b[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    # Source silhouette is not a material-point correspondence.
    # Tangential freedom survives even when station lies at an arc endpoint.
    new_point = local[k, 0] + station[:, None] * (local[k, 1] - local[k, 0])
    reliability = (error < cap).astype(float) * weight / sigma
    return new_point, target, normal, reliability


def paths(xy):
    return np.stack((xy[:-1], xy[1:]), axis=1)


def packet_from_fit(inputs, image, nav, model, row_camera, origin, rotation, *, cap=50.):
    """Recompute view-dependent source factors once per outer iteration."""
    n = len(origin)
    count = 26
    local = np.zeros((n, 2, count, 3), float)
    target = np.zeros((n, 2, count, 2), float)
    normal = np.zeros_like(target)
    weight = np.zeros((n, 2, count), float)
    curve = PoseCurve(inputs.times, origin, rotation)
    p, r = curve.at(inputs.times[:, None] + row_camera.offset)
    feet = np.arange(2)
    p = p[:, feet, feet]
    r = r[:, feet, feet]
    xyz = p[..., None, :] + model.points @ inverse(r)
    estimate = row_camera.project(xyz)
    for i in np.flatnonzero(~inputs.withheld):
        for f in range(2):
            rows = inputs.fit.get((i, f), [])
            mask = image["boot"][i, f]
            shell = estimate[i, f, :model.point_count]
            cur = 0
            boot_status = image["boot_status"][i, f]
            direct_boot = [r for r in rows if r["semantic"] in BOOT and r.get("evidence_status") == "observed"]
            if boot_status or direct_boot:
                ids = model.boundary(shell)
                if len(ids):
                    uv = shell[ids]
                    mesh = model.boot[f, ids]
                    if boot_status:
                        samples = mask[::6][:13]
                        # The lower half and heel/toe carry greater weight than
                        # the cuff, which can flex with the lower leg.
                        view_up = unit(nav.shin[i, f])
                        dist = (samples - nav.ankle[i, f]) @ view_up
                        q = np.where(dist > .16 * nav.size[i, f], .58, 1.)
                        q *= .42 if boot_status == 3 else 1.
                        q *= .72 if image["roi_partial"][i, f] or image["overlap"][i] else 1.
                        result = arc_factors(samples, uv, mesh, sigma=3.5, cap=cap, weight=q)
                        cur = len(samples)
                        for dest, value in zip((local, target, normal, weight), result):
                            dest[i, f, :cur] = value
                    if direct_boot:
                        pts = np.vstack([distribute(r["polyline_uv"], 2) for r in direct_boot[:3]])
                        result = arc_factors(pts, uv, mesh, sigma=3., cap=cap, weight=1.)
                        for dest, value in zip((local, target, normal, weight), result):
                            dest[i, f, cur:cur + len(pts)] = value
                        cur += len(pts)
            steel_status = image["steel_status"][i, f]
            if steel_status:
                # Sign comes from native view direction, never from a blade midline.
                camera_local = (row_camera.c[i, f] - p[i, f]) @ r[i, f]
                side = int(camera_local[1] >= 0)
                lines = estimate[i, f, model.point_count:].reshape(*model.runner.shape[1:-1], 2)
                candidates = image["steel"][i, f, ::2]
                meaning = image["steel_semantic"][i, f]
                options = [meaning - 1] if meaning else [0, 1]
                best = min(options, key=lambda j: np.median(masked_distance(candidates, lines[j, side])))
                strength = {1: .54, 2: 1.0, 3: .28}[int(steel_status)]
                result = arc_factors(candidates, paths(lines[best, side]), paths(model.runner[f, best, side]),
                                     sigma=3.4, cap=cap, weight=strength, ordered=True)
                for dest, value in zip((local, target, normal, weight), result):
                    dest[i, f, cur:cur + len(candidates)] = value
    return local, target, normal, weight


def screen_rotation(axis, up, tilt, bank):
    # These vectors are in native camera axes (x right, y down, z forward).
    front = unit(np.r_[axis, tilt])
    z = unit(np.r_[up, bank] - np.dot(np.r_[up, bank], front) * front)
    if np.linalg.norm(z) < .5:
        z = unit(np.cross(front, [0., 1., 0.]))
    y = unit(np.cross(z, front))
    return np.column_stack((front, y, np.cross(front, y)))


def screen_cost(model, f, uv, goal, steel, steel_semantic, toe):
    edges = model.boundary(uv[:model.point_count])
    if len(edges) == 0:
        return 1e3
    _, _, error = closest(goal, uv[edges])
    value = np.median(error) + .25 * np.percentile(error, 85)
    outline = uv[:model.point_count]
    # The projected size also constrains the start branch's distance.
    value += .22 * np.linalg.norm(np.ptp(goal, axis=0) - np.ptp(outline, axis=0))
    if np.isfinite(steel).all():
        rails = uv[model.point_count:].reshape(*model.runner.shape[1:-1], 2)
        if steel_semantic:
            rails = rails[steel_semantic - 1]
        value += .38 * min(np.median(masked_distance(steel, rail))
                           for rail in rails.reshape(-1, model.runner.shape[-2], 2))
    if toe is not None:
        tip = outline[model.boot[f, :, 0] > np.quantile(model.boot[f, :, 0], .85)].mean(axis=0)
        value += .28 * np.linalg.norm(tip - toe)
    return float(value)


def proposed_poses(inputs, image, nav, model, row_camera, prior_p, prior_r):
    n = len(inputs.pts)
    p = np.repeat(prior_p[:, :, None], 3, axis=2)
    r = np.repeat(prior_r[:, :, None], 3, axis=2)
    cost = np.full((n, 2, 3), 3.)
    visible = (image["boot_status"] > 0) & ~inputs.withheld[:, None]
    for i, f in zip(*np.where(visible)):
        poly = image["boot"][i, f]
        goal = poly[::5].astype(float)
        steel = image["steel"][i, f]
        steel_semantic = image["steel_semantic"][i, f]
        ends = image["heel_toe"][i, f]
        screen = unit(ends[1] - ends[0])
        boot_middle = (poly.min(axis=0) + poly.max(axis=0)) * .5
        ray = row_camera.rays([boot_middle], i, f)[0]
        camera_r, camera_p = row_camera.r[i, f], row_camera.c[i, f]
        intrinsic = row_camera.intr[i, f]
        # Shoe-up uses the image boot relative to the measured steel if visible.
        # With no steel, use the 2D shin side as the initial orientation.
        up = unit(nav.shin[i, f])
        if np.isfinite(steel).all():
            up = unit(boot_middle - steel.mean(axis=0))
        else:
            up = unit(up + np.array([-screen[1], screen[0]]) * (.2 if np.dot(up, [-screen[1], screen[0]]) >= 0 else -.2))
        toe = ends[1] if image["heel_toe_ordered"][i, f] else None
        available = [([], 1e9), ([], 1e9)]
        vertices = model.points[f] - model.pivot[f]
        observed_span = np.ptp(poly, axis=0)
        for branch, sign in enumerate((1, -1)):
            for tilt, bank in ((0., 0.), (.85, -.27), (-.85, -.27), (.35, .65), (-.35, .65)):
                rc = screen_rotation(sign * screen, up, tilt, bank)
                projected = vertices @ rc.T
                extent = np.ptp(projected[:model.point_count, :2] * intrinsic[:2], axis=0)
                inverse_depth = (extent @ observed_span) / max(extent @ extent, 1e-6)
                depth = np.clip(1. / max(inverse_depth, 1e-6), 2.2, 85.)
                ray_centre = ray * depth
                approx = (projected[:, :2] / depth + ray[None, :2]) * intrinsic[:2] + intrinsic[2:]
                delta = boot_middle - (approx[:model.point_count].min(axis=0) +
                                       approx[:model.point_count].max(axis=0)) * .5
                approx += delta
                ray_centre[:2] += delta * depth / intrinsic[:2]
                score = screen_cost(model, f, approx, goal, steel, steel_semantic, toe)
                if score < available[branch][1]:
                    rotation = camera_r.T @ rc
                    origin = camera_p + camera_r.T @ ray_centre - rotation @ model.pivot[f]
                    available[branch] = ((origin, rotation), score)
            candidate, score = available[branch]
            if candidate:
                p[i, f, branch], r[i, f, branch] = candidate
                cost[i, f, branch] = score / 5.
        # Body depth participates only as one trial branch; no depth residual.
        pc = (prior_p[i, f] - camera_p) @ camera_r.T
        local = model.points[f] @ (camera_r @ prior_r[i, f]).T + pc
        approx = local[:, :2] / local[:, 2:] * intrinsic[:2] + intrinsic[2:]
        cost[i, f, 2] = screen_cost(model, f, approx, goal, steel, steel_semantic, toe) / 5.
    return p, exp(log(r)), cost, visible
