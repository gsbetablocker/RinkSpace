"""Independent conversion of the frozen S01 d78 physical initialization.

Only historical data are imported.  The new forward evaluator uses SciPy cubic
B-splines, freshly decoded fixed-head parameters, Brown optics and an independent
predicted-row root.  It neither imports nor calls any legacy evaluator/optimizer.
A successful replay is a conversion check, never material or whole-shot approval.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from numbers import Integral
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline, PPoly
from scipy.special import roots_legendre
from scipy.spatial.transform import Rotation

VERSION = "rinksolve.d78_independent_forward.v1"
FIRST_PTS, LAST_PTS, STEP_PTS, PTS_DEN = 256256, 575488, 256, 12800
READ_MATH_FILES = (
    "calibration/rinkfit/src/rinkfit_start/native_camera.py",
    "calibration/rinkfit/src/rinkfit_start/native_scene.py",
    "calibration/rinkfit/cpp/include/rinkfit/physical.hpp",
    "calibration/rinkfit/cpp/include/rinkfit/event_model.hpp",
    "calibration/rinkfit/cpp/src/event_model.cpp",
)
CONTRACT = {
    "id": VERSION,
    "world": "one d78 L=1 scene gauge; no independently verified metre scale",
    "rotation": "R=Rot(b,beta) @ R0 @ Rot(n,alpha), world to camera",
    "center": "C=P+R.T@(d+[0,0,pupil_linear*z+pupil_quadratic*z*z])+axis_spacing*unit(n cross R.T@b)",
    "zoom": "z=log_f-log(f_ref); fx=exp(log_f+eta/2), fy=exp(log_f-eta/2)",
    "principal": "[cx,cy]+z*[cx_z,cy_z]",
    "Brown": "k1=k10+k11*z, k2=k20+k21*z; p1,p2 constant; x'=x*(1+k1*r2+k2*r4)+2*p1*x*y+p2*(r2+2*x*x), y' symmetric",
    "RS": "t=(nativePTS-256256)/12800+signed_readout*(predicted_v/1079-1/2); solve predicted_v=project_y(X,t)",
    "RS_observed_row_used": False,
    "trajectory": "uniform cardinal cubic alpha,beta,log_f; exact alpha(0)=beta(0)=0 dependency expanded before SciPy BSpline construction",
    "scene": "continuous closed superellipse board course, straight material centerlines, actual named finite branches; no image warp",
    "old_chart_constants": "initial chart witnesses only; do not imply physically fixed ad locations",
    "historical_ice_blocks": "initial geometry only; independent historic endpoints/circles do not become new structural constraints",
    "three_rigs": "only main has d78 gauge-compatible initial state; secondary and end unavailable, no fake positions",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length <= 1e-14:
        raise ValueError("undefined unit vector")
    return vector / length


def axis_rotation(axis, angle):
    """SO(3) from its analytic axis-angle definition; supports arrays of angles."""
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    if abs(float(axis @ axis) - 1) > 1e-10:
        raise ValueError("rotation axis is not unit length")
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    cosine = np.cos(angle)[..., None, None]
    sine = np.sin(angle)[..., None, None]
    return cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * skew


def decode_camera(model):
    """Turn historical chart coordinates into explicit new head/optics/control data."""
    blocks = [np.asarray(b["values"], dtype=np.float64) for b in model["blocks"]]
    g, c = blocks[0], model["camera"]
    if g.shape != (32,) or not np.isfinite(g).all():
        raise ValueError("frozen global32 missing or nonfinite")
    r0 = Rotation.from_rotvec(g[5:8]).as_matrix() @ np.asarray(c["R0_seed"]).reshape(3, 3)
    pan = unit(np.asarray(c["n_seed"]) + g[8] * np.asarray(c["n_tangent1"]) + g[9] * np.asarray(c["n_tangent2"]))
    q = r0 @ pan
    axis0 = unit(np.asarray(c["b_seed"]) - q * (np.asarray(c["b_seed"]) @ q))
    tilt = np.cos(g[10]) * axis0 + np.sin(g[10]) * np.cross(q, axis0)
    controls = np.stack(blocks[c["control_block_begin"]:c["control_block_begin"] + c["n_controls"]]).copy()
    if controls.shape != (c["n_controls"], 3) or not np.isfinite(controls).all():
        raise ValueError("complete frozen trajectory controls missing")
    indices, weights, dep = np.asarray(c["reference_indices"]), np.asarray(c["reference_weights"]), c["dependent_control"]
    pivot = weights[indices == dep]
    if pivot.shape != (1,) or abs(pivot[0]) <= 1e-12:
        raise ValueError("angle reference has no unique dependency")
    use = indices != dep
    original_dependency = controls[dep, :2].copy()
    controls[dep, :2] = -np.sum(weights[use, None] * controls[indices[use], :2], axis=0) / pivot[0]
    camera = {"R0": r0.tolist(), "pan_axis_world": pan.tolist(), "tilt_axis_camera": tilt.tolist(),
              "pivot_world": g[11:14].tolist(), "lever_camera": g[14:17].tolist(), "axis_spacing": float(g[17]),
              "pupil_polynomial": g[18:20].tolist(), "f_ref": c["f_ref"], "radius_max": c["radius_max"],
              "principal_point": g[20:22].tolist(), "eta": float(g[22]), "principal_zoom_slope": g[23:25].tolist(),
              "k1_coefficients": g[25:27].tolist(), "k2_coefficients": g[27:29].tolist(), "tangential": g[29:31].tolist(),
              "signed_readout_seconds": float(g[31]), "control_dt_seconds": c["dt"], "first_control_time_seconds": c["first_control_time"],
              "reference_dependency": {"index": dep, "expanded_angle_values": controls[dep, :2].tolist(), "inactive_stored_angle_values": original_dependency.tolist()},
              "rig_id": "main", "coordinate_system_id": "d78_L1", "initialization_only": True}
    if c.get("brown_wide_extension") is not None:
        raise ValueError("this frozen d78 importer does not silently add a wide Brown extension")
    return camera, controls


def decode_geometry(model):
    """Remove chart witnesses by composing actual finite paths in initial geometry.

    A frozen polyline vertex is a representation sample. The historical physical
    point flag, endpoint bounds and constant blocks are retained as provenance,
    never promoted into new landmark/locked-position constraints.
    """
    blocks = [np.asarray(b["values"], dtype=np.float64) for b in model["blocks"]]
    g = blocks[0]
    scene = {"coordinate_system_id": "d78_L1", "length_gauge": 1., "width": float(g[0]), "corner_rx": float(g[1]),
             "corner_ry": float(g[2]), "corner_exponent": float(g[3]), "board_height": float(g[4]),
             "metric_verified": False, "initial_geometry_only": True, "constraint_semantics_unassigned": True}
    paths = []
    for mi, material in enumerate(model["materials"]):
        for branch, old in enumerate(material["paths"]):
            kind = old["kind"]
            q = blocks[old["geometry_block"]] if old.get("geometry_block", -1) >= 0 else None
            new = {"id": "d78_material_%03d/path%d" % (mi, branch), "material_index": mi, "branch_index": branch,
                   "label": material["label"], "original_path_id": old["id"], "original_variant_id": old["variant_id"],
                   "closed": bool(old.get("closed", False)), "initial_geometry_only": True, "constraint_semantics_unassigned": True,
                   "vertices_are_landmarks": False, "historical_physical_point_flag": bool(old.get("physical_point", False)),
                   "historic_flag_is_new_semantic_authority": False, "display_only_initializer": bool(old.get("display_only_initializer", False)),
                   "station_coordinate": "frozen finite branch coordinate; not evidence or default new observation DOF"}
            if kind == 0:
                if q is None or q.shape != (4,):
                    raise ValueError("ice segment block malformed")
                new.update(kind="ice_segment", endpoints_xy=q.tolist())
            elif kind == 1:
                if q is None:
                    raise ValueError("ice circle block missing")
                new.update(kind="ice_circle", center_xy=q[:2].tolist(), radius=float(q[old.get("radius_index", 2)]),
                           angle_interval=[old.get("start_angle", 0.), old.get("end_angle", 2*np.pi)])
            elif kind == 2:
                points = np.asarray(old["points"], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1:
                    raise ValueError("board print points malformed")
                # Apply the two rows of the old affine chart once, then discard
                # its installation witness from the new numerical dependency.
                if q is not None:
                    if q.shape != (6,):
                        raise ValueError("print affine block malformed")
                    points = np.column_stack((points, np.ones(len(points)))) @ q.reshape(2, 3).T
                offset = old.get("advertisement_offset", 0)
                ad = blocks[old["advertisement_block"]][offset:offset+4]
                if ad.shape != (4,):
                    raise ValueError("board installation witness malformed")
                points = points * [ad[1], ad[3]-ad[2]] + [ad[0], ad[2]]
                new.update(kind="board_material_polyline", unwrapped_station_height=points.tolist(),
                           station_parameterization="linear by original vertex index, not Euclidean/arclength interpolation",
                           former_installation_block=model["blocks"][old["advertisement_block"]]["name"],
                           former_constant_witness=bool(model["blocks"][old["advertisement_block"]].get("constant", False)),
                           constant_witness_is_physical_lock=False)
            elif kind == 3:
                new.update(kind="board_boundary", course_segment=int(old.get("boundary_segment", -1)),
                           height_fraction=float(old.get("fixed_height_fraction", 0.)))
            else:
                raise ValueError("unsupported frozen primitive kind: " + repr(kind))
            paths.append(new)
    if len(model["materials"]) != 171 or len(paths) != 258:
        raise ValueError("unexpected d78 material/path inventory")
    return scene, paths


class ContinuousBoard:
    """Independent polar superellipse and continuous station-Z course.

    The integral uses freshly generated 128-node Gauss-Legendre quadrature,
    independent of the legacy composite 32-node rule and old dense table.
    Numerical checks against adaptive quadrature are recorded, not a survey.
    """
    def __init__(self, scene, order=128):
        self.width, self.rx, self.ry, self.exponent = (float(scene[k]) for k in ("width", "corner_rx", "corner_ry", "corner_exponent"))
        self.height = float(scene["board_height"])
        if not (0 < self.rx < .5 and 0 < self.ry < self.width/2 and self.exponent > 1 and self.height > 0):
            raise ValueError("invalid continuous board shape")
        self.nodes, self.weights = roots_legendre(order)
        self.corner_length = float(self.integral(np.pi/2))
        self.lengths = np.array([1-2*self.rx, self.corner_length, self.width-2*self.ry, self.corner_length]*2)
        self.breaks = np.r_[0., np.cumsum(self.lengths)]
        self.perimeter = float(self.breaks[-1])

    def polar(self, theta):
        theta = np.asarray(theta)
        cs = np.maximum(np.stack((np.cos(theta), np.sin(theta)), axis=-1), 0.)
        scale = np.sum(cs**self.exponent, axis=-1)**(-1/self.exponent)
        return cs * scale[..., None] * [self.rx, self.ry]

    def speed(self, theta):
        theta = np.asarray(theta)
        c, s = np.maximum(np.cos(theta), 0.), np.maximum(np.sin(theta), 0.)
        norm = c**self.exponent + s**self.exponent
        rho = norm**(-1/self.exponent)
        rho_log_derivative = (c**(self.exponent-1)*s-s**(self.exponent-1)*c) / norm
        dx = self.rx*rho*(-s+c*rho_log_derivative)
        dy = self.ry*rho*(c+s*rho_log_derivative)
        return np.hypot(dx, dy)

    def integral(self, theta):
        data = np.asarray(theta, dtype=np.float64)
        flat, result = data.reshape(-1), np.empty(data.size)
        for begin in range(0, len(flat), 2048):
            upper = flat[begin:begin+2048]
            abscissa = (self.nodes[None, :] + 1) * upper[:, None] / 2
            result[begin:begin+2048] = (self.speed(abscissa) @ self.weights) * upper / 2
        return result.reshape(data.shape)

    def angle_for_length(self, distance):
        target = np.asarray(distance, dtype=np.float64)
        if np.any(target < -1e-12) or np.any(target > self.corner_length+1e-12):
            raise ValueError("corner station outside its finite arc")
        target = np.clip(target, 0, self.corner_length)
        angle = target / self.corner_length * (np.pi/2)
        lower, upper = np.zeros_like(angle), np.full_like(angle, np.pi/2)
        for _ in range(28):
            residual = self.integral(angle) - target
            done = np.abs(residual) <= 4e-15
            if np.all(done):
                break
            lower = np.where(residual < 0, angle, lower)
            upper = np.where(residual > 0, angle, upper)
            candidate = angle - residual/self.speed(angle)
            candidate = np.where((candidate > lower) & (candidate < upper), candidate, (lower+upper)/2)
            angle = np.where(done, angle, candidate)
        if np.any(np.abs(self.integral(angle)-target) > 1e-11):
            raise ValueError("new corner integral inverse did not converge")
        return angle

    def xyz(self, station, height):
        station, height = np.broadcast_arrays(np.asarray(station, dtype=np.float64), np.asarray(height, dtype=np.float64))
        shape = station.shape
        s = np.mod(station.reshape(-1), self.perimeter)
        z = height.reshape(-1)
        segment = np.minimum(np.searchsorted(self.breaks[1:], s, side="left"), 7)
        along = s-self.breaks[segment]
        xyz = np.column_stack((np.zeros(len(s)), np.zeros(len(s)), z))
        for part in range(8):
            mask = segment == part
            value = along[mask]
            if not len(value):
                continue
            if part == 0:
                xy = np.column_stack((-.5+self.rx+value, np.full(len(value), -self.width/2)))
            elif part == 2:
                xy = np.column_stack((np.full(len(value), .5), -self.width/2+self.ry+value))
            elif part == 4:
                xy = np.column_stack((.5-self.rx-value, np.full(len(value), self.width/2)))
            elif part == 6:
                xy = np.column_stack((np.full(len(value), -.5), self.width/2-self.ry-value))
            else:
                inverse_target = self.corner_length-value if part in (1, 5) else value
                q = self.polar(self.angle_for_length(inverse_target))
                signs = {1: [1, -1], 3: [1, 1], 5: [-1, 1], 7: [-1, -1]}[part]
                corner_center = np.array([(.5-self.rx)*signs[0], (self.width/2-self.ry)*signs[1]])
                xy = corner_center+q*signs
            xyz[mask, :2] = xy
        return xyz.reshape(shape+(3,))

    def numerical_check(self):
        errors = []
        for angle in np.linspace(0, np.pi/2, 33):
            adaptive, estimated = quad(lambda x: float(self.speed(x)), 0., float(angle), epsabs=1e-13, epsrel=1e-12, limit=200)
            errors.append(abs(float(self.integral(angle))-adaptive)+estimated)
        return {"quadrature_order": len(self.nodes), "maximum_difference_plus_adaptive_error_scene_units": max(errors),
                "corner_arclength": self.corner_length, "perimeter": self.perimeter,
                "note": "numerical quadrature check, not interval proof or physical measurement"}


@dataclass
class D78Initialization:
    directory: Path
    metadata: dict
    camera_definition: dict
    controls: np.ndarray
    scene_definition: dict
    paths: list

    def __post_init__(self):
        self.board = ContinuousBoard(self.scene_definition)
        c = self.camera_definition
        n = len(self.controls)
        knots = c["first_control_time_seconds"]+(np.arange(n+4)-2)*c["control_dt_seconds"]
        self.trajectory = BSpline(knots, self.controls, 3, axis=0, extrapolate=False)
        self.derivative_polynomials = [PPoly.from_spline((knots, self.controls[:, j], 3)).derivative() for j in range(3)]
        self.domain = (float(knots[3]), float(knots[n]))
        self.path_lookup = {(p["material_index"], p["branch_index"]): p for p in self.paths}
        self.native_pts = np.arange(FIRST_PTS, LAST_PTS+STEP_PTS, STEP_PTS, dtype=np.int64)
        if not np.allclose(self.trajectory(0.)[:2], 0., rtol=0, atol=2e-14):
            raise ValueError("expanded cubic reference-angle gauge failed")
        r = np.asarray(c["R0"])
        pan, tilt = np.asarray(c["pan_axis_world"]), np.asarray(c["tilt_axis_camera"])
        if not np.allclose(r@r.T, np.eye(3), atol=1e-10, rtol=0) or abs(np.linalg.det(r)-1)>1e-10 or abs(tilt@r@pan)>1e-10:
            raise ValueError("new fixed-head frame is inconsistent")

    def frame_time(self, native_pts):
        data = np.asarray(native_pts)
        if data.dtype.kind not in "iu" or np.any(data < FIRST_PTS) or np.any(data > LAST_PTS) or np.any((data-FIRST_PTS)%STEP_PTS):
            raise ValueError("exact S01 official50 integer native PTS required")
        return (data-FIRST_PTS)/PTS_DEN

    def pose_at(self, time):
        time = np.asarray(time, dtype=np.float64)
        if not np.isfinite(time).all() or np.any(time < self.domain[0]-1e-12) or np.any(time > self.domain[1]+1e-12):
            raise ValueError("row time outside allocated cubic domain; no clamp")
        values = self.trajectory(time)
        alpha, beta, log_f = np.moveaxis(values, -1, 0)
        c = self.camera_definition
        rotation = axis_rotation(c["tilt_axis_camera"], beta) @ np.asarray(c["R0"]) @ axis_rotation(c["pan_axis_world"], alpha)
        z = log_f-np.log(c["f_ref"])
        camera_tilt_world = np.einsum("...ji,j->...i", rotation, np.asarray(c["tilt_axis_camera"]))
        shift_axis = np.cross(np.asarray(c["pan_axis_world"]), camera_tilt_world)
        shift_axis /= np.linalg.norm(shift_axis, axis=-1)[..., None]
        lever = np.broadcast_to(np.asarray(c["lever_camera"]), time.shape+(3,)).copy()
        lever[..., 2] += c["pupil_polynomial"][0]*z+c["pupil_polynomial"][1]*z*z
        center = np.asarray(c["pivot_world"])+np.einsum("...ji,...j->...i", rotation, lever)+c["axis_spacing"]*shift_axis
        intrinsics = np.zeros(time.shape+(3, 3))
        intrinsics[..., 0, 0], intrinsics[..., 1, 1] = np.exp(log_f+c["eta"]/2), np.exp(log_f-c["eta"]/2)
        principal = np.asarray(c["principal_point"])+z[..., None]*np.asarray(c["principal_zoom_slope"])
        intrinsics[..., :2, 2], intrinsics[..., 2, 2] = principal, 1.
        brown = np.empty(time.shape+(4,))
        brown[..., 0] = c["k1_coefficients"][0]+c["k1_coefficients"][1]*z
        brown[..., 1] = c["k2_coefficients"][0]+c["k2_coefficients"][1]*z
        brown[..., 2:] = c["tangential"]
        return {"R_cw": rotation, "C_w": center, "K": intrinsics, "brown_k1_k2_p1_p2": brown, "log_f": log_f, "zoom_offset": z}

    def project_at(self, xyz, time):
        xyz, time = np.asarray(xyz, dtype=np.float64), np.asarray(time, dtype=np.float64)
        shape = np.broadcast_shapes(xyz.shape[:-1], time.shape)
        xyz, time = np.broadcast_to(xyz, shape+(3,)), np.broadcast_to(time, shape)
        state = self.pose_at(time)
        ray = np.einsum("...ij,...j->...i", state["R_cw"], xyz-state["C_w"])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            x, y = np.moveaxis(ray[..., :2]/ray[..., 2, None], -1, 0)
            r2 = x*x+y*y
            k1, k2, p1, p2 = np.moveaxis(state["brown_k1_k2_p1_p2"], -1, 0)
            radial = 1+k1*r2+k2*r2*r2
            distorted = np.stack((x*radial+2*p1*x*y+p2*(r2+2*x*x), y*radial+p1*(r2+2*y*y)+2*p2*x*y), axis=-1)
            uv = distorted*np.stack((state["K"][..., 0, 0], state["K"][..., 1, 1]), axis=-1)+state["K"][..., :2, 2]
        valid = (ray[..., 2] > 0) & np.isfinite(uv).all(axis=-1) & (r2 <= self.camera_definition["radius_max"]**2)
        return {"uv": uv, "valid": valid, "depth": ray[..., 2], "camera_xyz": ray, "ray_radius": np.sqrt(r2), **state}

    def world_path(self, material_index, branch_index, station):
        p = self.path_lookup[(int(material_index), int(branch_index))]
        s = np.asarray(station, dtype=np.float64)
        if not np.isfinite(s).all() or np.any(s < 0) or np.any(s > 1):
            raise ValueError("finite path station outside [0,1]")
        kind = p["kind"]
        if kind == "ice_segment":
            ends = np.asarray(p["endpoints_xy"]).reshape(2, 2)
            xy = (1-s[..., None])*ends[0]+s[..., None]*ends[1]
            return np.concatenate((xy, np.zeros(s.shape+(1,))), axis=-1)
        if kind == "ice_circle":
            angle = p["angle_interval"][0]+s*(p["angle_interval"][1]-p["angle_interval"][0])
            xy = np.asarray(p["center_xy"])+p["radius"]*np.stack((np.cos(angle), np.sin(angle)), axis=-1)
            return np.concatenate((xy, np.zeros(s.shape+(1,))), axis=-1)
        if kind == "board_material_polyline":
            points = np.asarray(p["unwrapped_station_height"])
            if len(points) == 1:
                q = np.broadcast_to(points[0], s.shape+(2,))
            else:
                x = s*(len(points)-1)
                index = np.minimum(np.floor(x).astype(int), len(points)-2)
                u = (x-index)[..., None]
                q = points[index]*(1-u)+points[index+1]*u
            return self.board.xyz(q[..., 0], q[..., 1])
        if kind == "board_boundary":
            segment = p["course_segment"]
            if segment == -2:
                q = s*8
                i = np.minimum(np.floor(q).astype(int), 7)
                coordinate = self.board.breaks[i]+(q-i)*self.board.lengths[i]
            elif segment >= 0:
                coordinate = self.board.breaks[segment]+s*self.board.lengths[segment]
            else:
                coordinate = s*self.board.perimeter
            return self.board.xyz(coordinate, self.board.height*p["height_fraction"])
        raise ValueError("unknown new path kind")

    def world_rows(self, materials, branches, stations):
        materials, branches, stations = np.broadcast_arrays(materials, branches, stations)
        if materials.ndim != 1 or materials.dtype.kind not in "iu" or branches.dtype.kind not in "iu":
            raise ValueError("material and branch IDs require one-dimensional integer arrays")
        result = np.empty((len(stations), 3))
        for mi, branch in np.unique(np.column_stack((materials, branches)), axis=0):
            selected = (materials == mi) & (branches == branch)
            result[selected] = self.world_path(mi, branch, stations[selected])
        return result

    def derivative_envelope(self, frame_times):
        times, inverse = np.unique(np.asarray(frame_times, dtype=np.float64).reshape(-1), return_inverse=True)
        half = abs(self.camera_definition["signed_readout_seconds"])/2
        bounds = np.zeros((len(times), 3))
        if np.any(times-half < self.domain[0]) or np.any(times+half > self.domain[1]):
            raise ValueError("entire sensor readout outside allocated trajectory")
        for row, time in enumerate(times):
            lo, hi = time-half, time+half
            for channel, polynomial in enumerate(self.derivative_polynomials):
                edges = polynomial.x
                first = max(0, np.searchsorted(edges, lo, side="right")-1)
                last = min(len(edges)-2, np.searchsorted(edges, hi, side="left"))
                peak = 0.
                for cell in range(first, last+1):
                    a, b = max(lo, edges[cell]), min(hi, edges[cell+1])
                    if b < a or edges[cell+1] <= edges[cell]:
                        continue
                    qa, qb, qc = polynomial.c[:, cell]
                    x0, x1 = a-edges[cell], b-edges[cell]
                    values = [abs((qa*x0+qb)*x0+qc), abs((qa*x1+qb)*x1+qc)]
                    if qa != 0:
                        extremum = -qb/(2*qa)
                        if x0 < extremum < x1:
                            values.append(abs((qa*extremum+qb)*extremum+qc))
                    peak = max(peak, *values)
                bounds[row, channel] = np.nextafter(peak+2e-11*(1+peak), np.inf)
        return bounds[inverse].reshape(np.asarray(frame_times).shape+(3,))

    def rs_certificate(self, xyz, frame_time):
        """Analytic sufficient contraction bound over every complete sensor readout.

        Polynomial extrema bound angle/zoom velocities. Rigid-body and Brown
        Jacobian norm bounds then bound |tau*d(project_y)/dt/1079|. Failure is
        kept as uncertified, even when a numerical row root exists.
        """
        xyz, frame_time = np.asarray(xyz), np.asarray(frame_time)
        mid = self.project_at(xyz, frame_time)
        rates = self.derivative_envelope(frame_time)
        angular_rate = np.abs(rates[..., 0])+np.abs(rates[..., 1])
        zoom_rate = rates[..., 2]
        c = self.camera_definition
        half = abs(c["signed_readout_seconds"])/2
        z_abs = np.abs(mid["zoom_offset"])+zoom_rate*half
        y_rate = angular_rate*(np.linalg.norm(xyz-np.asarray(c["pivot_world"]), axis=-1)+abs(c["axis_spacing"]))
        y_rate += (abs(c["pupil_polynomial"][0])+2*abs(c["pupil_polynomial"][1])*z_abs)*zoom_rate
        span = y_rate*half
        depth_low = mid["depth"]-span
        safe = np.where(depth_low > 0, depth_low, 1.)
        ray_box = np.abs(mid["camera_xyz"])+span[..., None]
        x, y = np.moveaxis(ray_box[..., :2]/safe[..., None], -1, 0)
        x_rate, y_normal_rate = np.moveaxis(y_rate[..., None]*(safe[..., None]+ray_box[..., :2])/safe[..., None]**2, -1, 0)
        r2 = x*x+y*y
        radius = np.sqrt(r2)
        k1_abs = abs(c["k1_coefficients"][0])+abs(c["k1_coefficients"][1])*z_abs
        k2_abs = abs(c["k2_coefficients"][0])+abs(c["k2_coefficients"][1])*z_abs
        p1, p2 = np.abs(c["tangential"])
        radial_abs = 1+k1_abs*r2+k2_abs*r2*r2
        jac_yx = 2*x*y*(k1_abs+2*k2_abs*r2)+2*p1*x+2*p2*y
        jac_yy = radial_abs+2*y*y*(k1_abs+2*k2_abs*r2)+6*p1*y+2*p2*x
        distorted_y_abs = y*radial_abs+p1*(r2+2*y*y)+2*p2*x*y
        direct_zoom_rate = y*(abs(c["k1_coefficients"][1])*r2+abs(c["k2_coefficients"][1])*r2*r2)*zoom_rate
        focal_y_upper = np.exp(mid["log_f"]+zoom_rate*half-c["eta"]/2)
        speed = focal_y_upper*(distorted_y_abs*zoom_rate+jac_yx*x_rate+jac_yy*y_normal_rate+direct_zoom_rate)+abs(c["principal_zoom_slope"][1])*zoom_rate
        speed = np.nextafter(speed*(1+2e-10)+1e-8, np.inf)
        contraction = abs(c["signed_readout_seconds"])*speed/1079
        k1_low = mid["brown_k1_k2_p1_p2"][..., 0]-abs(c["k1_coefficients"][1])*zoom_rate*half
        k2_low = mid["brown_k1_k2_p1_p2"][..., 1]-abs(c["k2_coefficients"][1])*zoom_rate*half
        def quadratic_lower(a, b):
            result = np.minimum(1., 1+a*r2+b*r2*r2)
            vertex = np.divide(-a, 2*b, out=np.zeros_like(a), where=b != 0)
            inside = (b > 0) & (vertex > 0) & (vertex < r2)
            return np.minimum(result, np.where(inside, 1+a*vertex+b*vertex*vertex, np.inf))
        fold_lower = np.minimum(quadratic_lower(k1_low, k2_low), quadratic_lower(3*k1_low, 5*k2_low))-6*np.hypot(p1, p2)*radius
        certified = (depth_low > 0) & (radius <= c["radius_max"]) & (fold_lower > 1e-10) & np.isfinite(contraction) & (contraction < 1-1e-9)
        return {"certified_unique": certified, "contraction_upper": contraction, "depth_lower": depth_low,
                "ray_radius_upper": radius, "brown_fold_lower": fold_lower, "full_sensor_readout": True}

    def project_rs(self, xyz, native_pts, *, root_iterations=42):
        xyz = np.asarray(xyz, dtype=np.float64)
        frame_time = self.frame_time(native_pts)
        shape = np.broadcast_shapes(xyz.shape[:-1], frame_time.shape)
        xyz, frame_time = np.broadcast_to(xyz, shape+(3,)), np.broadcast_to(frame_time, shape)
        tau = self.camera_definition["signed_readout_seconds"]
        certificate = self.rs_certificate(xyz, frame_time)
        lower, upper = np.zeros(shape), np.full(shape, 1079.)
        a = self.project_at(xyz, frame_time-tau/2)
        b = self.project_at(xyz, frame_time+tau/2)
        bracket = a["valid"] & b["valid"] & (a["uv"][..., 1] >= 0) & (b["uv"][..., 1] <= 1079)
        # Root starts at the sensor bracket. Neither observed y nor saved row
        # witnesses seed its value. Bisection retains finite numerical failures.
        for _ in range(root_iterations):
            row = (lower+upper)/2
            at_row = self.project_at(xyz, frame_time+tau*(row/1079-.5))
            equation = row-at_row["uv"][..., 1]
            upper = np.where(equation >= 0, row, upper)
            lower = np.where(equation < 0, row, lower)
        row = (lower+upper)/2
        out = self.project_at(xyz, frame_time+tau*(row/1079-.5))
        equation = row-out["uv"][..., 1]
        numerical = out["valid"] & bracket & (np.abs(equation) < 1e-7)
        out.update(row=row, root_equality_px=equation, bracketed=bracket, numerical_root_valid=numerical,
                   valid=numerical & certificate["certified_unique"], certificate=certificate)
        return out


def load_initialization(directory):
    directory = Path(directory)
    metadata = read_json(directory/"initialization.json")
    if metadata.get("schema") != VERSION:
        raise ValueError("wrong d78 initialization schema")
    controls = np.load(directory/"trajectory_controls.npy", allow_pickle=False)
    if controls.shape != (1252, 3) or not np.isfinite(controls).all():
        raise ValueError("current initialization has incomplete trajectory")
    return D78Initialization(directory, metadata, read_json(directory/"camera.json"), controls,
                             read_json(directory/"geometry.json"), read_json(directory/"paths.json"))


def compare_frozen_predictions(package, predictions_path=None, material_path=None, *, chunk_size=8192):
    """Compare new geometry plus freshly solved trueRS against frozen row witnesses."""
    current_witness_mode = predictions_path is None
    if predictions_path is None:
        predictions_path = package.directory/"forward_comparison.npz"
        with np.load(predictions_path, allow_pickle=False) as archive:
            frozen = {"source_rows": archive["source_rows"], "native_pts": archive["native_pts"], "active_source_mask": archive["active"],
                      "uv": archive["frozen_uv"], "stations": archive["stations"], "branches": archive["branches"],
                      "root_rows": archive["frozen_row"], "profile_alpha": archive["frozen_profile_alpha"], "RS_valid": archive["frozen_RS_valid"]}
            material = archive["material_index"].copy()
    else:
        with np.load(predictions_path, allow_pickle=False) as archive:
            frozen = {key: archive[key] for key in archive.files}
        material = np.fromfile(material_path, dtype="<u4")
    n = len(frozen["source_rows"])
    if n != 182017 or material.shape != (n,) or not np.array_equal(frozen["source_rows"], np.arange(n)):
        raise ValueError("frozen source ID/material mapping is incomplete or reordered")
    package.frame_time(frozen["native_pts"])
    uv = np.full((n, 2), np.nan)
    rows, equality, valid, contraction = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool), np.full(n, np.nan)
    world = np.empty((n, 3))
    active = frozen["active_source_mask"]
    for start in range(0, n, chunk_size):
        indices = np.arange(start, min(start+chunk_size, n))
        # Excluded rows are preserved but not silently brought back into use.
        indices = indices[active[indices]]
        if not len(indices):
            continue
        world[indices] = package.world_rows(material[indices], frozen["branches"][indices], frozen["stations"][indices])
        out = package.project_rs(world[indices], frozen["native_pts"][indices])
        uv[indices], rows[indices], equality[indices], valid[indices] = out["uv"], out["row"], out["root_equality_px"], out["valid"]
        contraction[indices] = out["certificate"]["contraction_upper"]
    delta = np.linalg.norm(uv-frozen["uv"], axis=1)
    per_frame = []
    for pts in package.native_pts:
        hit = active & (frozen["native_pts"] == pts)
        finite = hit & np.isfinite(delta)
        per_frame.append({"native_pts": int(pts), "source_seconds": int(pts)/PTS_DEN, "active_rows": int(hit.sum()),
                          "new_certified_RS_valid": int((hit & valid).sum()), "maximum_px": float(np.max(delta[finite])) if finite.any() else None,
                          "rms_px": float(np.sqrt(np.mean(delta[finite]**2))) if finite.any() else None})
    finite = active & np.isfinite(delta)
    maximum = float(np.max(delta[finite]))
    frozen_reference = package.metadata.get("frozen_predictions_reference", {})
    report = {"schema": "rinksolve.d78_frozen_forward_comparison.v1", "source": frozen_reference.get("path", str(predictions_path)) if current_witness_mode else str(predictions_path),
              "source_hash": frozen_reference.get("sha256") if current_witness_mode else digest(predictions_path),
              "witnesses_loaded_from": str(predictions_path), "no_historical_input_read_required": current_witness_mode,
              "new_implementation": VERSION, "mathematical_stage": "complete geometry + Brown + fresh predicted-row RS, not a pinhole H check",
              "source_rows_preserved": n, "active_rows": int(active.sum()), "excluded_source_rows": frozen["source_rows"][~active].tolist(),
              "native_exposures": 1248, "new_certified_RS_valid_rows": int((active & valid).sum()),
              "missing_source_rows": frozen["source_rows"][active & ~np.isfinite(delta)].tolist(),
              "invalid_new_RS_source_rows": frozen["source_rows"][active & ~valid].tolist(),
              "maximum_euclidean_px": maximum, "rms_euclidean_px": float(np.sqrt(np.mean(delta[finite]**2))),
              "maximum_component_px": float(np.max(np.abs(uv[finite]-frozen["uv"][finite]))),
              "maximum_new_RS_equation_px": float(np.max(np.abs(equality[finite]))),
              "maximum_new_vs_saved_row_px": float(np.max(np.abs(rows[finite]-frozen["root_rows"][finite]))),
              "maximum_full_readout_contraction_bound": float(np.max(contraction[active])),
              "numerical_comparison_tolerance_px": 1e-4,
              "projection_arithmetic_equivalence_pass": bool(maximum <= 1e-4 and finite.sum() == active.sum()),
              "strict_all_sensor_row_roots_pass": bool(np.all(valid[active])),
              "border_event_source_rows": frozen["source_rows"][active & ~valid].tolist(),
              "border_event_is_global_engine_startup_failure": False,
              "profile_query_or_global_branch_optimality_reproved": False,
              "whole_shot_material_alignment_accepted": False, "next_shot_allowed": False,
              "limitations": ["Frozen station/branch values are forward comparison witnesses, not permission to reuse old semantic degrees of freedom.",
                              "Fresh predicted-row roots and full-readout contraction are checked; profile query construction/global profile optimum is not re-established.",
                              "Numerical conversion tolerance is an engineering check, not a physical fit acceptance threshold.",
                              "Three-dimensional artwork shape is retained as historical initialization; the old constant ad chart is not a physical location lock."],
              "frames": per_frame}
    np.savez_compressed(package.directory/"forward_comparison.npz", source_rows=frozen["source_rows"], native_pts=frozen["native_pts"],
                        active=active, material_index=material, branches=frozen["branches"], stations=frozen["stations"],
                        frozen_uv=frozen["uv"], new_uv=uv, new_row=rows, new_RS_equation=equality, new_RS_valid=valid,
                        full_readout_contraction_bound=contraction, euclidean_difference_px=delta,
                        frozen_profile_alpha=frozen["profile_alpha"], frozen_row=frozen["root_rows"], frozen_RS_valid=frozen["RS_valid"])
    write_json(package.directory/"forward_proof.json", report)
    return report


def import_d78(manifest_path, output, *, material_path=None, predictions_path=None):
    manifest = read_json(manifest_path)
    if manifest["scope_lock"]["active_shot"] != "S01":
        raise ValueError("d78 conversion is locked to S01")
    historic = manifest["d78"]
    model_path = Path(historic["model"]["path"])
    if digest(model_path) != historic["model"]["sha256"]:
        raise ValueError("selected frozen d78 data model changed")
    model = read_json(model_path)
    camera, controls = decode_camera(model)
    scene, paths = decode_geometry(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": VERSION, "shot_id": "S01", "source": manifest["source"], "clock": manifest["clock"],
                "projection_contract": CONTRACT, "original_model": historic["model"], "source_identity": model["source_identity"],
                "frozen_predictions_reference": {"path": item["review_path"], "sha256": item["sha256"]} if (item := next(x for x in historic["auxiliary_arrays"] if x["original_path"].endswith("candidate_predictions.npz"))) else None,
                "runtime_load_dependencies": ["initialization.json", "camera.json", "trajectory_controls.npy", "geometry.json", "paths.json"],
                "forward_proof_is_runtime_qualification_gate": False,
                "frozen_171_materials": 171, "finite_paths": len(paths), "initial_geometry_only": True, "constraint_semantics_unassigned": True,
                "rig_inventory": [{"id": "main", "status": "d78 same L1 gauge initialized"}, {"id": "secondary", "status": "missing, no proved d78 gauge transform"}, {"id": "end", "status": "missing, no proved d78 gauge transform"}],
                "no_legacy_solver_renderer_imported": True, "old_optimizer_duals_imported": False, "demo_scene_or_camera_mixed": False,
                "next_shot_allowed": False, "files": {"camera": "camera.json", "controls": "trajectory_controls.npy", "geometry": "geometry.json", "paths": "paths.json"}}
    write_json(output/"initialization.json", metadata)
    write_json(output/"camera.json", camera)
    write_json(output/"geometry.json", scene)
    write_json(output/"paths.json", paths)
    np.save(output/"trajectory_controls.npy", controls, allow_pickle=False)
    package = load_initialization(output)
    pose = package.pose_at(package.frame_time(package.native_pts))
    np.savez_compressed(output/"native_camera_states.npz", native_pts=package.native_pts, **pose)
    write_json(output/"board_numerical_check.json", package.board.numerical_check())
    project_root = Path(__file__).resolve().parents[3]
    audit = {"schema": "rinksolve.d78_read_math_not_code_reuse.v1", "files_actually_read": [],
             "new_implementation": "SciPy Rotation and BSpline/PPoly; independent Gauss-Legendre quadrature and safeguarded integral inverse; Brown and trueRS bisection from equations", "legacy_imports_or_calls": [], "new_source_path": str(Path(__file__).resolve()), "new_source_sha256": digest(Path(__file__)),
             "d78_old_semantic_or_fixedchart_automatically_accepted": False}
    for relative in READ_MATH_FILES:
        path = project_root/relative
        audit["files_actually_read"].append({"path": str(path), "sha256": digest(path), "use": "read-only mathematical understanding; not imported or executed"})
    write_json(output/"implementation_math_audit.json", audit)
    if predictions_path is None:
        predictions_path = next(item["review_path"] for item in historic["auxiliary_arrays"] if item["original_path"].endswith("candidate_predictions.npz"))
    if material_path is None:
        material_path = output/"resolved_material.u32"
    if Path(material_path).exists():
        report = compare_frozen_predictions(package, predictions_path, material_path)
        audit_border_events(package)
        export_typed_bundle(package, report)
    else:
        write_json(output/"forward_proof.json", {"status": "pending", "reason": "actual d78 resolved material mapping missing; no g7 row-order assumption", "exposures_initialized": 1248})
    return package



def audit_border_events(package):
    """Keep isolated sensor-boundary events separate from arithmetic equivalence."""
    comparison = package.directory/"forward_comparison.npz"
    with np.load(comparison, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    indices = np.flatnonzero(data["active"] & ~data["new_RS_valid"])
    raw_path = package.directory/"reference_native_uv.f64"
    raw = np.fromfile(raw_path, dtype="<f8").reshape(-1, 2) if raw_path.exists() else None
    records_path = package.directory/"border_original_source_records.json"
    records = {r["jsonl_index"]: r["record"] for r in read_json(records_path)} if records_path.exists() else {}
    previous_path = package.directory/"sensor_border_events.json"
    previous = {event["source_row"]: event for event in read_json(previous_path)["events"]} if previous_path.exists() else {}
    events = []
    for row in indices:
        material, branch = int(data["material_index"][row]), int(data["branches"][row])
        path = package.path_lookup[(material, branch)]
        xyz = package.world_path(material, branch, data["stations"][row])
        time = package.frame_time(data["native_pts"][row])
        sensor_upper = package.project_at(xyz, time+package.camera_definition["signed_readout_seconds"]/2)
        source = records.get(int(row))
        detail = {"source_row": int(row), "native_pts": int(data["native_pts"][row]), "material": path["label"],
                  "status": "border_roundoff_uncertified", "new_sensor_upper_equation_px": float(1079-sensor_upper["uv"][1]),
                  "saved_predicted_uv": data["frozen_uv"][row].tolist(), "saved_row_witness": float(data["frozen_row"][row]),
                  "new_predicted_uv": data["new_uv"][row].tolist(), "new_RS_equation_px": float(data["new_RS_equation"][row]),
                  "full_readout_contraction_bound": float(data["full_readout_contraction_bound"][row]),
                  "actual_source_uv": raw[row].tolist() if raw is not None else None,
                  "sensor_row_definition": "integer pixel centers [0,1079]; row_time uses predicted_v/1079-1/2",
                  "pixel_footprint_support_domain": "not defined by historical projection contract; no [-.5,1079.5] substitution made",
                  "source_classified_censored": False, "source_censor_status": "not established; original y is inside the center domain",
                  "no_UV_change_or_clamp_or_tolerance_extension": True,
                  "legacy_numpy_root_behavior": "bisection on [0,1079], equality tolerance 1e-8; no per-point sourceUV clamp in the read math",
                  "legacy_cpp_root_behavior": "sampled brackets on [0,1079], numerical endpoint detection/equality tolerances; no proof of exact endpoint root from saved decimals"}
        if int(row) in previous:
            for key in ("source_record", "source_semantics", "not_independent_raw_pixel_or_entity_corner"):
                if key in previous[int(row)]:
                    detail[key] = previous[int(row)][key]
        elif source is not None:
            source_path = Path(source["source_packet"]["path"])
            frame = read_json(source_path)
            feature = next((f for f in frame["features"] if f.get("id") == source["source_feature_id"] or f.get("target_id") == source["source_feature_id"]), None)
            if feature is not None:
                segment = feature["segments"][source["source_segment_index"]]
                detail["source_record"] = source
                detail["source_semantics"] = {"feature_id": feature["id"], "geometric_interpretation": feature.get("geometric_interpretation"),
                    "measurement_origin": segment.get("measurement_origin"), "representative_points_share_line_estimate": segment.get("representative_points_share_line_estimate"),
                    "individual_native_pixel_readouts": segment.get("individual_native_pixel_readouts"), "inferred_from_current_source_readouts": segment.get("inferred_from_current_source_readouts"),
                    "user_confirmed": feature.get("user_confirmed"), "source_evidence_group_id": segment.get("source_evidence_group_id"),
                    "point_id_if_present": segment.get("point_ids", [])[source["original_vertex_index"]] if len(segment.get("point_ids", [])) > source["original_vertex_index"] else None}
                detail["not_independent_raw_pixel_or_entity_corner"] = True
        events.append(detail)
    result = {"schema": "rinksolve.d78_sensor_border_events.v1", "events": events, "count": len(events),
              "whole_import_or_engine_startup_failure": False, "facts_rewritten": False,
              "new_problem_handling": "Explicit observation-local row auxiliary with the actual projection equality and declared sensor domain can represent a boundary feasibility step; no historical qualification replay is required.",
              "raw_source_semantics": "Shared line-fit representatives retain dependence/inference provenance; they are not independently measured pixel points or world landmarks."}
    write_json(package.directory/"sensor_border_events.json", result)
    return result


def export_typed_bundle(package, report):
    """Use the new schema for actual source/rig/geometry assets and proof.

    This is intentionally no prepared fit problem. Historical semantic flags,
    independent ice DOFs and constant chart witnesses are not reissued as new
    constraints, landmarks, observation associations or locked positions.
    """
    if __package__:
        from . import schema as sc
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rinksolve import schema as sc
    directory = package.directory
    stamp = sc.RevisionStamp(1, 1, 0, 1)
    source = package.metadata["source"]
    evidence = (
        sc.Evidence("source_official50", sc.EvidenceKind.NATIVE_VIDEO, source["path"], "Official ISU native50 programme source, unchanged original video.", source["sha256"]),
        sc.Evidence("d78_frozen_model", sc.EvidenceKind.COMPUTATION, package.metadata["original_model"]["path"], "Historical useful S01 ice initialization; no surveyed metric or accepted full advertising accuracy.", package.metadata["original_model"]["sha256"]),
        sc.Evidence("d78_new_geometry", sc.EvidenceKind.COMPUTATION, str(directory/"paths.json"), "All 171 materials / 258 finite paths independently decoded into one L1 gauge; representation samples have no landmark semantics.", digest(directory/"paths.json"), ("d78_frozen_model",)),
        sc.Evidence("d78_new_camera", sc.EvidenceKind.COMPUTATION, str(directory/"camera.json"), "New fixed-head/Brown/trajectory parameter package, no optimizer state.", digest(directory/"camera.json"), ("d78_frozen_model",)),
        sc.Evidence("d78_forward_report", sc.EvidenceKind.COMPUTATION, str(directory/"forward_proof.json"), "Complete independent frozen-forward numeric comparison; strict sensor border events are separate from arithmetic equality.", digest(directory/"forward_proof.json"), ("d78_frozen_model", "d78_new_geometry", "d78_new_camera")),
    )
    clock = sc.SourceClock("official50_clock", 1, PTS_DEN, "source_official50")
    frames = tuple(sc.SourceFrame("official50_pts%d" % int(pts), clock.id, int(pts), 1920, 1080, "source_official50") for pts in package.native_pts)
    frame_ids = tuple(f.id for f in frames)
    coordinate = sc.CoordinateSystem("d78_L1", sc.CoordinateKind.WORLD, 3, ("x", "y", "z"),
        tuple(sc.UnitRef(sc.Unit.SCENE_UNIT, "d78_L1") for _ in range(3)),
        "Original d78 full rink length gauge L=1, Z up, centered rink; no independently verified metres. Never mix demo scene units or unproved other-rig gauge.")
    scene = sc.Scene("ankara2026_d78_initial_scene", coordinate.id)
    rig = sc.CameraRig("main", scene.id, "Actual d78 fixed two-axis head, shared pivot/lever, zoom/Brown and signed predicted-row RS. Other rigs explicitly missing in initialization.json.", ("d78_frozen_model", "d78_new_camera"))
    shot = sc.Shot("S01", rig.id, clock.id, frame_ids)
    material_labels = {p["material_index"]: p["label"] for p in package.paths}
    entities = tuple(sc.GeometryEntity("d78_material_%03d" % i, scene.id, label,
        sc.MaterialScope.ICE_BODY if label.startswith("ice.") else sc.MaterialScope.BOARD_BODY) for i, label in material_labels.items())
    verifications = tuple(sc.EntityVerification("unconfirmed_"+e.id, e.id, sc.VerificationStatus.UNCONFIRMED,
        sc.MaterialScope.UNCONFIRMED, False, ("d78_frozen_model",), "Historical identity only. New semantic/source-material admission is not implied by this initializer.") for e in entities)
    primitives = tuple(sc.GeometryPrimitive(p["id"], "d78_material_%03d" % p["material_index"], coordinate.id,
        sc.RepresentationKind.MODEL, VERSION+"/"+p["kind"], (), "d78_new_geometry") for p in package.paths)
    source_contract = sc.ProjectionContract("d78_historical_physical_contract", "historical_frozen_prediction_data_only",
        CONTRACT["Brown"], CONTRACT["RS"], CONTRACT["rotation"]+"; "+CONTRACT["center"], ("d78_frozen_model",))
    new_contract = sc.ProjectionContract(VERSION, VERSION, CONTRACT["Brown"], CONTRACT["RS"],
        CONTRACT["rotation"]+"; "+CONTRACT["center"], ("d78_new_camera",))
    # This disabled fitting statement documents only the finite numerical
    # conversion test. It contributes no residual, physical relation or prior.
    criterion = sc.Constraint("d78_conversion_numeric_test", 1, sc.ConstraintKind.ESTIMATION_CONVENTION,
        sc.Authority.ENGINEERING_PROPOSAL, False, "conversion_report_max_euclidean_difference", (),
        (sc.NamedQuantity("tolerance_px", sc.Quantity(1e-4, sc.UnitRef(sc.Unit.PIXEL, "source_official50"))),),
        ("d78_forward_report",), "Numerical conversion check only; disabled for fitting. It is not physical material/whole-shot acceptance or source qualification.")
    proof = sc.ProjectionProof("d78_independent_arithmetic_equivalence", source_contract.id, new_contract.id, frame_ids,
        "mathematical_frozen_forward_arithmetic", "passed" if report.get("projection_arithmetic_equivalence_pass") else "failed",
        ("d78_forward_report",), (criterion.id,),
        "All active saved stations/branches are forward witnesses only. Four strict sensor-center boundary signs remain numerically uncertified; no UV clamp or semantic approval. No profile global optimum or whole-shot material validation claim.")
    camera_states = package.pose_at(package.frame_time(package.native_pts))
    np.savez_compressed(directory/"camera_columns.npz", native_pts=package.native_pts, C_w=camera_states["C_w"], R_cw=camera_states["R_cw"],
        fx=camera_states["K"][:, 0, 0], fy=camera_states["K"][:, 1, 1], cx=camera_states["K"][:, 0, 2], cy=camera_states["K"][:, 1, 2],
        brown=camera_states["brown_k1_k2_p1_p2"], alpha_controls=package.controls[:, 0], beta_controls=package.controls[:, 1], log_f_controls=package.controls[:, 2])
    specs = [("native_pts", (1248,), "<i8", sc.Unit.NATIVE_PTS, clock.id, None),
             ("C_w", (1248, 3), "<f8", sc.Unit.SCENE_UNIT, None, coordinate.id),
             ("R_cw", (1248, 3, 3), "<f8", sc.Unit.DIMENSIONLESS, None, None),
             *[(k, (1248,), "<f8", sc.Unit.PIXEL, None, None) for k in ("fx", "fy", "cx", "cy")],
             ("brown", (1248, 4), "<f8", sc.Unit.DIMENSIONLESS, None, None),
             ("alpha_controls", (1252,), "<f8", sc.Unit.RADIAN, None, None),
             ("beta_controls", (1252,), "<f8", sc.Unit.RADIAN, None, None),
             ("log_f_controls", (1252,), "<f8", sc.Unit.DIMENSIONLESS, None, None)]
    array_refs = tuple(sc.TypedArrayRef("d78_"+key, str((directory/"camera_columns.npz").resolve()), "npz", dtype, shape,
        sc.UnitRef(unit_kind, clock_id or coordinate_id), sc.ArrayRole.PARAMETER_STATE if key.endswith("controls") else sc.ArrayRole.COMPUTE_CACHE,
        True, 1, clock_id, coordinate_id, key) for key, shape, dtype, unit_kind, clock_id, coordinate_id in specs)
    bundle = sc.DataBundle("d78_independent_initialization_assets", stamp, "S01", evidence=evidence, clocks=(clock,), frames=frames,
        coordinate_systems=(coordinate,), scenes=(scene,), rigs=(rig,), shots=(shot,), entities=entities, verifications=verifications,
        primitives=primitives, constraints=(criterion,), projection_contracts=(source_contract, new_contract), projection_proofs=(proof,), array_refs=array_refs)
    issues = sc.validate_bundle(bundle, raise_on_error=False)+sc.validate_array_contracts(bundle, raise_on_error=False)
    write_json(directory/"typed_contract_validation.json", {"valid": not issues, "issues": [sc.to_json_value(i) for i in issues],
        "scope": "Source/clock/scene/main rig/all historical primitive assets/camera column units/numerical proof. No prepared fit, semantic landmark, association or variable-lock declaration inferred."})
    (directory/"typed_bundle.json").write_text(sc.dumps(bundle, indent=2)+"\n", encoding="utf-8")
    if issues:
        raise ValueError("new typed initialization assets have contract issues")
    return bundle


def render_selected_frames(package, native_pts, historical_video):
    """Few exact native source/history/new-model still comparisons; never video."""
    import av
    import cv2
    import re
    points, cuts = [], []
    for path in package.paths:
        if path["kind"] == "board_material_polyline":
            count = len(path["unwrapped_station_height"])
            station = np.unique(np.r_[np.linspace(0, 1, max(33, 4*count)), np.linspace(0, 1, count)])
        else:
            station = np.linspace(0, 1, 257 if path["kind"] == "ice_circle" else 129)
        xyz = package.world_path(path["material_index"], path["branch_index"], station)
        begin = sum(len(p) for p in points)
        points.append(xyz)
        cuts.append((begin, begin+len(xyz), path))
    world = np.vstack(points)
    directory = package.directory/"overlays"
    directory.mkdir(exist_ok=True)
    source_path = package.metadata["source"]["path"]
    def exact_frame(path, pts, expected_width, expected_height):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if Fraction(stream.time_base) != Fraction(1, PTS_DEN) or Fraction(stream.average_rate) != 50 or stream.width != expected_width or stream.height != expected_height:
                raise ValueError("video native format differs from explicit comparison contract")
            container.seek(int(pts), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts == int(pts):
                    return frame.to_ndarray(format="bgr24")
                if frame.pts is not None and frame.pts > int(pts):
                    break
        raise ValueError("video did not decode exact PTS " + str(pts))
    rows = []
    for pts in native_pts:
        package.frame_time(np.asarray(pts, dtype=np.int64))
        source = exact_frame(source_path, pts, 1920, 1080)
        history = exact_frame(historical_video, pts-FIRST_PTS, 2560, 1080)
        uv, valid = np.empty((len(world), 2)), np.zeros(len(world), bool)
        for begin in range(0, len(world), 8192):
            end = min(begin+8192, len(world))
            result = package.project_rs(world[begin:end], np.full(end-begin, pts, dtype=np.int64))
            uv[begin:end], valid[begin:end] = result["uv"], result["valid"]
        overlay = source.copy()
        labels = set()
        for begin, end, path in cuts:
            color = (0, 220, 255) if path["kind"].startswith("ice_") else (255, 170, 0) if path["kind"] == "board_boundary" else (100, 250, 90)
            for a in range(begin, end-1):
                b = a+1
                if not valid[a] or not valid[b] or np.max(np.abs(uv[[a, b]])) > 1e7:
                    continue
                p, q = tuple(int(x) for x in np.rint(uv[a])), tuple(int(x) for x in np.rint(uv[b]))
                hit, start, finish = cv2.clipLine((0, 0, 1920, 1080), p, q)
                if hit:
                    cv2.line(overlay, start, finish, color, 1, cv2.LINE_AA)
                    short = (re.match(r"Ad[0-9]+", path["label"]) or [path["label"]])[0]
                    if short not in labels:
                        cv2.putText(overlay, short, start, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                        labels.add(short)
        # Historical frame keeps its 1920px source panel; the additional 640px
        # historical top-down UI is separate and is not a source image crop.
        panes = [cv2.resize(image, (960, 540)) for image in (source, history[:, :1920], overlay)]
        for image, title in zip(panes, ("SOURCE", "FROZEN d78 VIDEO", "NEW INDEPENDENT d78 FORWARD")):
            cv2.rectangle(image, (0, 0), (960, 30), (0, 0, 0), -1)
            cv2.putText(image, title+" | PTS %d | %.2fs" % (pts, pts/PTS_DEN), (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        target = directory/("d78_pts%d_triple.jpg" % pts)
        if not cv2.imwrite(str(target), np.hstack(panes), [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError("comparison still export failed")
        rows.append({"native_pts": pts, "source_seconds": pts/PTS_DEN, "historical_video_pts": pts-FIRST_PTS,
                     "path": str(target), "model_path_count": len(package.paths), "sampled_points": len(world), "strict_RS_valid_samples": int(valid.sum())})
    report = {"schema": "rinksolve.d78_same_model_selected_stills.v1", "source": source_path, "historical_video": str(historical_video),
              "frames": rows, "whole_shot_visual_validation_completed": False, "next_shot_allowed": False,
              "limitations": ["Few still frames only; no continuous shot acceptance.", "No person/board occlusion mask; every finite path belongs to the same independently converted d78 state.", "Source/image panel is 1920x1080; historic 640px UI panel omitted only in the triptych.", "No demo geometry/camera, no other rig or metric conversion used."]}
    write_json(directory/"review_manifest.json", report)
    return report




def render_intersection_navigation(package, measurements_path, *, native_pts=575488):
    """Draw unbound image crossings next to named initial model curves, no matching."""
    import cv2
    rows = read_json(measurements_path)
    if isinstance(rows, dict):
        rows = [{"id": row["id"], "native_pts": row["native_pts"], "uv": row["uv_native_pixels"]} for row in rows["observations"]]
    if not isinstance(rows, list):
        raise ValueError("navigation expects registered image observations or a flat measurement draft list")
    rows = [r for r in rows if r["native_pts"] == native_pts]
    raw = package.directory.parent.parent/"solve2_review"/"s01_intersections"/("pts_%d.png" % native_pts)
    if not raw.exists():
        # Known source readout cache is adjacent to the reviewed measurements.
        raw = Path(measurements_path).parent/("pts_%d.png" % native_pts)
    source = cv2.imread(str(raw))
    if source is None or source.shape[:2] != (1080, 1920):
        raise ValueError("exact native source readout image is missing or has changed dimensions")
    output = source.copy()
    model_rows = []
    colors = {"C0": (0, 150, 255), "C": (20, 20, 255), "house": (255, 220, 0), "lane": (80, 240, 80), "other": (150, 150, 150)}
    # Render every same-model path, highlighting the four topological families.
    # No nearest-predicted-point or label identity is assigned to image crosses.
    for path in package.paths:
        label = path["label"]
        family = "C0" if label.startswith("ice.C0:") else "C" if label.startswith("ice.C:") else "house" if "house_connection" in label else "lane" if label.startswith("ice.lane") else "other"
        n = max(129, 4*len(path.get("unwrapped_station_height", [])))
        xyz = package.world_path(path["material_index"], path["branch_index"], np.linspace(0, 1, n))
        projection = package.project_rs(xyz, np.full(n, native_pts, dtype=np.int64))
        uv, valid = projection["uv"], projection["valid"]
        inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
        for a in range(n-1):
            b = a+1
            if not valid[a] or not valid[b] or np.max(np.abs(uv[[a, b]])) > 1e7:
                continue
            hit, start, finish = cv2.clipLine((0, 0, 1920, 1080), tuple(int(x) for x in np.rint(uv[a])), tuple(int(x) for x in np.rint(uv[b])))
            if hit:
                cv2.line(output, start, finish, colors[family], 2 if family != "other" else 1, cv2.LINE_AA)
        if family != "other" and inside.any():
            indices = np.flatnonzero(inside)
            label_index = indices[int(len(indices)*.67)]
            position = tuple(int(x) for x in np.rint(uv[label_index]))
            short = label.replace("ice.", "").replace(":segment0", "").replace("house_connection_", "axis_")
            cv2.putText(output, short, position, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output, short, position, cv2.FONT_HERSHEY_SIMPLEX, .5, colors[family], 1, cv2.LINE_AA)
            model_rows.append({"path_id": path["id"], "historical_model_label": label, "family": family, "label_uv": list(position)})
    candidates = []
    for index, crossing in enumerate(rows):
        label = "cross_"+chr(ord("a")+index)
        uv = np.asarray(crossing["uv"])
        point = tuple(int(x) for x in np.rint(uv))
        cv2.drawMarker(output, point, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 3, cv2.LINE_AA)
        target = (min(point[0]+45, 1600), max(point[1]-22, 60))
        cv2.line(output, point, target, (255, 255, 255), 1, cv2.LINE_AA)
        text = "%s (%.1f,%.1f) UNBOUND" % (label, uv[0], uv[1])
        cv2.putText(output, text, target, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, text, target, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        candidates.append({"id": label, "source_measurement_id": crossing["id"], "native_pts": native_pts, "uv": uv.tolist(), "model_identity_assigned": False})
    cv2.rectangle(output, (0, 0), (1920, 62), (0, 0, 0), -1)
    cv2.putText(output, "d78 SAME INITIAL MODEL | PTS %d | source %.2fs | CROSSINGS UNBOUND" % (native_pts, native_pts/PTS_DEN), (14, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    for i, family in enumerate(("C0", "C", "house", "lane")):
        cv2.putText(output, family+" model curve", (14+i*360, 50), cv2.FONT_HERSHEY_SIMPLEX, .55, colors[family], 1, cv2.LINE_AA)
    target = package.directory/"overlays"/("intersection_navigation_%d.png" % native_pts)
    if not cv2.imwrite(str(target), output):
        raise OSError("navigation image export failed")
    report = {"schema": "rinksolve.d78_unbound_intersection_navigation.v1", "path": str(target), "source_native_image": str(raw),
              "measurements": str(measurements_path), "native_pts": native_pts, "source_seconds": native_pts/PTS_DEN,
              "model_labels_are_historical_initialization_only": True, "image_candidates": candidates, "visible_highlighted_model_paths": model_rows,
              "no_nearest_prediction_or_semantic_binding_performed": True, "line_arc_identity_confirmed": False}
    write_json(target.with_suffix(".json"), report)
    return report



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("import")
    create.add_argument("--manifest", required=True, type=Path)
    create.add_argument("--out", required=True, type=Path)
    create.add_argument("--material", type=Path)
    create.add_argument("--predictions", type=Path)
    check = sub.add_parser("check")
    check.add_argument("--init-dir", required=True, type=Path)
    check.add_argument("--material", type=Path)
    check.add_argument("--predictions", type=Path)
    review = sub.add_parser("stills")
    review.add_argument("--init-dir", required=True, type=Path)
    review.add_argument("--historical-video", required=True, type=Path)
    review.add_argument("--pts", required=True, type=int, nargs="+")
    nav = sub.add_parser("navigation")
    nav.add_argument("--init-dir", required=True, type=Path)
    nav.add_argument("--measurements", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "import":
        package = import_d78(args.manifest, args.out, material_path=args.material, predictions_path=args.predictions)
        report = read_json(package.directory/"forward_proof.json")
    elif args.command == "check":
        package = load_initialization(args.init_dir)
        report = compare_frozen_predictions(package, args.predictions, args.material)
        audit_border_events(package)
        export_typed_bundle(package, report)
    elif args.command == "stills":
        package = load_initialization(args.init_dir)
        report = render_selected_frames(package, args.pts, args.historical_video)
    else:
        package = load_initialization(args.init_dir)
        report = render_intersection_navigation(package, args.measurements)
    print(json.dumps({k: v for k, v in report.items() if k not in {"frames", "limitations", "invalid_new_RS_source_rows"}}, ensure_ascii=False, indent=2))
    return 0 if args.command in {"stills", "navigation"} or report.get("projection_arithmetic_equivalence_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
