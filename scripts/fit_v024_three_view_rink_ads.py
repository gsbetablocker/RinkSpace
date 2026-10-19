"""Joint three-view rink, board-advertisement and full central-camera candidate.

Input observations are an immutable, source-linked manifest.  This script uses
v0.23 as the geometry/topology seed.  Each low frame has a separate camera,
including separate lens intrinsics.  Camera vectors contain optical centre,
Rodrigues pose, fx/fy/cx/cy/skew and all 14 OpenCV distortion coefficients.
Those 25 parameters per camera are all active in the final fitting stage.

The scene's user-confirmed structural relations remain exact through
SharedGeometry.  Advertisements are distinct vertical rectangles on identified
board segments, except partial edges and advertisement 5's *internal* graphic.
No equal dimensions or exact 180-degree position is imposed on counterpart ads.

This produces a development candidate.  Pixel residuals on fitted observations
do not independently validate geometry, a camera installation, or metric scale.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rinkspace.calibration_companion import Scene  # noqa: E402
from rinkspace.calibration_companion.joint_scene import JointSceneProblem, SharedGeometry  # noqa: E402
from rinkspace.calibration_companion.scene_camera import nearest_polyline_vectors  # noqa: E402

SCENE = _resolve_asset(ROOT, ROOT / "outputs/rink_model/ankara_2026_shared/v0.23-high-view-ad-corners/scene.json")
GEOMETRY_PROBLEM = _resolve_asset(ROOT, ROOT / "outputs/calibration/ankara_2026_wide_joint_20260923/A_pair_symmetry_A1_upper_A2_balance_v5/problem.json")
HIGH_ADS = _resolve_asset(ROOT, ROOT / "outputs/rink_model/ankara_2026_shared/v0.23-high-view-ad-corners/high_view_ad_corners_56218880.json")
LOW2_ADS = _resolve_asset(ROOT, ROOT / "evidence/low_end_reference_review_20260923/frame2_user_ad_corners_20260924.json")
LOW3_ADS = _resolve_asset(ROOT, ROOT / "evidence/low_end_reference_review_20260923/frame3_user_ad_corners_20260924.json")
HIGH_CAMERA = _resolve_asset(ROOT, ROOT / "outputs/rink_model/ankara_2026_shared/v0.20-wide-A-pair-upper-arc/scene.json")
LOW2_FIT = _resolve_asset(ROOT, ROOT / "outputs/calibration/ankara_2026_low_end_single_104693760_native_20260924/fit.json")
LOW3_FIT = _resolve_asset(ROOT, ROOT / "outputs/calibration/ankara_2026_low_end_single_205992960_full_lens_20260924/summary.json")
LOW3_NATIVE_FIT = _resolve_asset(ROOT, ROOT / "outputs/calibration/ankara_2026_low_end_single_205992960_lane_line_user_contact_20260924/fit.json")
DEFAULT_INPUT = _resolve_asset(ROOT, ROOT / "evidence/three_view_joint_20260924/input_observations.json")
DEFAULT_OUT = _resolve_asset(ROOT, ROOT / "outputs/calibration/ankara_2026_v024_three_view_rink_ads_20260924")
PTS = (56218880, 104693760, 205992960)
DIST_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
              "s1", "s2", "s3", "s4", "tauX", "tauY")
CAMERA_NAMES = ("center_x", "center_y", "center_z", "rvec_x", "rvec_y", "rvec_z",
                "log_fx", "log_fy", "cx", "cy", "skew") + DIST_NAMES
EYE = np.eye(3, dtype=np.float64)
ZERO = np.zeros(3, dtype=np.float64)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def camera_vector(camera: dict) -> np.ndarray:
    k = np.asarray(camera["K_px"], float)
    d = np.asarray(camera.get("distortion_opencv", []), float)
    if len(d) not in (0, 4, 5, 8, 12, 14):
        raise ValueError("unsupported camera distortion length")
    return np.r_[camera["center_xyz"], camera["rvec_world_to_camera"],
                 np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2], k[0, 1],
                 np.pad(d, (0, 14 - len(d)))]


def camera_seed(pts: int) -> np.ndarray:
    if pts == 56218880:
        scene = read(HIGH_CAMERA)
        frame = next(f for f in scene["frame_cameras"] if f["source_pts"] == pts)
        return camera_vector(frame["camera"])
    if pts == 104693760:
        fit = read(LOW2_FIT)
        frame = next(f for f in fit["frames"] if f["source_pts"] == pts)
        return camera_vector(frame["camera"])
    if pts == 205992960:
        return camera_vector(read(LOW3_FIT)["camera"])
    raise ValueError(f"unknown source PTS {pts}")


def camera_bounds(pts: int) -> tuple[np.ndarray, np.ndarray]:
    if pts == 56218880:
        lo_center, hi_center = [-250., -700., .5], [700., 250., 400.]
    else:
        lo_center, hi_center = [-300., -250., .5], [100., 250., 150.]
    lo = np.r_[lo_center, [-4., -4., -4.], [math.log(350.), math.log(350.)],
               [-1920., -1080., -3000.],
               [-3., -3., -1., -1., -3., -3., -3., -3., -1., -1., -1., -1., -.7, -.7]]
    hi = np.r_[hi_center, [4., 4., 4.], [math.log(30000.), math.log(30000.)],
               [3840., 2160., 3000.],
               [3., 3., 1., 1., 3., 3., 3., 3., 1., 1., 1., 1., .7, .7]]
    return lo, hi


def project(v: np.ndarray, xyz: np.ndarray, *, allow_behind: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Apply Brown14 before sensor scaling, including real (nonignored) skew."""
    points = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
    rot = cv2.Rodrigues(np.ascontiguousarray(v[3:6]))[0]
    q = np.ascontiguousarray((rot @ (points - v[:3]).T).T)
    depth = q[:, 2]
    if not np.isfinite(q).all() or (not allow_behind and np.any(depth <= .05)):
        raise ValueError("nonpositive or nonfinite camera depth")
    valid = depth > .05
    if allow_behind:
        normalized = q[:, :2] / np.maximum(depth[:, None], .05)
        valid &= np.max(np.abs(normalized), axis=1) < 8.
    if (np.count_nonzero(valid) < 2 if allow_behind else not np.all(valid)):
        raise ValueError("too few forward projected points")
    # projectPoints does not apply K[0,1].  Use identity K, then add skew.
    norm = cv2.projectPoints(np.ascontiguousarray(q[valid]), ZERO, ZERO, EYE,
                             np.ascontiguousarray(v[11:25]))[0].reshape(-1, 2)
    forward = np.column_stack([math.exp(v[6]) * norm[:, 0] + v[10] * norm[:, 1] + v[8],
                               math.exp(v[7]) * norm[:, 1] + v[9]])
    good = np.isfinite(forward).all(axis=1) & (np.max(np.abs(forward), axis=1) < 1e5)
    if (not allow_behind and not np.all(good)) or (allow_behind and np.count_nonzero(good) < 2):
        raise ValueError("nonfinite or runaway projection")
    uv = np.full((len(points), 2), np.nan)
    uv[np.flatnonzero(valid)[good]] = forward[good]
    return uv, depth


def distort_norm(xy: np.ndarray, d: np.ndarray) -> np.ndarray:
    q = np.column_stack([xy, np.ones(len(xy))]).astype(np.float64)
    return cv2.projectPoints(np.ascontiguousarray(q), ZERO, ZERO, EYE,
                             np.ascontiguousarray(d))[0].reshape(-1, 2)


def lens_guard(v: np.ndarray) -> np.ndarray:
    """Penalize folding and Brown rational poles over this camera's sensor rays.

    The historical wide-view lens is valid on its 1920x1080 sensor, although a
    fixed [-.95,.95] by [-.65,.65] rectangle extends well outside that sensor
    and spuriously rejects it.  Give the actual sensor a five-percent border.
    """
    fx, fy = math.exp(v[6]), math.exp(v[7])
    ys = np.array([-v[9]/fy, (1080.-v[9])/fy])
    xs = np.array([(u-v[8]-v[10]*y)/fx for u in (0., 1920.) for y in ys])
    xlo, xhi = float(xs.min()), float(xs.max())
    ylo, yhi = float(ys.min()), float(ys.max())
    xpad, ypad = .05*(xhi-xlo), .05*(yhi-ylo)
    xx, yy = np.meshgrid(np.linspace(xlo-xpad, xhi+xpad, 9),
                         np.linspace(ylo-ypad, yhi+ypad, 7))
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    eps = 1e-4
    xstep, ystep = np.array([eps, 0.]), np.array([0., eps])
    try:
        mapped = distort_norm(xy, v[11:])
        jx = (distort_norm(xy + xstep, v[11:]) - distort_norm(xy - xstep, v[11:])) / (2 * eps)
        jy = (distort_norm(xy + ystep, v[11:]) - distort_norm(xy - ystep, v[11:])) / (2 * eps)
        determinant = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
        r2 = np.sum(xy * xy, axis=1)
        k4, k5, k6 = v[16:19]
        denominator = 1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2
        if not np.isfinite(mapped).all() or not np.isfinite(determinant).all():
            raise ValueError("nonfinite lens map")
        return np.r_[30*np.maximum(.10-determinant, 0),
                     30*np.maximum(.10-denominator, 0),
                     2*np.maximum(np.max(np.abs(mapped), axis=1)-4., 0)]
    except (ValueError, cv2.error, FloatingPointError):
        return np.full(3*len(xy), 1000.)


def nearest_delta(observed: np.ndarray, projected: np.ndarray,
                  supporting_line: bool = False) -> np.ndarray:
    if supporting_line:
        forward = projected[np.isfinite(projected).all(axis=1)]
        if len(forward) < 2:
            raise ValueError("supporting line has fewer than two forward projected points")
        a, b = forward[0], forward[-1]
        direction = b-a
        denominator = max(float(direction @ direction), 1e-20)
        fraction = ((observed-a) @ direction) / denominator
        return observed - (a + fraction[:, None]*direction)
    # Keep source-curve adjacency; never connect across a horizon or lens pole.
    a, b = projected[:-1], projected[1:]
    direction = b-a
    good = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) &
            (np.linalg.norm(direction, axis=1) < 1200.))
    if not np.any(good):
        raise ValueError("no continuous finite projected curve segment")
    a, direction = a[good], direction[good]
    denominator = np.maximum(np.sum(direction*direction, axis=1), 1e-20)
    t = np.sum((observed[:, None, :]-a[None, :, :])*direction[None, :, :], axis=2) / denominator[None, :]
    foot = a[None, :, :] + np.clip(t, 0, 1)[:, :, None]*direction[None, :, :]
    differences = observed[:, None, :]-foot
    nearest = np.argmin(np.sum(differences*differences, axis=2), axis=1)
    return differences[np.arange(len(observed)), nearest]


def normalized_observations(manifest: dict) -> list[dict]:
    frame_by_id = {f["id"]: f for f in manifest["frames"]}
    if len(frame_by_id) != 3 or tuple(sorted(int(f["source_pts"]) for f in frame_by_id.values())) != tuple(sorted(PTS)):
        raise ValueError("manifest must contain the exact three source PTS, each with its own frame ID")
    for frame in manifest["frames"]:
        image = root_path(frame["image_path"])
        if sha(image) != frame["image_sha256"]:
            raise ValueError(f"source frame hash mismatch: {image}")
        if tuple(frame["image_size_wh"]) != (1920, 1080):
            raise ValueError("all three observations must be native 1920x1080 pixels")
    result, seen = [], set()
    for original in manifest["observations"]:
        row = copy.deepcopy(original)
        if row["id"] in seen or row["frame_id"] not in frame_by_id:
            raise ValueError("duplicate observation ID or unknown frame")
        seen.add(row["id"])
        if row["use"] not in ("fit", "check") or row["type"] not in ("curve", "point"):
            raise ValueError(f"invalid use/type: {row['id']}")
        uv = np.asarray(row["pixels_uv"], float).reshape(-1, 2)
        if (not len(uv) or not np.isfinite(uv).all() or
                np.any(uv < 0) or np.any(uv[:, 0] >= 1920) or np.any(uv[:, 1] >= 1080) or
                float(row["sigma_px"]) <= 0):
            raise ValueError(f"invalid source pixels or uncertainty: {row['id']}")
        if row["type"] == "point" and len(uv) != 1:
            raise ValueError("point observation must contain one pixel")
        # The low view's visible lane fragment establishes only its image line.
        target = row["target"]
        if target["kind"] == "polygon_edge" and target.get("image_supporting_line_only"):
            row["target"] = {"kind": "polygon_supporting_line",
                             "feature_id": target["feature_id"], "index": target["index"]}
            row["target_conversion"] = "visible finite fragment to infinite image support line"
        row["uv"] = uv
        row["frame_pts"] = int(frame_by_id[row["frame_id"]]["source_pts"])
        result.append(row)
    if not any(r["frame_pts"] == 205992960 and r["use"] == "fit" and
               r["target"].get("feature_id") == "F1" for r in result):
        raise ValueError("frame 3 F1 must enter the fit")
    contact = [r for r in result if r["frame_pts"] == 205992960 and
               r["target"].get("kind") == "board_side" and r["target"].get("edge") == "contact"]
    if len(contact) < 4 or sum(len(r["uv"]) for r in contact) != 22:
        raise ValueError("frame 3 must use the six groups / 22 points of the new user-traced contact")
    if not all("user" in r.get("evidence_role", "").lower() for r in contact):
        raise ValueError("frame 3 contact pixels must have user-trace provenance")
    return result


def ads_from_sources(manifest: dict) -> list[dict]:
    expected = {sha(root_path(entry["path"])): root_path(entry["path"]) for entry in manifest["advertisement_annotation_files"]}
    paths = [HIGH_ADS, LOW2_ADS, LOW3_ADS]
    if len(expected) != 3 or any(sha(p) not in expected or expected[sha(p)].resolve() != p.resolve() for p in paths):
        raise ValueError("advertisement annotation sources do not match the three existing JSON files")
    entries = []
    for pts, path in zip(PTS, paths):
        data = read(path)
        source = data["source_frame"]
        if source["source_pts"] != pts:
            raise ValueError("advertisement source PTS mismatch")
        for ad in data["advertisements"]:
            for corner in ad.get("corner_observations", ad.get("corners", [])):
                entries.append(dict(number=int(ad["user_number"]), pts=pts,
                                    letter=corner["letter"], role=corner["role"],
                                    uv=np.asarray(corner["source_frame_uv"], float),
                                    scope=ad["observed_feature_scope"],
                                    source_file=str(path.relative_to(ROOT))))
    if len(entries) != 26+13+16:
        raise ValueError("expected the user's 26 + 13 + 16 annotated advertisement corners")
    if any(not np.isfinite(r["uv"]).all() for r in entries):
        raise ValueError("nonfinite advertisement point")
    return entries


AD_SIDES = {1: "upper_right", 2: "upper_long_side", 3: "upper_long_side",
            4: "upper_long_side", 5: "upper_long_side", 6: "upper_long_side",
            7: "upper_left", 17: "lower_long_side", 19: "lower_long_side",
            20: "lower_right"}
AD_SCOPE = {5: "internal_graphic_only"}


def along_fraction(side: str, xyz: np.ndarray, params: dict) -> float:
    x0, y0, x1, y1 = map(float, params["bounds"])
    r = params["corner_radii"]
    if side == "upper_long_side":
        return float((xyz[0] - (x0+r["upper_left"])) /
                     (x1-r["upper_right"]-x0-r["upper_left"]))
    if side == "lower_long_side":
        return float((xyz[0] - (x0+r["lower_left"])) /
                     (x1-r["lower_right"]-x0-r["lower_left"]))
    if side in ("upper_left", "upper_right", "lower_right"):
        cx = x0+r[side] if "left" in side else x1-r[side]
        cy = y1-r[side] if "upper" in side else y0+r[side]
        angle = math.degrees(math.atan2(xyz[1]-cy, xyz[0]-cx)) % 360
        start = {"upper_right": 0., "upper_left": 90., "lower_right": 270.}[side]
        return float((angle-start)/90.)
    raise ValueError(side)


def board_xyz(side: str, fraction: float, z: float, params: dict) -> np.ndarray:
    x0, y0, x1, y1 = map(float, params["bounds"])
    r = params["corner_radii"]
    if side == "upper_long_side":
        return np.array([x0+r["upper_left"] + fraction*(x1-r["upper_right"]-x0-r["upper_left"]), y1, z])
    if side == "lower_long_side":
        return np.array([x0+r["lower_left"] + fraction*(x1-r["lower_right"]-x0-r["lower_left"]), y0, z])
    if side in ("upper_left", "upper_right", "lower_right"):
        cx = x0+r[side] if "left" in side else x1-r[side]
        cy = y1-r[side] if "upper" in side else y0+r[side]
        theta = math.radians({"upper_right": 0., "upper_left": 90., "lower_right": 270.}[side] + 90*fraction)
        return np.array([cx+r[side]*math.cos(theta), cy+r[side]*math.sin(theta), z])
    raise ValueError(side)


class ThreeViewProblem:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.manifest = read(manifest_path)
        if root_path(self.manifest["base_scene_path"]).resolve() != SCENE.resolve() or sha(SCENE) != self.manifest["base_scene_sha256"]:
            raise ValueError("manifest must target the exact corrected v0.23 scene")
        for source in self.manifest["sources"]:
            if sha(root_path(source["path"])) != source["sha256"]:
                raise ValueError(f"source observation hash mismatch: {source['path']}")
        self.scene = Scene(SCENE)
        source = read(GEOMETRY_PROBLEM)
        source["scene_sha256"] = self.scene.sha256
        source["base_scene_path"] = str(SCENE)
        geometry_base = JointSceneProblem(self.scene, source)
        self.geometry_base = geometry_base
        self.specs = geometry_base.scene_specs
        self.hard_constraints = geometry_base.hard_geometry_constraints
        self.geometry_count = geometry_base.scene_count
        self.geometry_bounds = tuple(b[:self.geometry_count].copy() for b in geometry_base.bounds)
        self.obs = normalized_observations(self.manifest)
        self.ad_obs = ads_from_sources(self.manifest)
        self.frames = {int(f["source_pts"]): f for f in self.manifest["frames"]}
        self.frame_id_to_pts = {f["id"]: int(f["source_pts"]) for f in self.manifest["frames"]}
        self.slices = {}
        values = list(geometry_base.x0[:self.geometry_count])
        lower = list(self.geometry_bounds[0]); upper = list(self.geometry_bounds[1])
        self.names = ["scene:"+s["id"] for s in self.specs]
        for pts in PTS:
            start = len(values)
            camera = camera_seed(pts)
            lo, hi = camera_bounds(pts)
            if not np.all((camera >= lo) & (camera <= hi)):
                raise ValueError(f"camera seed {pts} lies outside its broad numerical domain")
            values.extend(camera.tolist()); lower.extend(lo.tolist()); upper.extend(hi.tolist())
            self.slices[("camera", pts)] = slice(start, len(values))
            self.names.extend(f"camera:{pts}:{name}" for name in CAMERA_NAMES)
        params = self.geometry(np.asarray(values)).parameters
        high = read(HIGH_ADS)
        seed_ads = {int(a["user_number"]): a for a in high["advertisements"]}
        self.ad_roles = defaultdict(set)
        for row in self.ad_obs:
            self.ad_roles[row["number"]].add(row["role"])
        for number in sorted(self.ad_roles):
            if number not in AD_SIDES:
                raise ValueError(f"unknown advertisement board side for #{number}")
            side = AD_SIDES[number]
            has_right = any(role.startswith("right_") for role in self.ad_roles[number])
            if number in seed_ads:
                corners = seed_ads[number]["fit"]["fitted_corner_points"]
                left = along_fraction(side, np.asarray(corners["left_top"]["xyz_scene_units"]), params)
                right = (along_fraction(side, np.asarray(corners["right_top"]["xyz_scene_units"]), params)
                         if "right_top" in corners else None)
                top = corners["left_top"]["xyz_scene_units"][2] / params["board_height"]
                bottom = corners["left_bottom"]["xyz_scene_units"][2] / params["board_height"]
                if number == 1:
                    right = max(.05, left-.36)
            else:
                # Rotation by 180 degrees is a numerical seed only.  Counterpart
                # coordinates and sizes remain completely independent variables.
                counterpart = {17: 4, 19: 6, 20: 7}[number]
                old = seed_ads[counterpart]["fit"]["fitted_corner_points"]
                mirror = lambda role: -np.asarray(old[role]["xyz_scene_units"])
                left = along_fraction(side, mirror("left_top"), params)
                right = (along_fraction(side, mirror("right_top"), params)
                         if has_right and "right_top" in old else None)
                top = old["left_top"]["xyz_scene_units"][2] / params["board_height"]
                bottom = old["left_bottom"]["xyz_scene_units"][2] / params["board_height"]
            entries = [np.clip(left, .001, .999)]
            if has_right:
                entries.append(np.clip(right if right is not None else left+.1, .001, .999))
            entries += [np.clip(top, .001, .999), np.clip(bottom, .001, .999)]
            start = len(values)
            values.extend(entries); lower.extend([0.001]*len(entries)); upper.extend([.999]*len(entries))
            self.slices[("ad", number)] = slice(start, len(values))
            self.names.extend(f"ad:{number}:{key}" for key in (["left", "right"] if has_right else ["left"])+["top", "bottom"])
        self.x0 = np.asarray(values, float)
        self.bounds = (np.asarray(lower, float), np.asarray(upper, float))
        self.fit_obs = [o for o in self.obs if o["use"] == "fit"]
        self.check_obs = [o for o in self.obs if o["use"] == "check"]
        # A user's continuous contact stroke and multiple traced fragments of
        # the same physical feature are correlated evidence.  Normalize each
        # correlation ID to one group, then balance unique groups per frame.
        self.correlated_group_sizes = Counter(o["correlation_group_id"] for o in self.fit_obs)
        counts = {pts: len({o["correlation_group_id"] for o in self.fit_obs
                            if o["frame_pts"] == pts}) +
                       len({a["number"] for a in self.ad_obs if a["pts"] == pts})
                  for pts in PTS}
        self.frame_factor = {pts: 1/math.sqrt(counts[pts]) for pts in PTS}
        self.ad_frame_sizes = Counter((a["pts"], a["number"]) for a in self.ad_obs)
        self.n_data = 2*sum(len(o["uv"]) for o in self.fit_obs) + 2*len(self.ad_obs)
        self.n_guard = 3*9*7*3 + len(self.slices_ad()) + 10 + 2
        self.n_reg = 3*18
        self.residual_size = self.n_data+self.n_guard+self.n_reg
        self.initialization = []
        self._initialize_low2_from_shared_ads()
        self._initialize_low3_candidate_from_ads()

    def slices_ad(self):
        return {key: sl for key, sl in self.slices.items() if key[0] == "ad"}

    def _initialize_low2_from_shared_ads(self) -> None:
        """Seed low2 extrinsics from same physical corner IDs, never as truth."""
        pts = 104693760
        geo = self.geometry(self.x0)
        rows = [row for row in self.ad_obs if row["pts"] == pts and row["number"] in (4, 5, 6, 7)]
        world = np.asarray([self.ad_world(self.x0, geo, row["number"], row["role"])
                            for row in rows], np.float64)
        observed = np.asarray([row["uv"] for row in rows], np.float64)
        sl = self.slices[("camera", pts)]
        v = self.x0[sl].copy()
        K = np.array([[math.exp(v[6]), 0., v[8]], [0., math.exp(v[7]), v[9]], [0., 0., 1.]])
        R = cv2.Rodrigues(v[3:6])[0]
        t = -R @ v[:3]
        before = np.linalg.norm(project(v, world)[0]-observed, axis=1)
        success, rvec, tvec = cv2.solvePnP(world, observed, K,
                                            np.ascontiguousarray(v[11:25]),
                                            np.ascontiguousarray(v[3:6]).reshape(3, 1),
                                            np.ascontiguousarray(t).reshape(3, 1),
                                            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            raise ValueError("low2 same-corner PnP seed failed")
        candidate = v.copy()
        candidate[3:6] = rvec.ravel()
        candidate[:3] = -cv2.Rodrigues(rvec)[0].T @ tvec.ravel()
        after = np.linalg.norm(project(candidate, world)[0]-observed, axis=1)
        self.initialization.append(dict(frame_pts=pts, method="solvePnP_same_physical_ad_corners_seed_only",
            source_advertisement_numbers=sorted({row["number"] for row in rows}),
            n_corners=len(rows), before_rms_px=float(np.sqrt(np.mean(before**2))),
            after_rms_px=float(np.sqrt(np.mean(after**2))),
            source_world_geometry_status="conditional_high_view_v023_rectangle_candidates_not_measured"))
        if np.sqrt(np.mean(after**2)) < np.sqrt(np.mean(before**2)):
            self.x0[sl] = candidate

    def _initialize_low3_candidate_from_ads(self) -> None:
        """Use counterpart reflection *only* to seed a separate low3 camera.

        The counterpart rectangles remain independent variables.  A PnP pose
        using their reflected initial guesses has no evidentiary status and is
        selected only if the full balanced three-view data objective improves.
        """
        pts = 205992960
        geo = self.geometry(self.x0)
        rows = [row for row in self.ad_obs if row["pts"] == pts]
        world = np.asarray([self.ad_world(self.x0, geo, row["number"], row["role"])
                            for row in rows], np.float64)
        observed = np.asarray([row["uv"] for row in rows], np.float64)
        sl = self.slices[("camera", pts)]
        baseline = self.x0.copy()
        base_norm = float(np.linalg.norm(self._data_parts(baseline)[0]))
        native = read(LOW3_NATIVE_FIT)
        initial = camera_vector(next(f for f in native["frames"] if f["source_pts"] == pts)["camera"])
        initial[10] = 0.
        initial[11:25] = 0.
        K = np.array([[math.exp(initial[6]), 0., initial[8]],
                      [0., math.exp(initial[7]), initial[9]], [0., 0., 1.]])
        R = cv2.Rodrigues(initial[3:6])[0]
        t = -R @ initial[:3]
        success, rvec, tvec = cv2.solvePnP(world, observed, K, np.zeros(14),
                                            np.ascontiguousarray(initial[3:6]).reshape(3, 1),
                                            np.ascontiguousarray(t).reshape(3, 1),
                                            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            raise ValueError("low3 counterpart-seed PnP failed")
        candidate = initial.copy()
        candidate[3:6] = rvec.ravel()
        candidate[:3] = -cv2.Rodrigues(rvec)[0].T @ tvec.ravel()
        proposal = baseline.copy()
        proposal[sl] = candidate
        proposed_norm = float(np.linalg.norm(self._data_parts(proposal)[0]))
        after = np.linalg.norm(project(candidate, world)[0]-observed, axis=1)
        accepted = proposed_norm < base_norm
        self.initialization.append(dict(frame_pts=pts, method="solvePnP_counterpart_reflection_as_seed_only",
            source_advertisement_numbers=sorted({row["number"] for row in rows}),
            n_corners=len(rows), advertisement_rms_px=float(np.sqrt(np.mean(after**2))),
            balanced_data_norm_before=base_norm, balanced_data_norm_after=proposed_norm,
            accepted_as_initial=accepted, no_exact_symmetric_constraint=True))
        if accepted:
            self.x0 = proposal

    def geometry(self, x: np.ndarray) -> SharedGeometry:
        return SharedGeometry(self.scene, self.specs, x[:self.geometry_count],
                              hard_geometry_constraints=self.hard_constraints)

    def ad_world(self, x: np.ndarray, geo: SharedGeometry, number: int, role: str) -> np.ndarray:
        values = x[self.slices[("ad", number)]]
        has_right = len(values) == 4
        along = values[1] if role.startswith("right_") and has_right else values[0]
        if role.startswith("right_") and not has_right:
            raise ValueError(f"#{number} has an unobserved right edge")
        zfraction = values[-2] if role.endswith("top") else values[-1]
        return board_xyz(AD_SIDES[number], float(along), float(zfraction)*geo.parameters["board_height"],
                         geo.parameters)

    def curve_world(self, geo: SharedGeometry, row: dict) -> np.ndarray:
        target = row["target"]
        samples = 170 if target["kind"] == "board_side" else 250 if target["kind"] == "circle_outline" else 80
        return np.asarray(geo.points(target, row["type"], samples), float)

    def _data_parts(self, x: np.ndarray, with_details: bool = False):
        geo = self.geometry(x)
        parts, details = [], []
        for row in self.fit_obs if not with_details else self.obs:
            camera = x[self.slices[("camera", row["frame_pts"])]]
            world = self.curve_world(geo, row)
            prediction, _ = project(camera, world, allow_behind=row["type"] == "curve")
            # A straight painted world line generally projects to a *curved*
            # image line with Brown distortion.  Use its sampled image curve,
            # including only finite adjacent forward segments.
            delta = (prediction-row["uv"] if row["type"] == "point" else
                     nearest_delta(row["uv"], prediction))
            magnitude = np.linalg.norm(delta, axis=1)
            details.append(dict(id=row["id"], frame_pts=row["frame_pts"], target=row["target"],
                                use=row["use"], count=len(magnitude), sigma_px=float(row["sigma_px"]),
                                median_px=float(np.median(magnitude)), rms_px=float(np.sqrt(np.mean(magnitude**2))),
                                p95_px=float(np.percentile(magnitude, 95)), max_px=float(magnitude.max()),
                                evidence_role=row.get("evidence_role")))
            if row["use"] == "fit":
                weight = self.frame_factor[row["frame_pts"]] / (
                    float(row["sigma_px"])*math.sqrt(len(row["uv"])*
                    self.correlated_group_sizes[row["correlation_group_id"]]))
                parts.append((delta*weight).ravel())
        for row in self.ad_obs:
            camera = x[self.slices[("camera", row["pts"])]]
            world = self.ad_world(x, geo, row["number"], row["role"])
            prediction, _ = project(camera, world)
            delta = prediction[0]-row["uv"]
            magnitude = float(np.linalg.norm(delta))
            details.append(dict(id=f"ad:{row['number']}:{row['letter']}:{row['pts']}",
                                frame_pts=row["pts"], target={"kind": "advert_corner", "user_number": row["number"],
                                "role": row["role"], "letter": row["letter"]}, use="fit", count=1,
                                sigma_px=4., median_px=magnitude, rms_px=magnitude,
                                p95_px=magnitude, max_px=magnitude,
                                evidence_role="user_marked_advertisement_corner"))
            factor = self.frame_factor[row["pts"]] / math.sqrt(self.ad_frame_sizes[(row["pts"], row["number"])])
            parts.append(delta * factor / 4.)
        return np.concatenate(parts), details, geo

    def residual(self, x: np.ndarray, reg: float) -> np.ndarray:
        try:
            data, _, geo = self._data_parts(x)
            guards = []
            for pts in PTS:
                guards.append(lens_guard(x[self.slices[("camera", pts)]]))
            for (_, number), sl in self.slices_ad().items():
                values = x[sl]
                sense = 1. if AD_SIDES[number] == "upper_long_side" else -1.
                width_penalty = (max(.005-sense*(values[1]-values[0]), 0.)
                                 if len(values)==4 else 0.)
                guards.append(np.array([100*width_penalty +
                                        100*max(.005-(values[-2]-values[-1]), 0.)]))
            margins = list(geo.boundary_validity_margins().values())
            if len(margins) != 10:
                raise ValueError("unexpected number of boundary validity guards")
            guards.append(30*np.minimum(np.asarray(margins)-1e-5, 0.))
            # Both low views look into the rink from beyond its left end.
            # The camera centre must remain outside that end's boards as the
            # shared boundary moves during the joint fit.  This is a physical
            # soft guard, independently checked as a hard acceptance condition.
            left_end = float(geo.parameters["bounds"][0])
            guards.append(np.asarray([60.*max(x[self.slices[("camera", pts)]][0]
                                                - (left_end-.5), 0.)
                                      for pts in PTS[1:]]))
            regularizers = []
            scale = np.array([1, 1, .3, .3, 1, 1, 1, 1, .2, .2, .2, .2, .2, .2], float)
            for pts in PTS:
                cam = x[self.slices[("camera", pts)]]
                regularizers.append(reg*np.r_[cam[11:]/scale, cam[10]/1000.,
                                              (cam[8]-960.)/960., (cam[9]-540.)/540.,
                                              (cam[6]-cam[7])])
            result = np.r_[data, *guards, *regularizers]
            if len(result) != self.residual_size or not np.isfinite(result).all():
                raise ValueError("objective length/nonfinite mismatch")
            return result
        except (ValueError, cv2.error, FloatingPointError, OverflowError):
            return np.full(self.residual_size, 1000.)

    def camera_dict(self, x: np.ndarray, pts: int) -> dict:
        v = x[self.slices[("camera", pts)]]
        fx, fy = math.exp(v[6]), math.exp(v[7])
        return {"center_xyz": v[:3].tolist(), "rvec_world_to_camera": v[3:6].tolist(),
                "K_px": [[fx, float(v[10]), float(v[8])], [0., fy, float(v[9])], [0., 0., 1.]],
                "distortion_opencv": v[11:].tolist(),
                "projection_convention": "Brown14 then fx/fy/cx/cy/skew; Xcamera=R@(Xworld-center)"}

    def advertisement_estimates(self, x: np.ndarray) -> list[dict]:
        geo = self.geometry(x)
        result = []
        for (_, number), sl in sorted(self.slices_ad().items(), key=lambda pair: pair[0][1]):
            pars = x[sl]
            roles = sorted(self.ad_roles[number])
            corners = {role: self.ad_world(x, geo, number, role).tolist() for role in roles}
            result.append(dict(user_number=number, board_segment=AD_SIDES[number],
                               extent_scope=AD_SCOPE.get(number, "external_border" if len(pars)==4 else "observed_edge_only"),
                               fitted_roles=roles, corners_xyz_scene_units=corners,
                               along_fractions=pars[:2].tolist() if len(pars)==4 else [float(pars[0]), None],
                               top_z_scene_units=float(pars[-2]*geo.parameters["board_height"]),
                               bottom_z_scene_units=float(pars[-1]*geo.parameters["board_height"]),
                               width_status="estimated_from_observed_corners" if len(pars)==4 else "unobserved",
                               status="three_view_joint_candidate_not_independent_validation"))
        return result


def metrics(details: list[dict]) -> dict:
    result = {}
    for pts in PTS:
        for category in ("ice_and_board", "advertisement"):
            subset = [r for r in details if r["frame_pts"] == pts and
                      (r["target"]["kind"] == "advert_corner") == (category == "advertisement")]
            if subset:
                result[f"{pts}:{category}"] = dict(groups=len(subset),
                    median_group_rms_px=float(np.median([r["rms_px"] for r in subset])),
                    worst_group_rms_px=float(max(r["rms_px"] for r in subset)),
                    worst_point_px=float(max(r["max_px"] for r in subset)))
    return result


def stage_indices(problem: ThreeViewProblem, stage: int) -> list[int]:
    # All geometry and advertisement dimensions join at stage 1.  Final stage
    # releases every intrinsic, extrinsic and Brown14 term for all cameras.
    result = []
    if stage >= 1:
        result.extend(range(problem.geometry_count))
    camera_limit = (11, 11, 16, 23, 25)[stage]
    for pts in PTS:
        sl = problem.slices[("camera", pts)]
        result.extend(range(sl.start, sl.start+camera_limit))
    for sl in problem.slices_ad().values():
        result.extend(range(sl.start, sl.stop))
    return result


def solve(problem: ThreeViewProblem, stage_limits: list[int], reg: float,
          workers: int, output: Path, start_stage: int = 0,
          x_start: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    x = problem.x0.copy() if x_start is None else x_start.copy()
    records = [read(output / f"stage_{stage}.json")["record"] for stage in range(start_stage)
               if (output / f"stage_{stage}.json").exists()]
    for stage in range(start_stage, 5):
        max_nfev = stage_limits[stage]
        active = np.asarray(stage_indices(problem, stage), int)
        lo, hi = (b[active] for b in problem.bounds)
        def objective(values):
            full = x.copy()
            full[active] = values
            return problem.residual(full, reg if stage >= 2 else .05)
        options = dict(bounds=(lo, hi), method="trf", loss="soft_l1", f_scale=1.,
                       x_scale="jac", max_nfev=max_nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        before = float(np.linalg.norm(objective(x[active])))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fit = least_squares(objective, x[active], workers=pool.map, **options)
        else:
            fit = least_squares(objective, x[active], **options)
        x[active] = fit.x
        after = float(np.linalg.norm(objective(x[active])))
        data, details, geometry = problem._data_parts(x, with_details=True)
        record = dict(stage=stage, active_parameter_count=len(active), max_nfev=max_nfev,
                      evaluations=int(fit.nfev), status=int(fit.status), message=str(fit.message),
                      success=bool(fit.success), cost=float(fit.cost), objective_norm_before=before,
                      objective_norm_after=after, data_norm=float(np.linalg.norm(data)),
                      metrics=metrics(details),
                      camera_placement_checks=camera_placement_checks(problem, x, geometry))
        records.append(record)
        save(output / f"stage_{stage}.json", {"record": record,
            "parameter_names": problem.names, "parameter_values": x.tolist(),
            "camera_centers_xyz": {str(pts): problem.camera_dict(x, pts)["center_xyz"] for pts in PTS},
            "purpose": "reviewable optimization checkpoint; not physical validation"})
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return x, records


def camera_placement_checks(problem: ThreeViewProblem, x: np.ndarray, geometry: SharedGeometry) -> dict:
    left_end = float(geometry.parameters["bounds"][0])
    checks = {}
    for pts in PTS[1:]:
        center = x[problem.slices[("camera", pts)]][:3]
        margin = left_end-float(center[0])
        checks[str(pts)] = {"left_end_margin_scene_units": margin,
            "outside_left_end_boards": bool(margin >= .5-1e-4),
            "center_xyz": center.tolist(),
            "constraint": "camera center at least 0.5 scene units beyond left end boards",
            "numerical_tolerance_scene_units": 1e-4}
    return checks


def advertisement_validity_checks(problem: ThreeViewProblem, x: np.ndarray) -> list[dict]:
    checks = []
    for (_, number), sl in sorted(problem.slices_ad().items(), key=lambda pair: pair[0][1]):
        values = x[sl]
        direction = 1. if AD_SIDES[number] == "upper_long_side" else -1.
        width = float(direction*(values[1]-values[0])) if len(values) == 4 else None
        height = float(values[-2]-values[-1])
        checks.append({"user_number": number,
            "board_segment": AD_SIDES[number],
            "scope": AD_SCOPE.get(number, "external_border" if width is not None else "observed_edge_only"),
            "ordered_width_fraction": width,
            "vertical_height_fraction": height,
            "width_order_valid": bool(width is None or width > 1e-6),
            "height_order_valid": bool(height >= .005),
            "valid": bool((width is None or width > 1e-6) and height >= .005)})
    return checks


def export_candidate_scene(problem: ThreeViewProblem, x: np.ndarray, fit: dict,
                           output: Path) -> None:
    """Rebuild meshes/grid from the jointly changed geometry for independent QA."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_joint_scene_model import rebuild_scene

    geometry = problem.geometry(x)
    semantic = copy.deepcopy(problem.scene.data)
    semantic["parameters"] = copy.deepcopy(geometry.parameters)
    semantic["ice_features"] = copy.deepcopy(list(geometry.features.values()))
    frame_rows = []
    installation_rows = []
    for pts in PTS:
        source = problem.frames[pts]
        camera = problem.camera_dict(x, pts)
        frame_rows.append({"id": source["id"], "installation_id": source["id"],
            "source_pts": pts, "source_id": source.get("source_id"),
            "time_base": source.get("time_base"), "camera": camera,
            "camera_status": "independent_three_view_joint_candidate"})
        installation_rows.append({"id": source["id"], "center_xyz": camera["center_xyz"],
            "position_status": "joint_candidate_NOT_measured"})
    export_fit = {"frames": frame_rows, "installations": installation_rows,
                  "optimizer": {"success": all(s["success"] for s in fit["stage_results"]),
                                "stages": fit["stage_results"]}}
    scene, rebuilt_geometry, checks = rebuild_scene(semantic, export_fit,
        revision="v0.24-three-view-rink-ads-candidate")
    # rebuild_scene preserves the input revision's joint_refinement record.  It
    # must not make v0.23's old parameter changes or invalidated camera status
    # look like facts about the new three-view fit.
    base_vector = problem.geometry_base.x0.copy()
    base_vector[:problem.geometry_count] = x[:problem.geometry_count]
    joint = scene.setdefault("joint_refinement", {})
    joint.update({
        "base_scene_sha256": sha(SCENE),
        "coordinate_gauge": problem.geometry_base.coordinate_gauge_description(),
        "parameter_changes": problem.geometry_base.parameter_changes(base_vector),
        "retained_unobserved_parameters": copy.deepcopy(problem.geometry_base.retained_parameters),
        "board_surface_diagnostics": problem.geometry_base.board_surface_diagnostics(base_vector),
        "video_refitted_after_hard_structure": True,
        "camera_fit_status": "three_independent_full_Brown14_skew_joint_candidate_not_physically_validated",
        "prior_revision": {"revision": problem.scene.data["revision"],
                           "scene_path": str(SCENE.relative_to(ROOT)),
                           "scene_sha256": sha(SCENE)}})
    lane_spec = next((item for item in problem.specs if item["kind"] == "shared_lane_half_width"), None)
    if lane_spec is not None:
        joint["hard_shared_lane_width"]["half_width_scene_units"] = float(x[lane_spec["index"]])
    for feature in scene["ice_features"]:
        if feature.get("geometry_status") == "user_hard_relation_projection_not_video_refit":
            feature["geometry_status"] = "conditional_three_view_shared_geometry_candidate"
    estimates = {ad["user_number"]: ad for ad in fit["advertisement_estimates"]}
    for advertisement in scene["advertisements"]:
        number = advertisement.get("user_number")
        if number not in estimates:
            continue
        estimate = estimates[number]
        advertisement["physical_position_status"] = "conditional_three_view_fit_not_measured"
        advertisement["conditional_three_view_candidate"] = {
            "board_segment": estimate["board_segment"],
            "extent_scope": estimate["extent_scope"],
            "corners_xyz_scene_units": estimate["corners_xyz_scene_units"],
            "top_z_scene_units": estimate["top_z_scene_units"],
            "bottom_z_scene_units": estimate["bottom_z_scene_units"],
            "source_fit_path": str((output / "fit.json").relative_to(ROOT)),
            "physical_accuracy_verified": False,
            "note": "Advertisement 5 corners describe its internal graphic only" if number == 5 else
                    "Conditional visible advertisement extent; unobserved edges remain unresolved"}
    joint.update({
        "three_view_fit_path": str((output / "fit.json").relative_to(ROOT)),
        "three_view_fit_status": fit["status"],
        "three_view_manifest_sha256": fit["manifest_sha256"],
        "source_scene_sha256": fit["base_scene_sha256"],
        "physical_accuracy_verified": False})
    rebuilt_geometry["source"]["joint_refinement"] = copy.deepcopy(joint)
    scene["status"] = fit["status"]
    scene["uncertainty"]["advertisements"] = (
        "Conditional three-view visible-corner estimates are under conditional_three_view_candidate. "
        "Advertisement 5 is an internal graphic, and partial external extents remain unresolved. "
        "No advertisement position or size was independently measured.")
    save(output / "scene.json", scene)
    save(output / "scene_geometry.json", rebuilt_geometry)
    save(output / "scene_checks.json", checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-observations", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage-nfev", type=int, nargs=5, default=[15, 25, 20, 20, 20],
                        metavar=("CAM", "JOINT", "BROWN5", "BROWN12", "BROWN14"))
    parser.add_argument("--regularization", type=float, default=.35)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume-from", type=Path,
                        help="stage checkpoint with identical parameter names")
    parser.add_argument("--start-stage", type=int, default=0,
                        help="first stage to rerun; retain earlier stage records")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (args.workers < 1 or args.regularization < 0 or any(v < 1 for v in args.stage_nfev) or
            args.start_stage not in range(5) or bool(args.resume_from) != bool(args.start_stage)):
        parser.error("workers >= 1, regularization >= 0, and all stage limits >= 1 required")
    problem = ThreeViewProblem(args.input_observations.resolve())
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    warm_start = None
    if args.resume_from:
        checkpoint_path = args.resume_from.resolve()
        warm_start = {"checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                      "checkpoint_sha256": sha(checkpoint_path),
                      "interpretation": "numerical warm start only; all observations and bounds are re-evaluated"}
        previous_problem = checkpoint_path.parent / "problem.json"
        if previous_problem.exists():
            warm_start["previous_problem_path"] = str(previous_problem.relative_to(ROOT))
            warm_start["previous_problem_sha256"] = sha(previous_problem)
    provenance = dict(schema_version="rinkspace.v024-three-view-problem.v1",
        base_scene=str(SCENE.relative_to(ROOT)), base_scene_sha256=sha(SCENE),
        observation_manifest=str(problem.manifest_path.relative_to(ROOT)),
        observation_manifest_sha256=sha(problem.manifest_path),
        advertisement_annotation_files=[{"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
                                        for p in (HIGH_ADS, LOW2_ADS, LOW3_ADS)],
        source_observation_files=copy.deepcopy(problem.manifest["sources"]),
        source_frames=[{"source_pts": f["source_pts"], "image_path": f["image_path"],
                        "image_sha256": f["image_sha256"]} for f in problem.manifest["frames"]],
        camera_seed_files=[{"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
                           for p in (HIGH_CAMERA, LOW2_FIT, LOW3_FIT, LOW3_NATIVE_FIT)],
        initialization=problem.initialization,
        warm_start=warm_start,
        geometry_parameter_names=problem.names[:problem.geometry_count],
        all_parameter_names=problem.names,
        source_pts=list(PTS), independent_camera_ids=[problem.frames[pts]["id"] for pts in PTS],
        fit_observation_ids=[o["id"] for o in problem.fit_obs],
        check_observation_ids=[o["id"] for o in problem.check_obs],
        frame_group_balance=problem.frame_factor,
        correlated_subgroup_counts=dict(problem.correlated_group_sizes),
        user_confirmed_advertisements="identity, order, same-content pairs only; no exact world-point symmetry",
        camera_model="per-frame independent full central Brown14 with skew, all 25 dimensions active at final stage",
        excluded_claims=["verified metric scale", "camera installation identity", "independent physical accuracy"])
    save(output / "problem.json", provenance)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "parameters": len(problem.x0), "fit_groups": len(problem.fit_obs),
                          "ad_points": len(problem.ad_obs), "output": str(output)}, ensure_ascii=False))
        return
    x_start = None
    if args.resume_from:
        checkpoint = read(args.resume_from.resolve())
        if checkpoint["parameter_names"] != problem.names:
            raise ValueError("checkpoint parameter names do not match current input")
        x_start = np.asarray(checkpoint["parameter_values"], float)
        if not np.all((x_start >= problem.bounds[0]) & (x_start <= problem.bounds[1])):
            raise ValueError("checkpoint values outside current parameter bounds")
    x, stages = solve(problem, args.stage_nfev, args.regularization, args.workers,
                      output, args.start_stage, x_start)
    data, details, geometry = problem._data_parts(x, with_details=True)
    hard = geometry.hard_geometry_checks()
    boundary = geometry.boundary_validity_margins()
    circles = geometry.circle_validity_checks()
    rink_parameters = copy.deepcopy(geometry.parameters)
    rink_parameters["geometry_source"] = "semantic_joint_video_candidate; unchanged parameters remain schematic"
    ice_features = copy.deepcopy(list(geometry.features.values()))
    for feature in ice_features:
        if feature.get("geometry_status") == "user_hard_relation_projection_not_video_refit":
            feature["geometry_status"] = "conditional_three_view_shared_geometry_candidate"
    lens = {str(pts): dict(max_guard=float(lens_guard(x[problem.slices[("camera", pts)]]).max())) for pts in PTS}
    placement = camera_placement_checks(problem, x, geometry)
    ad_validity = advertisement_validity_checks(problem, x)
    valid = (hard is not None and hard["all_constraints_satisfied"] and
             all(v > 0 for v in boundary.values()) and all(r["valid"] for r in circles) and
             all(v["max_guard"] < 1e-6 for v in lens.values()) and
             all(v["outside_left_end_boards"] for v in placement.values()) and
             all(v["valid"] for v in ad_validity))
    if not valid:
        status = "REJECTED_INVALID_GEOMETRY_LENS_OR_CAMERA_PLACEMENT"
    elif not stages[-1]["success"]:
        status = "UNCONVERGED_THREE_VIEW_CANDIDATE_NOT_PHYSICALLY_VALIDATED"
    else:
        status = "three_view_joint_candidate_not_independently_validated"
    fit = dict(schema_version="rinkspace.v024-three-view-fit.v1",
        status=status,
        accepted_geometry_candidate=bool(valid), base_scene_sha256=sha(SCENE),
        manifest_sha256=sha(problem.manifest_path), parameter_names=problem.names,
        parameter_values=x.tolist(), stage_results=stages, camera_freedom="all 25 parameters per camera released",
        source_hashes={"scene": {"path": str(SCENE.relative_to(ROOT)), "sha256": sha(SCENE)},
            "manifest": {"path": str(problem.manifest_path.relative_to(ROOT)),
                          "sha256": sha(problem.manifest_path)},
            "observation_files": copy.deepcopy(problem.manifest["sources"]),
            "frame_images": provenance["source_frames"],
            "advertisement_files": provenance["advertisement_annotation_files"],
            "camera_seed_files": provenance["camera_seed_files"]},
        initialization=problem.initialization,
        warm_start=warm_start,
        cameras={str(pts): problem.camera_dict(x, pts) for pts in PTS},
        advertisement_estimates=problem.advertisement_estimates(x),
        rink_parameters=rink_parameters, hard_geometry_checks=hard,
        boundary_validity_margins=boundary, circle_validity_checks=circles,
        lens_guard=lens, camera_placement_checks=placement,
        advertisement_validity_checks=ad_validity,
        per_observation_residuals=details, observed_residual_metrics=metrics(details),
        fitted_normalized_data_norm=float(np.linalg.norm(data)),
        physical_accuracy_verified=False, scene_unit_scale_verified_metres=False)
    save(output / "fit.json", fit)
    save(output / "residuals.json", {"rows": details, "summary": metrics(details),
                                     "fitted_normalized_data_norm": float(np.linalg.norm(data)),
                                     "training_residuals_are_not_validation": True})
    save(output / "geometry.json", {"parameters": rink_parameters,
                                     "ice_features": ice_features,
                                     "hard_geometry_checks": hard,
                                     "boundary_validity_margins": boundary,
                                     "circle_validity_checks": circles})
    try:
        export_candidate_scene(problem, x, fit, output)
    except (ValueError, KeyError, TypeError) as error:
        save(output / "scene_export_failure.json", {"reason": str(error),
            "fit_path": str((output / "fit.json").relative_to(ROOT))})
        print(f"Scene mesh export failed: {error}", flush=True)
    print(json.dumps({"status": fit["status"], "output": str(output),
                      "observed_residual_metrics": fit["observed_residual_metrics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
