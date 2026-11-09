"""Fit the two rigid skate assemblies to native source pixels with a frozen camera."""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from rinkspace.review.render import source_frames


PROGRAMME = Path("events/ankara_2026/programmes/okada_mei_sp")
SIZE = np.array([1920, 1080])
LEGACY_PARTS = {
    "boot_sole": "outsole", "front_stanchion": "front_mount",
    "rear_stanchion": "rear_mount", "runner_lower_edge": "runner_lower_edge",
    "runner_upper_edge": "runner_upper_edge",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emit(kind, **values):
    print(json.dumps({"stage": kind, **values}, ensure_ascii=False), flush=True)


def local_points(world, origin, rotation):
    return (world - origin[..., None, :]) @ rotation


def uniform_path(points, count, *, closed=False):
    if closed:
        points = np.r_[points, points[:1]]
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=-1))]
    t = (np.arange(count) + .5) / count * distance[-1]
    return np.column_stack([np.interp(t, distance, xy) for xy in points.T])


class SkateGeometry:
    """A frozen rigid assembly, in the existing conditional world length units."""

    def __init__(self, result):
        origin = result["skate_origin_world_mid"]
        rotation = result["skate_R_world_mid"]
        self.lower = local_points(result["skate_lower_world_mid"][0, 0], origin[0, 0], rotation[0, 0])
        self.upper = np.median(local_points(result["skate_upper_world_mid"][::73],
                                            origin[::73], rotation[::73]), axis=(0, 1))
        # The tongue, shell, outsole, steel and mounts all share this rigid frame.
        self.lower_faces = result["skate_lower_faces"]
        self.upper_faces = result["skate_upper_faces"]
        self.parts = result["skate_lower_face_parts"]
        self.runner = local_points(result["skate_edge_world_mid"][0, 0], origin[0, 0], rotation[0, 0])
        steel = self.lower[np.unique(self.lower_faces[self.parts == "steel_runner"])]
        edge = self.runner[0]
        self.web = float(np.max(steel[:, 2] - np.interp(steel[:, 0], edge[:, 0], edge[:, 2])))
        self.prepare_features()

    @classmethod
    def from_candidate(cls, packet):
        model = cls.__new__(cls)
        model.lower = packet["lower_local_m"]
        model.upper = packet["upper_local_m"]
        model.lower_faces = packet["lower_faces"]
        model.upper_faces = packet["upper_faces"]
        model.parts = packet["lower_face_parts"]
        model.runner = packet["runner_local_m"]
        model.web = float(packet["blade_web_height_m"])
        model.prepare_features()
        return model

    def prepare_features(self):
        np.testing.assert_allclose(self.runner[0, :, 2], self.runner[1, :, 2], atol=1e-8)
        self.centre = self.upper.mean(axis=0)
        shell = slice(0, len(self.upper))
        self.sem = {"boot_shell": shell}
        self.length = float(np.ptp(self.upper[:, 0]))
        parts = [self.upper]
        for name, select in [("heel_shell", self.upper[:, 0] < .39*self.length),
                             ("toe_shell", self.upper[:, 0] > .59*self.length)]:
            points = self.upper[select]
            start = sum(map(len, parts))
            self.sem[name] = slice(start, start + len(points))
            parts.append(points)
        for name, owners in [
            ("outsole", ["outsole", "heel_block"]),
            ("front_mount", ["front_post", "front_nose_post", "front_mount_plate"]),
            ("rear_mount", ["rear_post", "rear_mount_plate"]),
        ]:
            ids = np.unique(self.lower_faces[np.isin(self.parts, owners)])
            start = sum(map(len, parts))
            self.sem[name] = slice(start, start + len(ids))
            parts.append(self.lower[ids])
        runner = self.runner.mean(axis=0)[::5]
        for name, points in [("runner_lower_edge", runner),
                             ("runner_upper_edge", runner + [0, 0, self.web])]:
            start = sum(map(len, parts))
            self.sem[name] = slice(start, start + len(points))
            parts.append(points)
        # Cuff opening, kept separate from the skate's laces and toe cap.
        sections = self.upper[:-4].reshape(-1, 6, 3)
        ring = sections[np.argmin(abs(sections[:, 0, 0] - .18*self.length))]
        opening = ring[[3, 4]].mean(axis=0)
        inside = self.upper.mean(axis=0)
        self.cuff = slice(sum(map(len, parts)), sum(map(len, parts))+5)
        parts.append(np.array([opening, opening + [0., 0., .045],
                               opening + [.030, 0., 0.], opening + [0., .025, 0.], inside]))
        self.feature_points = np.concatenate(parts) - self.centre

    @staticmethod
    def hull(points):
        return cv2.convexHull(np.ascontiguousarray(points, dtype=np.float32))[:, 0]

    def features(self, uv, traced):
        hull = self.hull(uv[self.sem["boot_shell"]])
        if len(hull) < 3:
            raise ValueError("Degenerate projected shell")
        perimeter = cv2.arcLength(hull, True)
        ring = uniform_path(hull, 32, closed=True)
        tangent = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=-1)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-7)
        normal *= np.where(np.sum(normal * (ring - hull.mean(axis=0)), axis=-1) > 0, 1, -1)[:, None]
        # Use the span of the source footprint to set blur/contrast sampling in pixels.
        step = np.clip(.022 * perimeter, 1.65, 4.0)
        outside, inside = ring + step*normal, ring - step*normal
        heel, toe = uv[self.sem["runner_lower_edge"]], uv[self.sem["runner_upper_edge"]]
        lower, upper = heel[2:-2], toe[2:-2]
        axis = heel[3:-1] - heel[1:-3]
        n = np.stack([axis[:, 1], -axis[:, 0]], axis=-1)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-7)
        n *= np.where(np.sum(n * (lower - upper), axis=-1) > 0, 1, -1)[:, None]
        span = np.linalg.norm(heel[-1] - heel[0])
        profile_offset = np.clip(.05 * span, 2., 4.5)
        cuff, up, across_x, across_y, shell_center = uv[self.cuff]
        vertical = up - cuff
        vertical /= max(float(np.linalg.norm(vertical)), 1e-7)
        across = np.array([-vertical[1], vertical[0]])
        half_width = np.clip(max(abs((across_x-cuff) @ across), abs((across_y-cuff) @ across)), 3., .15*perimeter)
        cuff_step = np.clip(.032*perimeter, 2.5, 6.5)
        cuff_line = cuff + np.linspace(-.66, .66, 7)[:, None]*half_width*across
        band = np.concatenate([outside, inside, (upper+lower)*.5,
                               lower + profile_offset*n, upper - profile_offset*n,
                               cuff_line + cuff_step*vertical, cuff_line - cuff_step*vertical,
                               .62*shell_center + .38*ring[::4]])
        paths = {}
        for semantic in traced:
            path = uv[self.sem[semantic]]
            if semantic.startswith("runner_"):
                paths[semantic] = path
            else:
                border = self.hull(path)
                if len(border) >= 2:
                    paths[semantic] = np.r_[border, border[:1]]
        # Steel hidden by its own boot contributes no image edge term.
        axis1 = np.roll(hull, -1, axis=0) - hull
        d = (upper+lower)[:, None]*.5 - hull[None]
        cross = axis1[:, 0]*d[..., 1] - axis1[:, 1]*d[..., 0]
        hidden = (cross > 0.).all(axis=-1) | (cross < 0.).all(axis=-1)
        return band, span, paths, ~hidden


def row_camera(camera, centre_uv, frames):
    n = len(frames)
    field = {key: np.empty((n,) + shape) for key, shape in
             (("rotation", (3, 3)), ("centre", (3,)), ("fx", ()), ("fy", ()),
              ("cx", ()), ("cy", ()), ("k1", ()), ("k2", ()), ("p1", ()), ("p2", ()))}
    for sel, shot, local in camera._groups(frames):
        p = shot.p
        times = p["motion_times"][local] + p["rig"][37] * (np.clip(centre_uv[sel, 1], 0, 1079)/1079 - .5)
        pose = shot.core.evaluate_motion_camera(p["rig"], p["controls"],
                                                float(p["first_control_time"]), float(p["control_dt"]), times)
        if pose["status"].any():
            raise ValueError("Native row camera evaluation failed")
        for key in field:
            field[key][sel] = pose[key]
    field["centre"] *= camera.info["metres_per_scene_unit"]
    return field


class RowProjector:
    def __init__(self, packet, index, crop_origin):
        self.R, self.C = packet["rotation"][index], packet["centre"][index]
        self.f = np.array([packet["fx"][index], packet["fy"][index]])
        self.c = np.array([packet["cx"][index], packet["cy"][index]]) - crop_origin
        self.D = np.array([packet[key][index] for key in ("k1", "k2", "p1", "p2")])
        self.K = np.diag([*self.f, 1.])
        self.K[:2, 2] = self.c

    def project(self, points):
        x, y = (points[:, :2] / points[:, 2:]).T
        rr = x*x + y*y
        k1, k2, p1, p2 = self.D
        radial = 1 + rr*(k1 + k2*rr)
        xy = np.column_stack([radial*x + 2*p1*x*y + p2*(rr+2*x*x),
                              radial*y + p1*(rr+2*y*y) + 2*p2*x*y])
        return xy*self.f + self.c

    def ray(self, uv):
        ray = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), self.K, self.D)[:, 0]
        return np.column_stack([ray, np.ones(len(ray))])

    def world(self, uv, depth, rotation, pivot):
        r = self.R.T @ rotation
        return self.C + self.R.T @ (self.ray(uv)[0]*depth) - r @ pivot, r


def traced_rows(document, pts, shape):
    by_frame = defaultdict(list)
    counts = defaultdict(int)
    allowed = set(map(int, pts))
    rows = list(document["programme_skate_evidence"]["ordered_contours"])
    for legacy in document.get("full_source_skate_structures", {}).get("rows", []):
        semantic = LEGACY_PARTS.get(legacy.get("structure_feature"))
        if not semantic or legacy.get("optimization_enabled") is False or legacy.get("bounded_ambiguous_boundary"):
            continue
        order = legacy.get("source_order", "")
        points = legacy["polyline_uv"]
        if order == "front_to_rear" and semantic.startswith("runner_"):
            points, order = points[::-1], "rear_to_front"
        rows.append(dict(legacy, raw_group_id=legacy["id"], semantic=semantic,
                         source_order=order, polyline_uv=points, radius_px=legacy["pixel_uncertainty_radius_px"]))
    for row in rows:
        if row["pts"] not in allowed or row["foot"] not in (0, 1) or row["semantic"] not in shape.sem:
            continue
        if row["role"] not in ("fit", "check") or row.get("source_status") == "ambiguous":
            continue
        points = np.asarray(row["polyline_uv"], float)
        if len(points) < 2 or not np.isfinite(points).all():
            continue
        by_frame[row["pts"], row["foot"], row["role"]].append(
            dict(row, points=uniform_path(points, min(16, max(5, len(points))))))
        counts[row["role"]] += 1
    return by_frame, counts


def row_distance(row, path, shifts, origin):
    a, d = path[:-1], np.diff(path, axis=0)
    x = row["points"][None, :, None, :] - origin - shifts[:, None, None, :] - a
    t = np.clip(np.sum(x*d, axis=-1)/np.maximum((d*d).sum(-1), 1e-10), 0., 1.)
    error = np.sqrt(np.min(np.sum((x-t[..., None]*d)**2, axis=-1), axis=-1))
    if row["semantic"].startswith("runner_"):
        obs = row["points"][-1]-row["points"][0]
        model = path[-1]-path[0]
        if np.linalg.norm(obs) >= 11 and np.linalg.norm(model) >= 6:
            cosine = float(obs @ model) / (np.linalg.norm(obs)*np.linalg.norm(model))
            error += (1 - (cosine if row.get("source_order") == "rear_to_front" else abs(cosine))) * .26*np.linalg.norm(obs)
        if row.get("complete_visible"):
            measured = row["points"][None] - origin - shifts[:, None]
            same = np.linalg.norm(measured[:, [0, -1]] - path[[0, -1]], axis=-1).mean(axis=-1)
            rev = np.linalg.norm(measured[:, [-1, 0]] - path[[0, -1]], axis=-1).mean(axis=-1)
            error += .5*(same if row.get("source_order") == "rear_to_front" else np.minimum(same, rev))[:, None]
    return np.sqrt(np.mean(error**2, axis=-1))


def remap(image, xy):
    return cv2.remap(image, xy[..., 0].copy(), xy[..., 1].copy(), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def skin_and_cuff(lab, hint, origin):
    """Use a saved image-space crop hint to read skin and the cuff from RGB."""
    knee, ankle = hint - origin
    if not np.isfinite(hint).all():
        return None, None, None
    motion = ankle-knee
    length = float(np.linalg.norm(motion))
    if length < 19:
        return None, None, None
    down = motion/length
    across = np.array([-down[1], down[0]])
    width = np.clip(.052*length, 3., 9.)
    xy = ankle - np.array([.20, .26, .33])[:, None, None]*length*down + np.linspace(-width, width, 9)[None, :, None]*across
    if not ((xy >= 1) & (xy <= np.array(lab.shape[1::-1])-2)).all():
        return None, None, None
    samples = remap(lab, xy.astype(np.float32)).astype(np.float32)
    colour = np.median(samples.reshape(-1, 3), axis=0)
    if colour[2] < 129.5 or colour[0] < 63 or np.median(np.linalg.norm(samples-colour, axis=-1)) > 12.:
        return None, None, None

    def similarity(pixels):
        diff = (pixels-colour)*[.34, 1.1, 1.2]
        return np.exp(-np.sum(diff**2, axis=-1)/85.).astype(np.float32)

    skin = similarity(lab.astype(np.float32))
    offsets = np.arange(-np.clip(.38*length, 12., 50.), np.clip(.42*length, 15., 55.))
    locations = ankle + offsets[:, None, None]*down + np.linspace(-width, width, 9)[None, :, None]*across
    source_skin = remap(skin, locations.astype(np.float32))
    gap = int(np.clip(length*.032, 3, 6))
    first, second = source_skin[:-2*gap], source_skin[2*gap:]
    skin_before = np.median(first, axis=1)
    contrast = np.median(first-second, axis=1)
    valid = (skin_before > .50) & (contrast > .30) & (((first-second) > .22).sum(axis=1) >= 6)
    if not valid.any():
        return skin, None, down
    off = offsets[gap:-gap]
    score = np.where(valid, contrast-.08*(off/max(13., .24*length))**2, -np.inf)
    j = int(score.argmax())
    if np.any((abs(off-off[j]) > .13*length) & (score > score[j]-.05)):
        return skin, None, down
    cue = ankle + off[j]*down
    if np.any(cue < 1) or np.any(cue > np.array(lab.shape[1::-1])-2):
        return skin, None, down
    return skin, cue, down


class ImageScore:
    def __init__(self, frame, origin, stop, rows, geometry, background, hint):
        x0, y0 = origin
        x1, y1 = stop
        crop = frame[y0:y1, x0:x1]
        self.lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        self.foreground = background.mask(self.lab, origin) if background is not None else None
        self.skin, self.cuff_uv, self.shin_direction = skin_and_cuff(self.lab, hint, origin)
        self.origin, self.rows, self.geometry = origin, rows, geometry
        self.names = {r["semantic"] for r in rows}

    def __call__(self, uv, shifts):
        samples, span, paths, visible = self.geometry.features(uv, self.names)
        xy = (samples[None] + shifts[:, None]).astype(np.float32)
        val = remap(self.lab, xy).astype(np.float32)
        outer, inner = val[:, :32], val[:, 32:64]
        delta = outer - inner
        response = np.sqrt(delta[..., 0]**2 + 1.55*np.sum(delta[..., 1:]**2, axis=-1))
        leather = np.clip((21.-np.linalg.norm(inner[..., 1:]-128., axis=-1))/15., 0., 1.)
        leather *= np.clip((inner[..., 0]-73.)/55., 0., 1.)
        boot = (np.clip(response/24., 0., 1.) * (.40+.60*leather)).mean(axis=-1)
        n = (len(samples)-86)//3
        steel_pixels = val[:, 64:64+3*n, 0]
        mid, low, high = np.split(steel_pixels, [n, 2*n], axis=-1)
        # A dark thin band bounded on both sides supports steel. Reflective
        # steel can have either polarity at its lower material edge.
        signed = low-mid
        steel_detail = .72*np.clip(signed/21., 0., 1.) + .28*np.clip(abs(signed)/23., 0., 1.)
        steel_detail *= .68 + .32*np.clip((high-mid)/17., 0., 1.)
        steel = (steel_detail*visible).sum(axis=-1) / max(5, int(visible.sum())) * min(span/16., 1.)
        above, below, interior = np.split(val[:, 64+3*n:], [7, 14], axis=1)
        contrast = (above[..., 2] - below[..., 2]) + .32*(below[..., 0] - above[..., 0])
        neutral_boot = np.clip((18. - np.linalg.norm(below[..., 1:] - 128., axis=-1))/12., 0., 1.)
        warm_leg = np.clip((above[..., 2] - 130.)/7., 0., 1.)
        cuff = (np.clip((contrast-2.)/15., 0., 1.) * neutral_boot * warm_leg).mean(axis=-1)
        wrong_material = np.clip((np.linalg.norm(interior[..., 1:] - 128., axis=-1)-8.)/18., 0., 1.)
        wrong_material = np.maximum(wrong_material, np.clip((115. - interior[..., 0])/75., 0., 1.))
        cost = -1.20*boot - 1.10*steel - .38*cuff - .12*leather.mean(axis=-1)
        cost += .70*wrong_material.mean(axis=-1)
        non_skin = np.ones((len(shifts), len(samples)), dtype=np.float32)
        if self.skin is not None:
            non_skin -= remap(self.skin, xy)
            cost += 1.5*(1-non_skin[:, -8:].mean(axis=-1))
            cost += .52*(1-non_skin[:, 64:64+n].mean(axis=-1))
        if self.cuff_uv is not None:
            centre, up, *_ = uv[self.geometry.cuff]
            error = np.linalg.norm(centre+shifts-self.cuff_uv, axis=-1)
            cost += 1.10*np.log1p((error/max(5., .11*span))**2)
            length = max(1., np.linalg.norm(up-centre))
            cost += .34*max(0., float(self.shin_direction @ (up-centre))/length)
        fg_support = np.zeros(len(shifts))
        if self.foreground is not None:
            fg = remap(self.foreground, xy)*non_skin
            outer_fg, inner_fg = fg[:, :32], fg[:, 32:64]
            boot_fg = fg[:, -8:].mean(axis=-1)
            steel_fg = fg[:, 64:64+n].mean(axis=-1)
            fg_support = .72*boot_fg + .28*steel_fg
            matte_edge = np.clip(inner_fg-outer_fg, 0., 1.).mean(axis=-1)
            cost -= 1.22*fg_support + .72*matte_edge
        if self.rows:
            evidence = []
            for row in self.rows:
                path = paths.get(row["semantic"])
                if path is None:
                    continue
                dist = row_distance(row, path, shifts, self.origin)
                evidence.append(np.log1p((dist / max(3., float(row["radius_px"])))**2) *
                                float(row.get("relative_weight", .85)))
            if evidence:
                cost += 1.25*np.mean(evidence, axis=0)
        inside = ((xy >= 1) & (xy <= np.array(self.lab.shape[1::-1])-2)).all(axis=-1).mean(axis=-1)
        cost += 1.25*(1-inside)
        return cost, np.column_stack([boot, steel, inside, cuff, fg_support])


def project_pose(cam, shape, uv, depth, r):
    return cam.project(shape.feature_points @ r.T + cam.ray(uv)[0]*depth)


def fit_one(frame, cam, shape, initial_uv, depth, r0, predicted, rows, other, background, hint):
    crop_origin = cam.crop_origin
    uv0 = initial_uv - crop_origin
    patch = ImageScore(frame, crop_origin, cam.crop_stop, rows, shape, background, hint)
    boot_size = max(18., shape.length * cam.f.mean() / depth)
    reach = np.clip(boot_size*.46, 13., 53.)
    prior_uv = uv0
    if predicted is not None:
        prior_uv = uv0 + np.clip(predicted[0], -reach, reach)
    previous_r = r0 if predicted is None else cam.R @ predicted[1]
    yaw = Rotation.from_euler("z", [-65., 65., 135., -135.], degrees=True).as_matrix()
    roll = Rotation.from_euler("z", [-42., 42., 90., -90.], degrees=True).as_matrix()
    seeds = [r0, previous_r, *(r0 @ yaw), *(roll @ r0)]
    grid = np.linspace(-reach, reach, 7)
    dx, dy = np.meshgrid(grid, grid)
    offsets = np.column_stack([dx.ravel(), dy.ravel()])
    offsets = np.unique(np.r_[offsets, (prior_uv - uv0)[None]], axis=0)
    weight_radius = max(17., .64 * boot_size)
    log = 0

    def score(projected, shifts, center, r, z):
        nonlocal log
        log += len(shifts)
        cost, quality = patch(projected, shifts)
        candidates = center + shifts
        cost += .12*np.sum(((candidates - prior_uv) / weight_radius)**2, axis=-1)
        cost += .09*(3. - np.sum(r*previous_r))
        drift = math.log(z/depth)
        cost += .17*(drift/.18)**2
        if predicted is not None:
            cost += .13*((drift-predicted[2])/.12)**2
        if other is not None:
            # Exclude stealing a clearly separated second shoe. True overlaps
            # remain possible and get their own ambiguity flag on delivery.
            separation = np.linalg.norm(other - initial_uv)
            if separation > .85*boot_size:
                gap = np.linalg.norm(candidates + crop_origin - other, axis=-1)
                cost += .30*np.clip(1.-gap / (.44*boot_size), 0., 1.)
        return cost, quality

    best = (float("inf"), uv0, depth, r0, np.zeros(5))
    base_cost, _ = score(project_pose(cam, shape, uv0, depth, r0), np.zeros((1, 2)), uv0, r0, depth)
    for r in seeds:
        uv = project_pose(cam, shape, uv0, depth, r)
        options = offsets
        if patch.cuff_uv is not None:
            shift = patch.cuff_uv-uv[shape.cuff][0]
            options = np.r_[options, np.clip(shift, -2*reach, 2*reach)[None]]
        if rows:
            parts = [r["points"].mean(axis=0)-crop_origin-uv[shape.sem[r["semantic"]]].mean(axis=0)
                     for r in rows]
            options = np.r_[options, np.clip(np.asarray(parts), -2*reach, 2*reach)]
        c, q = score(uv, options, uv0, r, depth)
        idx = int(c.argmin())
        if c[idx] < best[0]:
            best = (float(c[idx]), uv0 + options[idx], depth, r, q[idx])

    # Small 6D proposals around the source-driven winner. A depth change
    # rescales the complete physical skate; parts cannot slide independently.
    for step in (1., 1., .48, .24):
        cost, center, z, r, quality = best
        delta = max(2.6, .10*boot_size)*step
        xy = np.array([[0, 0], [-delta, 0], [delta, 0], [0, -delta], [0, delta]])
        angles = Rotation.from_rotvec(np.r_[np.eye(3), -np.eye(3)] * (.27*step)).as_matrix()
        alternatives = [(r, z), *((a @ r, z) for a in angles),
                        (r, z*np.exp(.09*step)), (r, z*np.exp(-.09*step))]
        for rot, scale in alternatives:
            projected = project_pose(cam, shape, center, scale, rot)
            c, q = score(projected, xy, center, rot, scale)
            i = int(c.argmin())
            if c[i] < best[0]:
                best = (float(c[i]), center + xy[i], scale, rot, q[i])
        # Verify the selected approximate XY shift through the actual ray.
        selected = project_pose(cam, shape, best[1], best[2], best[3])
        c, q = score(selected, np.zeros((1, 2)), best[1], best[3], best[2])
        best = (float(c[0]), *best[1:4], q[0])
    cost, center, z, rot, quality = best
    return cam.world(center, z, rot, shape.centre), quality, log, float(base_cost[0])-cost, math.log(z/depth)


def crop_for(pivot, boot_extent, rows):
    spread = np.clip(.95*boot_extent+22., 48., 155.)
    lo, hi = pivot - spread, pivot + spread + 1.
    if rows:
        u = np.concatenate([r["points"] for r in rows])
        lo, hi = np.minimum(lo, u.min(0)-.45*spread), np.maximum(hi, u.max(0)+.45*spread)
    return np.clip(np.floor(lo).astype(int), 0, SIZE-1), np.clip(np.ceil(hi).astype(int), 1, SIZE)


def projected_world(camera, vertices, origins, rotations, frames, length_ratio):
    # The native implicit row camera is used for the saved overlay. Pose is
    # frozen at each shoe's row centre within this exposure.
    points = origins[:, :, None] + vertices @ rotations.swapaxes(-1, -2)
    return camera.project((points / length_ratio).reshape(len(frames), -1, 3), frames).reshape(
        len(frames), 2, len(vertices), 2)


def camera_matrices(camera):
    mid = camera.midpoint
    k = np.repeat(np.eye(3)[None], len(mid["fx"]), axis=0)
    k[:, 0, 0], k[:, 1, 1] = mid["fx"], mid["fy"]
    k[:, 0, 2], k[:, 1, 2] = mid["cx"], mid["cy"]
    return k @ mid["rotation"]


def reference_frames(video, source, wanted):
    """Decode only same-shot background samples to a half-resolution image."""
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        base = Fraction(source["time_base_num"], source["time_base_den"])
        if stream.time_base != base:
            raise ValueError("Background sample clock differs")
        expected = iter(wanted)
        stamp = next(expected, None)
        container.seek(int(stamp), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts < stamp:
                continue
            if frame.pts != stamp:
                raise ValueError("Background source PTS gap")
            rgb = frame.to_ndarray(format="bgr24")
            yield cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
            stamp = next(expected, None)
            if stamp is None:
                return
    raise ValueError("Incomplete background decode")


class LocalBackground:
    """Use the frozen PTZ view to subtract stationary background pixels."""

    def __init__(self, samples, transforms):
        self.samples, self.transforms = samples, transforms

    def mask(self, observed, origin):
        if len(self.samples) < 3:
            return None
        height, width = observed.shape[:2]
        dim = ((width+1)//2, (height+1)//2)
        crop_to_native = np.array([[2., 0., origin[0]], [0., 2., origin[1]], [0., 0., 1.]])
        warped = []
        for image, h in zip(self.samples, self.transforms, strict=True):
            q = h @ crop_to_native
            q[:2] *= .5
            warped.append(cv2.warpPerspective(image, q, dim, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
        views = np.stack(warped)
        visible = (views.max(axis=-1) > 3).sum(axis=0)
        # A median handles other moments where the skater crosses the same
        # area. Uniformly unreliable background is never treated as evidence.
        bg = np.median(views, axis=0).astype(np.uint8)
        estimate = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
        raw = cv2.resize(observed, dim, interpolation=cv2.INTER_AREA).astype(np.float32)
        delta = raw - estimate
        distance = np.sqrt(delta[..., 0]**2 + 1.5*np.sum(delta[..., 1:]**2, axis=-1))
        signal = np.clip((distance-6.)/23., 0., 1.)
        signal *= (visible >= (len(self.samples)//2+1))
        # Foreground should be local. A global board shift, scoreboard
        # transition or a row which has never been visible gives no matte cue.
        if float(np.mean(signal > .26)) > .52 or float(signal.max()) < .30:
            return None
        return cv2.resize(signal, (width, height), interpolation=cv2.INTER_LINEAR)


def fit_span(video, source, timeline, matrix, shot, begin, pts, centres, seed_o, seed_r,
             shape, row_packet, rows, hints, pilot, owned_from):
    cv2.setNumThreads(1)
    started = time.perf_counter()
    oi = seed_o.copy()
    ri = seed_r.copy()
    quality = np.zeros((len(pts), 2, 5))
    gain = np.zeros((len(pts), 2))
    counts = np.zeros((len(pts), 2), np.int16)
    prev = [None, None]
    samples = []
    proposals = 0
    reference_steps = np.array([-72, -47, -26, 26, 47, 72])
    ref = np.arange(max(0, begin-75), min(len(timeline), begin+len(pts)+75), 5)
    ref_images = tuple(reference_frames(video, source, timeline[ref]))
    steps = max(1, pilot//4)
    for i, (stamp, frame) in enumerate(source_frames(video, source, pts)):
        index = begin+i
        targets = np.clip(index + reference_steps, 0, len(timeline)-1)
        choose = np.unique(np.abs(ref[None] - targets[:, None]).argmin(axis=1))
        choose = choose[abs(ref[choose] - index) >= 12]
        background = LocalBackground([ref_images[j] for j in choose], matrix[ref[choose]] @ np.linalg.inv(matrix[index]))
        for foot in range(2):
            key = i*2+foot
            use = rows.get((stamp, foot, "fit"), ())
            counts[i, foot] = len(use)
            pivot = seed_o[i, foot] + seed_r[i, foot] @ shape.centre
            dep = float(row_packet["rotation"][key, 2] @ (pivot - row_packet["centre"][key]))
            if dep < .5:
                prev[foot] = None
                continue
            extent = shape.length*row_packet["fx"][key]/dep
            lo, hi = crop_for(centres[i, foot], extent, use)
            if min(hi-lo) < 12:
                prev[foot] = None
                continue
            projector = RowProjector(row_packet, key, lo)
            projector.crop_origin, projector.crop_stop = lo, hi
            r0 = projector.R @ seed_r[i, foot]
            (o, r), q, n, d, log_depth = fit_one(frame, projector, shape, centres[i, foot], dep, r0,
                                                 prev[foot], use, centres[i, 1-foot], background, hints[i, foot])
            oi[i, foot], ri[i, foot] = o, r
            quality[i, foot], gain[i, foot] = q, d
            new_pivot = projector.project(((o+r @ shape.centre-projector.C) @ projector.R.T)[None])[0] + lo
            prev[foot] = (new_pivot-centres[i, foot], r, log_depth) if q[2] > .80 and q[0]+q[1] >= .32 else None
            proposals += n
        if pilot and i % steps == 0:
            samples.append((i, frame, centres[i]))
    kept = slice(owned_from, len(pts))
    emit("span", shot=str(shot), pts_start=int(pts[owned_from]), pts_end=int(pts[-1]),
         exposures=int(len(pts)-owned_from), proposals=proposals, fit_traces=int(counts[kept].sum()),
         source_scores=quality[kept].mean(axis=(0, 1)).round(3).tolist(),
         gain=float(gain[kept].mean()), seconds=round(time.perf_counter()-started, 2))
    return oi[kept], ri[kept], quality[kept], gain[kept], counts[kept], samples


def source_hints(path, expected):
    hints = np.full((len(expected), 2, 2, 2), np.nan)
    if path is not None:
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["pts"], expected)
            uv, labels, use = data["uv"], data["model_joint_ids"], data["use"]
        for foot in range(2):
            for j, joint in enumerate((13+foot, 15+foot)):
                label = (labels == joint) & use
                good = np.flatnonzero(label.sum(axis=1) == 1)
                hints[good, foot, j] = uv[good, label[good].argmax(axis=1)]
    return hints


def evaluate_traces(camera, shape, rows, out_pts, schedule, ratio, f_origin, f_rotation, seed_origin, seed_rotation):
    # Re-evaluate source traces on the fixed native projector and the same two
    # rigid assemblies before and after. Check curves never enter the fit.
    names = ["fit", "check"]
    trace_metrics = defaultdict(list)
    for role in names:
        indices = np.array([i for i, t in enumerate(out_pts) if any(rows[t, f, role] for f in (0, 1))], int)
        if not len(indices):
            continue
        v = shape.feature_points + shape.centre
        after = projected_world(camera, v, f_origin[indices], f_rotation[indices], schedule[indices], ratio)
        before = projected_world(camera, v, seed_origin[indices], seed_rotation[indices], schedule[indices], ratio)
        for j, i in enumerate(indices):
            for f in range(2):
                observed = rows[out_pts[i], f, role]
                kinds = {r["semantic"] for r in observed}
                if not kinds:
                    continue
                _, _, a, _ = shape.features(after[j, f], kinds)
                _, _, b, _ = shape.features(before[j, f], kinds)
                for r in observed:
                    k = r["semantic"]
                    if k in a and k in b:
                        trace_metrics[role].append(dict(
                            pts=int(out_pts[i]), foot=f, semantic=k, id=r["raw_group_id"],
                            candidate_px=float(row_distance(r, a[k], np.zeros((1, 2)), 0.)[0]),
                            seed_px=float(row_distance(r, b[k], np.zeros((1, 2)), 0.)[0])))
    return trace_metrics


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("project", "prepared", "source-evidence", "seed-result", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--pilot", type=int, default=0, help="Fit this many native exposures per shot for a short visual check")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--source-hints", type=Path, help="Saved native 2D crop hints for reading skin/cuff pixels")
    args = p.parse_args()
    if not 1 <= args.workers <= 8 or not 0 <= args.pilot <= 30:
        p.error("Use 1–8 workers and at most 30 pilot frames per shot")
    if not args.output.is_dir():
        raise FileNotFoundError("Create the new run directory before opening the solver log")
    started = time.perf_counter()
    cv2.setNumThreads(1)
    document = read(args.source_evidence)
    preparation = read(args.prepared/"preparation.json")
    video = Path(preparation["source_video"])
    if digest(video) != preparation["source_sha256"]:
        raise ValueError("Programme source media does not match the prepared clock")
    clock = read(args.project/PROGRAMME/"rebuild_scope.json")
    expected, shots, tb = required_clock(clock)
    with np.load(args.prepared/"input.npz", allow_pickle=False) as f:
        np.testing.assert_array_equal(f["pts"], expected)
        np.testing.assert_array_equal(f["shot_ids"], shots)
        packet = {key: f[key] for key in ("pts", "metres_per_scene_unit", "image_height")}
    record = document["programme_camera_candidate"]
    if digest(Path(record["record"])) != record["record_sha256"]:
        raise ValueError("Frozen camera record identity changed")
    camera = ProgrammeCamera(args.project, packet, shots, record_path=record["record"])
    if camera.info["state_sha256"] != document["programme_skate_evidence"]["camera_sha256"]:
        raise ValueError("Evidence and active frozen camera differ")
    camera.core.motion_projection_threads = 8
    with np.load(args.seed_result, allow_pickle=False) as f:
        np.testing.assert_array_equal(f["pts"], expected)
        np.testing.assert_array_equal(f["time_base"], tb)
        origin = f["skate_origin_world_mid"]
        rotation = f["skate_R_world_mid"]
        ratio = float(f["body_length_ratio"])
        shape = SkateGeometry(f)
    np.testing.assert_allclose(ratio, camera.info["length_export_ratio"], rtol=1e-10)
    rows, counts = traced_rows(document, expected, shape)
    hints = source_hints(args.source_hints, expected)
    schedule = np.arange(len(expected))
    if args.pilot:
        spans = {"s01":24.9, "s02":50.76, "s03":57.52, "s04":65., "s05":80.46,
                 "s06":99., "s07":129.22, "s08":139.1, "s09":161.48}
        dt = float(Fraction(*tb))
        chosen = []
        for shot, at in spans.items():
            ix = np.flatnonzero(shots == shot)
            start = int(np.clip(np.searchsorted(expected, at/dt)-args.pilot//2, ix[0], ix[-1]+1-args.pilot))
            chosen.extend(range(start, start+args.pilot))
        schedule = np.array(chosen)
    out_pts = expected[schedule]
    f_origin, f_rotation = origin[schedule].copy(), rotation[schedule].copy()
    seed_origin, seed_rotation = f_origin.copy(), f_rotation.copy()
    f_quality = np.zeros((len(schedule), 2, 5))
    f_gain = np.zeros((len(schedule), 2))
    fit_rows = np.zeros((len(schedule), 2), np.int16)
    pivot = seed_origin + seed_rotation @ shape.centre
    uv0 = camera.project(pivot/ratio, schedule)
    frames = np.repeat(schedule, 2)
    packet = row_camera(camera, uv0.reshape(-1, 2), frames)
    source = dict(width=int(SIZE[0]), height=int(SIZE[1]), time_base_num=int(tb[0]), time_base_den=int(tb[1]))
    emit("start", exposures=len(schedule), source_exposures=len(expected), shoes=2, fit_traces=counts["fit"],
         untouched_check_traces=counts["check"], camera=camera.info["state_sha256"], device="CPU / OpenCV",
         workers=1 if args.pilot else args.workers)
    change = (np.diff(schedule) != 1) | (shots[schedule[:-1]] != shots[schedule[1:]])
    edges = np.r_[0, np.flatnonzero(change)+1, len(schedule)]
    projection_matrices = camera_matrices(camera)
    jobs = []
    for first, last in zip(edges[:-1], edges[1:]):
        shot = shots[schedule[first]]
        shot_indices = np.flatnonzero(shots == shot)
        for i in range(first, last, 360):
            j = min(i+360, last)
            warm = max(first, i-6)
            use = slice(warm, j)
            work_pts = out_pts[use]
            active = {(t, f, "fit"): rows[t, f, "fit"] for t in work_pts for f in range(2) if rows[t, f, "fit"]}
            job = (video, source, expected[shot_indices], projection_matrices[shot_indices], shot,
                   int(schedule[warm]-shot_indices[0]), work_pts, uv0[use], seed_origin[use], seed_rotation[use],
                   shape, {k: v[2*warm:2*j] for k, v in packet.items()}, active, hints[schedule[use]],
                   args.pilot, i-warm)
            jobs.append((i, j, job))

    samples = []
    def accept(i, j, result):
        f_origin[i:j], f_rotation[i:j], f_quality[i:j], f_gain[i:j], fit_rows[i:j], checks = result
        samples.extend((i+k, frame, centres) for k, frame, centres in checks)

    if args.pilot or args.workers == 1:
        for i, j, job in jobs:
            accept(i, j, fit_span(*job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(fit_span, *job): (i, j) for i, j, job in jobs}
            for work in as_completed(pending):
                accept(*pending[work], work.result())
    f_inferred = (f_quality[..., 2] < .88) | (f_quality[..., 0] < .20) | (f_quality[..., 1] < .30)
    # A contact or foot identity claim cannot be inferred from a dark band.
    after_centres = camera.project((f_origin + f_rotation @ shape.centre)/ratio, schedule)
    scale_px = shape.length * packet["fx"].reshape(-1, 2)/np.linalg.norm(
        (pivot.reshape(-1, 3)-packet["centre"]).reshape(-1, 2, 3), axis=-1)
    overlap = np.linalg.norm(after_centres[:, 0]-after_centres[:, 1], axis=-1) < .36*scale_px.mean(axis=-1)
    f_inferred |= overlap[:, None]
    output = args.output/"shoes.npz"
    with output.open("xb") as stream:
        np.savez_compressed(stream, source_pts=out_pts, time_base=tb, shot_ids=shots[schedule],
            source_time_s=out_pts * float(Fraction(*tb)), frame_indices=schedule,
            foot_names=np.array(["left", "right"]), origin_world_m=f_origin, rotation_world=f_rotation,
            origin_world_seed_m=seed_origin, rotation_world_seed=seed_rotation, source_support_score=f_quality,
            inferred=f_inferred, identity_overlap=overlap, fit_row_count=fit_rows, native_image_cost_gain=f_gain,
            score_names=np.array(["shell_edge", "steel_band", "in_crop", "cuff", "moving_foreground"]),
            solver_metres_per_scene_unit=camera.scale,
            lower_local_m=shape.lower, upper_local_m=shape.upper, runner_local_m=shape.runner,
            lower_faces=shape.lower_faces, lower_face_parts=shape.parts, upper_faces=shape.upper_faces,
            blade_web_height_m=shape.web, camera_record=np.array(record["record"]),
            camera_sha256=np.array(camera.info["state_sha256"]), length_ratio=ratio,
            source_video_path=np.array(str(video)), source_video_sha256=np.array(preparation["source_sha256"]))
    trace_metrics = evaluate_traces(camera, shape, rows, out_pts, schedule, ratio,
                                    f_origin, f_rotation, seed_origin, seed_rotation)
    with (args.output/"trace_check.json").open("x", encoding="utf-8") as stream:
        json.dump(trace_metrics, stream, ensure_ascii=False, indent=2)
    if samples:
        for i, frame, old in samples:
            xy = projected_world(camera, shape.feature_points+shape.centre, f_origin[i:i+1],
                                 f_rotation[i:i+1], schedule[i:i+1], ratio)[0]
            before = projected_world(camera, shape.feature_points+shape.centre, seed_origin[i:i+1],
                                     seed_rotation[i:i+1], schedule[i:i+1], ratio)[0]
            patches = []
            for foot in range(2):
                a, b = before[foot], xy[foot]
                area = np.r_[a, b, old[foot][None]]
                lo = np.clip(np.floor(area.min(0)-20).astype(int), 0, SIZE-2)
                hi = np.clip(np.ceil(area.max(0)+20).astype(int), lo+1, SIZE)
                x0, y0 = lo
                x1, y1 = hi
                raw = frame[y0:y1, x0:x1]
                for u, color in ((None, None), (a, (60, 180, 255)), (b, (255, 240, 40))):
                    patch = raw.copy()
                    if u is not None:
                        shell = shape.hull(u[shape.sem["boot_shell"]]-lo)
                        cv2.polylines(patch, [np.rint(shell).astype(np.int32)], True, color, 1, cv2.LINE_AA)
                        for part in ("runner_lower_edge", "runner_upper_edge"):
                            path = np.rint(u[shape.sem[part]]-lo).astype(np.int32)
                            cv2.polylines(patch, [path], False, color, 1, cv2.LINE_AA)
                    patches.append(patch)
            h = max(x.shape[0] for x in patches)
            panel = np.concatenate([cv2.copyMakeBorder(x, 0, h-x.shape[0], 0, 6, cv2.BORDER_CONSTANT)
                                    for x in patches], axis=1)
            path = args.output/f"source_{out_pts[i]}.jpg"
            path.write_bytes(cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    emit("saved", path=str(output), exposures=len(schedule), inferred=int(f_inferred.sum()),
         elapsed_s=round(time.perf_counter()-started, 2),
         traces={key: dict(count=len(v), seed=float(np.median([r["seed_px"] for r in v])),
                           candidate=float(np.median([r["candidate_px"] for r in v]))) for key, v in trace_metrics.items()})


if __name__ == "__main__":
    main()
