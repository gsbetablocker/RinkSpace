"""Continuous native-camera body and skate assembly fit for the recovered programme."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .inputs import Inputs, PROJECT, atomic_json, atomic_npz, digest, distribute, emit
from calibration.program_tracking.body_tracking.absolute_motion_curve import _log
from calibration.program_tracking.body_tracking.absolute_motion_state import (
    absolute_reference, curve_from_packet, curve_metadata, evaluate_curve,
)
from calibration.program_tracking.body_tracking.programme_skate_evidence import programme_observation_path
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet, SHOE_SCHEMA
from calibration.program_tracking.body_tracking.skate_binding import SkateBinding, SHAPE_NAMES
from calibration.program_tracking.body_tracking.skate_integration import shared_bounds
from calibration.program_tracking.body_tracking.skate_physics import fk_state, batch_features
from calibration.program_tracking.body_tracking.skate_template import _profile, _rect_tube, profile_height
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

# Body root translation/rotation, then the two hip, knee and ankle chains.
JOINTS = (0, 1, 2, 4, 5, 7, 8)
LEGS = ((0, 1, 4, 7), (0, 2, 5, 8))
BODY_DIM = 3 + 3 * len(JOINTS)
STEEL = ("blade_width_m", "blade_web_height_m", "main_rocker_m", "secondary_rocker_m",
         "runner_rear_extension_m", "toe_head_length_m")
SHARED_DIM = 18 + len(STEEL)
NODES = np.linspace(-.5, .5, 5)
STRIDE = 3
PIXEL_HUBER = 3.
BODY_HUBER = 3.5
SCALE = 2
SAMPLES = 20
EPS = 1.e-4


class AdjustableUpper(SkateBinding):
    """One common height adjustment about the actual outsole interface."""

    def packet(self, x=None):
        packet = super().packet(x)
        if packet.get("upper_height_log") == self.upper_height_log:
            return packet
        # The base cache is fresh when its shape key changes. Clone before
        # scaling so the archived base packet is never modified.
        packet = dict(packet)
        factor = np.exp(self.upper_height_log)
        for name in ("upper", "inner"):
            vertices = packet[name].copy()
            floor = np.interp(vertices[:, 0], packet["xs"], packet["floor"])
            vertices[:, 2] = floor + factor * (vertices[:, 2] - floor)
            packet[name] = vertices
        packet["rise"] = packet["rise"] * factor
        adapter = packet["adapter"].copy()
        floor = np.interp(adapter[:14, 0], packet["xs"], packet["floor"])
        adapter[:14, 2] = floor + factor * (adapter[:14, 2] - floor)
        packet.update(adapter=adapter, upper_height_log=self.upper_height_log)
        self._packet = packet
        return packet

    def record(self):
        record = super().record()
        record["parameter_provenance"]["shared_upper_height_scale"] = float(np.exp(self.upper_height_log))
        return record


def revised_binding(seed, shared):
    binding = object.__new__(AdjustableUpper)
    binding.__dict__.update(seed.__dict__)
    binding.x = shared[:17].copy()
    binding.base_parameters = replace(seed.base_parameters, **{
        key: getattr(seed.base_parameters, key) * np.exp(change)
        for key, change in zip(STEEL, shared[17:17 + len(STEEL)])
    })
    binding.upper_height_log = float(shared[-1])
    binding._parameter_key = binding._packet = None
    return binding


class Assembly:
    def __init__(self, binding):
        self.binding = binding
        packet = binding.packet()
        self.upper_count = len(packet["upper"])
        template = packet["template"]
        p = packet["parameters"]
        _, _, (rear, _, join, front) = _profile(p)
        x = np.r_[np.linspace(rear, join, 10)[:-1], np.linspace(join, front, 8)]
        z = profile_height(x, p)
        runner = _rect_tube(x, np.full(len(x), p.blade_width_m * .5), z,
                            z + p.blade_web_height_m, "steel_runner")
        stop = template["part_ranges"]["steel_runner"][1]
        other = template["faces"]
        self.vertices = np.r_[packet["upper"], runner["vertices"], template["vertices"][stop:]]
        self.faces = np.asarray(np.r_[binding.faces, runner["faces"] + self.upper_count,
                               other[np.min(other, axis=1) >= stop] - stop
                               + self.upper_count + len(runner["vertices"])], np.int32)
        edges = np.sort(np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]), axis=1)
        self.edges = np.unique(edges, axis=0)

    def points(self, state, foot, selectors=None, transforms=None):
        b = self.binding
        relative = state["skate_relative_R"]
        if transforms is None:
            transforms = b.transforms(state["P"], state["Q"], state["native"], state["frames"],
                                      skate_relative_R=relative)
        rotation, origin = transforms
        alpha = b.deformation(state["native"], state["frames"], skate_relative_R=relative)
        upper = b.deform(self.vertices[:self.upper_count], alpha)[:, foot]
        lower = self.vertices[self.upper_count:]
        if selectors is None:
            local = np.concatenate((upper, np.broadcast_to(lower, (len(upper), *lower.shape))), axis=1)
        else:
            ix, fraction = selectors
            ix = np.broadcast_to(ix[:, None], (len(ix), 5, *ix.shape[1:])).reshape(len(upper), -1, 2)
            select = np.where((ix < self.upper_count)[..., None],
                              upper[np.arange(len(upper))[:, None, None], np.minimum(ix, self.upper_count - 1)],
                              lower[np.maximum(ix - self.upper_count, 0)])
            fraction = np.repeat(fraction, 5, axis=0)
            local = select[:, :, 0] + fraction[..., None] * (select[:, :, 1] - select[:, :, 0])
        return origin[:, foot, None] + np.einsum("nij,nkj->nki", rotation[:, foot], local)

    def raster(self, pixels):
        origin = np.floor(pixels.min(0) - 3)
        wh = np.ceil(pixels.max(0) - origin + 3).astype(int) * SCALE
        if np.any(wh <= 0) or np.max(wh) > 700:
            raise ValueError("Skate projection outside plausible raster extent")
        model = np.zeros((wh[1], wh[0]), np.uint8)
        px = np.rint((pixels - origin) * SCALE).astype(np.int32)
        for triangle in px[self.faces]:
            cv2.fillConvexPoly(model, triangle, 1)
        contours, _ = cv2.findContours(model, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("Empty model silhouette")
        # All visible exterior components are kept, including a subpixel
        # break between the toe pick and the post raster.
        contours = [c.reshape(-1, 2) / SCALE + origin for c in contours if len(c) >= 3]
        if not contours:
            contours = [pixels]
        return contours

    def associations(self, pixels, observed):
        contours = self.raster(pixels)
        boundary = np.concatenate(contours)
        edge_points = (pixels[self.edges[:, 0]] + pixels[self.edges[:, 1]]) / 2
        candidates = np.r_[pixels, edge_points]
        pair = np.r_[np.repeat(np.arange(len(pixels))[:, None], 2, axis=1), self.edges]
        fractions = np.r_[np.zeros(len(pixels)), np.full(len(edge_points), .5)]
        candidate = cKDTree(boundary).query(candidates)[0] < 1.55
        good = np.flatnonzero(candidate)
        if len(good) < 3:
            good = np.arange(len(candidates))
        predicted = cKDTree(candidates[good])
        source = distribute(observed, 64, closed=True)
        take = np.arange(0, len(source), max(1, len(source) // SAMPLES))[:SAMPLES]
        s2m = good[predicted.query(source[take])[1]]
        perimeter = distribute(max(contours, key=len), SAMPLES, closed=True)
        m2s = good[predicted.query(perimeter)[1]]
        near = cKDTree(source).query(candidates[m2s])[1]
        normals = np.roll(source, -1, axis=0) - np.roll(source, 1, axis=0)
        normals = normals[:, ::-1] * [-1., 1.]
        normals /= np.maximum(1.e-9, np.linalg.norm(normals, axis=1, keepdims=True))
        selected = np.r_[s2m, m2s]
        target = np.r_[take, near]
        return pair[selected], fractions[selected], source[target], normals[target]


def projection(camera, points, frames, jacobian=False):
    n = len(frames)
    return moving_projection(camera, points.reshape(n, 5, -1, 3), frames, jacobian=jacobian)


def huber(raw, cutoff):
    a = np.abs(raw)
    return np.where(a <= cutoff, .5 * raw**2, cutoff * (a - .5 * cutoff))


def rotation_sensitivity(points, pivot, axes):
    return np.cross(axes.swapaxes(-1, -2)[:, None], (points - pivot[:, None])[:, :, None]).swapaxes(-1, -2)


def append_columns(pieces, local, state, shape, row_shift, *, relative=None):
    """Chain local residuals through the four control slots at all five nodes."""
    n, k, _, _ = local.shape
    bt = state["body_eval"]
    idx = bt["control_indices"].reshape(n, 5, 4)
    w = bt["position_weights"].reshape(n, 5, 4)
    axes = bt["rotation_right_blocks"][:, :, JOINTS].reshape(n, 5, 4, len(JOINTS), 3, 3)
    all_rows = np.arange(n * k, dtype=np.int32).reshape(n, k) + row_shift

    def add(values, columns):
        rows = np.broadcast_to(all_rows[:, :, None], values.shape)
        columns = np.broadcast_to(columns[:, None], values.shape)
        active = np.abs(values) > 1.e-12
        pieces.append((values[active], rows[active], columns[active]))

    values = np.einsum("nkae,nas->nkse", local[..., :3], w)
    angular = np.einsum("nkajx,nasjxy->nksjy", local[..., 3:].reshape(n, k, 5, len(JOINTS), 3), axes)
    values = np.concatenate((values, angular.reshape(n, k, 4, -1)), axis=-1)
    # A readout node can cross a knot, in which case the slot index differs.
    # Accumulate those five supports independently to retain that exposure.
    if np.any(idx != idx[:, 2:3]):
        for node in range(5):
            val = local[:, :, node, None, :3] * w[:, node, None, :, None]
            ang = np.einsum("nkjx,nsjxy->nksjy",
                            local[:, :, node, 3:].reshape(n, k, len(JOINTS), 3), axes[:, node])
            add(np.concatenate((val, ang.reshape(n, k, 4, -1)), axis=-1).reshape(n, k, -1),
                (idx[:, node, :, None] * BODY_DIM + np.arange(BODY_DIM)).reshape(n, -1))
    else:
        add(values.reshape(n, k, -1), (idx[:, 2, :, None] * BODY_DIM + np.arange(BODY_DIM)).reshape(n, -1))
    if relative is not None:
        foot, derivatives = relative
        st = state["shoe_eval"]
        si = st["control_indices"].reshape(n, 5, 4)
        blocks = st["rotation_right_blocks"][:, :, foot].reshape(n, 5, 4, 3, 3)
        if np.any(si != si[:, 2:3]):
            for node in range(5):
                v = np.einsum("nki,nsij->nksj", derivatives[:, :, node], blocks[:, node])
                add(v.reshape(n, k, -1), (shape[0] + si[:, node, :, None] * 6 + 3 * foot
                                          + np.arange(3)).reshape(n, -1))
        else:
            v = np.einsum("nkai,nasij->nksj", derivatives, blocks)
            add(v.reshape(n, k, -1), (shape[0] + si[:, 2, :, None] * 6 + 3 * foot
                                      + np.arange(3)).reshape(n, -1))


def local_tangents(state, points, derivative, foot=None):
    n, k, _, _ = derivative.shape
    gradient = derivative.transpose(0, 2, 1, 3).reshape(n * 5, k, 3)
    blocks = [gradient]
    axes = state["Q"][:, None] @ state["frames"][:, JOINTS]
    for j, joint in enumerate(JOINTS):
        if foot is not None and joint not in LEGS[foot]:
            blocks.append(np.zeros_like(gradient))
            continue
        pivot = state["native_world"][:, joint]
        jac = rotation_sensitivity(points, pivot, axes[:, j])
        d = np.einsum("nki,nkij->nkj", gradient, jac)
        if foot is None and joint:
            # Local GN stencil for the exact regressed COCO likelihood.
            # Acceptance below reevaluates the original SMPL-X regressor.
            mask = {
                1: (11, 13, 15), 2: (12, 14, 16), 4: (13, 15),
                5: (14, 16), 7: (15,), 8: (16,),
            }[joint]
            keep = np.zeros((17, 2), bool)
            keep[list(mask)] = True
            d[:, ~keep.ravel()] = 0
        blocks.append(d)
    local = np.concatenate(blocks, axis=-1).reshape(n, 5, k, BODY_DIM).transpose(0, 2, 1, 3)
    if foot is None:
        return local, None
    rel = rotation_sensitivity(points, state["native_world"][:, 7 + foot], state["R"][:, foot])
    rel = np.einsum("nki,nkij->nkj", gradient, rel).reshape(n, 5, k, 3).transpose(0, 2, 1, 3)
    return local, (foot, rel)


class Problem:
    def __init__(self, visual, output, warm_controls=None):
        self.i = Inputs(verify_media=True)
        i = self.i
        self.visual, self.output = visual, output
        ref, self.camera, self.info, self.doc, _ = load_reference(PROJECT, i.prepared, i.document_path)
        if self.info["state_sha256"] != i.camera_sha:
            raise ValueError("Source camera changed")
        self.camera.core.motion_projection_threads = 8
        for camera in self.camera.cameras.values():
            camera.core.motion_projection_threads = 8
        with np.load(i.parent["parameter"]["path"], allow_pickle=False) as f:
            self.body = self.body_seed = curve_from_packet(f)
            self.shoe = self.shoe_seed = shoe_curve_from_packet(f)
            self.shared = self.shared_seed = np.r_[f["x"][:17], np.zeros(SHARED_DIM - 17)]
        self.ref = absolute_reference(ref, self.body)
        self.binding_seed = ref.binding
        self.ref.binding = self.binding_seed
        self.ref.binding.reference = self.ref
        self.shape = (len(self.body.position_controls) * BODY_DIM, len(self.shoe.position_controls) * 6)
        self.local_size = sum(self.shape)
        self.size = self.local_size + SHARED_DIM
        self.times = self.ref.times[:, None] + self.camera.readout_times(np.arange(len(i.pts)))[:, None] * NODES
        self.supports = [[] for _ in i.pts]
        lookup = dict(zip(i.pts, range(len(i.pts))))
        for row in self.doc.get("supports", []):
            if row["pts"] in lookup:
                self.supports[lookup[row["pts"]]].append(row)
        obs = programme_observation_path(i.prepared, i.document)
        with np.load(obs, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["pts"], i.pts)
            self.obs = {name: f[name] for name in ("uv", "use", "sigma_px", "model_joint_ids")}
        visual_path = visual / "whole_follow_evidence.npz"
        self.visual_sha256 = digest(visual_path)
        with np.load(visual_path, allow_pickle=False) as f:
            for key, target in (("source_pts", i.pts), ("shot_ids", i.shots),
                                ("source_video_sha256", i.media_sha), ("time_base", i.tb)):
                np.testing.assert_array_equal(f[key], target)
            self.region = f["region"]
            self.valid = ((f["status"] == 1) & (f["extent"] > 0) & ~f["partial_occlusion"]
                          & ~f["identity_unresolved"] & ~f["skate_occluded"])
        self.guard_visual()
        index = np.arange(len(i.pts))
        withheld = (index // 25 % 7 == 3) | i.withheld
        with_source_support = np.array([bool(s) for s in self.supports])
        self.direct_edges = {}
        for (j, f), rows in i.fit.items():
            if withheld[j]:
                continue
            for row in rows:
                if (row["semantic"] == "runner_lower_edge" and row.get("evidence_status") == "observed"
                        and row.get("source_image") and len(row["polyline_uv"]) >= 3):
                    self.direct_edges.setdefault((j, f), []).append(np.asarray(row["polyline_uv"], float))
        direct = np.isin(index, [j for j, _ in self.direct_edges])
        self.train = index[(index % STRIDE == 0) | (with_source_support & ~withheld) | direct]
        self.train_silhouette = ~withheld
        self.holdout = index[((index % 3 == 1) & withheld) | i.withheld]
        self.wear = set(self.train[::7].tolist())
        self.lo, self.hi = map(list, shared_bounds(self.binding_seed))
        self.lo = np.r_[self.lo, np.log([.72, .63, .63, .60, .63, .70, .78])]
        self.hi = np.r_[self.hi, np.log([1.45, 1.6, 1.60, 1.60, 1.55, 1.35, 1.22])]
        self.static_sigma = np.r_[np.tile([.024, .024, .024, .22, .22, .22], 2),
                                  [.12, .14, .23, .23, .23, .19, .23, .31, .30, .25, .20, .13]]
        self.prior = self.regularization()
        self.correction_origin = np.zeros(self.size)
        self.reviewed_source = np.zeros_like(self.valid)
        if warm_controls is not None:
            self.warm_start(warm_controls)

    def warm_start(self, path):
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["source_pts"], self.i.pts)
            np.testing.assert_array_equal(f["camera_sha256"], self.i.camera_sha)
            body, skate = curve_from_packet(f), shoe_curve_from_packet(f)
            self.shared = self.shared_seed + .24 * (f["shared"] - self.shared_seed)
        bp = .78 * (body.position_controls - self.body_seed.position_controls)
        br = .80 * _log(self.body_seed.rotation_controls.swapaxes(-1, -2) @ body.rotation_controls)
        sr = .55 * _log(self.shoe_seed.rotation_controls.swapaxes(-1, -2) @ skate.rotation_controls)
        self.body = self.body_seed.retracted(bp, br)
        self.shoe = self.shoe_seed.retracted(np.zeros_like(skate.position_controls), sr)

    def guard_visual(self):
        with np.load(self.i.parent_path, allow_pickle=False) as old:
            uv = old["equipment_uv"][:, :, :14]
        diag = np.linalg.norm(np.ptp(uv, axis=2), axis=-1)
        center = np.mean(uv, axis=2)
        reg = np.where(np.isfinite(self.region), self.region, 0)
        rd = np.linalg.norm(np.ptp(reg, axis=2), axis=-1)
        error = np.linalg.norm(reg.mean(axis=2) - center, axis=-1)
        self.valid &= (rd > .32 * diag) & (rd < 2.4 * diag) & (error < .95 * diag + 13)
        for index in np.flatnonzero(self.valid.all(axis=1)):
            l, r = self.region[index]
            if np.linalg.norm(l.mean(0) - r.mean(0)) < .11 * min(rd[index]):
                self.valid[index] = False

    def state(self, frames, binding, blocks=False):
        query = self.times[frames].ravel()
        e = evaluate_curve(self.body, query, with_blocks=blocks)
        s = evaluate_curve(self.shoe, query, with_blocks=blocks)
        r = {k: e[k] for k in ("P", "Q", "local_R")}
        r["skate_relative_R"] = s["rotations"]
        self.ref.binding = binding
        state = fk_state(self.ref, r, np.zeros((len(query), 69)))
        state.update(body_eval=e, shoe_eval=s)
        return state

    def regularization(self):
        matrices = []
        dims = (
            (len(self.body.position_controls), np.r_[[.062] * 3, [.115] * 3, [.17] * 18],
             np.r_[[3.8] * 3, [1.8] * 3, [1.6] * 18]),
            (len(self.shoe.position_controls), np.tile([.15, .17, .14], 2), np.full(6, 2.2)),
        )
        for n, sigma, weight in dims:
            # Second differences of the correction around the archived
            # moving trajectory, plus a broad weak initialization prior.
            # Constant velocities and real turns in the state are unpenalized.
            D = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
                             [0, 1, 2], shape=(n - 2, n))
            matrices.append(sparse.vstack((sparse.kron(.62 * sparse.eye(n), sparse.diags(1. / sigma)),
                                           sparse.kron(D, sparse.diags(weight / sigma))), format="csr"))
        independent_views = np.r_[[9.] * 17, 17., 12., 24., 15., 16., 15., 12.]
        matrices.append(sparse.diags(independent_views / self.static_sigma))
        return sparse.block_diag(matrices, format="csr")

    def correction(self):
        rb = _log(self.body_seed.rotation_controls[:, JOINTS].swapaxes(-1, -2)
                  @ self.body.rotation_controls[:, JOINTS])
        rs = _log(self.shoe_seed.rotation_controls.swapaxes(-1, -2) @ self.shoe.rotation_controls)
        return np.r_[np.concatenate((self.body.position_controls - self.body_seed.position_controls,
                                     rb.reshape(len(rb), -1)), axis=1).ravel(),
                     rs.ravel(), self.shared - self.shared_seed]

    def static(self, binding):
        result = binding.compatibility_rows(binding.neutral_native, binding.neutral_frames,
                                            binding.neutral_samples)
        return result.ravel()

    def body_sigma(self, frames, joint_ids, present):
        return np.where(present, np.maximum(2., self.obs["sigma_px"][frames]), 1.)

    def static_factor(self):
        return "neutral_foot_fit", 230.

    def build_matches(self, frames, binding, *, fitting=False):
        model = Assembly(binding)
        for start in range(0, len(frames), 48):
            f = frames[start:start + 48]
            matches = []
            for foot in range(2):
                use = self.valid[f, foot]
                if fitting:
                    use = use & self.train_silhouette[f]
                f1 = f[use]
                if not len(f1):
                    continue
                st = self.state(f1, binding)
                full = model.points(st, foot, transforms=(st["R"], st["O"]))
                uv = projection(self.camera, full, f1)["uv"]
                a = [model.associations(pix, self.region[j, foot]) for j, pix in zip(f1, uv)]
                ii, w, target, normal = (np.stack(arr) for arr in zip(*a))
                matches.append((f1, foot, (ii, w), target, normal))
            direct = self.edge_matches(f, binding) if fitting else []
            yield f, matches, direct

    @staticmethod
    def edge_world(binding, state, foot, selections):
        edge = binding.packet()["edges"].reshape(-1, 3)
        local = edge[selections].repeat(5, axis=0)
        return state["O"][:, foot, None] + np.einsum("nij,nkj->nki", state["R"][:, foot], local)

    def edge_matches(self, frames, binding):
        outputs = []
        for foot in range(2):
            selected = [j for j in frames if (j, foot) in self.direct_edges]
            if not selected:
                continue
            f = np.array(selected)
            state = self.state(f, binding)
            edges = binding.packet()["edges"]
            all_edges = np.broadcast_to(np.arange(np.prod(edges.shape[:-1])), (len(f), np.prod(edges.shape[:-1])))
            uv = projection(self.camera, self.edge_world(binding, state, foot, all_edges), f)["uv"]
            which, target, normals = [], [], []
            for j, p in zip(f, uv):
                curve = max(self.direct_edges[j, foot], key=lambda line: np.linalg.norm(np.diff(line, axis=0), axis=-1).sum())
                obs = distribute(curve, 14)
                score = [cKDTree(x).query(obs) for x in np.split(p, 2)]
                side = int(score[1][0].mean() < score[0][0].mean())
                d = np.gradient(obs, axis=0)
                normal = d[:, ::-1] * [-1., 1.]
                normal /= np.maximum(1.e-9, np.linalg.norm(normal, axis=1, keepdims=True))
                which.append(score[side][1] + side * edges.shape[1])
                target.append(obs)
                normals.append(normal)
            outputs.append((f, foot, np.stack(which), np.stack(target), np.stack(normals)))
        return outputs

    def system(self, matches, jac=False):
        bind = revised_binding(self.binding_seed, self.shared)
        model = Assembly(bind)
        variants = []
        if jac:
            for j in range(SHARED_DIM):
                d = self.shared.copy()
                d[j] += EPS
                v = Assembly(revised_binding(self.binding_seed, d))
                if len(v.vertices) != len(model.vertices):
                    raise ValueError("Shared shape changed the fitting mesh correspondence")
                variants.append(v)
        residuals, pieces, groups = [], [], {}
        row = 0
        body_norm = []
        skate_norm = []
        nmax = 0.

        def push(raw, cutoff, name, factor=1., derivatives=None, extras=None):
            nonlocal row
            val = np.asarray(raw)
            factor = np.broadcast_to(factor, val.shape)
            cost = huber(val, cutoff) * factor
            groups[name] = groups.get(name, 0.) + float(cost.sum())
            scale = np.sqrt(np.minimum(1., cutoff / np.maximum(np.abs(val), 1.e-12)) * factor)
            residuals.append((val * scale).ravel())
            if jac and derivatives is not None:
                local, st, relative = derivatives
                rel = None if relative is None else (relative[0], relative[1] * scale[..., None, None])
                append_columns(pieces, local * scale[..., None, None], st, self.shape, row, relative=rel)
            if jac and extras is not None:
                v = (extras * scale[..., None]).reshape(-1, SHARED_DIM)
                rr, cc = np.nonzero(np.abs(v) > 1.e-12)
                pieces.append((v[rr, cc], rr.astype(np.int32) + row, cc.astype(np.int32) + self.local_size))
            row += val.size

        for f, entries, direct in matches:
            st = self.state(f, bind, jac)
            hand = None
            if self.ref.hand_curves is not None:
                from calibration.program_tracking.body_tracking.hand_state import evaluate_hands
                hand = evaluate_hands(self.ref.hand_curves, self.times[f].ravel())
            coco = self.ref.body.forward_rotations(st["local_R"], hand_local_R=hand)[0]
            body_points = st["P"][:, None] + np.einsum("nij,nkj->nki", st["Q"], coco)
            ans = projection(self.camera, body_points, f, jac)
            joint_ids = self.obs["model_joint_ids"][f]
            safe_ids = np.clip(joint_ids, 0, 16)
            present = (self.obs["use"][f] & np.isfinite(self.obs["uv"][f]).all(axis=-1)
                       & np.isfinite(self.obs["sigma_px"][f]) & (joint_ids >= 0) & (joint_ids < 17))
            index = np.arange(len(f))[:, None]
            delta = ans["uv"][index, safe_ids] - np.where(present[..., None], self.obs["uv"][f], 0)
            body_norm.extend(np.linalg.norm(delta, axis=-1)[present])
            sigma = self.body_sigma(f, joint_ids, present)
            raw = np.where(present[..., None], delta / sigma[..., None], 0)
            derivatives = None
            if jac:
                # Compute in model order, then apply the measured identity mapping.
                native = ans["pixel_jac_nodes"]
                points = np.repeat(body_points, 2, axis=1)
                jacobian = native.reshape(len(f), 34, 5, 3)
                local, _ = local_tangents(st, points, jacobian)
                local = local.reshape(len(f), 17, 2, 5, BODY_DIM)[index, safe_ids]
                local *= (present / sigma)[..., None, None, None]
                derivatives = local.reshape(len(f), 34, 5, BODY_DIM), st, None
                nmax = max(nmax, ans["body_row_contraction"])
            push(raw.reshape(len(f), 34), BODY_HUBER, "body", 1.75, derivatives)
            contact = batch_features(bind, st["R"], st["O"], st)
            self.contact_factors(f, st, contact, variants, jac, push)
            wear = np.array([int(i) in self.wear for i in f])
            if np.any(wear):
                self.wear_factors(f[wear], bind, variants, jac, push)
            for foot_frames, foot, selectors, target, normal in entries:
                st1 = self.state(foot_frames, bind, jac)
                points = model.points(st1, foot, selectors, (st1["R"], st1["O"]))
                ans = projection(self.camera, points, foot_frames, jac)
                delta = ans["uv"] - target
                skate_norm.extend(np.linalg.norm(delta, axis=-1))
                raw = np.einsum("nki,nki->nk", delta, normal)
                derivatives, extra = None, None
                if jac:
                    nmax = max(nmax, ans["body_row_contraction"])
                    d = np.einsum("nkipl,nki->nkpl", ans["pixel_jac_nodes"], normal)
                    local, rel = local_tangents(st1, points, d, foot)
                    derivatives = local, st1, rel
                    extra = np.empty((*raw.shape, SHARED_DIM))
                    dxyz = d.transpose(0, 2, 1, 3).reshape(points.shape)
                    for j, assembly in enumerate(variants):
                        change = (assembly.points(st1, foot, selectors) - points) / EPS
                        extra[..., j] = np.einsum("nki,nki->nk", change, dxyz).reshape(len(foot_frames), 5, -1).sum(1)
                weight = np.where(self.reviewed_source[foot_frames, foot], 4.2, .36)
                push(raw, PIXEL_HUBER, "skate", weight[:, None], derivatives, extra)
            for f1, foot, indices, target, normal in direct:
                st1 = self.state(f1, bind, jac)
                points = self.edge_world(bind, st1, foot, indices)
                ans = projection(self.camera, points, f1, jac)
                raw = np.einsum("nki,nki->nk", ans["uv"] - target, normal)
                derivatives, extra = None, None
                if jac:
                    d = np.einsum("nkipl,nki->nkpl", ans["pixel_jac_nodes"], normal)
                    local, rel = local_tangents(st1, points, d, foot)
                    derivatives = local, st1, rel
                    dxyz = d.transpose(0, 2, 1, 3).reshape(points.shape)
                    extra = []
                    for assembly in variants:
                        b = assembly.binding
                        r, o = b.transforms(st1["P"], st1["Q"], st1["native"], st1["frames"],
                                            skate_relative_R=st1["skate_relative_R"])
                        shifted = self.edge_world(b, dict(R=r, O=o), foot, indices)
                        change = np.einsum("nki,nki->nk", (shifted - points) / EPS, dxyz)
                        extra.append(change.reshape(len(f1), 5, -1).sum(1))
                    extra = np.stack(extra, axis=-1)
                push(raw, 4., "direct_archived_runner", 3.5, derivatives, extra)

        base = self.static(bind)
        derivative = np.stack([(self.static(v.binding) - base) / EPS for v in variants], axis=-1) if jac else None
        name, weight = self.static_factor()
        push(base, 12., name, weight, extras=derivative)
        prior = self.prior @ (self.correction() - self.correction_origin)
        groups["motion_and_shared_prior"] = float(.5 * (prior @ prior))
        summary = dict(cost=groups, body_px=quantiles(body_norm), paired_skate_px=quantiles(skate_norm),
                       native_row_contraction=nmax)
        summary["total"] = sum(groups.values())
        if not jac:
            return summary
        vals, rows, cols = (np.concatenate(x) for x in zip(*pieces))
        J = sparse.coo_matrix((vals, (rows, cols)), shape=(row, self.size)).tocsr()
        return summary, sparse.vstack((J, self.prior), format="csr"), np.r_[np.concatenate(residuals), prior]

    def contact_factors(self, frames, state, base, variants, jac, push):
        ratio = self.binding_seed.ratio
        n = len(frames)
        h = base["height"].reshape(n, 5, 2, -1)[:, 2]
        world = base["local_point"].reshape(n, 5, 2, -1, 3)[:, 2]
        eligible = np.flatnonzero(base["eligible"])
        ids = np.argmin(h, axis=-1)
        active = np.ones((n, 2, 2), bool)
        active[:, :, 1] = False
        for j, frame in enumerate(frames):
            for record in self.supports[frame]:
                feet = list(record.get("feet", []))
                if not feet or record.get("mode") not in ("glide", "pivot", "toe", "stance", "spin"):
                    continue
                ix = np.argmin(h[j][np.ix_(feet, eligible)])
                foot = feet[ix // len(eligible)]
                active[j, foot, 1] = True
        variants_h = []
        if jac:
            for v in variants:
                b = v.binding
                r, o = b.transforms(state["P"], state["Q"], state["native"], state["frames"],
                                    skate_relative_R=state["skate_relative_R"])
                variants_h.append(batch_features(b, r, o, state)["height"].reshape(n, 5, 2, -1)[:, 2])
        for foot in range(2):
            select = np.c_[ids[:, foot], eligible[np.argmin(h[:, foot, eligible], axis=1)]]
            picked = np.take_along_axis(h[:, foot], select, axis=1)
            scale = np.array([.014, .025])
            raw = (picked + [.006, 0.]) / scale
            active[:, foot, 0] = raw[:, 0] < 0
            raw = np.where(active[:, foot], raw, 0.)
            deriv, extra = None, None
            if jac:
                points = np.take_along_axis(world[:, foot], select[..., None], axis=1)
                points = np.repeat(points[:, None], 5, axis=1).reshape(n * 5, 2, 3)
                points = state["O"][:, foot, None] + np.einsum("nij,nkj->nki", state["R"][:, foot], points)
                grad = np.zeros((n, 2, 5, 3))
                grad[:, :, 2, 2] = ratio / scale * active[:, foot]
                local, rel = local_tangents(state, points, grad, foot)
                deriv = local, state, rel
                extra = np.stack([(np.take_along_axis(v[:, foot], select, axis=1) - picked)
                                  / (EPS * scale) for v in variants_h], axis=-1)
                extra *= active[:, foot, :, None]
            push(raw, 4., "ice", 44., deriv, extra)

    def wear_factors(self, frames, binding, variants, jac, push):
        state = self.state(frames, binding, jac)
        mid = slice(2, None, 5)
        joints, rot = state["native"][mid], state["frames"][mid]
        relative = state["skate_relative_R"][mid]
        samples = binding.surface.sample_vertices(joints, rot, binding.sample_ids, double_precision=True)

        def values(b, q):
            metrics = b.foot_metrics(joints, rot, samples, skate_relative_R=q)
            ratio = b.ratio
            anchor = (metrics["anchors"] - metrics["targets"][None, None]) * ratio / [.012, .010, .012]
            cavity = (np.stack([v.max(axis=1) for v in metrics["violations"]], axis=1) * ratio - .0018) / .003
            seats = (metrics["support"] * ratio - .002) / .004
            return np.concatenate((anchor.reshape(len(frames), 2, 6),
                                   np.maximum(cavity, 0.), seats), axis=2)

        base = values(binding, relative)
        extra = np.stack([(values(v.binding, relative) - base) / EPS for v in variants], axis=-1) if jac else None
        for foot in range(2):
            derivatives = None
            if jac:
                rel = np.zeros((len(frames), base.shape[2], 5, 3))
                for axis in range(3):
                    r = relative.copy()
                    r[:, foot] = r[:, foot] @ Rotation.from_rotvec(EPS * np.eye(3)[axis]).as_matrix()
                    rel[:, :, 2, axis] = (values(binding, r)[:, foot] - base[:, foot]) / EPS
                derivatives = np.zeros((*rel.shape[:-1], BODY_DIM)), state, (foot, rel)
            push(base[:, foot], 9., "worn_foot_fit", 85., derivatives, None if extra is None else extra[:, foot])

    def proposed(self, step, fraction):
        def limit(vector, radius):
            scale = np.minimum(1., radius / np.maximum(np.linalg.norm(vector, axis=-1), 1.e-9))
            return vector * scale[..., None] * fraction

        b = step[:self.shape[0]].reshape(-1, BODY_DIM)
        d = np.zeros((*self.body.rotation_controls.shape[:2], 3))
        d[:, JOINTS] = limit(b[:, 3:].reshape(len(b), -1, 3), .19)
        s = limit(step[self.shape[0]:self.local_size].reshape(-1, 2, 3), .18)
        return (self.body.retracted(limit(b[:, :3], .070 / self.binding_seed.ratio), d),
                self.shoe.retracted(np.zeros_like(self.shoe.position_controls), s),
                np.clip(self.shared + fraction * step[self.local_size:], self.lo, self.hi))

    def checkpoint(self, label, summary):
        output = self.output
        atomic_npz(output / "controls.npz", **curve_metadata(self.body),
                   absolute_knots=self.body.knots, absolute_position_controls=self.body.position_controls,
                   absolute_rotation_controls=self.body.rotation_controls, relative_skate_schema=np.array(SHOE_SCHEMA),
                   relative_skate_knots=self.shoe.knots, relative_skate_rotation_controls=self.shoe.rotation_controls,
                   shared=self.shared, seed_shared=self.shared_seed, x=self.shared[:17],
                   source_pts=self.i.pts, source_time_base=self.i.tb, camera_sha256=np.array(self.i.camera_sha),
                   fitting_frames=self.train, held_out_frames=self.holdout,
                   silhouette_fitting_frames=self.train[self.train_silhouette[self.train]],
                   eligible_whole_skates=self.valid,
                   visual_path=np.array(str(self.visual / "whole_follow_evidence.npz")),
                   visual_sha256=np.array(self.visual_sha256),
                   initialization=np.array(self.i.parent["parameter"]["path"]),
                   optimization_step=np.array(label), fit_summary_json=np.array(json.dumps(summary)))
        emit("checkpoint", label=label, path=str(output / "controls.npz"), **summary)

    def inspect(self, frames):
        bind = revised_binding(self.binding_seed, self.shared)
        assembly = Assembly(bind)
        metrics = np.full((len(self.i.pts), 2, 3), np.nan, np.float32)
        for f, matches, _ in self.build_matches(frames, bind):
            for f1, foot, _, _, _ in matches:
                st = self.state(f1, bind)
                uv = projection(self.camera, assembly.points(st, foot, transforms=(st["R"], st["O"])), f1)["uv"]
                for j, pixels in zip(f1, uv):
                    old = distribute(self.region[j, foot], 96, closed=True)
                    boundary = np.concatenate(assembly.raster(pixels))
                    a = cKDTree(boundary).query(old)[0]
                    b = cKDTree(old).query(boundary)[0]
                    metrics[j, foot] = (a.mean(), b.mean(), .5 * (a.mean() + b.mean()))
        return metrics


def quantiles(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.quantile(x, .9)),
                mean=float(x.mean())) if x.size else dict(n=0)


def step_schur(J, residual, local_size, damping):
    local, common = J[:, :local_size], J[:, local_size:]
    H = (local.T @ local).tocsc()
    C = (local.T @ common).toarray()
    S = (common.T @ common).toarray()
    g = local.T @ residual
    q = common.T @ residual
    lu = splu(H + sparse.diags(damping * np.maximum(1., H.diagonal()), format="csc"),
              permc_spec="MMD_AT_PLUS_A")
    z = lu.solve(g)
    Y = lu.solve(C)
    v = np.linalg.solve(S + np.diag(damping * np.maximum(1., np.diag(S))) - C.T @ Y, -q + C.T @ z)
    return np.r_[-z - Y @ v, v]


def solve_native(problem, steps, active=None):
    accepted = 0
    damping = .017
    for k in range(steps):
        t = time.perf_counter()
        bind = revised_binding(problem.binding_seed, problem.shared)
        matches = list(problem.build_matches(problem.train, bind, fitting=True))
        emit("native_correspondences", iteration=k + 1, seconds=time.perf_counter() - t)
        current, J, residual = problem.system(matches, jac=True)
        emit("native_linearization", iteration=k + 1, rows=J.shape[0], nnz=J.nnz,
             seconds=time.perf_counter() - t, **current)
        if active is None:
            step = step_schur(J, residual, problem.local_size, damping)
        else:
            step = np.zeros(problem.size)
            step[active] = step_schur(J[:, active], residual,
                                      int(np.count_nonzero(active < problem.local_size)), damping)
        del J, residual
        previous = (problem.body, problem.shoe, problem.shared)
        for fraction in (1., .5, .25):
            try:
                problem.body, problem.shoe, problem.shared = problem.proposed(step, fraction)
                trial = problem.system(matches)
            except (ValueError, FloatingPointError) as e:
                emit("rejected_trial", iteration=k + 1, fraction=fraction, reason=str(e))
                problem.body, problem.shoe, problem.shared = previous
                continue
            emit("native_trial", iteration=k + 1, fraction=fraction,
                 seconds=time.perf_counter() - t, damping=damping, **trial)
            if trial["total"] < current["total"]:
                accepted += 1
                problem.checkpoint(k + 1, trial)
                damping = max(.007, damping * .75)
                break
            problem.body, problem.shoe, problem.shared = previous
        else:
            damping *= 4
    return accepted


def fit(visual, output, steps, warm_controls=None):
    cv2.setNumThreads(1)
    t0 = time.perf_counter()
    problem = Problem(visual, output, warm_controls)
    emit("native_source", exposures=len(problem.i.pts), fitting_exposures=len(problem.train),
         held_out_exposures=len(problem.holdout),
         usable_train_foot_exposures=int(problem.valid[problem.train[problem.train_silhouette[problem.train]]].sum()),
         usable_holdout_foot_exposures=int(problem.valid[problem.holdout].sum()),
         direct_runner_frames=len(problem.direct_edges), worn_geometry_exposures=len(problem.wear),
         camera_sha256=problem.i.camera_sha, native_library=problem.info["native_library"],
         body_controls=problem.shape[0], skate_controls=problem.shape[1], shared_controls=SHARED_DIM)
    initial = (problem.body_seed, problem.shoe_seed, problem.shared_seed)
    accepted = solve_native(problem, steps)
    if not accepted:
        raise RuntimeError("No native joint trial reduced the objective")
    final = (problem.body, problem.shoe, problem.shared)
    after = problem.inspect(problem.holdout)
    problem.body, problem.shoe, problem.shared = initial
    before = problem.inspect(problem.holdout)
    problem.body, problem.shoe, problem.shared = final
    atomic_npz(output / "held_out_silhouette.npz", before_px=before, after_px=after,
               source_pts=problem.i.pts, held_out_frames=problem.holdout)
    validation = dict(before=quantiles(before[..., 2]), after=quantiles(after[..., 2]),
                      by_shot={shot: dict(before=quantiles(before[problem.i.shots == shot, :, 2]),
                                          after=quantiles(after[problem.i.shots == shot, :, 2]))
                               for shot in np.unique(problem.i.shots)},
                      accepted=accepted, elapsed_seconds=time.perf_counter() - t0,
                      parameters=dict(zip(SHAPE_NAMES + STEEL + ("upper_height_scale",),
                                          np.exp(problem.shared[12:]).tolist())))
    atomic_json(output / "fit_validation.json", validation)
    emit("native_fit_complete", **validation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("visual", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warm-controls", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit(args.visual, args.output, args.steps, args.warm_controls)


if __name__ == "__main__":
    main()
