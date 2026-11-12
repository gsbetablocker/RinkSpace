"""Joint native-clock rigid pose refinement for both skate identities."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.linalg import spsolve

from evidence import colour_status, navigation
from inputs import BOOT, STEEL, Inputs, atomic_json, atomic_npz, emit, geometry, masked_distance, read
from model import RowNavigation, SkateModel, packet_from_fit, proposed_poses
from projection import Native, PoseCurve, exp, inverse, log, project_available
from track import Navigation
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection


def interpolate_gaps(p, r, observed):
    """Initialize every hidden exposure from both visible sides of the gap."""
    indexes = np.arange(len(p))
    for f in range(2):
        seen = indexes[observed[:, f]]
        if not len(seen):
            continue
        missing = indexes[~observed[:, f]]
        for branch in (0, 1):
            for j in range(3):
                p[missing, f, branch, j] = np.interp(missing, seen, p[seen, f, branch, j])
            k = np.searchsorted(seen, missing, side="left")
            a, b = seen[np.maximum(k - 1, 0)], seen[np.minimum(k, len(seen) - 1)]
            blend = (missing - a) / np.maximum(1, b - a)
            r0, r1 = r[a, f, branch], r[b, f, branch]
            r[missing, f, branch] = r0 @ exp(log(inverse(r0) @ r1) * blend[:, None])


def pair_path(times, p, r, unary):
    """Second-order dynamic programming over simultaneous two-shoe branches."""
    pair = np.array(list(product(range(p.shape[2]), repeat=2)))
    feet = np.arange(2)
    P, R = p[:, feet, pair], r[:, feet, pair]
    cost = unary[:, feet, pair].sum(axis=-1)
    dt = np.diff(times)
    count = len(pair)
    back = np.zeros((len(p), count, count), np.int8)
    work = cost[0, :, None] + cost[1, None]
    prev_v = (P[1, None] - P[0, :, None]) / dt[0]
    prev_w = log(R[1, None] @ inverse(R[0, :, None])) / dt[0]
    work[(np.linalg.norm(prev_w, axis=-1) * dt[0] >= np.pi - 1e-5).any(axis=-1)] = np.inf
    for i in range(2, len(p)):
        v = (P[i, None] - P[i - 1, :, None]) / dt[i - 1]
        w = log(R[i, None] @ inverse(R[i - 1, :, None])) / dt[i - 1]
        elapsed = (dt[i - 2] + dt[i - 1]) * .5
        dv = np.linalg.norm(v[None] - prev_v[:, :, None], axis=-1) / (145. * elapsed)
        dw = np.linalg.norm(w[None] - prev_w[:, :, None], axis=-1) / (430. * elapsed)
        # There is no cost on velocity itself. A constant high-speed slide or
        # turn pays zero interior acceleration cost.
        accel = .76 * huber_loss(dv, 1.5) + .52 * huber_loss(dw, 1.5)
        candidates = work[:, :, None] + accel.sum(axis=-1)
        back[i] = candidates.argmin(axis=0)
        work = candidates.min(axis=0) + cost[i][None]
        work[(np.linalg.norm(w, axis=-1) * dt[i - 1] >= np.pi - 1e-5).any(axis=-1)] = np.inf
        prev_v, prev_w = v, w
    path = np.empty(len(p), int)
    path[-2:] = np.unravel_index(work.argmin(), work.shape)
    for i in range(len(p) - 1, 1, -1):
        path[i - 2] = back[i, path[i - 1], path[i]]
    assignments = pair[path]
    frames = np.arange(len(p))[:, None]
    return p[frames, feet, assignments], r[frames, feet, assignments], assignments


def acceleration_matrix(times):
    """Finite differences on actual programme seconds, including cuts."""
    dt = np.diff(times)
    interval = (dt[1:] + dt[:-1]) * .5
    coeff = np.stack((1. / dt[:-1], -1. / dt[:-1] - 1. / dt[1:], 1. / dt[1:]), axis=1)
    coeff /= interval[:, None]
    scale = np.array([62., 62., 62., 285., 285., 285.] * 2)
    values = np.broadcast_to(coeff[:, :, None] / scale, (len(interval), 3, 12))
    columns = (np.arange(len(interval))[:, None, None] + np.arange(3)[None, :, None]) * 12 + np.arange(12)
    rows = np.broadcast_to(np.arange(len(interval) * 12).reshape(-1, 1, 12), values.shape)
    return coo_matrix((values.ravel(), (rows.ravel(), columns.ravel())),
                      shape=(len(interval) * 12, len(times) * 12)).tocsr()


def acceleration(times, p, r):
    dt = np.diff(times)
    v = np.diff(p, axis=0) / dt[:, None, None]
    w = log(r[1:] @ inverse(r[:-1])) / dt[:, None, None]
    interval = (dt[1:] + dt[:-1]) * .5
    return (np.concatenate((np.diff(v, axis=0) / 62., np.diff(w, axis=0) / 285.), axis=-1)
            / interval[:, None, None]).ravel()


def huber_loss(x, delta):
    a = np.abs(x)
    return np.where(a <= delta, x*x*.5, delta * (a - delta * .5))


def huber(x, delta=2.):
    return float(huber_loss(x, delta).sum())


def exposure_basis(times, query, frames):
    """Exact local Hermite position derivatives, and tangent-chart rotation weights."""
    step = times[1] - times[0]
    j = np.clip(np.searchsorted(times, query, side="right") - 1, 0, len(times) - 2)
    u = (query - times[j]) / step
    u2, u3 = u*u, u*u*u
    left = np.maximum(j - 1, 0)
    right = np.minimum(j + 2, len(times) - 1)
    alpha = (u3 - 2*u2 + u) / (j + 1 - left)
    beta = (u3 - u2) / (right - j)
    indices = np.stack((left, j, j + 1, right), axis=-1)
    values = np.stack((-alpha, 2*u3 - 3*u2 + 1 - beta, -2*u3 + 3*u2 + alpha, beta), axis=-1)
    # Linear extension matches PoseCurve outside the first and last exposure.
    before = query < times[0]
    after = query > times[-1]
    values[before] = np.stack((np.zeros(before.sum()), 1. - u[before], u[before],
                               np.zeros(before.sum())), axis=-1)
    values[after] = np.stack((np.zeros(after.sum()), 1. - u[after], u[after],
                              np.zeros(after.sum())), axis=-1)
    offsets = indices - frames[:, None, None] + 2
    if np.any((offsets < 0) | (offsets > 4)):
        raise ValueError("Exposure Jacobian exceeds local support")
    result = np.zeros((*query.shape, 5))
    for k in range(4):
        np.add.at(result, (np.arange(len(query))[:, None], np.arange(query.shape[1])[None], offsets[..., k]),
                  values[..., k])
    return result


def exact_factors(native, row_camera, times, frame_map, p, r, obs, *, jac=False):
    local, target, normal, weight = obs
    curve = PoseCurve(times, p, r)
    frame_count, feet, points = weight.shape
    active = weight > 0
    used = np.flatnonzero(active.any(axis=(1, 2)))
    residuals = []
    columns = []
    values = []
    use_native_jac = native.camera.core.supports_motion_xyz_jacobian
    for start in range(0, len(used), 256):
        active_frames = used[start:start + 256]
        frames = frame_map[active_frames]
        query = times[active_frames, None] + native.tau[frames, None] * np.linspace(-.5, .5, 5)
        pos, rot = curve.at(query)
        relative = np.einsum("nsfji,nfki->nsfkj", rot, local[active_frames])
        xyz = pos[..., None, :] + relative
        keep = active[active_frames]
        packed = xyz.reshape(len(frames), 5, feet * points, 3)
        kept = keep.reshape(len(frames), -1)
        valid_xyz = packed[np.arange(len(frames)), :, kept.argmax(axis=1)].copy()
        packed[:] = np.where(kept[:, None, :, None], packed, valid_xyz[:, :, None])
        node_uv = moving_projection(
            native.camera, packed / native.ratio, frames,
            jacobian=jac and use_native_jac)
        uv = node_uv["uv"].reshape(len(frames), feet, points, 2)
        nrm = normal[active_frames] * weight[active_frames, ..., None]
        residuals.append(np.sum((uv - target[active_frames]) * nrm, axis=-1)[keep])
        if not jac:
            continue
        if use_native_jac:
            native_jac = node_uv["pixel_jac_nodes"].reshape(len(frames), feet, points, 2, 5, 3)
            gradient = np.einsum("nfki,nfkisj->nfksj", nrm, native_jac) / native.ratio
        else:
            # Some frozen DLLs predate the analytic XYZ binding. In that case
            # use the local camera tangent and test every step on native pixels.
            world = node_uv["world"].reshape(len(frames), feet, points, 3) * native.ratio
            Rcw, centre = row_camera.r[frames], row_camera.c[frames]
            xyz_cam = (world - centre[:, :, None]) @ inverse(Rcw)
            focal = row_camera.intr[frames, :, None, :2]
            scaled = nrm * focal / np.maximum(xyz_cam[..., 2:], 1e-4)
            dz = -(scaled * xyz_cam[..., :2]).sum(axis=-1) / np.maximum(xyz_cam[..., 2], 1e-4)
            g = np.concatenate((scaled, dz[..., None]), axis=-1) @ Rcw
            row_w = node_uv["row_weights"].reshape(len(frames), feet, points, 5)
            gradient = g[..., None, :] * row_w[..., None]
        arm = relative.transpose(0, 2, 3, 1, 4)
        jac_nodes = np.concatenate((gradient, np.cross(arm, gradient)), axis=-1)
        basis = exposure_basis(times, query, active_frames)
        jacobian = np.einsum("nfksj,nsm->nfkmj", jac_nodes, basis)[keep]
        node = np.clip(active_frames[:, None] + np.arange(-2, 3)[None], 0, frame_count - 1)
        frame, foot, _ = np.where(keep)
        column = node[frame, :, None]*12 + foot[:, None, None]*6 + np.arange(6)
        columns.append(column.reshape(-1, 30))
        values.append(jacobian.reshape(-1, 30))
    residual = np.concatenate(residuals) if residuals else np.empty(0)
    if not jac:
        return residual
    column = np.concatenate(columns)
    values = np.concatenate(values)
    index = np.broadcast_to(np.arange(len(residual))[:, None], column.shape)
    matrix = coo_matrix((values.ravel(), (index.ravel(), column.ravel())),
                        shape=(len(residual), frame_count*12)).tocsr()
    matrix.eliminate_zeros()
    return residual, matrix


def continuous(native, row_camera, times, frame_map, start_p, start_r, refresh, *, rounds, label):
    accel_matrix = acceleration_matrix(times)
    p, r = start_p.copy(), start_r.copy()
    damping = .13
    last = None
    for step in range(rounds):
        began = time.perf_counter()
        obs = refresh(p, r)
        if not (obs[-1] > 0).any():
            raise ValueError("No qualified visual observations for continuous solve")
        e, j = exact_factors(native, row_camera, times, frame_map, p, r, obs, jac=True)
        a = acceleration(times, p, r)
        w = np.sqrt(np.minimum(1, 2. / np.maximum(np.abs(np.r_[e, a]), 1e-7)))
        matrix = vstack((j, accel_matrix), format="csr").multiply(w[:, None]).tocsr()
        weighted = np.r_[e, a] * w
        g = matrix.T @ weighted
        H = (matrix.T @ matrix).tocsc()
        diag = np.maximum(H.diagonal(), 1e-4)
        initial = huber(e) + huber(a)
        committed = False
        reasons = []
        linearized = time.perf_counter()
        for trial in range(5):
            delta = spsolve(H + diags(diag * damping + 1e-6), -g).reshape(len(times), 2, 6)
            # Bound each local retraction; one unobservable outlier cannot
            # shrink the numerical step everywhere else on the programme.
            trust = np.maximum(1., np.maximum(np.linalg.norm(delta[..., :3], axis=-1) / .65,
                                             np.linalg.norm(delta[..., 3:], axis=-1) / .37))
            delta /= trust[..., None]
            q = p + delta[..., :3]
            s = exp(delta[..., 3:]) @ r
            try:
                b = exact_factors(native, row_camera, times, frame_map, q, s, obs)
                movement = acceleration(times, q, s)
                score = huber(b) + huber(movement)
            except ValueError as error:
                reasons.append(str(error))
                score = np.inf
            if score < initial:
                p, r = q, s
                committed = True
                damping = max(.005, damping * .50)
                break
            damping *= 3.2
        emit(label, iteration=step+1, accepted=committed, pixel_rows=len(e), unknown_exposures=int(
                 np.sum(~(obs[-1] > 0).any(axis=(1, 2)))),
             before_pixel=huber(e), before_motion=huber(a),
             after_pixel=huber(b) if committed else None, after_motion=huber(movement) if committed else None,
             damping=damping, trial_domain_failures=reasons, sparse_nonzero=matrix.nnz,
             linearization_s=round(linearized-began, 2), elapsed_s=round(time.perf_counter()-began, 2))
        last = obs
        if not committed or (initial - score) / max(initial, 1e-8) < .0005:
            break
    return p, r, last


def synthetic_closure(inputs, native, rows, model, p, r, rounds, *, smooth=False):
    """Use the actual PTS, shoe materials and DLL, with an artificial occlusion."""
    start = int(np.searchsorted(inputs.times, 80.1))
    frames = np.arange(start, start + 71)
    t = inputs.times[frames]
    known_p, known_r = p[frames].copy(), r[frames].copy()
    if smooth:
        dt = t - t[len(t) // 2]
        v = np.array([[1.7, -.45, .18], [1.55, -.49, .21]])
        known_p[:] = known_p[len(t) // 2] + dt[:, None, None] * v
        omega = np.array([[.12, .34, 7.2], [.2, -.15, -5.5]])
        known_r[:] = exp(dt[:, None, None] * omega) @ known_r[len(t) // 2]
    shell = model.boot[:, np.linspace(0, model.boot.shape[1] - 1, 16).astype(int)]
    local = np.concatenate((shell, model.steel[:, :, ::7].reshape(2, -1, 3)), axis=1)
    truth = native.project(PoseCurve(t, known_p, known_r), local, frames)
    loc = np.repeat(np.broadcast_to(local[None], (len(t), *local.shape)), 2, axis=2)
    target = np.repeat(truth, 2, axis=2)
    normals = np.tile(np.eye(2), (len(t), 2, local.shape[1], 1))
    weight = np.ones(target.shape[:-1])
    weight[:2] = weight[-2:] = 0.
    weight[25:35, 0] = 0.
    weight[39:50, 1] = 0.
    phase = np.arange(len(t))[:, None, None]
    perturb = .07 * np.sin(phase * .27 + np.array([0., 1., 2.])) + .06
    p0 = known_p + perturb
    r0 = exp(np.broadcast_to(.16 * np.cos(phase * .24 + np.array([0., .6, .9])),
                             (len(t), 2, 3))) @ known_r
    observations = (loc, target, normals, weight)
    solve_p, solve_r, _ = continuous(native, rows, t, frames, p0, r0, lambda p, r: observations,
                                          rounds=rounds, label="synthetic_native")
    visible = weight > 0
    e0 = exact_factors(native, rows, t, frames, p0, r0, observations)
    e1 = exact_factors(native, rows, t, frames, solve_p, solve_r, observations)
    hidden_weight = 1. - weight
    hidden_weight[:2] = hidden_weight[-2:] = 0.
    hidden = loc, target, normals, hidden_weight
    h0 = exact_factors(native, rows, t, frames, p0, r0, hidden)
    h1 = exact_factors(native, rows, t, frames, solve_p, solve_r, hidden)
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    return dict(mode="constant_world_velocities" if smooth else "historical_motion_stress",
                pts=inputs.pts[frames].tolist(), coordinate_px_rms_visible=[rms(e0), rms(e1)],
                coordinate_px_rms_hidden=[rms(h0), rms(h1)],
                position_p50_m=[float(np.median(np.linalg.norm(q-known_p, axis=-1))) for q in (p0, solve_p)],
                rotation_p50_deg=[float(np.median(np.rad2deg(np.linalg.norm(log(q @ inverse(known_r)), axis=-1))))
                                  for q in (r0, solve_r)],
                known_material_points=int(len(e0)), visibility_gap_foot_exposures=int(10 + 11))


def action_diagnostics(inputs, p, r, image):
    """Candidate direction/contact relationships on the same world clock."""
    n = len(p)
    velocity = np.gradient(p, inputs.times, axis=0)
    angular = log(r[1:] @ inverse(r[:-1])) / np.diff(inputs.times)[:, None, None]
    longitudinal = np.sum(velocity * r[..., 0], axis=-1)
    lateral = np.sum(velocity * r[..., 1], axis=-1)
    side = r[:, :, 2, 1]
    support = np.zeros((n, 2), np.uint8)
    flight = np.zeros(n, bool)
    grasp = np.zeros((n, 2), bool)
    turning = np.zeros(n, bool)
    for event in inputs.action:
        mask = (inputs.pts >= event["start_pts"]) & (inputs.pts <= event["end_pts_inclusive"])
        if event["kind"] == "flight_visible_candidate" and event["role"] == "fit":
            flight |= mask
        if event["kind"] == "rotation_source_path":
            turning |= mask
        for f, foot in enumerate(("left", "right")):
            state = event["per_foot_state"].get(foot)
            if state in ("supported", "lifted") and event["role"] == "fit":
                support[mask, f] = 1 if state == "supported" else 2
            if state == "related":
                grasp[mask, f] = True
    support[flight] = 3
    usable = (image["boot_status"] == 1) & (image["steel_status"] == 2) & image["heel_toe_ordered"]
    reliable_direction = usable & (support == 1)
    travel = np.where(reliable_direction, np.sign(longitudinal), 0).astype(np.int8)
    # Signed blade inclination and travel are conditional candidates. We do not
    # assign a figure name where edge identity and transition are unobserved.
    signed_tilt = np.arcsin(np.clip(side, -1, 1))
    return dict(world_velocity=velocity, world_angular_velocity=angular,
                blade_along_velocity=longitudinal, blade_lateral_velocity=lateral,
                support_source_phase=support, flight_source_phase=flight,
                grasp_source_candidate=grasp, turn_source_phase=turning,
                travel_direction_candidate=travel, signed_tilt_rad=signed_tilt)


def measure_checks(inputs, model, native, p, r, old_p, old_r):
    frames = np.array(sorted({inputs.index[row["pts"]] for row in inputs.checks}))
    lookup = {k: j for j, k in enumerate(frames)}
    pixels = [project_available(native, PoseCurve(inputs.times, pp, rr), model.points, frames)
              for pp, rr in ((old_p, old_r), (p, r))]
    result = []
    for row in inputs.checks:
        foot = row["foot"]
        if foot not in (0, 1) or row["semantic"] not in BOOT | STEEL:
            continue
        i = inputs.index[row["pts"]]
        sample = np.asarray(row["polyline_uv"])
        errors = []
        for uv, pp, rr in zip(pixels, (old_p, p), (old_r, r), strict=True):
            shape = uv[lookup[i], foot]
            if not np.isfinite(shape).all():
                errors.append(None)
                continue
            if row["semantic"] in BOOT:
                poly = model.source_outline(shape[:model.point_count])
                error = float(np.median(masked_distance(sample, poly, closed=True))) if poly is not None else None
            else:
                local_view = (native.centres[i] - pp[i, foot]) @ rr[i, foot]
                side = int(local_view[1] >= 0)
                edge = int(row["semantic"] == "runner_upper_edge")
                curve = shape[model.point_count:].reshape(-1, 2, model.runner.shape[-2], 2)[edge, side]
                error = float(np.median(masked_distance(sample, curve)))
            errors.append(error)
        result.append(dict(pts=int(row["pts"]), foot=int(foot), semantic=row["semantic"], old_px=errors[0],
                           new_px=errors[1], direct=row.get("evidence_status") == "observed",
                           source_radius_px=float(row.get("radius_px") or 6.)))
    return result


def main(output, rounds, closure_only):
    cv2.setNumThreads(1)
    inputs = Inputs()
    native = Native(inputs)
    shape, p0, r0 = geometry(inputs)
    model = SkateModel(shape)
    nav = Navigation(inputs)
    initial_rows = nav.centre[:, :, 1]
    row = RowNavigation(native, initial_rows)
    if closure_only:
        tests = [synthetic_closure(inputs, native, row, model, p0, r0, rounds, smooth=smooth)
                 for smooth in (False, True)]
        atomic_json(output / "native_recovery.json", tests)
        for report in tests:
            emit("native_recovery", **{k: v for k, v in report.items() if k != "pts"})
        return
    with np.load(output / "image_evidence.npz", allow_pickle=False) as f:
        image = {key: f[key] for key in f.files}
    np.testing.assert_array_equal(image["source_pts"], inputs.pts)
    if str(image["source_video_sha256"]) != inputs.media_sha or str(image["camera_sha256"]) != inputs.camera_sha:
        raise ValueError("Visual source/camera binding changed")
    chosen_row = np.where(image["boot_status"] > 0, image["boot"][..., 1].mean(axis=-1), initial_rows)
    row = RowNavigation(native, chosen_row)
    before = time.perf_counter()
    emit("proposal_start", visual_frames=int((image["boot_status"] > 0).any(axis=1).sum()),
         prior_depth_constraint=False, camera_sha256=inputs.camera_sha, template_sha256=str(shape["template_sha256"]))
    bank_p, bank_r, bank_cost, visible = proposed_poses(inputs, image, nav, model, row, p0, r0)
    interpolate_gaps(bank_p, bank_r, visible)
    seed_p, seed_r, branch = pair_path(inputs.times, bank_p, bank_r, bank_cost)
    emit("proposal_complete", seconds=round(time.perf_counter() - before, 2),
         image_selected=int(np.sum(branch < 2)), parent_start_selected=int(np.sum(branch == 2)))
    refresh = lambda p, r: packet_from_fit(inputs, image, nav, model, row, p, r)
    p, r, last = continuous(native, row, inputs.times, np.arange(len(p0)), seed_p, seed_r, refresh,
                                   rounds=rounds, label="programme_visual")
    status = np.maximum(image["boot_status"], image["steel_status"])
    status[inputs.withheld] = 0
    sensitivity = (last[-1] > 0).sum(axis=-1)
    status[sensitivity == 0] = 0
    diagnostics = action_diagnostics(inputs, p, r, image)
    frame_rot = np.rad2deg(np.linalg.norm(log(r[1:] @ inverse(r[:-1])), axis=-1))
    alt_branch = np.where(branch == 0, 1, 0)
    alt_p = bank_p[np.arange(len(p))[:, None], np.arange(2), alt_branch]
    alt_r = bank_r[np.arange(len(p))[:, None], np.arange(2), alt_branch]
    atomic_npz(output / "visual_shoes.npz", source_pts=inputs.pts, pts=inputs.pts, time_base=inputs.tb,
               shot_ids=inputs.shots, source_seconds=inputs.times, origin_world_m=p, rotation_world=r,
               seed_origin_world_m=seed_p, seed_rotation_world=seed_r, start_branch=branch,
               alternative_origin_world_m=alt_p, alternative_rotation_world=alt_r,
               alternative_semantics=np.array("unrefined competing image initialization; spread is not a confidence interval"),
               silhouette_status=image["boot_status"], steel_status=image["steel_status"], observation_status=status,
               visual_factor_counts=sensitivity, uncertainty_origin=np.array("monocular: weakly identified along viewing ray"),
               check_protection=inputs.withheld, camera_sha256=np.array(inputs.camera_sha),
               native_dll_sha256=np.array(native.camera.info["native_library_sha256"]),
               source_video_sha256=np.array(inputs.media_sha), template_sha256=shape["template_sha256"],
               length_export_ratio=np.array(native.ratio), world_length_domain=np.array("conditional programme metres"),
               **diagnostics)
    report = measure_checks(inputs, model, native, p, r, p0, r0)
    stats = dict(check=report, edges_gt_60deg=int((frame_rot > 60.).sum()),
                 same_shot_edges_gt_60deg=int(((frame_rot > 60.) & (inputs.shots[1:] == inputs.shots[:-1])[:, None]).sum()),
                 turns_p50_p95_deg=np.percentile(frame_rot, [50, 95]).tolist(),
                 max_turn_deg=float(frame_rot.max()),
                 displacement_p50_p95_m=np.percentile(np.linalg.norm(np.diff(p, axis=0), axis=-1), [50, 95]).tolist(),
                 observation_status_counts={colour_status(s): int((status == s).sum()) for s in range(4)})
    atomic_json(output / "pose_checks.json", stats)
    rows = read(output / "image_checks.json")["rows"]
    flag = np.r_[False, (frame_rot > 60.).any(axis=1)]
    atomic_json(output / "navigation.json", navigation(inputs, image, rows, pose_flags=flag))
    emit("visual_shoes", file=str(output / "visual_shoes.npz"), **{k: v for k, v in stats.items() if k != "check"})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--native-recovery", action="store_true")
    args = parser.parse_args()
    main(args.output, args.rounds, args.native_recovery)
