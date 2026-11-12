"""Seat a fixed anatomical ankle inside the image-fitted boot using two-link leg IK."""
from argparse import ArgumentParser
from copy import copy
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .body_skate_fit import Assembly, Problem, projection, quantiles
from .foot_attachment import ANKLES, CHECKS, KNEES, check_values
from .inputs import atomic_json, atomic_npz, digest, distribute, emit, read
from calibration.program_tracking.body_tracking.skate_physics import fk_state


HIPS = (1, 2)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1.e-12)


def root_shift(centres, radius):
    """Closest common hip translation satisfying both femur/tibia reaches."""
    distance = np.linalg.norm(centres, axis=-1)
    candidates = centres * np.maximum(0., 1. - radius / np.maximum(distance, 1.e-12))[..., None]
    crosses = np.linalg.norm(candidates - centres[:, ::-1], axis=-1) <= radius[:, ::-1] + 1.e-10
    result = candidates[:, 0].copy()
    result[~crosses[:, 0]] = candidates[~crosses[:, 0], 1]
    both = ~crosses.any(axis=1)
    if not np.any(both):
        return result
    c0, c1 = np.moveaxis(centres[both], 1, 0)
    r0, r1 = radius[both].T
    d = np.linalg.norm(c1 - c0, axis=-1)
    if np.any(d > r0 + r1 + 1.e-9) or np.any(d < np.abs(r0 - r1) - 1.e-9):
        raise ValueError("Ankle positions cannot be reached at the fixed leg lengths")
    axis = (c1 - c0) / d[:, None]
    offset = (r0**2 - r1**2 + d**2) / (2. * d)
    center = c0 + offset[:, None] * axis
    rim = np.sqrt(np.maximum(0., r0**2 - offset**2))
    u = -center + (center * axis).sum(axis=-1, keepdims=True) * axis
    bad = np.linalg.norm(u, axis=-1) < 1.e-10
    if np.any(bad):
        basis = np.eye(3)[np.argmin(np.abs(axis[bad]), axis=-1)]
        u[bad] = np.cross(axis[bad], basis)
    result[both] = center + rim[:, None] * unit(u)
    return result


def rotate_bone(old, new):
    a, b = unit(old), unit(new)
    q = np.concatenate((np.cross(a, b), 1. + (a * b).sum(-1, keepdims=True)), axis=-1)
    return Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)


def socket_binding(binding, shift):
    out = copy(binding)
    mount = binding.parameters()[0]
    out.x = binding.x.copy()
    x = out.x[:12].reshape(2, 6)
    x[:, :3] -= np.einsum("fij,fj->fi", mount, shift)
    out.respect_open_collar = True
    return out


def articulated_sample(ref, original, state, shift):
    native, frames = state["native"], state["frames"]
    h, k, a = (native[:, ids] for ids in (HIPS, KNEES, ANKLES))
    w = np.einsum("nfij,fj->nfi", state["R"], shift)
    desired = a + np.einsum("nji,nfj->nfi", state["Q"], w)
    thigh, shin = k - h, a - k
    l1, l2 = (np.linalg.norm(v, axis=-1) for v in (thigh, shin))
    total = l1 + l2
    slack = np.maximum(0., total - np.linalg.norm(a - h, axis=-1))
    # Retain a fraction of the original knee bend near full extension.
    centres = desired - h
    padding = np.minimum(.003 / ref.binding.ratio, slack * .5)
    between = np.linalg.norm(centres[:, 0] - centres[:, 1], axis=-1)
    room = np.maximum(0., total.sum(axis=1) - between)
    fraction = np.minimum(1., .9 * room / np.maximum(padding.sum(axis=1), 1.e-12))
    translation = root_shift(centres, total - fraction[:, None] * padding)
    target = desired - translation[:, None]
    ray = target - h
    distance = np.linalg.norm(ray, axis=-1)
    if np.any(distance < np.abs(l1 - l2)) or np.any(distance > total + 1.e-8):
        raise ValueError("Leg IK outside its two-link reach")
    direction = unit(ray)
    along = (l1*l1 - l2*l2 + distance*distance) / (2*distance)
    off = thigh - (thigh * direction).sum(-1, keepdims=True) * direction
    flat = np.linalg.norm(off, axis=-1) < 1.e-9
    if np.any(flat):
        off[flat] = np.cross(frames[:, KNEES, :, 0][flat], direction[flat])
    rad = np.sqrt(np.maximum(0., l1*l1 - along*along))
    new_knee = h + along[..., None] * direction + rad[..., None] * unit(off)
    hip_rot = rotate_bone(thigh, new_knee - h) @ frames[:, HIPS]
    knee_rot = rotate_bone(shin, target - new_knee) @ frames[:, KNEES]
    parent = frames[:, ref.body.parents[list(HIPS)]]
    local = original["local_R"].copy()
    local[:, np.array(HIPS) - 1] = parent.swapaxes(-1, -2) @ hip_rot
    local[:, np.array(KNEES) - 1] = hip_rot.swapaxes(-1, -2) @ knee_rot
    local[:, np.array(ANKLES) - 1] = knee_rot.swapaxes(-1, -2) @ frames[:, ANKLES]
    result = dict(original, local_R=local,
                  P=original["P"] + np.einsum("nij,nj->ni", state["Q"], translation))
    return result, translation, new_knee - k


def apply_socket_reference(ref, shift):
    """Use one shared boot-local ankle point and preserve the same outer skate SE(3)."""
    shift = np.asarray(shift, float)
    if shift.shape != (2, 3) or not np.isfinite(shift).all():
        raise ValueError("Invalid shared skate-local ankle offset")
    before = SimpleNamespace(body=ref.body, binding=ref.binding)
    at = ref.at
    ref.binding = socket_binding(ref.binding, shift)
    ref.binding.reference = ref
    ref.socket_shift = shift

    def sample(t):
        r = at(t)
        old = fk_state(before, r, np.zeros((len(r["P"]), 69)))
        return articulated_sample(before, r, old, shift)[0]

    def derivative(t, order):
        t = np.atleast_1d(t).astype(float)
        h = .001
        plus, minus = sample(t + h)["P"], sample(t - h)["P"]
        if order == 1:
            return (plus - minus) / (2. * h)
        if order == 2:
            return (plus - 2. * sample(t)["P"] + minus) / h**2
        raise ValueError("Socket reference supports first and second root derivatives")

    ref.at, ref.root_derivative = sample, derivative
    return ref


def compatibility(ref, original, state, shift):
    trial = SimpleNamespace(body=ref.body, binding=socket_binding(ref.binding, shift))
    r, move, knee = articulated_sample(ref, original, state, shift)
    pose = fk_state(trial, r, np.zeros((len(r["P"]), 69)))
    np.testing.assert_allclose(pose["O"], state["O"], atol=5.e-9, rtol=0.)
    np.testing.assert_allclose(pose["R"], state["R"], atol=5.e-9, rtol=0.)
    binding = trial.binding
    metric = binding.foot_metrics(pose["native"], pose["frames"], skate_relative_R=pose["skate_relative_R"])
    walls = np.stack([v.max(axis=1) for v in metric["violations"]], axis=1) * binding.ratio
    footbed = metric["support"] * binding.ratio
    wall_scale = np.r_[.0018, np.full(walls.shape[-1] - 1, .0035)]
    residual = np.r_[(np.maximum(walls - .001, 0.) / wall_scale).ravel(),
                     np.maximum(footbed - .008, 0.).ravel() / .008,
                     (move * binding.ratio).ravel() / .016, (knee * binding.ratio).ravel() / .019]
    return residual, walls, move, footbed, pose


def observed_body(problem, ref, frames):
    q = ref.at(problem.times[frames].ravel())
    coco = ref.body.forward_rotations(q["local_R"], hand_local_R=q.get("hand_local_R"))[0]
    world = q["P"][:, None] + np.einsum("nij,nkj->nki", q["Q"], coco)
    image = projection(problem.camera, world, frames)["uv"]
    ids = problem.obs["model_joint_ids"][frames]
    use = problem.obs["use"][frames] & np.isfinite(problem.obs["uv"][frames]).all(axis=2)
    ordered = image[np.arange(len(frames))[:, None], ids]
    px = np.linalg.norm(ordered - problem.obs["uv"][frames], axis=-1)
    return dict(all=quantiles(px[use]), exposed=quantiles(px[use & (ids < 15)]),
                hips_and_knees=quantiles(px[use & (ids >= 11) & (ids < 15)]))


def calibration_body_rows(problem, ref, frames, pose):
    local = ref.body.forward_rotations(pose["local_R"], hand_local_R=pose.get("hand_local_R"))[0]
    world = pose["P"][:, None] + np.einsum("nij,nkj->nki", pose["Q"], local)
    image = problem.camera.project(world, frames)
    ids = problem.obs["model_joint_ids"][frames]
    observed = problem.obs["uv"][frames]
    sigma = problem.obs["sigma_px"][frames]
    use = problem.obs["use"][frames] & np.isfinite(observed).all(axis=2) & (ids < 15)
    err = image[np.arange(len(frames))[:, None], ids] - observed
    return (err[use] / np.maximum(2., sigma[use, None])).ravel()


def silhouette_check(problem, ref):
    model = Assembly(ref.binding)
    values = np.full((len(problem.i.pts), 2, 3), np.nan, np.float32)
    for start in range(0, len(problem.holdout), 48):
        frames = problem.holdout[start:start + 48]
        for foot in range(2):
            use = frames[problem.valid[frames, foot]]
            if not len(use):
                continue
            t = problem.times[use].ravel()
            s = fk_state(ref, ref.at(t), np.zeros((len(t), 69)))
            pixels = projection(problem.camera, model.points(s, foot, transforms=(s["R"], s["O"])), use)["uv"]
            for j, pix in zip(use, pixels):
                observed = distribute(problem.region[j, foot], 96, closed=True)
                model_uv = np.concatenate(model.raster(pix))
                a = cKDTree(model_uv).query(observed)[0].mean()
                b = cKDTree(observed).query(model_uv)[0].mean()
                values[j, foot] = [a, b, (a+b)*.5]
    return values


def reach_envelope(problem, ref):
    """Clock-wide bound on a single constant two-boot registration change."""
    delta, rotation, diameter = [], [], []
    for clock in np.array_split(problem.times.ravel(), 32):
        pose = fk_state(ref, ref.at(clock), np.zeros((len(clock), 69)))
        h, k, a = (pose["native_world"][:, js] for js in (HIPS, KNEES, ANKLES))
        d = a - h
        delta.append(d[:, 0] - d[:, 1])
        rotation.append(pose["R"])
        diameter.append((np.linalg.norm(k-h, axis=-1) + np.linalg.norm(a-k, axis=-1)).sum(axis=1))
    return np.concatenate(delta), np.concatenate(rotation), np.concatenate(diameter)


def feasible_socket(shift, envelope):
    d, R, span = envelope
    change = np.einsum("nij,j->ni", R[:, 0], shift[0]) - np.einsum("nij,j->ni", R[:, 1], shift[1])
    aa = np.einsum("ni,ni->n", change, change)
    bb = np.einsum("ni,ni->n", d, change)
    cc = np.minimum(0., np.einsum("ni,ni->n", d, d) - span**2)
    valid = aa > 1.e-18
    maximum = np.ones(len(d))
    maximum[valid] = (np.sqrt(np.maximum(0., bb[valid]**2 - aa[valid]*cc[valid])) - bb[valid]) / aa[valid]
    return shift * np.clip(.998 * maximum.min(), 0., 1.) if np.any(maximum < 1.) else shift


def fit(visual, mounted, output):
    from .deliver_body_skate_fit import saved_ref

    p = Problem(visual, output)
    checkpoint = mounted / "controls.npz"
    r = saved_ref(p, checkpoint)
    train = np.arange(0, len(p.i.pts), 19)
    t = r.times[train]
    old = r.at(t)
    state = fk_state(r, old, np.zeros((len(t), 69)))
    limits_mm = np.array([22., 12., 24.] * 2)
    zero = np.zeros((2, 3))
    _, old_walls, _, _, _ = compatibility(r, old, state, zero)
    envelope = reach_envelope(p, r)

    def residual(mm):
        q = feasible_socket(mm.reshape(2, 3) / (1000 * r.binding.ratio), envelope)
        v, _, _, _, pose = compatibility(r, old, state, q)
        body = calibration_body_rows(p, r, train, pose)
        return np.r_[v, 1.45 * body, mm / 7., (mm - q.ravel()*1000*r.binding.ratio) / 5.]

    emit("ankle_socket_calibration_start", sampled_frames=len(train), source_parent=str(checkpoint),
         previous_cavity_mm=quantiles(old_walls.max(-1).ravel() * 1000))
    result = least_squares(residual, np.zeros(6), bounds=(-limits_mm, limits_mm), jac="2-point",
                           diff_step=1.e-4, loss="soft_l1", f_scale=3., max_nfev=30, verbose=1)
    shift = feasible_socket(result.x.reshape(2, 3) / (1000 * r.binding.ratio), envelope)
    _, walls, _, _, _ = compatibility(r, old, state, shift)
    candidate = saved_ref(p, checkpoint)
    apply_socket_reference(candidate, shift)
    check = np.arange(1, len(p.i.pts), 11)
    tk = r.times[check]
    old_sample = r.at(tk)
    old_pose = fk_state(r, old_sample, np.zeros((len(tk), 69)))
    _, held_wall, delta, held_lift, _ = compatibility(r, old_sample, old_pose, shift)
    _, held_old, _, old_held_lift, _ = compatibility(r, old_sample, old_pose, zero)
    audit = dict(shift_boot_xyz_mm=(shift * 1000 * r.binding.ratio).tolist(),
                 held_out_cavity_before_mm=quantiles(held_old.max(-1).ravel()*1000),
                 held_out_cavity_after_mm=quantiles(held_wall.max(-1).ravel()*1000),
                 held_out_sole_before_mm=quantiles(held_old[..., 0].ravel()*1000),
                 held_out_sole_after_mm=quantiles(held_wall[..., 0].ravel()*1000),
                 held_out_heel_bed_before_mm=quantiles(old_held_lift[..., 0].ravel()*1000),
                 held_out_heel_bed_after_mm=quantiles(held_lift[..., 0].ravel()*1000),
                 root_shift_mm=quantiles(np.linalg.norm(delta, axis=-1)*r.binding.ratio*1000),
                 body_before_px=observed_body(p, r, check[::2]),
                 body_after_px=observed_body(p, candidate, check[::2]))
    validation = read(mounted / "fit_validation.json")
    with np.load(mounted / "held_out_silhouette.npz", allow_pickle=False) as f:
        before = f["before_px"]
        mounted_px = f["after_px"]
    after = silhouette_check(p, candidate)
    audit.update(mounted_whole_px=quantiles(mounted_px[..., 2]), new_whole_px=quantiles(after[..., 2]),
                 fitted_train_cavity_mm=quantiles(walls.max(-1).ravel()*1000))
    emit("ankle_socket_calibration_review", **audit)
    atomic_json(output / "socket_review.json", audit)
    if audit["held_out_cavity_after_mm"]["median"] >= audit["held_out_cavity_before_mm"]["median"]:
        raise RuntimeError("Socket shift did not improve interior agreement")
    validation.update(after=quantiles(after[..., 2]),
                      by_shot={shot: dict(before=quantiles(before[p.i.shots == shot, :, 2]),
                                          after=quantiles(after[p.i.shots == shot, :, 2]))
                               for shot in np.unique(p.i.shots)},
                      cases_after=check_values(p, after), socket_alignment=audit,
                      socket_source_controls_sha256=digest(checkpoint))
    with np.load(checkpoint, allow_pickle=False) as f:
        data = dict(f)
    data.update(socket_shift_boot_units=shift, fit_summary_json=np.array(json.dumps(validation)),
                socket_kinematics=np.array("two_link_leg_ik_to_fixed_boot_local_ankle_v1"))
    atomic_npz(output / "controls.npz", **data)
    atomic_npz(output / "held_out_silhouette.npz", before_px=before, after_px=after,
               source_pts=p.i.pts, held_out_frames=p.holdout)
    atomic_json(output / "fit_validation.json", validation)
    emit("ankle_socket_calibration_complete", output=str(output))


def main():
    parser = ArgumentParser()
    parser.add_argument("visual", type=Path)
    parser.add_argument("mounted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit(args.visual, args.mounted, args.output)


if __name__ == "__main__":
    main()
