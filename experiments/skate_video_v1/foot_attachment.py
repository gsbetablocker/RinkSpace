"""Fit the observed assembly and articulate the foot inside it on the same body."""
from argparse import ArgumentParser
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from .body_skate_fit import (
    Assembly, BODY_DIM, EPS, JOINTS, Problem, SHAPE_NAMES, STEEL, projection, quantiles, revised_binding, solve_native,
)
from .inputs import atomic_json, atomic_npz, digest, distribute, emit, read
from .refine_body_skate_source import controls_overlapping
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve
from calibration.program_tracking.body_tracking.absolute_motion_state import (
    absolute_reference, curve_from_packet, curve_metadata, evaluate_curve,
)
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state


ANKLES = (7, 8)
TOES = (10, 11)
KNEES = (4, 5)
CHECKS = ((66.86, 0), (90.12, 0), (97.58, 0), (151.02, 1), (186.84, 1))
WINDOWS = ((66.50, 67.22), (89.66, 90.70), (97.18, 97.90), (150.47, 151.53))
# Hidden articulation lives around the existing joint centres.
WEAR_LIMIT = np.array([.24, .11, .16, .36])
WEAR_PRIOR = np.array([.18, .12, .15, .28])


def ankle_curve(curve):
    ref = curve.edge_reference_quaternion
    return AbsoluteMotionCurve(curve.knots, np.zeros_like(curve.position_controls),
                               curve.rotation_controls[:, ANKLES], None if ref is None else ref[:, ANKLES])


def ankle_curve_from_packet(packet):
    if "wear_reference_ankle_controls" not in packet:
        return None
    rot = packet["wear_reference_ankle_controls"]
    return AbsoluteMotionCurve(packet["absolute_knots"], np.zeros((len(rot), 3)), rot,
                               packet.get("wear_reference_ankle_chart"))


def compensate_skate(relative, present, reference, mount):
    # World boot rotation is world ankle rotation times mount times relative.
    # The boot origin stays fixed as well because registration pivots at ankle.
    return mount.swapaxes(-1, -2) @ present.swapaxes(-1, -2) @ reference @ mount @ relative


def apply_wear_reference(ref, reference):
    if reference is None:
        return ref
    ref.wear_reference = reference
    at = ref.at

    def sample(t):
        state = at(t)
        anchor = evaluate_curve(reference, t)["rotations"]
        mount = ref.binding.parameters()[0]
        state["skate_relative_R"] = compensate_skate(
            state["skate_relative_R"], state["local_R"][:, np.array(ANKLES) - 1], anchor, mount
        )
        return state

    ref.at = sample
    return ref


def restore(problem, path):
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_pts"], problem.i.pts)
        np.testing.assert_array_equal(saved["camera_sha256"], problem.i.camera_sha)
        body = curve_from_packet(saved)
        shoe = shoe_curve_from_packet(saved)
        reference = ankle_curve_from_packet(saved)
        shared = saved["shared"]
    if reference is not None:
        rotations = body.rotation_controls.copy()
        rotations[:, ANKLES] = reference.rotation_controls
        body = AbsoluteMotionCurve(body.knots, body.position_controls, rotations, body.edge_reference_quaternion)
    problem.body, problem.shoe, problem.shared = body, shoe, shared
    problem.correction_origin = problem.correction()


def window_mask(times):
    return np.logical_or.reduce([(times >= lo) & (times <= hi) for lo, hi in WINDOWS])


def attach_source_outlines(problem, parent):
    path = parent / "source_correction.json"
    if not path.is_file():
        return []
    doc = read(path)
    if doc["source_sha256"] != problem.i.media_sha:
        raise ValueError("Reviewed source outline clock changed")
    frames = []
    for row in doc["source_outlines"]:
        i, f = problem.i.index[row["pts"]], row["foot"]
        if not problem.train_silhouette[i]:
            raise ValueError("Held-out outline present in reviewed fitting source")
        if not problem.valid[i, f]:
            problem.region[i, f] = distribute(np.asarray(row["polygon_uv"], float),
                                               problem.region.shape[2], closed=True)
            problem.valid[i, f] = problem.reviewed_source[i, f] = True
        frames.append(i)
    return frames


def source_admission(problem, parent, subset=False):
    reviewed = attach_source_outlines(problem, parent)
    region = window_mask(problem.i.times)
    if subset:
        fit = np.flatnonzero(region & (np.arange(len(region)) % 2 == 0))
    else:
        fit = problem.train
    observed = np.flatnonzero(region & problem.train_silhouette & problem.valid.any(axis=1))
    problem.train = np.unique(np.r_[fit, observed, reviewed])
    problem.wear = set(problem.train[::(3 if subset else 7)].tolist())


def local_controls(problem):
    seconds = problem.i.times[0]

    def get(curve):
        return np.unique(np.concatenate([controls_overlapping(curve, lo - seconds, hi - seconds)
                                          for lo, hi in WINDOWS]))

    body = (get(problem.body)[:, None] * BODY_DIM + np.arange(BODY_DIM)).ravel()
    skate = problem.shape[0] + (get(problem.shoe)[:, None] * 6 + np.arange(6)).ravel()
    return np.r_[body, skate, np.arange(problem.local_size, problem.size)]


def socket_geometry(binding):
    p = binding.packet()
    x = p["parameters"].sole_length_m
    collar = p["inner"][np.unique(p["collar"])]
    r, t, _ = binding.parameters()
    ankle = -np.einsum("fi,fij->fj", t, r)
    return p, x, collar, ankle


class MountProblem(Problem):
    """Only a socket and the shin entrance govern placement of a visible boot."""

    def __init__(self, visual, output, *, uncertain_ankle=True):
        super().__init__(visual, output)
        self.uncertain_ankle = uncertain_ankle

    def body_sigma(self, frames, joint_ids, present):
        sigma = super().body_sigma(frames, joint_ids, present)
        if not self.uncertain_ankle:
            return sigma
        # A detector's ankle falls under the cuff in the source. Its positional
        # uncertainty includes the hidden part of the visible boot.
        for foot, joint in enumerate((15, 16)):
            visible = self.valid[frames, foot] & self.train_silhouette[frames]
            size = np.linalg.norm(np.ptp(np.nan_to_num(self.region[frames, foot]), axis=1), axis=1)
            variance = (.14 * size[:, None]) ** 2
            hidden_ankle = visible[:, None] & present & (joint_ids == joint)
            sigma = np.sqrt(sigma**2 + variance * hidden_ankle)
        return sigma

    def static(self, binding):
        p, length, collar, ankle = socket_geometry(binding)
        xx, yy, zz = ankle.T
        floor = np.interp(xx, p["xs"], p["floor"])
        half = .8 * np.interp(xx, p["xs"], p["width"])
        lo = np.c_[.08 * length - xx, np.abs(yy) - half, floor + .020 / binding.ratio - zz]
        hi = np.c_[xx - .47 * length, zz - np.median(collar[:, 2]) + .003 / binding.ratio]
        return np.maximum(0., np.c_[lo, hi]).ravel() * binding.ratio / .008

    def static_factor(self):
        return "ankle_inside_boot", 300.

    @staticmethod
    def cuff_values(binding, rotation, knee_minus_ankle):
        p, length, collar, ankle = socket_geometry(binding)
        shank = np.einsum("nfji,nfj->nfi", rotation, knee_minus_ankle)
        z = np.quantile(collar[:, 2], .40)
        height = z - ankle[:, 2]
        fraction = np.clip(height[None] / np.maximum(shank[..., 2], 1.e-5), 0., .45)
        enter = ankle[None] + fraction[..., None] * shank
        x, y = enter[..., 0], enter[..., 1]
        # Open collar has clearance. A crossing anywhere inside the opening
        # has zero cost, so it does not pull the shank to a chosen centre.
        room = .74 * np.interp(np.clip(x, 0., length), p["xs"], p["width"]) + .004 / binding.ratio
        margins = np.stack((collar[:, 0].min() - x, x - collar[:, 0].max(),
                            np.abs(y) - room, z - enter[..., 2] - .008 / binding.ratio), axis=-1)
        return np.maximum(0., margins) * binding.ratio / .016

    def wear_factors(self, frames, binding, variants, jac, push):
        st = self.state(frames, binding, jac)
        mid = slice(2, None, 5)
        direction = (st["native_world"][:, KNEES] - st["native_world"][:, ANKLES])[mid]
        rot = st["R"][mid]
        base = self.cuff_values(binding, rot, direction)
        extra = None
        if jac:
            v = []
            for variant in variants:
                b = variant.binding
                r, _ = b.transforms(st["P"][mid], st["Q"][mid], st["native"][mid], st["frames"][mid],
                                    skate_relative_R=st["skate_relative_R"][mid])
                v.append((self.cuff_values(b, r, direction) - base) / EPS)
            extra = np.stack(v, axis=-1)
            turns = Rotation.from_rotvec(EPS * np.eye(3)).as_matrix()
            fd = np.stack([(self.cuff_values(binding, rot @ turn, direction) - base) / EPS
                           for turn in turns], axis=-1)
            ankle_axes = st["Q"][mid, None] @ st["frames"][mid][:, ANKLES]
        for foot in range(2):
            derivatives = None
            if jac:
                rel = np.zeros((len(frames), base.shape[-1], 5, 3))
                rel[:, :, 2] = fd[:, foot]
                local = np.zeros((*rel.shape[:-1], BODY_DIM))
                tangent = (ankle_axes[:, foot].swapaxes(-1, -2) @ rot[:, foot]).swapaxes(-1, -2)
                at = 3 + 3 * JOINTS.index(ANKLES[foot])
                local[:, :, 2, at:at + 3] = fd[:, foot] @ tangent
                derivatives = local, st, (foot, rel)
            push(base[:, foot], 5., "shank_at_cuff", 19., derivatives, None if extra is None else extra[:, foot])


def current_state(p):
    return p.body, p.shoe, p.shared


def check_values(problem, scores):
    return {f"{sec:.2f}:{f}": float(scores[problem.i.index[round(sec * 12800)], f, 2])
            for sec, f in CHECKS if np.isfinite(scores[problem.i.index[round(sec * 12800)], f, 2])}


def ablate(visual, parent, output, steps):
    reports = {}
    for name, problem_class in (("previous_cavity", Problem), ("socket_and_cuff", MountProblem)):
        sub = output / name
        sub.mkdir(exist_ok=True)
        problem = (problem_class(visual, sub, uncertain_ankle=False) if name == "socket_and_cuff"
                   else problem_class(visual, sub))
        restore(problem, parent / "controls.npz")
        source_admission(problem, parent, subset=True)
        check = problem.holdout[window_mask(problem.i.times)[problem.holdout]]
        base = problem.inspect(check)
        # Keep shared geometry active in both runs under the same mild prior.
        regularizer = np.r_[np.full(problem.local_size, 3.), np.full(problem.size - problem.local_size, 240.)]
        problem.prior = sparse.vstack((problem.prior, sparse.diags(regularizer)), format="csr")
        emit("mount_ablation_start", model=name, frames=len(problem.train), test_exposures=len(check),
             active_geometry=problem.size - problem.local_size)
        accepted = solve_native(problem, steps, local_controls(problem))
        after = problem.inspect(check)
        reports[name] = dict(accepted=accepted, before=quantiles(base[..., 2]), after=quantiles(after[..., 2]),
                             checks_before=check_values(problem, base), checks_after=check_values(problem, after))
        emit("mount_ablation_end", model=name, **reports[name])
    atomic_json(output / "attachment_ablation.json", reports)


def hidden_wear(problem, steps=3):
    """Optimise only hidden foot rotations while preserving every boot pose."""
    time0 = time.perf_counter()
    ref = absolute_reference(problem.ref, problem.body)
    ref.relative_skate_curve = problem.shoe
    binding = revised_binding(problem.binding_seed, problem.shared)
    ref.binding = binding
    n = len(problem.body.position_controls)
    reference = ankle_curve(problem.body)
    source = np.unique(np.r_[np.arange(0, len(problem.i.pts), 5), problem.holdout[::5]])
    times = problem.ref.times[source]
    B = problem.body.basis_sparse(times).tocsr()
    mount = binding.parameters()[0]
    relative = evaluate_curve(problem.shoe, times)["rotations"]
    anchor = evaluate_curve(reference, times)["rotations"]
    zeros = np.zeros((len(source), 69))
    coeff = np.zeros((n, 2, 4))
    prior = sparse.diags(np.tile(.40 / WEAR_PRIOR, (n, 1)).T.ravel(), format="csc")
    D = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n))
    regularizer = sparse.vstack((prior, sparse.kron(sparse.diags(1.2 / WEAR_PRIOR), D)), format="csr")
    guard = regularizer.T @ regularizer
    joint_lookup = np.array([*ANKLES, *TOES]) - 1

    def body_from(q):
        d = np.zeros((*problem.body.rotation_controls.shape[:2], 3))
        d[:, ANKLES] = q[:, :, :3]
        d[:, TOES, 0] = q[:, :, 3]
        return problem.body.retracted(np.zeros_like(problem.body.position_controls), d)

    def adjust(sample, param=None):
        result = dict(sample)
        if param is not None:
            local = sample["local_R"].copy()
            a = np.zeros((2, 3))
            toe = np.zeros((2, 3))
            if param < 3:
                a[:, param] = EPS
            else:
                toe[:, 0] = EPS
            local[:, joint_lookup] = local[:, joint_lookup] @ Rotation.from_rotvec(np.r_[a, toe]).as_matrix()
            result["local_R"] = local
        result["skate_relative_R"] = compensate_skate(
            relative, result["local_R"][:, joint_lookup[:2]], anchor, mount
        )
        return result

    def skin(sample):
        st = fk_state(ref, sample, zeros)
        m = binding.foot_metrics(st["native"], st["frames"], skate_relative_R=st["skate_relative_R"])
        outside = np.stack([v.max(axis=1) for v in m["violations"]], axis=1) * binding.ratio
        # Equal-size clearance by wall, including the insole. The old patch
        # centres no longer have to coincide with invented seat coordinates.
        clearance = np.maximum(0., outside - .001) / .003
        support = m["support"] * binding.ratio
        lifted = np.maximum(0., support - [.012, .010]) / .010
        return np.concatenate((clearance, lifted), axis=2), outside, st

    def state_at(q):
        e = evaluate_curve(body_from(q), times)
        return {key: e[key] for key in ("P", "Q", "local_R")}

    def score(v, q):
        return sum(float((regularizer @ q[:, f].T.ravel()) @ (regularizer @ q[:, f].T.ravel()))
                   for f in range(2)) + float((v**2).sum())

    sample = state_at(coeff)
    _, previous_definition, _ = skin(adjust(sample))
    binding.respect_open_collar = True
    res, before, state_before = skin(adjust(sample))
    accepted = 0
    for k in range(steps):
        jac = np.stack([(skin(adjust(sample, j))[0] - res) / EPS for j in range(4)], axis=-1)
        m = res.shape[-1]
        basis = B[np.repeat(np.arange(len(source)), m)]
        proposed = coeff.copy()
        for f in range(2):
            J = sparse.hstack([basis.multiply(jac[:, f, :, j].reshape(-1, 1)) for j in range(4)], format="csr")
            H = (J.T @ J + guard + sparse.eye(4 * n) * .75).tocsc()
            gradient = J.T @ res[:, f].ravel() + guard @ coeff[:, f].T.ravel()
            d = spsolve(H, -gradient).reshape(4, n).T
            proposed[:, f] += np.clip(d, -.13, .13)
        proposed = np.clip(proposed, -WEAR_LIMIT, WEAR_LIMIT)
        current = score(res, coeff)
        for f in (1., .5, .25):
            test = coeff + f * (proposed - coeff)
            trial = state_at(test)
            cand, after, state_after = skin(adjust(trial))
            value = score(cand, test)
            emit("hidden_wear_trial", iteration=k + 1, fraction=f, current_cost=current, trial_cost=value,
                 penetration_mm=quantiles(after.max(-1).ravel() * 1000))
            if value < current:
                coeff, sample, res = test, trial, cand
                accepted += 1
                break
        else:
            break
    _, after, state_after = skin(adjust(sample))
    np.testing.assert_allclose(state_after["R"], state_before["R"], rtol=0., atol=5.e-12)
    np.testing.assert_allclose(state_after["O"], state_before["O"], rtol=0., atol=5.e-12)
    body = body_from(coeff)
    pixels = verify_wear_projection(problem, ref, body, reference, binding)
    stats = dict(accepted=accepted, model_foot_observations=len(source) * 2,
                 closed_collar_definition_mm=quantiles(previous_definition.max(-1).ravel()*1000),
                 previous_cavity_mm=quantiles(before.max(-1).ravel()*1000),
                 final_cavity_mm=quantiles(after.max(-1).ravel()*1000),
                 previous_floor_mm=quantiles(before[..., 0].ravel()*1000),
                 final_floor_mm=quantiles(after[..., 0].ravel()*1000),
                 max_pose_orientation_difference=float(np.max(np.abs(state_after["R"] - state_before["R"]))),
                 max_origin_difference_m=float(np.max(np.abs(state_after["O"] - state_before["O"])) * binding.ratio),
                 max_native_projection_difference_px=pixels,
                 seconds=time.perf_counter() - time0)
    emit("hidden_wear_complete", **stats)
    return body, reference, stats


def verify_wear_projection(problem, ref, body, reference, binding):
    after = absolute_reference(problem.ref, body)
    after.relative_skate_curve, after.binding = problem.shoe, binding
    apply_wear_reference(after, reference)
    i = problem.i
    frames = np.unique(np.r_[np.arange(0, len(i.pts), 139),
                             [i.index[round(sec * 12800)] for sec, _ in CHECKS]])
    query = problem.times[frames].ravel()
    zero = np.zeros((len(query), 69))
    mesh = Assembly(binding)
    a, b = (fk_state(r, r.at(query), zero) for r in (ref, after))
    error = 0.
    for foot in range(2):
        ua, ub = (projection(problem.camera, mesh.points(x, foot, transforms=(x["R"], x["O"])), frames)["uv"]
                  for x in (a, b))
        error = max(error, float(np.max(np.abs(ua - ub))))
    if error >= 1.e-7:
        raise ValueError(f"Hidden-foot fit moved the exterior in the native source image: {error}")
    return error


def publish_wear(problem, checkpoint, validation):
    body, reference, wear = hidden_wear(problem)
    validation["hidden_foot_fit"] = wear
    with np.load(checkpoint, allow_pickle=False) as saved:
        packet = dict(saved)
    packet.update(curve_metadata(body))
    packet.update(absolute_rotation_controls=body.rotation_controls,
                  wear_reference_ankle_controls=reference.rotation_controls,
                  wear_model=np.array("compensated_ankle_and_forefoot_v1"),
                  wear_respect_open_collar=np.array(True),
                  fit_summary_json=np.array(json.dumps(validation)))
    if reference.edge_reference_quaternion is not None:
        packet["wear_reference_ankle_chart"] = reference.edge_reference_quaternion
    atomic_npz(checkpoint, **packet)


def recheck_interior(visual, parent, output, steps):
    p = MountProblem(visual, output)
    checkpoint = output / "controls.npz"
    restore(p, checkpoint)
    validation_path = output / "fit_validation.json"
    validation = read(validation_path)
    if validation["fit_parent_sha256"] != digest(parent / "controls.npz"):
        raise ValueError("Fitting parent changed")
    publish_wear(p, checkpoint, validation)
    atomic_json(validation_path, validation)


def fit(visual, parent, output, steps):
    t0 = time.perf_counter()
    p = MountProblem(visual, output)
    restore(p, parent / "controls.npz")
    source_admission(p, parent)
    start = current_state(p)
    before = p.inspect(p.holdout)
    emit("mounted_visual_fit_start", native_library=p.info["native_library"], frames=len(p.train),
         usable_foot_exposures=int(p.valid[p.train[p.train_silhouette[p.train]]].sum()),
         local_measurements=int(p.reviewed_source.sum()), shared_geometry=p.size-p.local_size,
         held_out=quantiles(before[..., 2]), source_pts=[int(p.i.pts[0]), int(p.i.pts[-1])])
    accepted = solve_native(p, steps)
    after = p.inspect(p.holdout)
    emit("mounted_visual_fit_review", accepted=accepted, before=quantiles(before[..., 2]),
         after=quantiles(after[..., 2]), checks_before=check_values(p, before), checks_after=check_values(p, after))
    if not accepted:
        p.body, p.shoe, p.shared = start
        raise RuntimeError("No mounted visual update reduced the joint objective")
    # Retain the already accepted exterior and compute the foot inside it.
    validation = dict(before=quantiles(before[..., 2]), after=quantiles(after[..., 2]),
                      by_shot={shot: dict(before=quantiles(before[p.i.shots == shot, :, 2]),
                                          after=quantiles(after[p.i.shots == shot, :, 2]))
                               for shot in np.unique(p.i.shots)},
                      cases_before=check_values(p, before), cases_after=check_values(p, after),
                      fit_parent=str(parent / "controls.npz"), fit_parent_sha256=digest(parent / "controls.npz"),
                      accepted=accepted, parameters=dict(zip(SHAPE_NAMES + STEEL + ("upper_height_scale",),
                                                             np.exp(p.shared[12:]).tolist())),
                      attachment_model="open cuff crossing and anatomical ankle joint; adjustable shared outer geometry",
                      elapsed_seconds=time.perf_counter() - t0)
    checkpoint = output / "controls.npz"
    publish_wear(p, checkpoint, validation)
    atomic_npz(output / "held_out_silhouette.npz", before_px=before, after_px=after,
               source_pts=p.i.pts, held_out_frames=p.holdout)
    atomic_json(output / "fit_validation.json", validation)
    emit("mounted_fit_complete", before=validation["before"], after=validation["after"],
         output=str(output), seconds=validation["elapsed_seconds"])


def main():
    parser = ArgumentParser()
    parser.add_argument("visual", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--interior-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    function = ablate if args.ablation else recheck_interior if args.interior_only else fit
    function(args.visual, args.parent, args.output, args.steps)


if __name__ == "__main__":
    main()
