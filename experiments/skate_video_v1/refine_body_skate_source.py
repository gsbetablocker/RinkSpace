"""Use sparse, source-reviewed assembly outlines to repair a continuous local failure."""
from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .body_skate_fit import Problem, revised_binding, quantiles, step_schur
from .inputs import atomic_json, atomic_npz, digest, distribute, emit, read
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet


def controls_overlapping(curve, low, high):
    return np.flatnonzero((curve.knots[:-4] < high) & (curve.knots[4:] > low))


def refit(visual, seed, source, output, steps):
    p = Problem(visual, output)
    document = read(source)
    if document["source_sha256"] != p.i.media_sha:
        raise ValueError("Wrong source video for local measurements")
    checkpoints = seed / "controls.npz"
    with np.load(checkpoints, allow_pickle=False) as f:
        np.testing.assert_array_equal(f["source_pts"], p.i.pts)
        np.testing.assert_array_equal(f["camera_sha256"], p.i.camera_sha)
        p.body, p.shoe, p.shared = curve_from_packet(f), shoe_curve_from_packet(f), f["shared"]
    old = (p.body_seed, p.shoe_seed, p.shared_seed)
    base = (p.body, p.shoe, p.shared)
    p.correction_origin = p.correction()
    additional = np.r_[np.full(p.shape[0], 5.), np.full(p.shape[1], 8.), np.full(p.size - p.local_size, 340.)]
    p.prior = sparse.vstack((p.prior, sparse.diags(additional)), format="csr")
    train_pts = []
    for row in document["source_outlines"]:
        j, f = p.i.index[row["pts"]], row["foot"]
        if not p.train_silhouette[j]:
            raise ValueError("A held-out silhouette cannot become a fitting row")
        if p.valid[j, f]:
            continue
        p.region[j, f] = distribute(np.asarray(row["polygon_uv"], float), p.region.shape[2], closed=True)
        p.valid[j, f] = p.reviewed_source[j, f] = True
        train_pts.append(j)
    center = float(document["check_seconds"])
    limits = np.asarray(document["window_source_seconds"], float)
    in_window = (p.i.times >= limits[0]) & (p.i.times <= limits[1])
    p.train = np.unique(np.r_[np.flatnonzero(in_window & (np.arange(len(p.i.pts)) % 2 == 0)), train_pts])
    p.wear = set(p.train[::3].tolist())
    eval_frames = p.holdout[in_window[p.holdout]]
    pre = p.inspect(eval_frames)
    low, high = limits - p.i.times[0]
    nk = p.shape[0] // len(p.body.position_controls)
    a = (controls_overlapping(p.body, low, high)[:, None] * nk + np.arange(nk)).ravel()
    b = (p.shape[0] + controls_overlapping(p.shoe, low, high)[:, None] * 6 + np.arange(6)).ravel()
    active = np.r_[a, b, np.arange(p.local_size, p.size)]
    emit("local_native_feedback", window_source_seconds=limits.tolist(), exact_source_anchors=len(train_pts),
         active_body_controls=len(a), active_skate_controls=len(b), active_geometry=len(active) - len(a) - len(b),
         fit_pts=[int(p.i.pts[i]) for i in train_pts],
         held_out_check_pts=int(p.i.pts[np.argmin(abs(p.i.times - center))]))
    accepted = 0
    report = None
    damping = .024
    t = time.perf_counter()
    for n in range(steps):
        matches = list(p.build_matches(p.train, revised_binding(p.binding_seed, p.shared), fitting=True))
        state, J, residual = p.system(matches, jac=True)
        step = np.zeros(p.size)
        step[active] = step_schur(J[:, active], residual, len(a) + len(b), damping)
        previous = (p.body, p.shoe, p.shared)
        for f in (1., .5, .25):
            try:
                p.body, p.shoe, p.shared = p.proposed(step, f)
                trial = p.system(matches)
            except ValueError as error:
                emit("local_trial_rejected", reason=str(error))
            else:
                emit("local_native_trial", iteration=n + 1, fraction=f, before=state["total"], after=trial["total"])
                if trial["total"] < state["total"]:
                    accepted += 1
                    report = trial
                    damping *= .84
                    break
            p.body, p.shoe, p.shared = previous
        else:
            damping *= 3.
    post = p.inspect(eval_frames)
    focus = int(np.argmin(abs(p.i.times - center)))
    foot = int(document["foot"])
    local = dict(before=quantiles(pre[..., 2]), after=quantiles(post[..., 2]),
                 center_seconds=center, center_foot=foot, center_before_px=float(pre[focus, foot, 2]),
                 center_after_px=float(post[focus, foot, 2]),
                 new_source_outlines=source.as_posix(), new_source_outlines_sha256=digest(source),
                 parent_controls=checkpoints.as_posix(), parent_controls_sha256=digest(checkpoints),
                 accepted=accepted, elapsed_seconds=time.perf_counter() - t)
    atomic_json(output / "local_validation.json", local)
    emit("local_check", **local)
    if not accepted or local["center_after_px"] >= local["center_before_px"]:
        raise RuntimeError("Local native correction did not improve the withheld target")
    p.checkpoint(f"localized_{accepted}", report)
    final = (p.body, p.shoe, p.shared)
    after = p.inspect(p.holdout)
    p.body, p.shoe, p.shared = old
    before = p.inspect(p.holdout)
    p.body, p.shoe, p.shared = final
    previous = read(seed / "fit_validation.json")
    previous.update(accepted=previous["accepted"] + accepted, before=quantiles(before[..., 2]),
                    after=quantiles(after[..., 2]), local_review=local,
                    by_shot={shot: dict(before=quantiles(before[p.i.shots == shot, :, 2]),
                                        after=quantiles(after[p.i.shots == shot, :, 2]))
                             for shot in np.unique(p.i.shots)})
    atomic_npz(output / "held_out_silhouette.npz", before_px=before, after_px=after,
               source_pts=p.i.pts, held_out_frames=p.holdout)
    atomic_json(output / "fit_validation.json", previous)
    emit("local_correction_complete", before=previous["before"], after=previous["after"])


def main():
    parser = ArgumentParser()
    parser.add_argument("visual", type=Path)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    refit(args.visual, args.seed, args.source, args.output, args.steps)


if __name__ == "__main__":
    main()
