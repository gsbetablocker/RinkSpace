"""Select a continuous path of independent skate poses using the source evidence."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from rinkspace.review.render import source_frames
from scripts.fit_programme_shoes import (
    PROGRAMME, SIZE, ImageScore, LocalBackground, RowProjector, SkateGeometry, camera_matrices,
    crop_for, emit, evaluate_traces, project_pose, read, reference_frames, row_camera, source_hints, traced_rows,
)


# Retain alternate image-supported views instead of treating the best
# per-exposure pose as a unique orientation/depth measurement.
NEIGHBOURS = np.array([0, -1, 1, -3, 3, -6, 6])
BRANCHES = len(NEIGHBOURS)+4


def world_depth(camera, points):
    return np.sum(camera["rotation"][:, 2]*(points.reshape(-1, 3)-camera["centre"]), axis=-1).reshape(-1, 2)


def smooth_depth(raw, seed, shots):
    answer = seed.copy()
    for shot in dict.fromkeys(shots.tolist()):
        take = np.flatnonzero(shots == shot)
        base = gaussian_filter1d(seed[take], 2.0, axis=0, mode="nearest")
        relative = np.clip(np.log(raw[take]/seed[take]), -.085, .085)
        trend = median_filter(relative, size=(21, 1), mode="nearest")
        answer[take] = base*np.exp(gaussian_filter1d(trend, 5.5, axis=0, mode="nearest"))
    return answer


def alternatives(rotations, seed, index):
    neighbours = rotations[np.clip(index+NEIGHBOURS, 0, len(rotations)-1)]
    score = np.einsum("nij,mij->nm", neighbours, neighbours)
    centre = int(np.sum(score > 2.0, axis=1).argmax())
    consistent = score[centre] > 2.0
    smooth = Rotation.from_matrix(neighbours[consistent]).mean().as_matrix()
    local_roll = Rotation.from_rotvec(np.array([[0., 0., .21], [0., 0., -.21]])).as_matrix()
    return np.concatenate((neighbours, seed[None], smooth[None], smooth @ local_roll))


def pose_options(frame, shape, camera, crop, ref_depth, uv, seed_uv, rotations, rows, background, hint, ice):
    lo, hi = crop
    origins = np.empty((BRANCHES, 3))
    source_scores = np.zeros((BRANCHES, 5))
    positions = np.empty((BRANCHES, 2))
    cost = np.zeros(BRANCHES)
    centre = uv - lo
    seed = seed_uv - lo
    step = np.clip(shape.length*camera.f.mean()/ref_depth*.095, 3., 10.)
    xs, ys = np.meshgrid([-2*step, -step, 0., step, 2*step], [-step, 0., step])
    offset = np.r_[np.column_stack((xs.ravel(), ys.ravel())), (seed-centre)[None]]
    use_image = min(hi-lo) >= 12
    patch = ImageScore(frame, lo, hi, rows, shape, background, hint) if use_image else None
    for b, R in enumerate(rotations):
        rot = camera.R @ R
        projected = project_pose(camera, shape, centre, ref_depth, rot)
        shifts = offset
        if patch is not None and patch.cuff_uv is not None:
            shifts = np.r_[shifts, (patch.cuff_uv-projected[shape.cuff][0])[None]]
        if rows:
            source_centres = [r["points"].mean(axis=0)-lo-projected[shape.sem[r["semantic"]]].mean(axis=0)
                              for r in rows]
            shifts = np.r_[shifts, np.clip(np.asarray(source_centres), -step*6., step*6.)]
        if patch is None:
            pos = centre
        else:
            score, quality = patch(projected, shifts)
            score += .10*np.sum((shifts/(3.5*step))**2, axis=-1)
            win = int(score.argmin())
            pos = centre + shifts[win]
            exact = project_pose(camera, shape, pos, ref_depth, rot)
            px_cost, px_quality = patch(exact, np.zeros((1, 2)))
            cost[b] = px_cost[0] + .10*np.sum((shifts[win]/(3.5*step))**2)
            source_scores[b] = px_quality[0]
        origins[b], _ = camera.world(pos, ref_depth, rot, shape.centre)
        positions[b] = pos + lo
        # A broad soft term operates only over the bounded rink footprint.
        # It is a model prior, never an assessment of physical penetration.
        edge = shape.runner[:, ::8] @ R.T + origins[b]
        finite = (abs(edge[..., :2]) <= ice).all(axis=-1)
        if finite.any():
            low = float(edge[..., 2][finite].min())
            cost[b] += .36*(max(0., -.07-low)/.18)**2
    return origins, source_scores, positions, cost


def gather_span(video, source, timeline, matrix, shot, start, stop, shape, depths, fit_uv, seed_uv,
                rotations, seed_rotations, camera_packet, hints, trace_rows, ice):
    cv2.setNumThreads(1)
    started = time.perf_counter()
    count = stop-start
    P = np.empty((count, 2, BRANCHES, 3))
    Q = np.empty((count, 2, BRANCHES, 3, 3))
    cost = np.empty((count, 2, BRANCHES))
    scores = np.empty((count, 2, BRANCHES, 5))
    uv = np.empty((count, 2, BRANCHES, 2))
    ref_steps = np.array([-72, -47, -26, 26, 47, 72])
    ref = np.arange(max(0, start-75), min(len(timeline), stop+75), 5)
    ref_frames = tuple(reference_frames(video, source, timeline[ref]))
    for i, (stamp, frame) in enumerate(source_frames(video, source, timeline[start:stop]), start=start):
        target = np.clip(i+ref_steps, 0, len(timeline)-1)
        choose = np.unique(np.abs(ref[None]-target[:, None]).argmin(axis=1))
        choose = choose[abs(ref[choose]-i) >= 12]
        bg = LocalBackground([ref_frames[j] for j in choose], matrix[ref[choose]] @ np.linalg.inv(matrix[i]))
        for foot in range(2):
            key = i*2+foot
            direct = trace_rows.get((stamp, foot, "fit"), ())
            size = shape.length*camera_packet["fx"][key]/depths[i, foot]
            crop = crop_for(fit_uv[i, foot], size, direct)
            cam = RowProjector(camera_packet, key, crop[0])
            j = i-start
            Q[j, foot] = alternatives(rotations[:, foot], seed_rotations[i, foot], i)
            P[j, foot], scores[j, foot], uv[j, foot], cost[j, foot] = pose_options(
                frame, shape, cam, crop, depths[i, foot], fit_uv[i, foot], seed_uv[i, foot],
                Q[j, foot], direct, bg, hints[i, foot], ice)
    emit("image_branches", shot=str(shot), first_pts=int(timeline[start]), last_pts=int(timeline[stop-1]),
         exposures=int(count), poses=int(count*2*BRANCHES),
         best_mean=float(cost.min(axis=-1).mean()), seconds=round(time.perf_counter()-started, 2))
    return P, Q, scores, uv, cost


def choose_path(origin, rotation, cost, pivot):
    """Shortest temporal path within one shot, with exact backtracking."""
    steps, nodes = cost.shape
    centre = origin + rotation @ pivot
    r = rotation.reshape(steps, nodes, 9)
    forward = cost[0].copy()
    back = np.zeros((steps, nodes), dtype=np.intp)
    for i in range(1, steps):
        distance = np.linalg.norm(centre[i-1, :, None] - centre[i, None, :], axis=-1)
        cosine = np.clip((r[i-1] @ r[i].T-1.)/2., -1., 1.)
        theta = np.arccos(cosine)
        delta = .41*(theta/.39)**2 + .30*(distance/.20)**2
        total = forward[:, None] + delta
        back[i] = total.argmin(axis=0)
        forward = cost[i] + total[back[i], np.arange(nodes)]
    state = np.empty(steps, dtype=np.intp)
    state[-1] = forward.argmin()
    for i in range(steps-1, 0, -1):
        state[i-1] = back[i, state[i]]
    return state


def motion_report(origin, rotation, shots):
    continuous = shots[:-1] == shots[1:]
    d = np.linalg.norm(np.diff(origin, axis=0), axis=-1)
    cos = np.clip((np.sum(rotation[1:]*rotation[:-1], axis=(-1,-2))-1)/2, -1, 1)
    turn = np.degrees(np.arccos(cos))
    return dict(turn_p50_p95_deg=np.percentile(turn[continuous], [50,95]).tolist(),
                adjacent_turn_gt_60=int((turn[continuous] > 60).sum()),
                position_p50_p95_m=np.percentile(d[continuous], [50,95]).tolist())


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("project", "candidate", "source-evidence", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--source-hints", type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error("Use 1–8 workers")
    if (args.output/"shoes.npz").exists() or not args.output.is_dir():
        raise ValueError("Choose a new output directory")
    begin = time.perf_counter()
    with np.load(args.candidate, allow_pickle=False) as data:
        packet = {k: data[k] for k in data.files}
    pts, shots, tb = required_clock(read(args.project/PROGRAMME/"rebuild_scope.json"))
    np.testing.assert_array_equal(packet["source_pts"], pts)
    np.testing.assert_array_equal(packet["shot_ids"], shots)
    np.testing.assert_array_equal(packet["time_base"], tb)
    document = read(args.source_evidence)
    camera_record = Path(str(packet["camera_record"]))
    if digest(camera_record) != document["programme_camera_candidate"]["record_sha256"]:
        raise ValueError("Source evidence belongs to a different camera record")
    camera = ProgrammeCamera(args.project, dict(pts=pts, image_height=int(SIZE[1]),
        metres_per_scene_unit=float(packet["solver_metres_per_scene_unit"])), shots, record_path=camera_record)
    if camera.info["state_sha256"] != str(packet["camera_sha256"]):
        raise ValueError("Frozen camera state changed")
    media = Path(str(packet["source_video_path"]))
    if digest(media) != str(packet["source_video_sha256"]):
        raise ValueError("Candidate source video changed")
    with np.load(read(camera_record)["state"]["compact_path"], allow_pickle=False) as data:
        ice = np.array(data["board"][:2])*camera.info["metres_per_scene_unit"]
    ratio = float(packet["length_ratio"])
    geometry = SkateGeometry.from_candidate(packet)
    raw_o, raw_r = packet["origin_world_m"], packet["rotation_world"]
    seed_o, seed_r = packet["origin_world_seed_m"], packet["rotation_world_seed"]
    raw_p, seed_p = raw_o + raw_r @ geometry.centre, seed_o + seed_r @ geometry.centre
    timeline = np.arange(len(pts))
    raw_uv = camera.project(raw_p/ratio, timeline)
    seed_uv = camera.project(seed_p/ratio, timeline)
    row = row_camera(camera, raw_uv.reshape(-1, 2), np.repeat(timeline, 2))
    dep = smooth_depth(world_depth(row, raw_p), world_depth(row, seed_p), shots)
    rows, counts = traced_rows(document, pts, geometry)
    hints = source_hints(args.source_hints, pts)
    source = dict(width=int(SIZE[0]), height=int(SIZE[1]), time_base_num=tb[0], time_base_den=tb[1])
    matrices = camera_matrices(camera)
    P = np.empty((len(pts), 2, BRANCHES, 3))
    Q = np.empty((len(pts), 2, BRANCHES, 3, 3))
    E = np.empty((len(pts), 2, BRANCHES))
    qual = np.empty((len(pts), 2, BRANCHES, 5))
    uv = np.empty((len(pts), 2, BRANCHES, 2))
    emit("start_link", exposures=len(pts), fit_traces=counts["fit"], check_traces=counts["check"],
         branches=BRANCHES, workers=args.workers, device="CPU", input=str(args.candidate))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future = {}
        for shot in dict.fromkeys(shots.tolist()):
            ix = np.flatnonzero(shots == shot)
            start = int(ix[0])
            select = slice(start, int(ix[-1])+1)
            row_select = {k:v[2*start:2*(int(ix[-1])+1)] for k,v in row.items()}
            for a in range(0, len(ix), 360):
                b = min(a+360,len(ix))
                traces = {(t,f,"fit"): rows[t,f,"fit"] for t in pts[ix[a:b]]
                          for f in range(2) if rows[t,f,"fit"]}
                job = pool.submit(gather_span, media, source, pts[select], matrices[select], shot, a, b,
                                  geometry, dep[select], raw_uv[select], seed_uv[select], raw_r[select],
                                  seed_r[select], row_select, hints[select], traces, ice)
                future[job] = slice(start+a,start+b)
        for job in as_completed(future):
            select = future[job]
            P[select], Q[select], qual[select], uv[select], E[select] = job.result()
    for array in (P, Q, E):
        if not np.isfinite(array).all():
            raise ValueError("Invalid shoe proposal")
    choice = np.empty((len(pts),2), dtype=np.intp)
    for shot in dict.fromkeys(shots.tolist()):
        ix = np.flatnonzero(shots == shot)
        for side in range(2):
            choice[ix, side] = choose_path(P[ix,side],Q[ix,side],E[ix,side],geometry.centre)
    idx = (timeline[:,None], np.arange(2)[None,:], choice)
    origin, rotation = P[idx], Q[idx]
    scores = qual[idx]
    joined_uv = uv[idx]
    overlap = np.linalg.norm(joined_uv[:,0]-joined_uv[:,1],axis=-1) < .36*geometry.length*(
        row["fx"].reshape(-1,2)/dep).mean(axis=-1)
    weak = (scores[...,2] < .88) | (scores[...,0] < .20) | (scores[...,1] < .30) | overlap[:,None]
    report = motion_report(origin,rotation,shots)
    trace = evaluate_traces(camera, geometry, rows, pts, timeline, ratio, origin, rotation, seed_o, seed_r)
    file = args.output/"shoes.npz"
    packet.update(origin_world_m=origin, rotation_world=rotation, source_support_score=scores,
                  inferred=weak, identity_overlap=overlap, selected_branch=choice,
                  linked_image_cost=E[idx], conditional_centre_depth_m=dep,
                  framewise_candidate_path=np.array(str(args.candidate)))
    with file.open("xb") as output:
        np.savez_compressed(output,**packet)
    with (args.output/"trace_check.json").open("x",encoding="utf-8") as output:
        json.dump(trace,output,ensure_ascii=False,indent=2)
    result = dict(framewise=motion_report(raw_o,raw_r,shots),linked=report,
                  candidate=str(file), exposures=len(pts), weak_shoe_exposures=int(weak.sum()),
                  trace={role: {name: dict(count=len(use), seed_p50_px=float(np.median([r["seed_px"] for r in use])),
                                          candidate_p50_px=float(np.median([r["candidate_px"] for r in use])))
                                for name, use in (("all", entries),("steel", [r for r in entries if r["semantic"].startswith("runner_")]))
                                if use} for role, entries in trace.items()},
                  seconds=round(time.perf_counter()-begin,2))
    with (args.output/"linked_receipt.json").open("x",encoding="utf-8") as output:
        json.dump(result,output,indent=2)
    emit("linked", **result)


if __name__ == "__main__":
    main()
