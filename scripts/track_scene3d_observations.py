"""Track image-measured scene curves across complete source-PTS shot intervals.

Camera-independent image tracking only. Frame homographies are search proposals;
exported curve pixels must be observed local intensity transitions. They are not
model projections and tracked frames are not independent validation evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rinkspace.storage import AssetStore

OUT = AssetStore(ROOT).directory(
    'outputs/calibration/ankara_2026_v015_joint/tracking',
    'calibration/ankara_2026_v015_joint/tracking',
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def image_at(path, pts):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(pts, stream=s, backward=True)
        for f in c.decode(s):
            if f.pts == pts:
                return f.to_ndarray(format="bgr24")
            if f.pts is not None and f.pts > pts:
                break
    raise ValueError(f"PTS not decoded: {pts}")


def decode_shot(source, shot, seed_pts):
    """Decode every native frame; retain grayscale at <=30 fps for motion tracking."""
    arrays, pts, all_pts = [], [], []
    with av.open(str(source)) as c:
        s = c.streams.video[0]
        c.seek(shot["start_pts"], stream=s, backward=True)
        for f in c.decode(s):
            if f.pts is None or f.pts < shot["start_pts"]:
                continue
            if f.pts >= shot["end_pts"]:
                break
            all_pts.append(f.pts)
            if len(all_pts) % 2 == 1 or f.pts == seed_pts:
                # Luma is measured from decoded source; no image enhancement.
                gray = f.to_ndarray(format="gray")
                arrays.append(cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA))
                pts.append(f.pts)
    return arrays, np.array(pts), all_pts


def sample(gray, xy):
    xy = np.asarray(xy, np.float32)
    return cv2.remap(gray.astype(np.float32), xy[..., 0], xy[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def normals(points, target):
    p = np.asarray(points, np.float32)
    if target["kind"] == "circle_outline" and len(p) >= 5:
        ellipse = cv2.fitEllipse(p)
        center, axes, angle = ellipse
        a = np.deg2rad(angle)
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        local = (p - center) @ r
        n = (local / np.maximum(np.array(axes) / 2, 1) ** 2) @ r.T
    else:
        d = np.gradient(p, axis=0)
        n = np.column_stack([-d[:, 1], d[:, 0]])
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)).astype(np.float32)


def edge_resnap(gray, points, norm, polarity, strength):
    # 3 native pixel radius. Contrast 4 native pixels across each edge.
    offsets = np.linspace(-1.5, 1.5, 13, dtype=np.float32)
    loc = points[:, None, :] + norm[:, None, :] * offsets[None, :, None]
    a = sample(gray, loc + norm[:, None, :])
    b = sample(gray, loc - norm[:, None, :])
    score = (a - b) * polarity[:, None]
    # Prefer the nearby genuine edge over an unrelated stronger parallel line.
    score -= np.abs(offsets)[None, :] * 1.25
    ix = np.argmax(score, axis=1)
    displacement = offsets[ix]
    raw_strength = ((a - b) * polarity[:, None])[np.arange(len(points)), ix]
    p = points + norm * displacement[:, None]
    inside = (p[:, 0] > 4) & (p[:, 0] < 955) & (p[:, 1] > 4) & (p[:, 1] < 535)
    ok = inside & (raw_strength >= np.maximum(3.5, strength * .22))
    # Limit to a local search: a maximum at the search boundary is not measured.
    ok &= (ix > 0) & (ix < len(offsets) - 1)
    return p, ok, raw_strength


def motion(prev, curr):
    mask = np.full(prev.shape, 255, np.uint8)
    mask[:95] = 0  # spectator heads, not stable rink structure
    mask[490:] = 0
    mask[:140, :330] = 0  # broadcast scoreboard
    p = cv2.goodFeaturesToTrack(prev, 650, .008, 8, mask=mask, blockSize=5)
    if p is None or len(p) < 12:
        return None, {"status": "too_few_features", "inliers": 0}
    kw = dict(winSize=(25, 25), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, .02))
    q, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, p, None, **kw)
    back, sb, _ = cv2.calcOpticalFlowPyrLK(curr, prev, q, None, **kw)
    fb = np.linalg.norm(back[:, 0] - p[:, 0], axis=1)
    good = (st[:, 0] == 1) & (sb[:, 0] == 1) & (fb < .7) & (err[:, 0] < 30)
    if good.sum() < 12:
        return None, {"status": "low_fb_support", "inliers": int(good.sum())}
    h, inliers = cv2.findHomography(p[good], q[good], cv2.RANSAC, 1.25, maxIters=700)
    if h is None:
        return None, {"status": "homography_failed", "inliers": 0}
    n = int(inliers.sum())
    ratio = n / good.sum()
    good_h = n >= 15 and ratio >= .45 and .8 < np.linalg.det(h[:2, :2]) < 1.25
    return (h if good_h else None), {
        "status": "image_motion_search_proposal" if good_h else "inconsistent_motion",
        "inliers": n, "inlier_fraction": float(ratio),
        "fb_median_native_px": float(np.median(fb[good]) * 2),
        "geometry_assumption": "single image homography conditional on static camera centre; not a ground-plane fit",
    }


def warp(points, h):
    return cv2.perspectiveTransform(np.asarray(points, np.float32)[:, None, :], h)[:, 0]


def moving_occlusion_reject(prev, curr, points, proposal):
    """Reject locally trackable pixels moving differently from the static scene.

    Smooth line pixels may have an aperture ambiguity, so absence of a reliable
    local LK solution is not itself a foreground segmentation decision.
    """
    p = np.asarray(points, np.float32)[:, None, :]
    initial = np.asarray(proposal, np.float32)[:, None, :].copy()
    kw = dict(winSize=(19, 19), maxLevel=2,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02))
    q, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, p, initial,
                                       flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    b, sb, _ = cv2.calcOpticalFlowPyrLK(curr, prev, q, None, **kw)
    deviation = np.linalg.norm(q[:, 0] - proposal, axis=1)
    fb = np.linalg.norm(b[:, 0] - points, axis=1)
    locally_trackable = (st[:, 0] == 1) & (sb[:, 0] == 1)
    reject = locally_trackable & ((deviation > 2.0) | (fb > 1.2) | (err[:, 0] > 35))
    return reject


def initial_state(gray, obs):
    states = []
    for o in obs:
        # Preserve spatial support but limit redundant dense samples.
        p = np.array(o["pixels_uv"], np.float32)[::max(1, len(o["pixels_uv"]) // 110)] / 2
        n = normals(p, o["target"])
        contrast = (sample(gray, (p + n)[:, None]) - sample(gray, (p - n)[:, None]))[:, 0]
        states.append({"points": p, "normals": n, "polarity": np.where(contrast >= 0, 1, -1),
                       "strength": np.abs(contrast), "missing": np.zeros(len(p), int),
                       "valid": np.ones(len(p), bool), "seed_observation": o})
    return states


def track_direction(grays, pts, seed_index, obs, direction):
    states = initial_state(grays[seed_index], obs)
    rows, curves = {}, {}
    rows[int(pts[seed_index])] = {"source_pts": int(pts[seed_index]), "seed": True}
    curves[int(pts[seed_index])] = copy.deepcopy(states)
    indices = range(seed_index + direction, len(grays) if direction > 0 else -1, direction)
    for ix in indices:
        h, quality = motion(grays[ix - direction], grays[ix])
        counts = []
        if h is None:
            # No invented identity after an untracked transition. Continue decode audit.
            for s in states:
                s["valid"][:] = False
                s["missing"] += 1
        else:
            for s in states:
                proposal = warp(s["points"], h)
                tip = warp(s["points"] + s["normals"], h)
                n = tip - proposal
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-5)
                measured, valid, strength = edge_resnap(grays[ix], proposal, n, s["polarity"], s["strength"])
                reject = moving_occlusion_reject(grays[ix - direction], grays[ix], s["points"], proposal)
                valid &= ~reject
                s["local_motion_or_occlusion_rejected"] = int(reject.sum())
                # Identity can bridge brief occlusion only, not arbitrary disappearance.
                valid &= s["missing"] < 15
                s["points"] = np.where(valid[:, None], measured, proposal)
                s["normals"] = n
                s["missing"] = np.where(valid, 0, s["missing"] + 1)
                s["valid"] = valid
        for s in states:
            counts.append({"observation_id": s["seed_observation"]["id"],
                           "measured_points": int(s["valid"].sum()),
                           "seed_points": len(s["points"]),
                           "local_motion_or_occlusion_rejected": s.get("local_motion_or_occlusion_rejected", 0),
                           "out_of_view": int(((s["points"][:, 0] < 0) | (s["points"][:, 0] >= 960) |
                                                (s["points"][:, 1] < 0) | (s["points"][:, 1] >= 540)).sum())})
        rows[int(pts[ix])] = {"source_pts": int(pts[ix]), "seconds": float(pts[ix] / 15360),
                              "motion": quality, "curves": counts}
        # Store approximately 1 fps plus first/last for QA; every frame still tracked.
        if ix % 30 == 0 or ix in (0, len(grays) - 1):
            curves[int(pts[ix])] = copy.deepcopy(states)
        if ix % 300 == 0:
            print(f"track pts={pts[ix]} direction={direction} measured={sum(x['measured_points'] for x in counts)}", flush=True)
    return rows, curves


COLORS = [(0, 230, 255), (255, 100, 40), (190, 80, 255), (40, 230, 80), (220, 200, 60)]


def render(frame, states, title):
    img = frame.copy()
    for i, s in enumerate(states):
        color = COLORS[i % len(COLORS)]
        for p in s["points"][s["valid"]] * 2:
            cv2.circle(img, tuple(np.rint(p).astype(int)), 3, color, -1, cv2.LINE_AA)
        name = s["seed_observation"].get("feature_id", s["seed_observation"]["id"].split("boards_")[-1])
        text = f"{name}: {int(s['valid'].sum())}/{len(s['valid'])}"
        cv2.putText(img, text, (20, 85 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, .85, color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (1920, 47), (15, 15, 15), -1)
    cv2.putText(img, title, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def process(seed_frame, obs, shot, out, hash_verified):
    shot_dir = out / shot["id"]
    shot_dir.mkdir(parents=True, exist_ok=True)
    grays, pts, all_pts = decode_shot(seed_frame["source_path"], shot, seed_frame["source_pts"])
    print(f"{shot['id']}: decoded={len(all_pts)}, tracking={len(pts)}", flush=True)
    seed_index = int(np.where(pts == seed_frame["source_pts"])[0][0])
    rows1, curves1 = track_direction(grays, pts, seed_index, obs, 1)
    rows2, curves2 = track_direction(grays, pts, seed_index, obs, -1)
    rows, curves = rows1 | rows2, curves1 | curves2
    frames_out, obs_out, panels = [], [], []
    chosen_qa = [min(curves), min(curves, key=lambda p: abs(p - (shot['start_pts'] + shot['end_pts']) / 2)), max(curves)]
    for pts_i in sorted(curves):
        states = curves[pts_i]
        # Native decoded pixels saved only for selected measured frames / QA.
        native = image_at(seed_frame["source_path"], pts_i)
        frame_id = f"track_{shot['id']}_{pts_i}"
        path = shot_dir / f"{frame_id}.jpg"
        cv2.imwrite(str(path), native, [cv2.IMWRITE_JPEG_QUALITY, 97])
        f = copy.deepcopy(seed_frame)
        for inherited in ("decoded_rgb_identical_to_existing_png", "decoded_rgb_sha256", "derivation"):
            f.pop(inherited, None)
        f.update(id=frame_id, source_pts=pts_i, source_seconds=pts_i / 15360,
                 image_path=str(path.resolve()), image_sha256=sha(path),
                 source_hash_verified_this_run=hash_verified,
                 image_derivation="native-size source decode saved JPEG quality97; observations measured on decoded grayscale, not JPEG",
                 pixel_frame_id=f"{f['media_track_id']}:pts:{pts_i}:1920x1080",
                 decoded_bgr_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                 derivation="Native source PTS decoded; original dimensions preserved; JPEG is for visual review. No prior seed RGB/hash assertion is inherited.",
                 principal_point_px=[959.5, 539.5], installation_id="women_sp_main")
        frames_out.append(f)
        for s in states:
            valid = s["valid"]
            o0 = s["seed_observation"]
            # At least 8 measured pixels and a 30-native-pixel spatial span.
            if valid.sum() < 8 or np.linalg.norm(np.ptp(s["points"][valid], axis=0)) < 15:
                continue
            o = {k: copy.deepcopy(o0[k]) for k in ["target", "feature_id", "physical_group_id", "type", "surface_class", "identity_status", "xyz_status"] if k in o0}
            o.update(id=f"{frame_id}_{o0['id']}", frame_id=frame_id, use="fit",
                     historical_seed_use=o0.get("use"), source_seed_observation_id=o0["id"],
                     source_seed_pts=seed_frame["source_pts"], pixels_uv=np.round(s["points"][valid] * 2, 4).tolist(),
                     sigma_px=max(4.0, float(o0.get("sigma_px", 3))),
                     method="image-only LK forward/backward background-motion proposal plus signed local intensity-edge remeasurement; local LK disagreement rejects possible moving occlusion",
                     measurement_origin="direct_local_edge_remeasurement_with_temporally_transferred_identity",
                     pure_transfer_points_exported=False,
                     independence="same complete-shot temporal chain; not independent validation; do not count each pixel/frame as independent evidence",
                     seed_point_indices=np.where(valid)[0].tolist(),
                     source_frame={"source_id": f["source_id"], "source_pts": pts_i, "time_base": f["time_base"], "image_path": str(path.resolve())})
            obs_out.append(o)
        if pts_i in chosen_qa:
            panel = render(native, states, f"{shot['id']} source {pts_i / 15360:.3f}s PTS {pts_i}; measured curve pixels")
            cv2.imwrite(str(shot_dir / f"qa_{pts_i}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
            panels.append(cv2.resize(panel, (960, 540)))
    contact = np.concatenate(panels, axis=0)
    contact_path = shot_dir / "start_middle_end.jpg"
    cv2.imwrite(str(contact_path), contact, [cv2.IMWRITE_JPEG_QUALITY, 92])
    summaries = []
    for o in obs:
        counts = [(p, next((x["measured_points"] for x in r.get("curves", []) if x["observation_id"] == o["id"]), 0)) for p, r in sorted(rows.items())]
        good = [p for p, n in counts if n >= 8]
        summaries.append({"seed_observation_id": o["id"], "surface_class": o.get("surface_class"),
                          "trackable_frame_fraction": len(good) / len(rows),
                          "first_measured_pts": min(good) if good else None, "last_measured_pts": max(good) if good else None,
                          "coverage_meaning": "fraction of processed 30fps samples with >=8 locally measured edge pixels; not whole object visibility"})
    summary = {"shot": shot, "seed_frame": seed_frame["id"], "source_sha256": seed_frame["source_sha256"],
               "decoded_frame_count": len(all_pts), "decoded_first_pts": all_pts[0], "decoded_last_pts": all_pts[-1],
               "all_native_frames_decoded": True, "tracking_rate_hz": 30, "tracking_processed_frames": len(rows),
               "exported_frames": len(frames_out), "exported_observations": len(obs_out),
               "edge_pixel_sigma_native_px": 4, "sampling": "all frames decoded, alternate frames tracked; about 1Hz complementary measurements exported",
               "limitations": ["LK homography is image-only proposal under a conditional fixed camera centre assumption", "brief disappearance may recover within 0.5s; longer disappearance is marked lost", "same-shot samples are correlated, not independent evidence"],
               "observations": summaries, "qa_contact": str(contact_path.resolve())}
    dump(shot_dir / "source_pts_decode_audit.json", {"decoded_native_pts": all_pts, "processed_pts": pts.tolist()})
    dump(shot_dir / "per_frame_tracking.json", [rows[p] for p in sorted(rows)])
    dump(shot_dir / "summary.json", summary)
    dump(shot_dir / "observations.json", {"frames": frames_out, "observations": obs_out, "summary": summary})
    return frames_out, obs_out, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shots", nargs="*", default=["S008", "S025", "S027", "S032"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify-source-hash", action="store_true")
    p.add_argument("--publish-reviewed", action="store_true")
    a = p.parse_args()
    cv2.setNumThreads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    old = read(ROOT / "outputs/calibration/ankara_2026_v014_main/real_observations.json")
    extra = ROOT / "outputs/calibration/ankara_2026_v015_joint/additional_seeds.json"
    if extra.exists():
        more = read(extra)
        old["frames"] += more["frames"]
        old["observations"] += more["observations"]
    trial = read(ROOT / "outputs/camera_classification/trial_010800_013600/result.json")
    shots = {s["id"]: s for s in trial["intervals"]}
    assert not set(a.shots) & {"S085", "S087", "S089"}, "Held-out clips cannot enter training tracker"
    verified = False
    if a.verify_source_hash:
        f = old["frames"][0]
        verified = sha(f["source_path"]) == f["source_sha256"]
        if not verified:
            raise ValueError("source SHA256 mismatch")
    all_f, all_o, reports = [], [], []
    for f in old["frames"]:
        sid = f["shot_id"]
        if sid not in a.shots:
            continue
        saved = OUT / sid / "observations.json"
        if saved.exists() and not a.overwrite:
            data = read(saved)
            ff, oo, report = data["frames"], data["observations"], data["summary"]
        else:
            obs = [o for o in old["observations"] if o["frame_id"] == f["id"]]
            ff, oo, report = process(f, obs, shots[sid], OUT, verified)
        all_f += ff
        all_o += oo
        reports.append(report)
        dump(OUT / "observations.json", {"frames": all_f, "observations": all_o, "shots": reports,
             "coordinate_space": "native 1920x1080 source pixels", "training_only": True})
        used = {o["frame_id"] for o in all_o}
        dump(OUT / "observations_fit_ready.json", {"frames": [f for f in all_f if f["id"] in used],
             "observations": all_o, "coordinate_space": "native 1920x1080 source pixels", "training_only": True,
             "source_hash_verified_this_run": verified,
             "limitations": "Same-shot frames are correlated. Export excludes pure homography transfer. Foreground rejection is partial; visual review and robust curve residuals remain necessary."})
        print(f"DONE {sid}: frames={len(ff)} observations={len(oo)}", flush=True)
    if a.publish_reviewed:
        # Publishing is an explicit final action so a joint solver never reads
        # an input being progressively overwritten by the tracking loop.
        ready = read(OUT / "observations_fit_ready.json")
        for f in ready["frames"]:
            for inherited in ("decoded_rgb_identical_to_existing_png", "decoded_rgb_sha256"):
                f.pop(inherited, None)
            f["derivation"] = "Native source PTS decoded; original dimensions preserved; JPEG is for visual review. No prior seed RGB/hash assertion is inherited."
        ready["review"] = {"version": "local_motion_occlusion_rejection_v2", "source_sha256_recomputed": verified,
            "complete_shots_decoded": [s["shot"]["id"] for s in reports],
            "human_style_visual_inspection": "assistant viewed native-derived first/middle/end overlays for all 4 shots; S032 viewed again after motion-occlusion rejection",
            "limitations": "Automatic occlusion rejection is partial. Robust residuals and object-level weighting required; curve pixels and temporal samples are correlated."}
        assert all(len(o["pixels_uv"]) >= 8 and not o["pure_transfer_points_exported"] for o in ready["observations"])
        ids = {f["id"] for f in ready["frames"]}
        assert all(o["frame_id"] in ids for o in ready["observations"])
        assert not {"S085", "S087", "S089"} & {f["shot_id"] for f in ready["frames"]}
        dump(OUT / "observations_reviewed.json", ready)
        print(f"PUBLISHED {OUT / 'observations_reviewed.json'}", flush=True)


if __name__ == "__main__":
    main()
