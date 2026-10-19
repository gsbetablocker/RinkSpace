"""Local adjacent-frame model diagnosis; never reads manual endpoint checks.

For three difficult backward edges, compare three target-camera models with
the same measured ice/board correspondences. Also compare a jointly fitted
shared optical-centre lever arm against d=0 after re-attaching source rays to
their physical surfaces for each candidate d. A deterministic track subset is
held out from each local fit; this is a model diagnosis, not independent
ice-placement validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from camera import ROOT, anchor_camera, attach_to_surface, fast_project, from_state, read_json

EDGES = (40, 30, 20)


def metrics(errors: np.ndarray) -> dict:
    e = np.asarray(errors, float)
    return {"count": len(e),
            "median_px": float(np.median(e)) if len(e) else None,
            "rms_px": float(np.sqrt(np.mean(e * e))) if len(e) else None,
            "p95_px": float(np.percentile(e, 95)) if len(e) else None,
            "max_px": float(np.max(e)) if len(e) else None}


def select_edge(z: np.lib.npyio.NpzFile, poses: dict, scene: dict, t: int) -> dict:
    pick = np.flatnonzero((z["from_index"] == t + 1) & (z["to_index"] == t))
    pick = pick[z["fb_error_native"][pick] <= 0.5]
    xy_from = np.asarray(z["xy_from"][pick], float)
    xy_to = np.asarray(z["xy_to"][pick], float)
    surface = np.asarray(z["surface"][pick]).astype(str)
    from_camera = np.asarray(poses["frames"][t + 1]["camera_vector"], float)
    xyz, good = attach_to_surface(scene, from_camera, xy_from, surface)
    pick, xy_from, xy_to, surface, xyz = (
        arr[good] for arr in (pick, xy_from, xy_to, surface, xyz))
    if len(xy_to) < 40 or len(set(surface)) < 2:
        raise ValueError(f"Too little two-depth evidence at edge {t + 1}->{t}")
    # Full temporal tracks never enter manual endpoint checks. This holdout is
    # only for comparing local model capacity, not for accuracy claims.
    test = (pick % 5) == 0
    train = ~test
    if min(train.sum(), test.sum()) < 12:
        raise ValueError("Insufficient deterministic local model holdout")
    return {"t": t, "pick": pick, "xy_from": xy_from, "xy_to": xy_to,
            "surface": surface, "xyz_zero": xyz, "train": train, "test": test,
            "base_to": np.asarray(poses["frames"][t]["camera_vector"], float),
            "source_state": np.asarray(poses["frames"][t + 1]["pan_tilt_logzoom"], float),
            "target_state": np.asarray(poses["frames"][t]["pan_tilt_logzoom"], float)}


def balanced_weight(labels: np.ndarray) -> np.ndarray:
    w = np.ones(len(labels))
    for label in ("ice", "board"):
        mask = labels == label
        if mask.any():
            w[mask] = np.sqrt(len(labels) / (2 * mask.sum()))
    return w


def free_camera(base: np.ndarray, delta: np.ndarray, translate: bool) -> np.ndarray:
    v = base.copy()
    if translate:
        v[:3] += delta[:3]
        rot = delta[3:6]
        zoom = delta[6]
    else:
        rot = delta[:3]
        zoom = delta[3]
    r0 = cv2.Rodrigues(base[3:6])[0]
    v[3:6] = cv2.Rodrigues(cv2.Rodrigues(rot)[0] @ r0)[0].ravel()
    v[6:8] += zoom
    v[10] *= np.exp(zoom)
    return v


def fit_single(edge: dict, v0: np.ndarray, kind: str) -> dict:
    train = edge["train"]
    xyz, target = edge["xyz_zero"], edge["xy_to"]
    w = balanced_weight(edge["surface"][train])
    if kind == "ptz_fixed_centre":
        x0 = edge["target_state"]
        lo = x0 + [-0.04, -0.04, -0.05]
        hi = x0 + [0.04, 0.04, 0.05]
        make = lambda x: from_state(v0, x)
    elif kind == "free_rotation_zoom_fixed_centre":
        x0 = np.zeros(4)
        lo = [-0.05, -0.05, -0.05, -0.06]
        hi = [0.05, 0.05, 0.05, 0.06]
        make = lambda x: free_camera(edge["base_to"], x, False)
    elif kind == "free_translation_rotation_zoom":
        x0 = np.zeros(7)
        lo = [-2., -2., -2., -0.05, -0.05, -0.05, -0.06]
        hi = [2., 2., 2., 0.05, 0.05, 0.05, 0.06]
        make = lambda x: free_camera(edge["base_to"], x, True)
    else:
        raise ValueError(kind)
    def residual(x: np.ndarray) -> np.ndarray:
        p = fast_project(make(x), xyz[train])
        return ((p - target[train]) * w[:, None]).ravel()
    result = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1",
                           f_scale=2., max_nfev=60,
                           xtol=1e-8, ftol=1e-8, gtol=1e-8)
    predicted = fast_project(make(result.x), xyz)
    errors = np.linalg.norm(predicted - target, axis=1)
    return {"kind": kind, "parameters": result.x.tolist(),
            "optimizer_success": bool(result.success), "nfev": result.nfev,
            "fit": metrics(errors[train]), "heldout": metrics(errors[~train]),
            "heldout_ice": metrics(errors[(~train) & (edge["surface"] == "ice")]),
            "heldout_board": metrics(errors[(~train) & (edge["surface"] == "board")])}


def fit_image_homography(edge: dict) -> dict:
    """Image-only capacity reference, not a physical camera or rink fit."""
    train, test = edge["train"], edge["test"]
    matrix, inliers = cv2.findHomography(
        edge["xy_from"][train], edge["xy_to"][train], cv2.RANSAC, 2.0)
    if matrix is None:
        raise ValueError("Image homography could not be estimated")
    predicted = cv2.perspectiveTransform(
        edge["xy_from"].astype(np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - edge["xy_to"], axis=1)
    return {"kind": "unconstrained_image_homography_diagnostic_only",
            "train_inliers": int(inliers.sum()), "train_count": int(train.sum()),
            "heldout": metrics(errors[test]),
            "heldout_ice": metrics(errors[test & (edge["surface"] == "ice")]),
            "heldout_board": metrics(errors[test & (edge["surface"] == "board")])}


def small_balanced_indices(edge: dict) -> np.ndarray:
    selected = []
    for label in ("ice", "board"):
        idx = np.flatnonzero(edge["surface"] == label)
        if len(idx) > 65:
            idx = idx[np.linspace(0, len(idx) - 1, 65, dtype=int)]
        selected.extend(idx.tolist())
    return np.asarray(sorted(set(selected)), int)


def fit_shared_pivot(edges: list[dict], scene: dict, v0: np.ndarray,
                     *, release_d: bool) -> dict:
    subset = [small_balanced_indices(e) for e in edges]
    chosen = []
    for e, idx in zip(edges, subset):
        chosen.append({key: e[key][idx] for key in ("xy_from", "xy_to", "surface", "train", "test")}
                      | {"source_state": e["source_state"], "target_state": e["target_state"]})
    base_states = np.concatenate([e["target_state"] for e in chosen])
    if release_d:
        x0 = np.r_[np.zeros(3), np.zeros(9)]
        lo = np.r_[[-5., -5., -5.], np.tile([-.04, -.04, -.06], 3)]
        hi = np.r_[[5., 5., 5.], np.tile([.04, .04, .06], 3)]
    else:
        x0 = np.zeros(9)
        lo = np.tile([-.04, -.04, -.06], 3)
        hi = np.tile([.04, .04, .06], 3)

    def evaluate(x: np.ndarray, training: bool) -> tuple[np.ndarray, list[dict]]:
        d = x[:3] if release_d else np.zeros(3)
        delta = x[3:] if release_d else x
        chunks = []
        rows = []
        for j, e in enumerate(chosen):
            camera_from = from_state(v0, e["source_state"], d)
            camera_to = from_state(v0, e["target_state"] + delta[3*j:3*j+3], d)
            mask = e["train"] if training else e["test"]
            xyz, good = attach_to_surface(scene, camera_from,
                                           e["xy_from"][mask], e["surface"][mask])
            pred = fast_project(camera_to, np.nan_to_num(xyz, nan=0.))
            err_xy = pred - e["xy_to"][mask]
            err_xy[~good] = 80.
            if training:
                err_xy *= balanced_weight(e["surface"][mask])[:, None]
            chunks.append(err_xy.ravel())
            err = np.linalg.norm(err_xy, axis=1)
            rows.append({"edge": f"{EDGES[j]+1}->{EDGES[j]}",
                         "metrics": metrics(err), "attached": int(good.sum()),
                         "attempted": len(good)})
        return np.concatenate(chunks), rows

    opt = least_squares(lambda x: evaluate(x, True)[0], x0,
                        bounds=(lo, hi), loss="soft_l1", f_scale=2.,
                        max_nfev=40, xtol=1e-7, ftol=1e-7, gtol=1e-7)
    train_err, train_rows = evaluate(opt.x, True)
    test_err, test_rows = evaluate(opt.x, False)
    return {"offset_released": release_d, "offset_camera_xyz":
            opt.x[:3].tolist() if release_d else [0., 0., 0.],
            "parameters": opt.x.tolist(), "optimizer_success": bool(opt.success),
            "nfev": int(opt.nfev), "fit_weighted_component_rms_px":
            float(np.sqrt(np.mean(train_err**2))),
            "heldout_component_rms_px": float(np.sqrt(np.mean(test_err**2))),
            "heldout_by_edge": test_rows, "fit_by_edge": train_rows,
            "d_near_bound": bool(release_d and np.any(np.abs(opt.x[:3]) >= 4.9))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "calibration" / "shot_tracking" / "runs" / "low_end_pm1")
    p.add_argument("--scene", type=Path, default=ROOT / "calibration" / "deliverables" / "scene.json")
    args = p.parse_args()
    poses_path = args.run / "poses.json"
    pair_path = args.run / "pair_tracks.npz"
    scene = read_json(args.scene)
    poses = read_json(poses_path)
    pairs = np.load(pair_path, allow_pickle=False)
    v0 = anchor_camera(scene, int(poses["anchor_pts"]))
    edges = [select_edge(pairs, poses, scene, t) for t in EDGES]
    single = []
    for edge in edges:
        print(f"diagnosing edge {edge['t']+1}->{edge['t']}", flush=True)
        variants = [fit_single(edge, v0, name) for name in (
            "ptz_fixed_centre", "free_rotation_zoom_fixed_centre",
            "free_translation_rotation_zoom")]
        single.append({"edge": f"{edge['t']+1}->{edge['t']}",
                       "surface_attached": len(edge["xy_to"]),
                       "variants": variants,
                       "image_homography_capacity_reference": fit_image_homography(edge)})
    print("diagnosing shared lever arm", flush=True)
    shared0 = fit_shared_pivot(edges, scene, v0, release_d=False)
    shared1 = fit_shared_pivot(edges, scene, v0, release_d=True)
    report = {
        "schema_version": "rinkspace.shot_tracking.pivot_diagnostic.v1",
        "scope": "three original adjacent-frame edges, t=40/30/20, prev=t+1 to t",
        "source_pair_tracks_sha256": hashlib.sha256(pair_path.read_bytes()).hexdigest(),
        "input_pose_sha256": hashlib.sha256(poses_path.read_bytes()).hexdigest(),
        "source_scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "manual_endpoint_readings_used": False,
        "disclaimer": "Local tracked-pair model capacity only; no full-shot or independent ice fit claim.",
        "single_edges": single,
        "joint_shared_pivot": {"zero_offset": shared0, "released_offset": shared1},
    }
    out = args.run / "pivot_diagnostic.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out),
                      "joint_heldout_zero": shared0["heldout_component_rms_px"],
                      "joint_heldout_released": shared1["heldout_component_rms_px"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
