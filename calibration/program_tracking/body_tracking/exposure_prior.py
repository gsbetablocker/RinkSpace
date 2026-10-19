"""Share cached body priors within confirmed official source exposures.

This CPU-only repair preserves presentation PTS and the original prior file.
It reads the actual representative's cached body prior, including its existing
interpolation bracket, rather than assigning aliases new posture times.  It
does not change raw H observations, camera poses, world translations or any
values across different exposures.  Missing representatives stay unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .camera_bridge import FrozenCameraStream
except ImportError:
    from camera_bridge import FrozenCameraStream


SEGMENT_START_PTS = 230031360
SEGMENT_END_PTS_EXCLUSIVE = 230261760
PTS_STEP = 256
BODY_ARRAY_SHAPES = {
    "coco17_world_offset_m": (17, 3),
    "native22_world_offset_m": (22, 3),
    "coco17_body_offset_m": (17, 3),
    "native22_body_offset_m": (22, 3),
    "body_world_R": (3, 3),
    "smplx_native_pelvis_at_zero_transl_m": (3,),
}
SHARED_FRAME_FIELDS = ("prior_valid", "prior_origin", "model_slot_pts",
                       "interpolation_left_source_pts",
                       "interpolation_right_source_pts", "interpolation_alpha",
                       *BODY_ARRAY_SHAPES)
IDENTITY_FIELDS = ("pts", "shot_index", "official_exposure_representative_pts")
GLOBAL_FIELDS = ("shared_betas", "coco17_names", "native22_names")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _exact_indices(available: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    available = np.asarray(available)
    requested = np.asarray(requested)
    if (available.ndim != 1 or not np.issubdtype(available.dtype, np.integer)
            or np.any(np.diff(available) <= 0)):
        raise ValueError(f"{label} input PTS must be strictly increasing integers")
    if not np.issubdtype(requested.dtype, np.integer):
        raise ValueError(f"{label} requested PTS must be exact integers")
    indices = np.searchsorted(available, requested)
    if (np.any(indices >= len(available)) or
            not np.array_equal(available[np.minimum(indices, len(available)-1)], requested)):
        raise ValueError(f"{label} does not contain every exact requested source PTS")
    return indices


def _distribution(values: list[float]) -> dict:
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    return dict(count=int(len(a)), median=float(np.median(a)) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


def same_exposure_differences(prior: dict) -> dict:
    """Adjacent equal-exposure pairs only; missing pairs remain counted."""
    pairs = np.flatnonzero(
        prior["official_exposure_representative_pts"][:-1] ==
        prior["official_exposure_representative_pts"][1:])
    valid = np.asarray(prior["prior_valid"], bool)
    finite_pairs = pairs[valid[pairs] & valid[pairs+1]]
    body_gap, angle, exact_same = [], [], 0
    for i in finite_pairs:
        a, b = prior["coco17_world_offset_m"][[i, i+1]]
        qa, qb = prior["body_world_R"][[i, i+1]]
        body_gap.append(float(np.linalg.norm(a.astype(float)-b, axis=1).max()))
        angle.append(float(np.degrees((Rotation.from_matrix(qa).inv()
                                        * Rotation.from_matrix(qb)).magnitude())))
        exact_same += int(all(np.array_equal(prior[k][i], prior[k][i+1], equal_nan=True)
                              for k in BODY_ARRAY_SHAPES))
    return dict(adjacent_same_exposure_pairs=int(len(pairs)),
                both_valid_pairs=int(len(finite_pairs)),
                missing_either_prior_pairs=int(len(pairs)-len(finite_pairs)),
                exact_equal_all_body_array_pairs=exact_same,
                coco17_world_max_joint_difference_model_m=_distribution(body_gap),
                body_world_rotation_geodesic_deg=_distribution(angle))


def share_exposure_prior(prior: dict, camera, start_pts: int,
                         end_pts_exclusive: int) -> tuple[dict, dict]:
    """Return only this segment, sharing each confirmed representative cache.

    Inputs cover the original programme, so a group representative just outside
    the requested segment can be read without inventing or recomputing it.
    Camera differences are reported, never edited or quietly folded together.
    """
    if not isinstance(start_pts, (int, np.integer)) or not isinstance(
            end_pts_exclusive, (int, np.integer)):
        raise ValueError("Segment limits must be exact integer source PTS")
    if end_pts_exclusive <= start_pts or (end_pts_exclusive-start_pts) % PTS_STEP:
        raise ValueError("Segment must be a positive exact 256-PTS interval")
    required = set(IDENTITY_FIELDS + SHARED_FRAME_FIELDS + GLOBAL_FIELDS)
    if not required.issubset(prior):
        raise ValueError(f"Original prior lacks {sorted(required-set(prior))}")
    pts = np.asarray(prior["pts"])
    if not np.array_equal(pts, camera.pts):
        raise ValueError("Prior and frozen camera must have exact ordered original source PTS")
    n = len(pts)
    for name in ("prior_valid", "prior_origin", "model_slot_pts",
                 "interpolation_left_source_pts", "interpolation_right_source_pts",
                 "interpolation_alpha", "shot_index", "official_exposure_representative_pts"):
        if np.asarray(prior[name]).shape != (n,):
            raise ValueError(f"Original {name} must cover every input source PTS")
    for name, shape in BODY_ARRAY_SHAPES.items():
        if np.asarray(prior[name]).shape != (n, *shape):
            raise ValueError(f"Original {name} has an invalid body-array shape")
    if (not np.array_equal(prior["shot_index"], camera.shot_index) or
            not np.array_equal(prior["official_exposure_representative_pts"],
                               camera.official_exposure_representative_pts)):
        raise ValueError("Prior source shots or confirmed exposure identities differ from camera")
    requested = np.arange(start_pts, end_pts_exclusive, PTS_STEP, dtype=np.int64)
    selected = _exact_indices(pts, requested, "Segment")
    representatives = camera.official_exposure_representative_pts[selected]
    canonical = _exact_indices(pts, representatives, "Exposure representative")
    if (not np.array_equal(camera.official_exposure_representative_pts[canonical], representatives)
            or not np.array_equal(camera.shot_index[canonical], camera.shot_index[selected])):
        raise ValueError("Confirmed representatives must point to themselves in the same source shot")

    output = {k: np.asarray(prior[k])[selected].copy() for k in IDENTITY_FIELDS}
    output.update({k: np.asarray(prior[k]).copy() for k in GLOBAL_FIELDS})
    canonical_valid = np.asarray(prior["prior_valid"], bool)[canonical].copy()
    declared_valid = canonical_valid.copy()
    for name in BODY_ARRAY_SHAPES:
        canonical_valid &= np.isfinite(np.asarray(prior[name])[canonical]).reshape(len(selected), -1).all(axis=1)
    for name in SHARED_FRAME_FIELDS:
        output["original_"+name] = np.asarray(prior[name])[selected].copy()
        output[name] = np.asarray(prior[name])[canonical].copy()
    output["prior_valid"] = canonical_valid
    output["prior_origin"] = output["prior_origin"].astype("<U64")
    alias = requested != representatives
    output["prior_origin"][alias & canonical_valid] = "confirmed_exposure_shared_cached_prior"
    output["prior_origin"][~canonical_valid] = "missing_canonical_exposure_prior"
    for name in BODY_ARRAY_SHAPES:
        output[name][~canonical_valid] = np.nan
    for name in ("model_slot_pts", "interpolation_left_source_pts", "interpolation_right_source_pts"):
        output[name][~canonical_valid] = -1
    output["interpolation_alpha"][~canonical_valid] = np.nan
    output["exposure_canonical_source_pts"] = representatives.copy()
    output["exposure_canonical_prior_origin"] = np.asarray(prior["prior_origin"])[canonical].copy()
    output["exposure_canonical_prior_valid"] = canonical_valid.copy()
    output["exposure_prior_alias_share"] = alias.copy()
    output["exposure_canonical_outside_segment"] = ((representatives < start_pts) |
                                                    (representatives >= end_pts_exclusive))
    output["source_pose_representative_pts"] = camera.source_pose_representative_pts[selected].copy()
    output["source_camera25"] = camera.camera25[selected].copy()
    camera_equal = np.all(camera.camera25[selected] == camera.camera25[canonical], axis=1)
    output["exposure_camera_equal_to_canonical"] = camera_equal
    differences = np.max(np.abs(camera.camera25[selected]-camera.camera25[canonical]), axis=1)
    anomalies = [dict(source_pts=int(requested[i]), representative_pts=int(representatives[i]),
                      max_camera_parameter_difference=float(differences[i]))
                 for i in np.flatnonzero(~camera_equal)]
    before = {k: np.asarray(prior[k])[selected] for k in IDENTITY_FIELDS+SHARED_FRAME_FIELDS}
    changed = np.zeros(len(selected), bool)
    for name in BODY_ARRAY_SHAPES:
        a, b = output["original_"+name], output[name]
        equal = (a == b) | (np.isnan(a) & np.isnan(b))
        changed |= ~equal.reshape(len(selected), -1).all(axis=1)
    changed_network_slots = ((output["original_prior_origin"] == "network_model_slot") & changed)
    if changed_network_slots.any():
        raise ValueError("Canonical sharing would alter an existing real network-model slot")
    audit = dict(source_pts_start=int(start_pts), source_pts_end_exclusive=int(end_pts_exclusive),
                 frame_count=int(len(requested)), distinct_exposure_count=int(len(np.unique(representatives))),
                 alias_presentations=int(alias.sum()), body_changed_presentations=int(changed.sum()),
                 original_network_model_slot_count=int(np.sum(output["original_prior_origin"] == "network_model_slot")),
                 changed_network_model_slot_count=int(changed_network_slots.sum()),
                 canonical_reads_outside_segment=sorted(set(map(int, representatives[output["exposure_canonical_outside_segment"]]))),
                 missing_canonical_prior_frames=int((~canonical_valid).sum()),
                 declared_valid_but_nonfinite_canonical_frames=int((declared_valid & ~canonical_valid).sum()),
                 camera_inconsistent_with_canonical_frames=int((~camera_equal).sum()),
                 camera_parameter_max_difference=float(differences.max()),
                 camera_anomalies=anomalies,
                 before=same_exposure_differences(before), after=same_exposure_differences(output))
    return output, audit


def repair_cached_segment(prior_path: Path, observations_path: Path, camera,
                          output_dir: Path, *, start_pts=SEGMENT_START_PTS,
                          end_pts_exclusive=SEGMENT_END_PTS_EXCLUSIVE) -> dict:
    prior_path, observations_path, output_dir = map(Path, (prior_path, observations_path, output_dir))
    prior_sidecar = prior_path.with_suffix(".json")
    metadata = json.loads(prior_sidecar.read_text(encoding="utf-8-sig"))
    prior_sha, sidecar_sha, observation_sha = map(sha256, (prior_path, prior_sidecar, observations_path))
    expected = dict(source_sha256=camera.source["source_sha256_from_timeline"],
                    scene_sha256=camera.scene_sha256, camera_stream_sha256=camera.poses_sha256,
                    source_shots_sha256=camera.source_shots_sha256,
                    observation_sha256=observation_sha, body_prior_sha256=prior_sha)
    if metadata.get("schema") != "rinkspace.smplx_native_pelvis_prior.v1":
        raise ValueError("Unexpected original cached body-prior schema")
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Cached prior {name} binding differs from frozen original input")
    with np.load(prior_path, allow_pickle=False) as packet:
        prior = {k: packet[k].copy() for k in packet.files}
    repaired, audit = share_exposure_prior(prior, camera, start_pts, end_pts_exclusive)
    output_path = output_dir / "segment_prior.npz"
    summary_path = output_dir / "segment_prior.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("Existing segment repair must be inspected/reused; no automatic overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **repaired)
    for path, original_sha in ((prior_path, prior_sha), (prior_sidecar, sidecar_sha),
                               (observations_path, observation_sha)):
        if sha256(path) != original_sha:
            raise ValueError(f"Frozen original input changed while repairing: {path}")
    origins, counts = np.unique(repaired["prior_origin"], return_counts=True)
    canonical_origins, canonical_counts = np.unique(
        repaired["exposure_canonical_prior_origin"], return_counts=True)
    summary = {**metadata, **audit,
               "repair_schema": "rinkspace.confirmed_exposure_cached_prior_share.v1",
               "source_pts_count": len(repaired["pts"]),
               "prior_valid_count": int(repaired["prior_valid"].sum()),
               "body_prior_sha256": sha256(output_path),
               "original_body_prior_path": str(prior_path.resolve()),
               "original_body_prior_sha256": prior_sha,
               "original_body_prior_sidecar_sha256": sidecar_sha,
               "original_prior_materialized_shots": metadata.get("materialized_shots"),
               "materialized_shots": np.unique(repaired["shot_index"]).astype(int).tolist(),
               "interpolated_source_frames": int(np.sum(
                   repaired["exposure_canonical_prior_origin"] == "interpolated_model_prior")),
               "interpolated_source_frame_count_semantics": "presentations whose shared representative has an already cached interpolated prior",
               "prior_origin_counts": dict(zip(origins.tolist(), counts.astype(int).tolist())),
               "exposure_canonical_prior_origin_counts": dict(zip(
                   canonical_origins.tolist(), canonical_counts.astype(int).tolist())),
               "original_observations_path": str(observations_path.resolve()),
               "original_inputs_unchanged_after_repair": True,
               "repair_scope": {
                   "source_pts_start": int(start_pts),
                   "source_pts_end_exclusive": int(end_pts_exclusive),
                   "source_frame_count": len(repaired["pts"]),
                   "covers_entire_input_prior": bool(
                       start_pts == int(camera.pts[0]) and
                       end_pts_exclusive == int(camera.pts[-1]+PTS_STEP))},
               "repair_method": "read actual official representative cached prior for every group; invalid representative stays unknown",
               "raw_h_observations_modified": False,
               "camera_modified": False, "network_inference_rerun": False,
               "trajectory_smoothing": False, "world_translation_in_prior": False,
               "different_exposure_priors_modified": False,
               "note": "Original arrays/provenance retained as original_*; no model inference, temporal smoothing, cross-gap prediction or changed source PTS"}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                        allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-prior", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--source-shots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--full-scope", action="store_true",
                        help="Share the full selected camera/prior source scope instead of the historical [60,75) segment")
    parser.add_argument("--start-pts", type=int,
                        help="Exact source PTS start; requires --end-pts-exclusive")
    parser.add_argument("--end-pts-exclusive", type=int,
                        help="Exact exclusive source PTS end; requires --start-pts")
    args = parser.parse_args()
    if args.full_scope and (args.start_pts is not None or args.end_pts_exclusive is not None):
        parser.error("--full-scope cannot be combined with explicit source limits")
    if (args.start_pts is None) != (args.end_pts_exclusive is None):
        parser.error("Explicit source limits require both --start-pts and --end-pts-exclusive")
    camera = FrozenCameraStream(args.poses, args.scene, args.source_shots)
    start, end = ((int(camera.pts[0]), int(camera.pts[-1]+PTS_STEP)) if args.full_scope else
                  (args.start_pts, args.end_pts_exclusive) if args.start_pts is not None else
                  (SEGMENT_START_PTS, SEGMENT_END_PTS_EXCLUSIVE))
    print(json.dumps(repair_cached_segment(args.body_prior, args.observations, camera,
                                           args.output, start_pts=start,
                                           end_pts_exclusive=end), ensure_ascii=False))


if __name__ == "__main__":
    main()
