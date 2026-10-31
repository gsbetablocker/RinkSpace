"""Source-time GVHMR/SMPL-X prior for the frozen Ankara programme.

Run stages in order: plan (CPU), prepare (CPU), infer (project-owned body
CUDA environment, one shot at a time), materialize (same environment).  No
stage downloads weights, installs packages, writes the upstream repository,
or treats network global translation as a measured rink trajectory.

The explicit historical nearest_25hz policy chooses distinct exposure frames
in native presentation time. The official_25hz candidate instead chooses every
second verified official 50 Hz frame, breaking context at unknown/missing
exposures. It retains every original 60 Hz source PTS in the
final table with marked between-slot pose interpolation.  A full official
exposure (~50 Hz) branch can diagnose rhythm sensitivity on a complete short
shot.  The GVHMR training rhythm is not established by its mixed 25/50 fps
dataset code, so both branches are labelled unverified.  Windows, model
inputs and shape are never joined across cuts or unknown-body gaps.
"""
from __future__ import annotations

from rinkspace.storage import resolve_asset

import argparse
import hashlib
import json
import os
from pathlib import Path

import av
import cv2
import numpy as np

try:
    from .camera_bridge import (FrozenCameraStream, V15_POSES_SHA256, V15_SCENE_SHA256,
                                native_to_rectified,
                                rectify_patch, rectified_intrinsics)
    from .observations import iter_exact_frames
except ImportError:
    import sys
    from camera_bridge import (FrozenCameraStream, V15_POSES_SHA256, V15_SCENE_SHA256,
                               native_to_rectified,
                               rectify_patch, rectified_intrinsics)
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from three_frame_refit.program_tracking.body_tracking.observations import iter_exact_frames


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GVHMR_ROOT = resolve_asset(PROJECT_ROOT, PROJECT_ROOT / "environments/body/GVHMR")
GVHMR_CHECKPOINT = GVHMR_ROOT / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
HMR2_CHECKPOINT = GVHMR_ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
SMPLX_MODEL = GVHMR_ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
TIME_BASE = 15360
OBSERVATION_MASK_POLICY = "native_joint_candidates_keep_in_frame_face_v2"
COCO_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
              "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
              "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle")
NATIVE22_NAMES = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee",
                  "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
                  "left_foot", "right_foot", "neck", "left_collar", "right_collar",
                  "head", "left_shoulder", "right_shoulder", "left_elbow",
                  "right_elbow", "left_wrist", "right_wrist")
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _camera_metadata(camera: FrozenCameraStream) -> dict:
    """Bind every camera-dependent stage to the same files and release."""
    if hasattr(camera, "binding_metadata"):
        binding = camera.binding_metadata()
    else:  # small CPU numerical fixtures, never selected by the CLI
        binding = dict(camera_profile="synthetic_fixture",
                       source_sha256=camera.source["source_sha256_from_timeline"],
                       camera_stream_sha256=camera.poses_sha256,
                       scene_sha256=camera.scene_sha256,
                       source_shots_sha256=camera.source_shots_sha256)
    result = {key: binding[key] for key in (
        "source_sha256", "camera_stream_sha256", "scene_sha256", "source_shots_sha256")}
    result["camera_binding"] = binding
    if hasattr(camera, "poses_path"):
        result["camera_inputs"] = dict(
            poses=str(camera.poses_path.resolve()), scene=str(camera.scene_path.resolve()),
            source_shots=str(camera.source_shots_path.resolve()),
            release=str(camera.release_path.resolve()) if camera.release_path else None,
            profile=camera.profile)
    return result


def _validate_camera_metadata(metadata: dict, camera: FrozenCameraStream,
                              label: str, *, legacy_full: bool = False) -> None:
    expected = _camera_metadata(camera)
    binding = metadata.get("camera_binding")
    if binding is None:
        # The historical prepare/raw files saved only the source and v15 pose
        # hash. They remain usable under the strict v15 profile; the old slot
        # plan additionally binds scene and source-shot hashes at assembly.
        if (getattr(camera, "profile", None) != "v15" or
                camera.poses_sha256 != V15_POSES_SHA256 or
                camera.scene_sha256 != V15_SCENE_SHA256):
            raise ValueError(f"{label} lacks camera_binding for this camera release")
        required = ("source_sha256", "camera_stream_sha256")
        if legacy_full:
            required += ("scene_sha256", "source_shots_sha256")
    else:
        if not isinstance(binding, dict):
            raise ValueError(f"{label} camera_binding must be an object")
        required = ("source_sha256", "camera_stream_sha256", "scene_sha256",
                    "source_shots_sha256")
        # A moved copy of the same release is equivalent; its file hash,
        # revision, status, coordinate domains and source scope must match.
        for key, value in expected["camera_binding"].items():
            if key != "release_path" and binding.get(key) != value:
                raise ValueError(f"{label} camera_binding {key} differs from selected camera")
    for key in ("source_sha256", "camera_stream_sha256", "scene_sha256",
                "source_shots_sha256"):
        if (key in required or key in metadata) and metadata.get(key) != expected[key]:
            raise ValueError(f"{label} {key} differs from selected camera")


def _prepared_camera(metadata: dict, camera: FrozenCameraStream | None
                     ) -> FrozenCameraStream:
    """Restore a new cache's checked inputs; preserve standalone legacy v15 inference."""
    if camera is None and metadata.get("camera_binding") is not None:
        inputs = metadata.get("camera_inputs")
        if not isinstance(inputs, dict) or not all(inputs.get(k) for k in (
                "poses", "scene", "source_shots")):
            raise ValueError("Prepared cache needs checked camera inputs for inference")
        camera = FrozenCameraStream(inputs["poses"], inputs["scene"], inputs["source_shots"],
                                    release_path=inputs.get("release"),
                                    profile=inputs.get("profile"))
    if camera is None:
        if metadata.get("camera_stream_sha256") != V15_POSES_SHA256:
            raise ValueError("Standalone legacy inference requires the frozen v15 camera cache")
        try:
            from .manifest import V15
        except ImportError:
            from manifest import V15
        camera = FrozenCameraStream(V15 / "candidate_poses.npz", V15 / "candidate_scene.json",
                                    Path(__file__).resolve().parents[1] / "source_shots.json")
    _validate_camera_metadata(metadata, camera, "Prepared cache")
    return camera


def _validate_slot_cameras(data: dict, camera: FrozenCameraStream, label: str) -> None:
    n = len(data["source_pts"])
    if data["R_w2c"].shape != (n, 3, 3) or data["K_rect"].shape != (n, 3, 3):
        raise ValueError(f"{label} camera arrays must cover every source PTS")
    for stamp, rotation, intrinsic in zip(data["source_pts"], data["R_w2c"], data["K_rect"]):
        c = camera.camera_at(int(stamp))
        if (not np.allclose(rotation, cv2.Rodrigues(c[3:6])[0], rtol=0., atol=1e-5) or
                not np.array_equal(intrinsic, rectified_intrinsics(c).astype(np.float32))):
            raise ValueError(f"{label} source PTS {int(stamp)} has a camera inconsistent with selected inputs")


def _load_observations(path: Path, camera: FrozenCameraStream) -> dict:
    with np.load(path, allow_pickle=False) as z:
        required = {"pts", "coco17", "valid", "bbox_native_xyxy", "shot_index"}
        if not required.issubset(z.files):
            raise ValueError(f"Observation NPZ lacks {sorted(required - set(z.files))}")
        obs = {name: z[name].copy() for name in z.files}
    if not np.array_equal(obs["pts"], camera.pts):
        raise ValueError("Original H observations must cover all 10123 included source PTS")
    n = len(camera.pts)
    if (obs["coco17"].shape != (n, 17, 3) or
            obs["bbox_native_xyxy"].shape != (n, 4) or
            obs["valid"].shape != (n,) or
            not np.array_equal(obs["shot_index"], camera.shot_index)):
        raise ValueError("Original H array shape or shot identity is inconsistent")
    return obs


def original_observation_mask(observations: dict, native_size_wh=(1920, 1080)
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Gate constraints without changing the stored original-H candidates.

    In-image heatmap candidates are not independently reviewed visibility.
    A lower-body prediction in a large, bottom-truncated source portrait is
    specifically excluded from measured joint constraints.  Its network 3D
    prediction may remain a labelled body prior, never height support.
    """
    joints = np.asarray(observations["coco17"], np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (17, 3):
        raise ValueError("Original H joints must have shape [N,17,3]")
    width, height = native_size_wh
    native = (np.isfinite(joints).all(axis=2) & (joints[:, :, 2] > 0)
              & (joints[:, :, 0] >= 0) & (joints[:, :, 0] < width)
              & (joints[:, :, 1] >= 0) & (joints[:, :, 1] < height))
    for name in ("keypoint_valid", "keypoint_in_source"):
        if name in observations:
            mask = np.asarray(observations[name], bool)
            if mask.shape != native.shape:
                raise ValueError(f"{name} must have shape [N,17]")
            native &= mask
    truncated = np.asarray(observations.get(
        "source_clipped_target", np.zeros(len(joints), bool)), bool).copy()
    if truncated.shape != (len(joints),):
        raise ValueError("source_clipped_target must have shape [N]")
    if "bbox_native_xyxy" in observations:
        boxes = np.asarray(observations["bbox_native_xyxy"], np.float64)
        if boxes.shape != (len(joints), 4):
            raise ValueError("Original target boxes must have shape [N,4]")
        finite = np.isfinite(boxes).all(axis=1)
        top = finite & (boxes[:, 1] <= 24)
        bottom_portrait = (finite & (boxes[:, 3] >= height-24)
                           & (boxes[:, 3]-boxes[:, 1] > height/2))
        truncated |= (finite & ((boxes[:, 0] <= 24) | top
                                | (boxes[:, 2] >= width-24)
                                | (boxes[:, 3] >= height-24)))
        # The box can touch the top because of a raised arm. It does not
        # establish that the face is cropped. Face candidates retain their
        # own source-domain/validity gates; reviewed exclusions and score
        # gates are applied by the caller. Keep the frame truncation flag.
        native[bottom_portrait, 11:17] = False
    return native, truncated


def _choose_25hz(indices: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Nearest *real distinct exposure* to 40 ms presentation-time targets."""
    if not len(indices):
        return indices
    times = pts[indices].astype(np.float64) / TIME_BASE
    targets = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * 25)) + 1) / 25.
    right = np.searchsorted(times, targets)
    right = np.clip(right, 0, len(indices) - 1)
    left = np.clip(right - 1, 0, len(indices) - 1)
    nearest = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets),
                       left, right)
    # Reusing one exposure for two model slots would make false observations.
    # Keep the last real exposure as a boundary slot even if the preceding
    # 40 ms target landed earlier; no extrapolation is then needed at the cut.
    return indices[np.unique(np.r_[nearest, len(indices)-1])]


def build_slot_plan(camera: FrozenCameraStream, observations: dict,
                    *, time_policy: str, observation_sha256: str,
                    source_video: Path | None = None,
                    official_evidence: tuple[dict, list[dict], dict] | None = None,
                    official_evidence_paths: tuple[Path, Path] | None = None) -> dict:
    if time_policy not in ("official_unique", "nearest_25hz", "official_25hz"):
        raise ValueError("Unknown body-prior time_policy")
    pts = camera.pts
    valid = np.asarray(observations["valid"], bool)
    aliases = camera.official_exposure_representative_pts
    slots = []
    for shot in range(1, 10) if time_policy != "official_25hz" else ():
        member = np.flatnonzero(camera.shot_index == shot)
        representatives = np.unique(aliases[member])
        group_indices = []
        for representative in representatives:
            group = member[aliases[member] == representative]
            if not len(group):
                raise ValueError("Exposure representative has no source frame")
            # Prefer the representative itself.  A confirmed duplicate exposure
            # is allowed only if its actual original-H observation exists.
            usable = group[valid[group]]
            selected = (int(representative) if
                        np.any(pts[usable] == representative) else
                        int(pts[usable[0]]) if len(usable) else int(representative))
            selected_index = int(np.searchsorted(pts, selected))
            group_indices.append(selected_index)
        group_indices = np.asarray(group_indices, np.int64)
        if time_policy == "nearest_25hz":
            group_indices = _choose_25hz(group_indices, pts)
        for shot_slot, idx in enumerate(group_indices):
            source_pts = int(pts[idx])
            representative = int(aliases[idx])
            slots.append(dict(slot_id=len(slots), shot_index=shot,
                              shot_slot=shot_slot, source_pts=source_pts,
                              official_exposure_representative_pts=representative,
                              source_presentation_time_s=source_pts / TIME_BASE,
                              model_sample_time_s=source_pts / TIME_BASE,
                              model_time_semantics="source presentation time; true capture time unknown",
                              observation_valid=bool(valid[idx]),
                              observation_reuses_confirmed_exposure=
                              source_pts != representative,
                              camera_pose_representative_pts=int(
                                  camera.source_pose_representative_pts[idx])))
    cadence = {}
    if time_policy == "official_25hz":
        if official_evidence is None:
            raise ValueError("official_25hz requires checked complete official exposure evidence")
        from three_frame_refit.program_tracking.body_tracking.official_cadence import build_official_slots
        cadence = build_official_slots(camera, observations, *official_evidence)
        slots = cadence.pop("slots")
        if official_evidence_paths is not None:
            cadence["official_evidence_inputs"] = dict(
                groups=str(official_evidence_paths[0].resolve()),
                mapping=str(official_evidence_paths[1].resolve()))
    if any(a["source_pts"] >= b["source_pts"] for a, b in zip(slots, slots[1:])):
        raise ValueError("Model slots must advance in original source PTS")
    videosha = None
    if source_video is not None:
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        videosha = _sha256(source_video)
        if videosha != camera.source["source_sha256_from_timeline"]:
            raise ValueError("Input video differs from frozen source SHA256")
    shot_deltas = {}
    for shot in range(1, 10):
        times = [row["model_sample_time_s"] for row in slots if row["shot_index"] == shot]
        delta = np.diff(times)
        shot_deltas[str(shot)] = dict(count=len(times),
                                      min_s=float(delta.min()) if len(delta) else None,
                                      median_s=float(np.median(delta)) if len(delta) else None,
                                      max_s=float(delta.max()) if len(delta) else None)
    plan = dict(schema="rinkspace.body_prior_slots.v1",
                time_policy=time_policy,
                training_rhythm_compatibility="unverified_mixed_25fps_video_and_50hz_pose_code",
                no_synthetic_or_duplicated_model_frames=True,
                source_time_base="1/15360", full_source_pts_count=len(pts),
                excluded_first_shot_frame_count=len(camera.excluded_pts),
                shot_count=9, slots=slots, by_shot_time_delta_s=shot_deltas,
                source_video_sha256_verified=videosha,
                observations_sha256=observation_sha256,
                **_camera_metadata(camera), **cadence)
    if time_policy == "official_25hz":
        from three_frame_refit.program_tracking.body_tracking.official_cadence import temporal_plan_digest
        plan["temporal_plan_sha256"] = temporal_plan_digest(plan)
    return plan


def _bbox_to_rect_xys(native_xyxy: np.ndarray, camera25: np.ndarray) -> np.ndarray | None:
    x0, y0, x1, y1 = map(float, native_xyxy)
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        return None
    t = np.linspace(0., 1., 9)
    border = np.concatenate((np.c_[x0 + (x1-x0)*t, np.full(9, y0)],
                             np.c_[x0 + (x1-x0)*t, np.full(9, y1)],
                             np.c_[np.full(9, x0), y0 + (y1-y0)*t],
                             np.c_[np.full(9, x1), y0 + (y1-y0)*t]))
    mapped, valid = native_to_rectified(border, camera25)
    if np.count_nonzero(valid) < 18:
        return None
    lo, hi = mapped[valid].min(axis=0), mapped[valid].max(axis=0)
    center = (lo + hi) / 2.
    side = max(hi - lo) * 1.2  # upstream get_bbx_xys_from_xyxy enlargement
    if not np.isfinite(side) or side < 16.:
        return None
    return np.r_[center, side]


def _rectified_crop(bgr: np.ndarray, camera25: np.ndarray,
                    bbox_rect_xys: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Sample the 256-square virtual camera crop through native Brown rays.

    The crop K is an isotropic affine transform of the saved full-image
    K_rect.  Thus its pixels represent precisely the same rectified rays that
    an upstream affine crop of a full rectified frame would represent, while
    only 256² rays are evaluated.  We preserve both K values and the mapping.
    """
    cx, cy, side = map(float, bbox_rect_xys)
    k_rect_full = rectified_intrinsics(camera25)
    a = 255. / side
    left, top = cx-side/2., cy-side/2.
    k_crop = np.array([[a*k_rect_full[0, 0], 0., a*(k_rect_full[0, 2]-left)],
                       [0., a*k_rect_full[1, 1], a*(k_rect_full[1, 2]-top)],
                       [0., 0., 1.]], np.float64)
    # G1 get_batch(img_ds=1) blurs before downsampling a large bbox.  Apply
    # that small anti-alias filter to native source before the one-step Brown
    # ray remap; this differs subtly from blur-after-full-rectification, so
    # the actual kernel/order are saved for provenance.
    d = side / 256. / 2.
    if d > 1.1:
        bgr = cv2.GaussianBlur(bgr, (5, 5), (d - 1.) / 2.)
    crop_bgr, valid, returned_k_crop = rectify_patch(
        bgr, camera25, (0, 0, 256, 256), K_rect=k_crop)
    if crop_bgr.shape[:2] != (256, 256) or not np.allclose(returned_k_crop, k_crop):
        raise ValueError("Direct rectified crop has unexpected geometry")
    full_to_crop = np.array([[a, 0., -a*left], [0., a, -a*top], [0., 0., 1.]])
    return crop_bgr[:, :, ::-1].copy(), float(np.mean(valid)), dict(
        rect_patch_xywh=[0, 0, 256, 256],
        full_rect_to_crop_affine=full_to_crop.tolist(),
        K_rect_full=k_rect_full.tolist(), K_rect_crop=k_crop.tolist(),
        anti_alias_native_gaussian_sigma=(d-1.)/2. if d > 1.1 else 0.,
        crop_domain="rectified virtual image; RGB uint8, 256x256",
        normalization="RGB/255, subtract ImageNet mean, divide ImageNet std")


def prepare_shot(plan: dict, shot_index: int, observations: dict,
                 camera: FrozenCameraStream, video: Path, output: Path) -> dict:
    """Produce exact-PTS rectified HMR2 crops without loading GPU models."""
    _validate_camera_metadata(plan, camera, "Slot plan", legacy_full=True)
    from three_frame_refit.program_tracking.body_tracking import official_cadence
    official_cadence.plan_binding(plan)
    official_cadence.validate_plan_evidence(plan, camera, observations)
    if _sha256(video) != plan["source_sha256"]:
        raise ValueError("Prepare video differs from the slot plan source SHA256")
    rows = [row for row in plan["slots"] if row["shot_index"] == shot_index]
    usable = [row for row in rows if row["observation_valid"]]
    wanted = [row["source_pts"] for row in usable]
    if not wanted:
        raise ValueError(f"Shot {shot_index} has no usable official-exposure slots")
    index = {int(p): i for i, p in enumerate(observations["pts"])}
    image_iter = iter_exact_frames(video, wanted)
    prepared = []
    crops = []
    rect_joints = []
    bbox_rect = []
    k_rect = []
    rw2c = []
    original_valid, source_truncated = original_observation_mask(
        observations, camera.native_size_wh)
    for row in usable:
        stamp, bgr = next(image_iter)
        if stamp != row["source_pts"]:
            raise ValueError("Decoder returned a frame for the wrong model slot")
        c = camera.camera_at(stamp)
        source = observations["coco17"][index[stamp]]
        bbox = _bbox_to_rect_xys(observations["bbox_native_xyxy"][index[stamp]], c)
        if bbox is None:
            continue
        mapped, good = native_to_rectified(source[:, :2], c)
        model_obs = np.zeros((17, 3), np.float32)
        model_obs[:, :2] = bbox[:2]
        usable_joint = good & original_valid[index[stamp]]
        model_obs[usable_joint, :2] = mapped[usable_joint]
        model_obs[usable_joint, 2] = source[usable_joint, 2]
        crop_rgb, valid_fraction, mapping = _rectified_crop(bgr, c, bbox)
        if valid_fraction < .5:
            continue
        prepared.append(dict(**row, crop_valid_fraction=valid_fraction,
                             crop_mapping=mapping,
                              source_truncation_unreviewed=bool(source_truncated[index[stamp]]),
                             rectified_joint_count=int(np.count_nonzero(usable_joint))))
        crops.append(crop_rgb)
        rect_joints.append(model_obs)
        bbox_rect.append(bbox.astype(np.float32))
        k_rect.append(rectified_intrinsics(c).astype(np.float32))
        rw2c.append(cv2.Rodrigues(c[3:6])[0].astype(np.float32))
    if not prepared:
        raise ValueError(f"Shot {shot_index} has no valid rectified crops")
    output.mkdir(parents=True, exist_ok=True)
    name = f"prepared_shot_{shot_index:02d}"
    temporal_arrays = {}
    if plan["time_policy"] == official_cadence.POLICY:
        temporal_arrays = dict(
            official_pts=np.asarray([r["official_pts"] for r in prepared], np.int64),
            sequence_id=np.asarray([r["sequence_id"] for r in prepared], np.int32),
            model_sample_time_s=np.asarray([r["model_sample_time_s"] for r in prepared], np.float64))
    np.savez_compressed(output / f"{name}.npz",
                        source_pts=np.asarray([r["source_pts"] for r in prepared], np.int64),
                        official_exposure_representative_pts=np.asarray(
                            [r["official_exposure_representative_pts"] for r in prepared], np.int64),
                        shot_slot=np.asarray([r["shot_slot"] for r in prepared], np.int32),
                        rgb_crop256=np.stack(crops),
                        rect_coco17=np.stack(rect_joints),
                        bbox_rect_xys=np.stack(bbox_rect),
                        K_rect=np.stack(k_rect), R_w2c=np.stack(rw2c),
                        crop_valid_fraction=np.asarray(
                            [r["crop_valid_fraction"] for r in prepared], np.float32),
                        **temporal_arrays)
    _save_json(output / f"{name}.json", dict(
        schema="rinkspace.rectified_body_prior_input.v1", shot_index=shot_index,
        slots=prepared, **_camera_metadata(camera),
        prepared_sha256=_sha256(output / f"{name}.npz"),
        observation_sha256=plan["observations_sha256"],
        observation_mask_policy=OBSERVATION_MASK_POLICY,
        pixel_runtime=dict(pyav=av.__version__, opencv=cv2.__version__,
            ffmpeg_libraries={name:list(version) for name,version in av.library_versions.items()},
            decoded_format="bgr24", cv2_threads=cv2.getNumThreads()),
        time_policy=plan["time_policy"],
        **official_cadence.cache_binding(plan, prepared),
        note="Missing/invalid slots remain gaps; no crop or body pose is fabricated"))
    return dict(shot_index=shot_index, planned_slots=len(rows),
                rectified_real_exposures=len(prepared),
                missing_or_invalid_slots=len(rows)-len(prepared),
                prepared_path=str(output / f"{name}.npz"))


def _windows(shot_slot: np.ndarray, *, max_length: int = 256,
             overlap: int = 64) -> list[np.ndarray]:
    """Contiguous observed runs only; no cut/gap padding or copied frames."""
    if max_length < 2 or overlap < 0 or overlap >= max_length:
        raise ValueError("Invalid temporal window length or overlap")
    if not len(shot_slot) or np.any(np.diff(shot_slot) <= 0):
        raise ValueError("Prepared shot slots must be strictly increasing")
    breaks = np.r_[0, np.flatnonzero(np.diff(shot_slot) != 1) + 1, len(shot_slot)]
    result = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b-a < 2:
            continue  # upstream compute_cam_angvel needs a real adjacent pair
        start = a
        while start < b:
            end = min(start+max_length, b)
            if end-start < 2:
                break
            result.append(np.arange(start, end))
            if end == b:
                break
            start = end-overlap
    return result


def _upstream_runtime():
    import sys
    if not GVHMR_CHECKPOINT.is_file() or not HMR2_CHECKPOINT.is_file() or not SMPLX_MODEL.is_file():
        raise FileNotFoundError("Existing verified project GVHMR weights are missing")
    expected_env = resolve_asset(PROJECT_ROOT, PROJECT_ROOT / ".venv-body")
    if not Path(sys.prefix).resolve().samefile(expected_env.resolve()):
        raise RuntimeError(f"Use existing GVHMR interpreter: {expected_env / 'Scripts/python.exe'}")
    if str(GVHMR_ROOT) not in sys.path:
        sys.path.insert(0, str(GVHMR_ROOT))
    os.chdir(GVHMR_ROOT)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Real GVHMR stage requires the existing CUDA environment")
    return torch


def infer_shot(prepared_path: Path, output: Path, *, max_length: int = 256,
               overlap: int = 64, camera: FrozenCameraStream | None = None) -> dict:
    """Real HMR2 features -> GVHMR incam pose, never global-world translation."""
    prepared_path, output = prepared_path.resolve(), output.resolve()
    prepared_sidecar = json.loads(prepared_path.with_suffix(".json").read_text(encoding="utf-8"))
    camera = _prepared_camera(prepared_sidecar, camera)
    if ((prepared_sidecar.get("camera_binding") is not None or
         prepared_sidecar.get("prepared_sha256") is not None) and
            prepared_sidecar.get("prepared_sha256") != _sha256(prepared_path)):
        raise ValueError("Prepared NPZ changed after camera-bound preparation")
    with np.load(prepared_path, allow_pickle=False) as z:
        data = {name: z[name].copy() for name in z.files}
    pts = data["source_pts"]
    slots = data["shot_slot"]
    if (data["rgb_crop256"].shape != (len(pts), 256, 256, 3)
            or data["rect_coco17"].shape != (len(pts), 17, 3)
            or data["bbox_rect_xys"].shape != (len(pts), 3)
            or data["K_rect"].shape != (len(pts), 3, 3)
            or data["R_w2c"].shape != (len(pts), 3, 3)):
        raise ValueError("Prepared rectified arrays have wrong shapes")
    from three_frame_refit.program_tracking.body_tracking import official_cadence
    official_cadence.validate_cache(prepared_sidecar, data, "Prepared cache", camera=camera)
    wins = _windows(slots, max_length=max_length, overlap=overlap)
    if not wins:
        raise ValueError("No real adjacent slots support GVHMR in this shot")
    shot = int(prepared_path.stem.split("_")[-1])
    if prepared_sidecar["shot_index"] != shot or len(prepared_sidecar["slots"]) != len(pts):
        raise ValueError("Prepared body-crop sidecar does not match the NPZ")
    _validate_slot_cameras(data, camera, "Prepared cache")
    torch = _upstream_runtime()
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    import hmr4d.model.gvhmr.gvhmr_pl_demo  # Hydra registration
    import hmr4d.model.gvhmr.utils.endecoder
    import hmr4d.network.gvhmr.relative_transformer
    torch.set_grad_enabled(False)
    feature_unit = Extractor(tqdm_leave=False)
    features = np.full((len(pts), 1024), np.nan, np.float32)
    covered = np.unique(np.concatenate(wins))
    for part in np.array_split(covered, max(1, int(np.ceil(len(covered)/32)))):
        rgb = data["rgb_crop256"][part].astype(np.float32)
        crops = torch.from_numpy(((rgb / 255. - IMAGE_MEAN) / IMAGE_STD).transpose(0, 3, 1, 2))
        bbox = torch.from_numpy(data["bbox_rect_xys"][part].copy())
        with torch.inference_mode():
            features[part] = feature_unit.extract_video_features(crops, bbox).numpy()
    del feature_unit
    torch.cuda.empty_cache()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=rinkspace_body_prior"])
    unit = instantiate(cfg.model, _recursive_=False)
    unit.load_pretrained_model(GVHMR_CHECKPOINT)
    unit.eval().requires_grad_(False).cuda()
    body_pose = np.full((len(pts), 63), np.nan, np.float32)
    betas = np.full((len(pts), 10), np.nan, np.float32)
    orient = np.full((len(pts), 3), np.nan, np.float32)
    transl = np.full((len(pts), 3), np.nan, np.float32)
    chosen_window = np.full(len(pts), -1, np.int32)
    edge_distance = np.full(len(pts), -1, np.int32)
    overlap_comparisons = []
    for window_id, indices in enumerate(wins):
        l = len(indices)
        r = torch.from_numpy(data["R_w2c"][indices]).float()
        # The official function computes R[t+1] @ R[t].T internally and pads
        # the last delta.  Do not feed it precomputed deltas.
        angvel = compute_cam_angvel(r)
        if angvel.shape != (l, 6):
            raise ValueError("Upstream camera angular velocity has wrong shape")
        box = torch.from_numpy(data["bbox_rect_xys"][indices]).float()
        joints = torch.from_numpy(data["rect_coco17"][indices]).float()
        batch = dict(length=torch.tensor([l], dtype=torch.long),
                     obs=normalize_kp2d(joints, box)[None],
                     bbx_xys=box[None],
                     K_fullimg=torch.from_numpy(data["K_rect"][indices]).float()[None],
                     cam_angvel=angvel[None],
                     f_imgseq=torch.from_numpy(features[indices]).float()[None])
        batch = {key: value.cuda() for key, value in batch.items()}
        with torch.inference_mode():
            outputs = unit.pipeline.forward(batch, train=False,
                                            postproc=False, static_cam=False)
        incam = outputs["pred_smpl_params_incam"]
        current = {"body_pose": incam["body_pose"][0].cpu().numpy(),
                   "betas": incam["betas"][0].cpu().numpy(),
                   "global_orient": incam["global_orient"][0].cpu().numpy(),
                   "transl": incam["transl"][0].cpu().numpy()}
        if any(not np.isfinite(value).all() for value in current.values()):
            raise ValueError(f"Nonfinite real GVHMR output in window {window_id}")
        for local, global_i in enumerate(indices):
            # Select the prediction with the most real context on both sides.
            # Window overlap is diagnosed below and never world-aligned by a
            # per-window translation.
            distance = min(local, l-1-local)
            if chosen_window[global_i] >= 0:
                overlap_comparisons.append(dict(source_pts=int(pts[global_i]),
                    previous_window=int(chosen_window[global_i]), new_window=window_id,
                    body_pose_l2=float(np.linalg.norm(
                        body_pose[global_i]-current["body_pose"][local])),
                    global_orient_l2=float(np.linalg.norm(
                        orient[global_i]-current["global_orient"][local]))))
            if distance > edge_distance[global_i]:
                body_pose[global_i] = current["body_pose"][local]
                betas[global_i] = current["betas"][local]
                orient[global_i] = current["global_orient"][local]
                transl[global_i] = current["transl"][local]
                chosen_window[global_i] = window_id
                edge_distance[global_i] = distance
    output.mkdir(parents=True, exist_ok=True)
    temporal_arrays = {key: data[key] for key in (
        "official_pts", "sequence_id", "model_sample_time_s")
        if prepared_sidecar["time_policy"] == official_cadence.POLICY}
    np.savez_compressed(output / f"raw_shot_{shot:02d}.npz",
                        source_pts=pts, shot_slot=slots,
                        official_exposure_representative_pts=
                            data["official_exposure_representative_pts"],
                        R_w2c=data["R_w2c"], K_rect=data["K_rect"],
                        body_pose=body_pose, betas=betas, global_orient=orient,
                        transl_incam_network=transl,
                        chosen_window=chosen_window, edge_distance=edge_distance,
                        **temporal_arrays)
    _save_json(output / f"raw_shot_{shot:02d}.json", dict(
        schema="rinkspace.gvhmr_incam_raw.v1", shot_index=shot,
        prepared_sha256=_sha256(prepared_path),
        raw_sha256=_sha256(output / f"raw_shot_{shot:02d}.npz"),
        **_camera_metadata(camera),
        observation_sha256=prepared_sidecar["observation_sha256"],
        time_policy=prepared_sidecar["time_policy"],
        **{key: prepared_sidecar[key] for key in (
            "temporal_plan_sha256", "model_time_contract", "official_evidence_sha256",
            "official_evidence_inputs", "temporal_slots", "temporal_slots_sha256")
           if prepared_sidecar["time_policy"] == official_cadence.POLICY},
        checkpoint_sha256=_sha256(GVHMR_CHECKPOINT),
        hmr2_checkpoint_sha256=_sha256(HMR2_CHECKPOINT),
        smplx_model_sha256=_sha256(SMPLX_MODEL),
        postproc=False, static_cam=False,
        global_translation_used_for_world=False,
        network_time_compatibility="unverified",
        model_slots=len(pts), predicted_slots=int(np.sum(chosen_window >= 0)),
        windows=[dict(start_source_pts=int(pts[w[0]]),
                      end_source_pts=int(pts[w[-1]]), length=len(w)) for w in wins],
        overlap_comparisons=overlap_comparisons,
        note="Single-slot observed runs have no verified GVHMR prediction"))
    return dict(shot_index=shot, slots=len(pts), predicted=int(np.sum(chosen_window >= 0)),
                windows=len(wins))


def _shared_beta(raws: list[tuple[int, dict]]) -> tuple[np.ndarray, dict]:
    by_shot = {}
    for shot, data in raws:
        beta = np.asarray(data["betas"], np.float64)
        finite = np.isfinite(beta).all(axis=1)
        if not finite.any():
            continue
        by_shot[str(shot)] = np.median(beta[finite], axis=0)
    if len(by_shot) < 2:
        raise ValueError("Cannot establish one robust shared body shape from multiple shots")
    shared = np.median(np.stack(list(by_shot.values())), axis=0)
    return shared.astype(np.float32), {k: v.tolist() for k, v in by_shot.items()}


def _slerp_rotvec(left: np.ndarray, right: np.ndarray,
                  fraction: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path rotations, preserving the fixed body shape."""
    from scipy.spatial.transform import Rotation
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    t = np.asarray(fraction, np.float64)
    if a.shape != b.shape or a.shape[-1] != 3 or t.shape != a.shape[:-2]:
        raise ValueError("Rotation interpolation needs [frame,joint,3] and [frame] times")
    q0 = Rotation.from_rotvec(a.reshape(-1, 3)).as_quat().reshape(*a.shape[:-1], 4)
    q1 = Rotation.from_rotvec(b.reshape(-1, 3)).as_quat().reshape(*a.shape[:-1], 4)
    dot = np.sum(q0*q1, axis=-1)
    q1 = np.where(dot[..., None] < 0., -q1, q1)
    dot = np.clip(np.abs(dot), -1., 1.)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    f = t[:, None]
    small = sin_angle < 1e-7
    denominator = np.where(small, 1., sin_angle)
    w0 = np.where(small, 1.-f, np.sin((1.-f)*angle)/denominator)
    w1 = np.where(small, f, np.sin(f*angle)/denominator)
    q = w0[..., None]*q0 + w1[..., None]*q1
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(a.shape).astype(np.float32)


def materialize(plan: dict, raw_dir: Path, camera: FrozenCameraStream,
                output: Path, *, shots: tuple[int, ...] = tuple(range(1, 10)),
                shared_betas_from: Path | None = None) -> dict:
    """Rerun real SMPL-X with one shared beta on all supported source PTS.

    At model slots, use the real GVHMR estimate.  Between adjacent, valid
    slots of the *same shot*, interpolate local pose rotations and fixed-world
    root orientation in actual PTS time for historical policies, or verified
    canonical official exposure time for official_25hz, then run SMPL-X.
    Interpolated frames are never counted as extra network observations.
    """
    raw_dir, output = raw_dir.resolve(), output.resolve()
    if shared_betas_from is not None:
        shared_betas_from = shared_betas_from.resolve()
    _validate_camera_metadata(plan, camera, "Slot plan", legacy_full=True)
    from three_frame_refit.program_tracking.body_tracking import official_cadence
    official_cadence.plan_binding(plan)
    official_evidence = official_cadence.validate_plan_evidence(plan, camera)
    if not shots or len(set(shots)) != len(shots) or any(s not in range(1, 10) for s in shots):
        raise ValueError("Materialized shots must be distinct reviewed indices 1..9")
    if len(shots) != 9 and shared_betas_from is None:
        raise ValueError("Partial-shot sensitivity needs the primary programme-shared betas")
    raws = []
    for shot in shots:
        path = raw_dir / f"raw_shot_{shot:02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        _validate_camera_metadata(sidecar, camera, f"Raw GVHMR shot {shot}")
        if ((sidecar.get("camera_binding") is not None or sidecar.get("raw_sha256") is not None)
                and sidecar.get("raw_sha256") != _sha256(path)):
            raise ValueError(f"Raw GVHMR shot {shot} NPZ changed after inference")
        if (sidecar["shot_index"] != shot or
                sidecar["camera_stream_sha256"] != camera.poses_sha256 or
                sidecar["source_sha256"] != plan["source_sha256"] or
                sidecar["observation_sha256"] != plan["observations_sha256"] or
                sidecar["time_policy"] != plan["time_policy"]):
            raise ValueError(f"Raw GVHMR shot {shot} is bound to another input")
        with np.load(path, allow_pickle=False) as z:
            data = {name: z[name].copy() for name in z.files}
        official_cadence.validate_cache(sidecar, data, f"Raw shot {shot}", plan)
        planned = {(row["shot_slot"], row["source_pts"],
                    row["official_exposure_representative_pts"])
                   for row in plan["slots"] if row["shot_index"] == shot}
        actual = set(zip(data["shot_slot"].tolist(), data["source_pts"].tolist(),
                         data["official_exposure_representative_pts"].tolist()))
        if not actual.issubset(planned) or len(actual) != len(data["source_pts"]):
            raise ValueError(f"Raw shot {shot} has an invented or repeated model slot")
        _validate_slot_cameras(data, camera, f"Raw shot {shot}")
        raws.append((shot, data))
    if shared_betas_from is None:
        beta, shot_betas = _shared_beta(raws)
        shared_beta_source = "robust_median_of_nine_shot_medians"
    else:
        beta_sidecar = json.loads(shared_betas_from.with_suffix(".json").read_text(encoding="utf-8"))
        _validate_camera_metadata(beta_sidecar, camera, "Shared body shape", legacy_full=True)
        if (beta_sidecar.get("body_prior_sha256") != _sha256(shared_betas_from) or
                beta_sidecar.get("observation_sha256") != plan["observations_sha256"]):
            raise ValueError("Shared body shape differs from primary camera-bound prior")
        with np.load(shared_betas_from, allow_pickle=False) as z:
            beta = np.asarray(z["shared_betas"], np.float32).copy()
        if beta.shape != (10,) or not np.isfinite(beta).all():
            raise ValueError("Primary shared beta must have 10 finite values")
        shot_betas = {}
        shared_beta_source = f"primary_body_prior_sha256:{_sha256(shared_betas_from)}"
    # The EnDecoder owns exactly the upstream SmplxLiteV437Coco17 regressor
    # and native 22-joint FK used by the verified G1 chain.
    torch = _upstream_runtime()
    from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
    model = EnDecoder().eval().requires_grad_(False).cuda()
    n = len(camera.pts)
    pose = np.full((n, 63), np.nan, np.float32)
    world_orient = np.full((n, 3, 3), np.nan, np.float32)
    prior_origin = np.full(n, "missing_body_prior", dtype="<U48")
    model_slot_pts = np.full(n, -1, np.int64)
    bracket_left_pts = np.full(n, -1, np.int64)
    bracket_right_pts = np.full(n, -1, np.int64)
    interpolation_alpha = np.full(n, np.nan, np.float32)
    # Convert incam root orientations into one frozen world frame at generation
    # time.  A later camera optimizer must not recompute Q from its trial R.
    from scipy.spatial.transform import Rotation
    exposure_to_model = {}
    for shot, data in raws:
        valid = np.isfinite(data["body_pose"]).all(axis=1)
        valid &= np.isfinite(data["global_orient"]).all(axis=1)
        valid &= data["chosen_window"] >= 0
        raw_indices = np.flatnonzero(valid)
        if not len(raw_indices):
            continue
        q_world_raw = np.full((len(data["source_pts"]), 3, 3), np.nan, np.float32)
        orient_c = Rotation.from_rotvec(data["global_orient"][raw_indices]).as_matrix()
        q_world_raw[raw_indices] = (np.swapaxes(data["R_w2c"][raw_indices], 1, 2)
                                     @ orient_c).astype(np.float32)
        for raw_i in raw_indices:
            rep = int(data["official_exposure_representative_pts"][raw_i])
            if rep in exposure_to_model:
                raise ValueError("One confirmed exposure was inferred twice")
            exposure_to_model[rep] = dict(
                shot=shot, source_pts=int(data["source_pts"][raw_i]),
                body_pose=data["body_pose"][raw_i],
                q_world=q_world_raw[raw_i])
        member = np.flatnonzero(camera.shot_index == shot)
        for i in member:
            item = exposure_to_model.get(int(camera.official_exposure_representative_pts[i]))
            if item is None or item["shot"] != shot:
                continue
            pose[i] = item["body_pose"]
            world_orient[i] = item["q_world"]
            model_slot_pts[i] = item["source_pts"]
            bracket_left_pts[i] = item["source_pts"]
            bracket_right_pts[i] = item["source_pts"]
            interpolation_alpha[i] = 0.
            prior_origin[i] = ("network_model_slot" if int(camera.pts[i]) == item["source_pts"]
                               else "confirmed_same_exposure_alias")
        # Fill only between *adjacent planned slots* that both have real
        # predictions.  Unknown target/crop or a one-slot temporal run breaks
        # this relation even when later timestamps are close.
        for left_raw, right_raw in zip(raw_indices[:-1], raw_indices[1:]):
            if int(data["shot_slot"][right_raw] - data["shot_slot"][left_raw]) != 1:
                continue
            left_pts = int(data["source_pts"][left_raw])
            right_pts = int(data["source_pts"][right_raw])
            gap = member[(camera.pts[member] > left_pts) &
                         (camera.pts[member] < right_pts) &
                         (prior_origin[member] == "missing_body_prior")]
            if not len(gap):
                continue
            if plan["time_policy"] == official_cadence.POLICY:
                alpha = official_cadence.interpolation_fractions(
                    camera, official_evidence[1], left_pts, right_pts, camera.pts[gap])
                supported = np.isfinite(alpha)
                gap, alpha = gap[supported], alpha[supported]
                if not len(gap):
                    continue
            else:
                alpha = ((camera.pts[gap]-left_pts)/(right_pts-left_pts)).astype(np.float32)
            left_body = np.repeat(data["body_pose"][left_raw].reshape(1, 21, 3), len(gap), axis=0)
            right_body = np.repeat(data["body_pose"][right_raw].reshape(1, 21, 3), len(gap), axis=0)
            pose[gap] = _slerp_rotvec(left_body, right_body, alpha).reshape(len(gap), 63)
            q_left = Rotation.from_matrix(q_world_raw[left_raw]).as_rotvec()
            q_right = Rotation.from_matrix(q_world_raw[right_raw]).as_rotvec()
            qrv = _slerp_rotvec(np.repeat(q_left.reshape(1, 1, 3), len(gap), axis=0),
                                np.repeat(q_right.reshape(1, 1, 3), len(gap), axis=0),
                                alpha)[:, 0]
            world_orient[gap] = Rotation.from_rotvec(qrv).as_matrix().astype(np.float32)
            prior_origin[gap] = "interpolated_model_prior"
            bracket_left_pts[gap] = left_pts
            bracket_right_pts[gap] = right_pts
            interpolation_alpha[gap] = alpha
    coco_world = np.full((n, 17, 3), np.nan, np.float32)
    native_world = np.full((n, 22, 3), np.nan, np.float32)
    coco_body = np.full((n, 17, 3), np.nan, np.float32)
    native_body = np.full((n, 22, 3), np.nan, np.float32)
    body_world_R = np.full((n, 3, 3), np.nan, np.float32)
    incam_pelvis = np.full((n, 3), np.nan, np.float32)
    active = np.flatnonzero(np.isfinite(pose).all(axis=1) &
                            np.isfinite(world_orient).all(axis=(1,2)))
    for batch_indices in np.array_split(active,
                                        max(1, int(np.ceil(len(active)/96)))):
        if not len(batch_indices):
            continue
        length = len(batch_indices)
        r_w2c = np.stack([cv2.Rodrigues(camera.camera25[i, 3:6])[0]
                          for i in batch_indices])
        orient_c = r_w2c @ world_orient[batch_indices]
        orient_vec = Rotation.from_matrix(orient_c).as_rotvec().astype(np.float32)
        params = dict(body_pose=torch.from_numpy(pose[batch_indices])[None].cuda(),
                      betas=torch.from_numpy(np.tile(beta, (length, 1)))[None].cuda(),
                      global_orient=torch.from_numpy(orient_vec)[None].cuda(),
                      transl=torch.zeros((1, length, 3), device="cuda"))
        with torch.inference_mode():
            _, coco = model.smplx_model(**params)
            native = model.fk_v2(**params)
        coco = coco[0].cpu().numpy()
        native = native[0].cpu().numpy()
        for local, i in enumerate(batch_indices):
            root = native[local, 0].copy()  # true native SMPL-X pelvis index 0
            coco_c = coco[local] - root
            native_c = native[local] - root
            q = world_orient[i]
            body_coco = (orient_c[local].T @ coco_c.T).T
            body_native = (orient_c[local].T @ native_c.T).T
            coco_world[i] = (q @ body_coco.T).T
            native_world[i] = (q @ body_native.T).T
            coco_body[i] = body_coco
            native_body[i] = body_native
            body_world_R[i] = q
            incam_pelvis[i] = root
    output.mkdir(parents=True, exist_ok=True)
    materialized_packet = dict(
                        pts=camera.pts, prior_valid=np.isfinite(coco_world).all(axis=(1,2)),
                        prior_origin=prior_origin, model_slot_pts=model_slot_pts,
                        interpolation_left_source_pts=bracket_left_pts,
                        interpolation_right_source_pts=bracket_right_pts,
                        interpolation_alpha=interpolation_alpha,
                        coco17_world_offset_m=coco_world,
                        native22_world_offset_m=native_world,
                        coco17_body_offset_m=coco_body,
                        native22_body_offset_m=native_body,
                        body_world_R=body_world_R,
                        smplx_native_pelvis_at_zero_transl_m=incam_pelvis,
                        shared_betas=beta,
                        coco17_names=np.array(COCO_NAMES),
                        native22_names=np.array(NATIVE22_NAMES),
                        official_exposure_representative_pts=
                            camera.official_exposure_representative_pts,
                        shot_index=camera.shot_index)
    # Two presentations of one confirmed exposure must not receive different
    # interpolated postures. Normalize the completed cache to its actual
    # representative, retaining the original arrays and missingness provenance.
    # This does not freeze movement across distinct physical exposures.
    try:
        from .exposure_prior import share_exposure_prior
    except ImportError:
        from exposure_prior import share_exposure_prior
    materialized_packet, exposure_audit = share_exposure_prior(
        materialized_packet, camera, int(camera.pts[0]),
        int(camera.pts[-1]) + 256)
    np.savez_compressed(output / "body_prior.npz", **materialized_packet)
    summary = dict(schema="rinkspace.smplx_native_pelvis_prior.v1",
        source_pts_count=n, prior_valid_count=int(materialized_packet["prior_valid"].sum()),
        reference_point="SMPL-X native joint index 0 pelvis",
        coco_joints="original upstream SMPL-X-to-SMPL COCO17 vertex regressor",
        shared_betas=beta.tolist(), median_beta_per_shot=shot_betas,
        shared_beta_source=shared_beta_source, materialized_shots=list(shots),
        body_joints_model_unit="model_metre", camera_world_unit="conditional_scene_unit",
        world_composition="X_world=P_world+s*Q_world*b_local; s shared over programme",
        network_global_translation_used=False,
        explicit_postprocessing=True,
        explicit_postprocessing_method="confirmed_exposure_cached_prior_share",
        upstream_temporal_prior_remains=True,
        network_time_compatibility="unverified",
        camera_status=camera.binding_metadata()["camera_release_status"],
        **_camera_metadata(camera),
        observation_sha256=plan["observations_sha256"],
        body_prior_sha256=_sha256(output / "body_prior.npz"),
        metric_scale_status="unverified_scene_units",
        time_policy=plan["time_policy"],
        materialization_time_policy=("verified_canonical_official_exposure_clock_v1"
            if plan["time_policy"] == official_cadence.POLICY else "source_presentation_clock_legacy"),
        **official_cadence.plan_binding(plan),
        missing_frames_preserved=bool(not np.any(
            ~materialized_packet["original_prior_valid"] & materialized_packet["prior_valid"])),
        original_missing_prior_frames=int((~materialized_packet["original_prior_valid"]).sum()),
        missing_aliases_read_from_valid_cached_representative=int(np.sum(
            ~materialized_packet["original_prior_valid"] & materialized_packet["prior_valid"])),
        exposure_identity_policy="share_actual_cached_representative_including_interpolated_prior",
        exposure_prior_consistency=exposure_audit,
        interpolated_source_frames=int(np.sum(
            materialized_packet["exposure_canonical_prior_origin"] == "interpolated_model_prior")),
        interpolated_source_frame_count_semantics="presentations whose exposure representative has an interpolated prior",
        note="Repeated presentations share their actual exposure's cached prior; different exposures retain their motion. No cross-cut model context, root world translation or trajectory smoothing.")
    _save_json(output / "body_prior.json", summary)
    return summary


def _cli_time_policy(metadata: dict, requested: str | None, stage: str) -> str:
    actual = metadata.get("time_policy")
    if actual not in ("official_unique", "nearest_25hz", "official_25hz"):
        raise ValueError(f"{stage} cache has no recognized actual time_policy")
    if requested is not None and requested != actual:
        raise ValueError(f"{stage} requested time_policy {requested} differs from actual cache policy {actual}")
    print(json.dumps(dict(stage="body_prior_time_policy", requested_stage=stage,
        requested_time_policy=requested, actual_time_policy=actual,
        temporal_plan_sha256=metadata.get("temporal_plan_sha256"))), flush=True)
    return actual


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("plan", "prepare", "infer", "materialize"))
    p.add_argument("--poses", type=Path)
    p.add_argument("--scene", type=Path)
    p.add_argument("--source-shots", type=Path)
    p.add_argument("--release", type=Path,
                   help="Versioned grid/body camera release; omitted for the frozen v15 baseline")
    p.add_argument("--observations", type=Path)
    p.add_argument("--source-video", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--time-policy", choices=("official_unique", "nearest_25hz", "official_25hz"),
                   default=None,
                   help="Required for plan: use official_25hz for new candidates; nearest_25hz is explicit historical reproduction. Other stages verify it against their cache when supplied.")
    p.add_argument("--official-exposure-groups", type=Path)
    p.add_argument("--official-mapping", type=Path)
    p.add_argument("--shot-index", type=int)
    p.add_argument("--prepared", type=Path)
    p.add_argument("--raw-dir", type=Path)
    p.add_argument("--materialize-shots", type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--shared-betas-from", type=Path)
    p.add_argument("--max-window-length", type=int, default=256)
    p.add_argument("--window-overlap", type=int, default=64)
    a = p.parse_args()
    if a.stage == "plan" and a.time_policy is None:
        p.error("plan requires an explicit --time-policy; new candidates use official_25hz, legacy nearest_25hz is only explicit historical reproduction")
    if a.stage == "infer":
        if not a.prepared:
            p.error("infer requires --prepared")
        prepared_metadata = json.loads(a.prepared.with_suffix(".json").read_text(encoding="utf-8"))
        _cli_time_policy(prepared_metadata, a.time_policy, "infer")
        camera = None
        if any((a.poses, a.scene, a.source_shots, a.release)):
            if not all((a.poses, a.scene, a.source_shots)):
                p.error("Explicit infer camera requires --poses, --scene and --source-shots")
            camera = FrozenCameraStream(a.poses, a.scene, a.source_shots, release_path=a.release)
        print(json.dumps(infer_shot(a.prepared, a.output,
                                    max_length=a.max_window_length,
                                     overlap=a.window_overlap, camera=camera)))
        return
    if not all((a.poses, a.scene, a.source_shots)):
        p.error("stage requires --poses, --scene and --source-shots")
    camera = FrozenCameraStream(a.poses, a.scene, a.source_shots, release_path=a.release)
    if a.stage == "plan":
        _cli_time_policy(dict(time_policy=a.time_policy), a.time_policy, "plan")
        if not a.observations:
            p.error("plan requires --observations")
        obs = _load_observations(a.observations, camera)
        official_evidence = None
        if a.time_policy == "official_25hz":
            if not all((a.official_exposure_groups, a.official_mapping)):
                p.error("official_25hz requires --official-exposure-groups and --official-mapping")
            from three_frame_refit.program_tracking.body_tracking.exposure_observations import load_confirmed_evidence
            official_evidence = load_confirmed_evidence(
                a.official_exposure_groups, a.official_mapping, a.source_shots)
        plan = build_slot_plan(camera, obs, time_policy=a.time_policy,
                               observation_sha256=_sha256(a.observations),
                               source_video=a.source_video, official_evidence=official_evidence,
                               official_evidence_paths=(a.official_exposure_groups, a.official_mapping)
                               if official_evidence is not None else None)
        a.output.mkdir(parents=True, exist_ok=True)
        _save_json(a.output / "slot_plan.json", plan)
        print(json.dumps(dict(slots=len(plan["slots"]),
                              by_shot=plan["by_shot_time_delta_s"])))
        return
    plan_path = a.output / "slot_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _cli_time_policy(plan, a.time_policy, a.stage)
    _validate_camera_metadata(plan, camera, "Slot plan", legacy_full=True)
    if a.stage == "prepare":
        if not a.observations or not a.source_video or a.shot_index not in range(1, 10):
            p.error("prepare requires --observations, --source-video, --shot-index 1..9")
        if _sha256(a.observations) != plan["observations_sha256"]:
            raise ValueError("Original H observations changed after slot planning")
        obs = _load_observations(a.observations, camera)
        print(json.dumps(prepare_shot(plan, a.shot_index, obs, camera,
                                      a.source_video, a.output)))
        return
    if not a.raw_dir:
        p.error("materialize requires --raw-dir")
    print(json.dumps(materialize(plan, a.raw_dir, camera, a.output,
                                 shots=tuple(a.materialize_shots),
                                 shared_betas_from=a.shared_betas_from)))


if __name__ == "__main__":
    main()
