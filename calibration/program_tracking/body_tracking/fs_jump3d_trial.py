"""Lossless full-trial FS/AP interchange; no data, camera or truth is inferred.

Schema fs_jump3d_trial_v1 is illustrated by the synthetic test fixture.
Arrays source_frame/source_pts/world_points/valid_mask/names may instead be
{"npz": "relative.npz", "key": "array_name", "sha256": "..."}; pickle is forbidden.
All original frames/names must match source_frame_range/point_inventory.
These declarations need original-file checking: validation is not truth QA.

Coordinates: world_points keep metric_unit and world_frame_id. Cameras use
Xc = R_w2c @ Xworld + t_w2c (or t=-R_w2c @ C_world), in camera.metric_unit.
K/distortion precede affine pixel_transform. Evaluation applies one fixed
trial-wide declared_once rigid transform after conversion to metres; cameras
stay in the original world frame. video_sync.source_pts explicitly pairs each
trial sample with actual video PTS; repeats represent the same video exposure,
not independent evidence. No FPS clock, marker remapping or fitting.
Theia points are multiview references, not blade/contact measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA = "fs_jump3d_trial_v1"
UNITS_TO_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}


class TrialContractError(ValueError):
    """Missing, ambiguous or inconsistent declared trial input."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrialContractError(message)


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field}: nonempty string required")
    return value


def _mapping(value: Any, field: str) -> Mapping:
    _require(isinstance(value, Mapping), f"{field}: object required")
    return value


def _field(data: Mapping, key: str) -> Any:
    _require(key in data, f"missing {key}")
    return data[key]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _array(value: Any, field: str, shape: tuple, kind: str = "float") -> np.ndarray:
    try:
        a = np.asarray(value)
    except (ValueError, TypeError) as e:
        raise TrialContractError(f"{field}: invalid array") from e
    _require(a.shape == shape, f"{field}: expected shape {shape}, got {a.shape}")
    if kind == "int":
        _require(a.dtype.kind in "iu", f"{field}: integer array required (no rounding)")
    elif kind == "bool":
        _require(a.dtype.kind == "b", f"{field}: boolean mask required")
    else:
        _require(a.dtype.kind in "iuf", f"{field}: real numeric array required")
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _finite_array(value: Any, field: str, shape: tuple) -> np.ndarray:
    a = _array(value, field, shape)
    _require(bool(np.isfinite(a).all()), f"{field}: finite values required")
    return a


def _rotation(value: Any, field: str) -> np.ndarray:
    a = _finite_array(value, field, (3, 3))
    _require(bool(np.allclose(a.T @ a, np.eye(3), atol=1e-7, rtol=0))
             and bool(np.isclose(np.linalg.det(a), 1, atol=1e-7, rtol=0)),
             f"{field}: fixed proper orthonormal rotation required")
    return a


def _time_base(value: Any, field: str) -> Fraction:
    a = _array(value, field, (2,), "int")
    _require(bool((a > 0).all()), f"{field}: positive numerator/denominator required")
    return Fraction(int(a[0]), int(a[1]))


def _unit(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value in UNITS_TO_METRES, f"{field}: explicit m/cm/mm required")
    return value


def _sha256(value: Any, field: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdefABCDEF" for c in value), f"{field}: SHA256 required")
    return value.lower()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class InputView:
    """Single system input: no 3D reference or holdout-camera material."""
    trial_id: str
    subject_id: str
    split: str
    source_frame: np.ndarray
    source_pts: np.ndarray
    time_base: Fraction
    camera: Mapping
    sources: Mapping


@dataclass(frozen=True)
class CompleteTrial:
    """Evaluation record retaining original point slots and all input metadata."""
    metadata: Mapping
    source_frame: np.ndarray
    source_pts: np.ndarray
    time_base: Fraction
    world_points: np.ndarray
    names: tuple[str, ...]
    valid_mask: np.ndarray
    cameras: tuple[Mapping, ...]
    rotation: np.ndarray
    translation_m: np.ndarray

    def evaluation_points_m(self) -> np.ndarray:
        """Apply the one declared transform, leaving original points intact."""
        return (self.world_points * UNITS_TO_METRES[self.metadata["metric_unit"]]) @ self.rotation.T + self.translation_m

    def input_view(self) -> InputView:
        camera = next(c for c in self.cameras if c["role"] == "input")
        keys = (camera["calibration_source_key"], camera["video_sync"]["video_source_key"],
                camera["video_sync"]["sync_source_key"])
        sources = MappingProxyType({k: self.metadata["source_hashes"][k] for k in keys})
        return InputView(self.metadata["trial_id"], self.metadata["subject_id"], self.metadata["split"],
                         self.source_frame, self.source_pts, self.time_base, camera, sources)

    def holdout_camera_ids(self) -> tuple[str, ...]:
        return tuple(c["camera_id"] for c in self.cameras if c["role"] == "holdout")

    def project_reference(self, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Evaluation-only projection in declared output pixels.

        Returns pixels[T,P,2], usable[T,P]; invalid/behind-camera points are NaN.
        Geometric closure alone cannot establish real cross-view accuracy.
        """
        camera = next((c for c in self.cameras if c["camera_id"] == camera_id), None)
        _require(camera is not None, f"unknown camera_id: {camera_id}")
        r = np.asarray(camera["R_w2c"], dtype=float)
        t = (np.asarray(camera["t_w2c"], dtype=float) if "t_w2c" in camera
             else -r @ np.asarray(camera["C_world"], dtype=float))
        scale = UNITS_TO_METRES[self.metadata["metric_unit"]] / UNITS_TO_METRES[camera["metric_unit"]]
        xyz = (self.world_points * scale) @ r.T + t
        valid = self.valid_mask & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            xy = xyz[..., :2] / xyz[..., 2:3]
            d = camera["distortion"]
            if d["model"] == "opencv_brown_conrady":
                k1, k2, p1, p2, *rest = d["coefficients"]
                k3 = rest[0] if rest else 0.0
                x, y = xy[..., 0], xy[..., 1]
                r2 = x*x + y*y
                radial = 1 + r2*(k1 + r2*(k2 + r2*k3))
                xy = np.stack((x*radial + 2*p1*x*y + p2*(r2 + 2*x*x),
                               y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y), axis=-1)
            hom = np.concatenate((xy, np.ones(xy.shape[:-1] + (1,))), axis=-1)
            pixels = hom @ np.asarray(camera["K"], dtype=float).T @ np.asarray(camera["pixel_transform"], dtype=float).T
            pixels = pixels[..., :2] / pixels[..., 2:3]
        valid &= np.isfinite(pixels).all(axis=-1)
        return np.where(valid[..., None], pixels, np.nan), valid


def validate_trial(payload: Mapping) -> CompleteTrial:
    """Validate declarations; load_trial separately checks original-file hashes.

    physical_trial_id and trial_aliases identify FS/AP duplicates from source
    evidence, not per-camera/crop IDs. Call validate_trial_splits on the complete
    registry before fitting. A single record cannot prove absence of leakage.
    """
    p = _mapping(payload, "trial")
    _require(_field(p, "schema_version") == SCHEMA, f"schema_version must be {SCHEMA}")
    for key in ("trial_id", "physical_trial_id", "subject_id", "dataset_id", "world_frame_id"):
        _text(_field(p, key), key)
    _require(_field(p, "complete_trial") is True, "complete_trial must be true")
    _require(_field(p, "split") in ("train", "validation", "test"), "split: train/validation/test required")
    aliases = _field(p, "trial_aliases")
    _require(isinstance(aliases, (list, tuple)) and bool(aliases), "trial_aliases required, including own dataset/trial")
    alias_keys = []
    for alias in aliases:
        a = _mapping(alias, "trial_aliases item")
        alias_keys.append((_text(_field(a, "dataset_id"), "alias.dataset_id"), _text(_field(a, "trial_id"), "alias.trial_id")))
    _require((p["dataset_id"], p["trial_id"]) in alias_keys, "trial_aliases must include this dataset/trial")
    _unit(_field(p, "metric_unit"), "metric_unit")
    names_raw = _field(p, "names")
    _require(isinstance(names_raw, (list, tuple, np.ndarray)), "names: ordered original point names required")
    names = tuple(_text(n, "names item") for n in names_raw)
    _require(bool(names), "names must not be empty")
    inventory = _mapping(_field(p, "point_inventory"), "point_inventory")
    _require(_field(inventory, "selection") == "all_source_points", "point_inventory must retain all_source_points")
    _require(tuple(_field(inventory, "original_names")) == names, "names must exactly retain original_names, including order/duplicates")
    frames_raw = _field(p, "source_frame")
    _require(isinstance(frames_raw, (list, tuple, np.ndarray)) and len(frames_raw) > 0, "source_frame: nonempty original frame sequence required")
    count = len(frames_raw)
    frames = _array(frames_raw, "source_frame", (count,), "int")
    frame_range = _array(_field(p, "source_frame_range"), "source_frame_range", (2,), "int")
    _require(int(frame_range[0]) >= 0 and int(frame_range[1]) >= int(frame_range[0]), "source_frame_range: inclusive nonnegative range required")
    _require(count == int(frame_range[1])-int(frame_range[0])+1
             and bool(np.array_equal(frames, np.arange(int(frame_range[0]), int(frame_range[1])+1))),
             "source_frame must retain every original frame in complete trial (use mask for missing points)")
    pts = _array(_field(p, "source_pts"), "source_pts", (count,), "int")
    _require(all(int(b) > int(a) for a, b in zip(pts, pts[1:])), "source_pts must strictly increase")
    base = _time_base(_field(p, "time_base"), "time_base")
    world = _array(_field(p, "world_points"), "world_points", (count, len(names), 3))
    mask = _array(_field(p, "valid_mask"), "valid_mask", (count, len(names)), "bool")
    _require(not bool(np.isinf(world).any()) and bool(np.isfinite(world[mask]).all()), "world_points: valid points must be finite; NaN only permitted under false mask")
    sources = _mapping(_field(p, "source_hashes"), "source_hashes")
    for key, source in sources.items():
        _text(key, "source key")
        _mapping(source, f"source_hashes.{key}")
        _text(_field(source, "path"), f"source_hashes.{key}.path")
        _sha256(_field(source, "sha256"), f"source_hashes.{key}.sha256")
        _text(_field(source, "kind"), f"source_hashes.{key}.kind")

    def source_ref(key: Any, kind: str) -> None:
        _require(isinstance(key, str) and key in sources, f"missing {kind} source reference: {key}")
        _require(sources[key]["kind"] == kind, f"{key}: expected source kind {kind}")

    source_ref(_field(p, "world_source_key"), "world_points")
    source_ref(_field(p, "identity_source_key"), "identity")
    licence = _mapping(_field(p, "licence_origin"), "licence_origin")
    for key in ("dataset", "licence", "source_uri"):
        _text(_field(licence, key), f"licence_origin.{key}")
    source_ref(_field(licence, "terms_source_key"), "licence")
    transform = _mapping(_field(p, "world_transform"), "world_transform")
    _require(_field(transform, "method") == "declared_once", "world_transform: only declared_once, no per-frame fitting")
    _text(_field(transform, "declaration_id"), "world_transform.declaration_id")
    _require(_field(transform, "source_world_frame") == p["world_frame_id"], "world_transform source frame mismatch")
    _text(_field(transform, "evaluation_world_frame"), "world_transform.evaluation_world_frame")
    rotation = _rotation(_field(transform, "R_source_to_evaluation"), "world_transform.R_source_to_evaluation")
    translation = _finite_array(_field(transform, "translation_m"), "world_transform.translation_m", (3,))
    raw_cameras = _field(p, "cameras")
    _require(isinstance(raw_cameras, (list, tuple)) and len(raw_cameras) >= 2, "complete multiview trial requires input and holdout cameras")
    camera_ids, input_ids = [], []
    for cam in raw_cameras:
        c = _mapping(cam, "camera")
        camera_id = _text(_field(c, "camera_id"), "camera.camera_id")
        camera_ids.append(camera_id)
        role = _field(c, "role")
        _require(role in ("input", "holdout"), "camera.role: input or holdout required")
        if role == "input":
            input_ids.append(camera_id)
        _require(_field(c, "world_frame_id") == p["world_frame_id"], "camera and raw world frame mismatch")
        _unit(_field(c, "metric_unit"), "camera.metric_unit")
        _rotation(_field(c, "R_w2c"), "camera.R_w2c")
        _require(("t_w2c" in c) != ("C_world" in c), "camera requires exactly one of t_w2c/C_world")
        key = "t_w2c" if "t_w2c" in c else "C_world"
        _finite_array(c[key], f"camera.{key}", (3,))
        k = _finite_array(_field(c, "K"), "camera.K", (3, 3))
        _require(k[0, 0] > 0 and k[1, 1] > 0 and bool(np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0))
                 and k[1, 0] == 0, "camera.K: upper triangular intrinsics with positive focal lengths required")
        pix = _finite_array(_field(c, "pixel_transform"), "camera.pixel_transform", (3, 3))
        _require(bool(np.allclose(pix[2], [0, 0, 1], atol=1e-12, rtol=0)) and abs(np.linalg.det(pix[:2, :2])) > 1e-12,
                 "pixel_transform: explicit invertible affine transform required")
        size = _array(_field(c, "image_size"), "camera.image_size", (2,), "int")
        _require(bool((size > 0).all()), "camera.image_size: positive output width/height required")
        d = _mapping(_field(c, "distortion"), "camera.distortion")
        model, coefficients = _field(d, "model"), _field(d, "coefficients")
        _require(isinstance(coefficients, (list, tuple)), "distortion.coefficients: ordered list required")
        _require((model == "none" and len(coefficients) == 0) or
                 (model == "opencv_brown_conrady" and len(coefficients) in (4, 5)),
                 "distortion: none/[] or opencv_brown_conrady/[k1,k2,p1,p2,(k3)] required")
        _finite_array(coefficients, "distortion.coefficients", (len(coefficients),))
        source_ref(_field(c, "calibration_source_key"), "camera")
        sync = _mapping(_field(c, "video_sync"), "video_sync")
        _require(_field(sync, "mapping") == "explicit_pts", "video_sync.mapping: explicit_pts required; no nominal FPS synthesis")
        _require(_field(sync, "evidence_kind") in ("measured", "source_metadata"), "video_sync: measured/source_metadata evidence required")
        video_pts = _array(_field(sync, "source_pts"), "video_sync.source_pts", (count,), "int")
        _require(all(int(b) >= int(a) for a, b in zip(video_pts, video_pts[1:])), "video_sync PTS must not decrease")
        _time_base(_field(sync, "time_base"), "video_sync.time_base")
        uncertainty = _field(sync, "uncertainty_seconds")
        _require(isinstance(uncertainty, (int, float)) and not isinstance(uncertainty, bool)
                 and np.isfinite(uncertainty) and uncertainty >= 0, "video_sync: nonnegative measured/declared uncertainty required")
        source_ref(_field(sync, "video_source_key"), "video")
        source_ref(_field(sync, "sync_source_key"), "sync")
    _require(len(set(camera_ids)) == len(camera_ids), "camera_id must be unique")
    _require(len(input_ids) == 1 and input_ids[0] == _field(p, "input_camera_id"), "exactly one declared input_camera_id; all others holdout")
    normalized = json.loads(json.dumps(p, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else a))
    return CompleteTrial(_freeze(normalized), frames, pts, base, world, names, mask,
                         tuple(_freeze(c) for c in normalized["cameras"]), rotation, translation)


def load_trial(manifest: str | Path, *, verify_sources: bool = False) -> CompleteTrial:
    """Load JSON plus hashed NPZ arrays, without C3D readers or downloads.

    NPZ hashes always checked; verify_sources=True checks all original files.
    Default validation does not assert originals exist. Relative source paths
    resolve beside the manifest, never by guessing an asset location.
    """
    manifest = Path(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    _mapping(payload, "manifest")
    for key in ("source_frame", "source_pts", "world_points", "valid_mask", "names"):
        descriptor = payload.get(key)
        if isinstance(descriptor, Mapping):
            npz_name = _text(_field(descriptor, "npz"), f"{key}.npz")
            _require(not Path(npz_name).is_absolute() and not Path(npz_name).drive
                     and ":" not in npz_name, "npz: manifest-relative local path required")
            npz_path = (manifest.parent / npz_name).resolve()
            _require(npz_path.is_relative_to(manifest.parent.resolve()), "npz must stay inside manifest directory")
            _require(_hash_file(npz_path) == _sha256(_field(descriptor, "sha256"), f"{key}.sha256"), f"{key}: NPZ SHA256 mismatch")
            with np.load(npz_path, allow_pickle=False) as arrays:
                name = _text(_field(descriptor, "key"), f"{key}.key")
                _require(name in arrays, f"{key}: missing NPZ key {name}")
                payload[key] = arrays[name].tolist()
    trial = validate_trial(payload)
    if verify_sources:
        for key, source in trial.metadata["source_hashes"].items():
            source_path = Path(source["path"])
            if not source_path.is_absolute():
                source_path = manifest.parent / source_path
            _require(_hash_file(source_path) == source["sha256"].lower(), f"{key}: original source SHA256 mismatch")
    return trial


def validate_trial_splits(trials: Iterable[CompleteTrial]) -> None:
    """Reject role leakage across views, versions and declared FS/AP aliases.

    Run over the full registry, including both datasets. Unknown equivalences
    cannot be inferred from renamed files; source identity evidence is required.
    """
    owners: dict[tuple, str] = {}
    for trial in trials:
        p = trial.metadata
        keys = [("physical", p["physical_trial_id"])]
        keys.extend(("alias", a["dataset_id"], a["trial_id"]) for a in p["trial_aliases"])
        keys.append(("raw", p["source_hashes"][p["world_source_key"]]["sha256"].lower(),
                     p["subject_id"], tuple(p["source_frame_range"])))
        for key in keys:
            previous = owners.setdefault(key, p["split"])
            _require(previous == p["split"], f"whole physical trial split leakage: {key}: {previous}/{p['split']}")
