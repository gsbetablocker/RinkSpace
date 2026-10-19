"""Version-bound camera bridge for body observations in native source pixels.

The 25 values are [world optical centre C(3), world-to-camera Rodrigues(3),
log(fx), log(fy), cx, cy, skew, OpenCV Brown14].  World coordinates and heights
are conditional rink-scene units, not verified metres.  The virtual rectified
camera has the *same* C and rotation; only its intrinsics change.  No plane
homography or temporal smoothing is used here.
"""
from __future__ import annotations

import hashlib
import json
import operator
from fractions import Fraction
from pathlib import Path
from rinkspace.storage import AssetStore, resolve_asset

PROJECT = Path(__file__).resolve().parents[3]

try:
    from ..projection import project_central_valid
    from ...shot_tracking.pipeline.camera import world_rays
except ImportError:  # direct execution from program_tracking/body_tracking
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    from projection import project_central_valid
    from camera import world_rays

import cv2
import numpy as np
from rinkspace.calibration_companion.wide_angle_camera import brown_central_inverse
from types import SimpleNamespace


FIRST_VALID_PTS = 229114112
END_PTS_EXCLUSIVE = 231705600
PTS_STEP = 256
VALID_FRAME_COUNT = 10123
SOURCE_SHA256 = "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891"
V15_SCENE_SHA256 = "5e4faf7b525e4a16f660a81e45b97aafa6fac39531617927dde11776694cc63d"
V15_POSES_SHA256 = "5ac09663b320fa42e3722a2e31df93bcd200229248e781546baaccd0d1c5e8fb"
DEFAULT_NATIVE_SIZE_WH = (1920, 1080)
CAMERA_PARAMETERIZATION = (
    "camera25_world_center_world_to_camera_rodrigues_log_fxy_cxy_skew_brown14")
CAMERA_RELEASE_SCHEMA = "rinkspace.body-camera-release.v1"
NATIVE_COORDINATE_DOMAIN = "native_brown_pixels"
WORLD_COORDINATE_DOMAIN = "conditional_scene_units"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    path = resolve_asset(PROJECT, path)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_release_binding(poses_path, scene_path, source_shots_path, *,
                           revision: str, status: str,
                           camera_group_status: dict | None = None) -> dict:
    """Describe explicit camera inputs; this records provenance, not accuracy.

    Save this object as a release, or under ``body_camera_binding`` in a grid
    release.  The loader independently checks every file hash and domain.
    """
    source = json.loads(resolve_asset(PROJECT, source_shots_path).read_text(encoding="utf-8-sig"))
    scene = json.loads(resolve_asset(PROJECT, scene_path).read_text(encoding="utf-8-sig"))
    coordinate = scene.get("coordinate_system", {})
    binding = dict(
        schema=CAMERA_RELEASE_SCHEMA, revision=revision, status=status,
        camera_stream_sha256=_sha256(Path(poses_path)),
        scene_sha256=_sha256(Path(scene_path)),
        source_shots_sha256=_sha256(Path(source_shots_path)),
        source_sha256=source["source_sha256_from_timeline"],
        source_time_base=source["source_time_base"],
        native_size_wh=source["source_resolution"],
        coordinate_domain=NATIVE_COORDINATE_DOMAIN,
        camera_parameterization=CAMERA_PARAMETERIZATION,
        world_coordinate_domain=WORLD_COORDINATE_DOMAIN,
        world_coordinate_system_id=coordinate.get("id"))
    if camera_group_status is not None:
        binding["camera_group_status"] = dict(camera_group_status)
    return binding


def _camera(camera25) -> np.ndarray:
    camera = np.asarray(camera25, dtype=np.float64)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("camera25 must contain 25 finite values")
    if camera[2] <= 0 or np.any(np.abs(camera[23:25]) > 1e-12):
        raise ValueError("Camera centre must be above ice; Brown sensor tilt must be zero")
    with np.errstate(over="ignore"):
        focal = np.exp(camera[6:8])
    if not np.isfinite(focal).all() or np.any(focal <= 0):
        raise ValueError("Camera focal lengths must be finite and positive")
    return camera


def _points(values, dimensions: int) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != dimensions:
        raise ValueError(f"Expected an (N,{dimensions}) array")
    return points


def _size(size_wh) -> tuple[int, int]:
    width, height = tuple(size_wh)
    if (not isinstance(width, (int, np.integer)) or
            not isinstance(height, (int, np.integer)) or width <= 0 or height <= 0):
        raise ValueError("Native size must contain positive integer width and height")
    return int(width), int(height)


def _inside(uv: np.ndarray, size_wh) -> np.ndarray:
    width, height = _size(size_wh)
    return (np.isfinite(uv).all(axis=1) &
            (uv[:, 0] >= 0) & (uv[:, 0] < width) &
            (uv[:, 1] >= 0) & (uv[:, 1] < height))


def rectified_intrinsics(camera25) -> np.ndarray:
    """Return K_rect: geometric-mean focal, square pixels, zero skew.

    cx and cy stay at their native values.  The rectified image has the native
    width and height, so severe Brown distortion can leave invalid borders.
    """
    camera = _camera(camera25)
    focal = float(np.exp((camera[6] + camera[7]) / 2.))
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Rectified focal length is invalid")
    return np.array([[focal, 0., camera[8]],
                     [0., focal, camera[9]],
                     [0., 0., 1.]], dtype=np.float64)


def _rect_k(camera25, K_rect) -> np.ndarray:
    if K_rect is None:
        return rectified_intrinsics(camera25)
    k = np.asarray(K_rect, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or
            k[0, 0] <= 0 or not np.isclose(k[0, 0], k[1, 1], rtol=0, atol=1e-9) or
            not np.allclose(k[[0, 1, 2], [1, 0, 0]], 0., rtol=0, atol=1e-12) or
            not np.allclose(k[2], [0., 0., 1.], rtol=0, atol=1e-12)):
        raise ValueError("K_rect must have square pixels, zero skew and standard last row")
    return k


def project_world(world_xyz, camera25) -> tuple[np.ndarray, np.ndarray]:
    """Project conditional scene XYZ into native Brown pixels.

    Optical-domain/depth failures are NaN with valid=False.  The mask describes
    the physical central branch, even when a projected pixel is off-screen.
    """
    camera = _camera(camera25)
    points = _points(world_xyz, 3)
    uv = np.full((len(points), 2), np.nan)
    valid = np.zeros(len(points), bool)
    finite = np.isfinite(points).all(axis=1)
    if finite.any():
        projected, _, good = project_central_valid(camera, points[finite])
        valid[finite] = good
        uv[finite] = projected
    uv[~valid] = np.nan
    return uv, valid


def _precise_rays(camera: np.ndarray, native_uv: np.ndarray,
                  tolerance_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Tight Brown inverse for rays whose legacy .005 px inverse did not close."""
    fx, fy = np.exp(camera[6:8])
    deskewed = native_uv.copy()
    deskewed[:, 0] -= camera[10] * (deskewed[:, 1] - camera[9]) / fy
    k = np.array([[fx, 0., camera[8]], [0., fy, camera[9]], [0., 0., 1.]])
    inverse = brown_central_inverse(
        SimpleNamespace(K_px=k, distortion_opencv=camera[11:23]),
        deskewed, tolerance_px=max(tolerance_px / 10., 1e-9))
    rotation = cv2.Rodrigues(camera[3:6])[0]
    world = np.c_[inverse["normalized_xy"], np.ones(len(native_uv))] @ rotation
    return world, np.asarray(inverse["valid"], bool)


def rays(native_uv, camera25, *, inverse_tolerance_px: float = 1e-4
         ) -> tuple[np.ndarray, np.ndarray]:
    """Invert native Brown pixels to world rays with explicit closure checks.

    Each world ray has camera-Z component 1 rather than Euclidean unit length.
    The returned mask checks positive central Brown branch and native-pixel
    round trip.  A failed inverse is never clamped onto a nearby valid ray.
    """
    camera = _camera(camera25)
    pixels = _points(native_uv, 2)
    if not np.isfinite(inverse_tolerance_px) or inverse_tolerance_px <= 0:
        raise ValueError("inverse_tolerance_px must be finite and positive")
    result = np.full((len(pixels), 3), np.nan)
    valid = np.zeros(len(pixels), bool)
    finite = np.isfinite(pixels).all(axis=1)
    if not finite.any():
        return result, valid
    indices = np.flatnonzero(finite)
    initial_rays, initial_good = world_rays(camera, pixels[indices])
    candidate = np.asarray(initial_rays, np.float64)
    good = np.asarray(initial_good, bool) & np.isfinite(candidate).all(axis=1)
    if good.any():
        check, projected_good = project_world(camera[None, :3] + candidate[good], camera)
        good_indices = np.flatnonzero(good)
        closure = np.linalg.norm(check - pixels[indices[good]], axis=1)
        good[good_indices] = projected_good & (closure <= inverse_tolerance_px)
    retry = ~good
    if retry.any():
        refined, refined_good = _precise_rays(camera, pixels[indices[retry]],
                                               inverse_tolerance_px)
        retry_indices = np.flatnonzero(retry)
        candidate[retry_indices] = refined
        good[retry_indices] = refined_good & np.isfinite(refined).all(axis=1)
        if good[retry_indices].any():
            verify_indices = retry_indices[good[retry_indices]]
            check, projected_good = project_world(camera[None, :3] + candidate[verify_indices], camera)
            closure = np.linalg.norm(check - pixels[indices[verify_indices]], axis=1)
            good[verify_indices] &= projected_good & (closure <= inverse_tolerance_px)
    valid[indices] = good
    result[indices[good]] = candidate[good]
    return result, valid


def native_to_rectified(native_uv, camera25, *, K_rect=None,
                        native_size_wh=DEFAULT_NATIVE_SIZE_WH,
                        inverse_tolerance_px: float = 1e-4
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Map native pixels to the zero-distortion virtual camera's pixels."""
    camera = _camera(camera25)
    pixels = _points(native_uv, 2)
    k = _rect_k(camera, K_rect)
    direction, valid = rays(pixels, camera, inverse_tolerance_px=inverse_tolerance_px)
    valid &= _inside(pixels, native_size_wh)
    result = np.full((len(pixels), 2), np.nan)
    if valid.any():
        rotation = cv2.Rodrigues(camera[3:6])[0]
        cam = direction[valid] @ rotation.T
        valid_indices = np.flatnonzero(valid)
        nz = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 0)
        valid[valid_indices] &= nz
        if nz.any():
            normalized = cam[nz, :2] / cam[nz, 2, None]
            result[valid_indices[nz]] = normalized * k[0, 0] + k[:2, 2]
    valid &= np.isfinite(result).all(axis=1)
    result[~valid] = np.nan
    return result, valid


def rectified_to_native(rectified_uv, camera25, *, K_rect=None,
                        native_size_wh=DEFAULT_NATIVE_SIZE_WH
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Map square-pixel virtual camera coordinates to native Brown pixels."""
    camera = _camera(camera25)
    pixels = _points(rectified_uv, 2)
    k = _rect_k(camera, K_rect)
    result = np.full((len(pixels), 2), np.nan)
    valid = np.zeros(len(pixels), bool)
    finite = np.isfinite(pixels).all(axis=1)
    if finite.any():
        normalized = (pixels[finite] - k[:2, 2]) / k[0, 0]
        cam = np.column_stack((normalized, np.ones(len(normalized))))
        rotation = cv2.Rodrigues(camera[3:6])[0]
        world = camera[:3] + cam @ rotation
        projected, good = project_world(world, camera)
        valid[finite] = good & _inside(projected, native_size_wh)
        result[finite] = projected
    result[~valid] = np.nan
    return result, valid


def rectify_image(bgr, camera25, *, K_rect=None,
                  interpolation: int = cv2.INTER_LINEAR,
                  row_chunk: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rectified image, valid-source mask, K_rect in that image's pixels.

    Pixels outside the native frame or central Brown branch remain black with
    mask=False.  The mask is required when a model crop touches invalid borders.
    """
    image = np.asarray(bgr)
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Expected a nonempty native image")
    height, width = image.shape[:2]
    return rectify_patch(image, camera25, (0, 0, width, height),
                         K_rect=K_rect, interpolation=interpolation,
                         row_chunk=row_chunk)


def rectify_patch(bgr, camera25, rect_xywh, *, K_rect=None,
                  interpolation: int = cv2.INTER_LINEAR,
                  row_chunk: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a virtual-camera patch directly from native pixels.

    ``rect_xywh`` uses integer full rectified-image coordinates.  Returns
    ``(patch, mask, K_rect)`` with K_rect in full rectified-image coordinates.
    A crop-local K is made by subtracting x/y origin from its principal point.
    This avoids rectifying unused full-frame pixels for a body crop.
    """
    image = np.asarray(bgr)
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Expected a nonempty native image")
    if not isinstance(row_chunk, int) or row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    if len(rect_xywh) != 4 or any(not isinstance(x, (int, np.integer)) for x in rect_xywh):
        raise ValueError("rect_xywh must be four integers")
    origin_x, origin_y, patch_width, patch_height = map(int, rect_xywh)
    if patch_width <= 0 or patch_height <= 0:
        raise ValueError("Patch width and height must be positive")
    camera = _camera(camera25)
    k = _rect_k(camera, K_rect)
    source_height, source_width = image.shape[:2]
    map_x = np.full((patch_height, patch_width), -1., np.float32)
    map_y = np.full((patch_height, patch_width), -1., np.float32)
    valid_mask = np.zeros((patch_height, patch_width), bool)
    x = np.arange(origin_x, origin_x + patch_width, dtype=np.float64)
    for start in range(0, patch_height, row_chunk):
        stop = min(patch_height, start + row_chunk)
        xx, yy = np.meshgrid(x, np.arange(origin_y + start,
                                           origin_y + stop, dtype=np.float64))
        rect_uv = np.column_stack((xx.ravel(), yy.ravel()))
        native_uv, good = rectified_to_native(
            rect_uv, camera, K_rect=k,
            native_size_wh=(source_width, source_height))
        good &= _inside(rect_uv, (source_width, source_height))
        native_uv[~good] = -1.
        map_x[start:stop] = native_uv[:, 0].reshape(stop - start, patch_width)
        map_y[start:stop] = native_uv[:, 1].reshape(stop - start, patch_width)
        valid_mask[start:stop] = good.reshape(stop - start, patch_width)
    rectified = cv2.remap(image, map_x, map_y, interpolation,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rectified, valid_mask, k


def height_conditioned_ground(native_uv, camera25, height_scene,
                              *, inverse_tolerance_px: float = 1e-4,
                              min_vertical_fraction: float = 1e-6
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return G(h)=(x,y,0), dG/dh and valid for a pelvis pixel and height.

    This is a conditional ray/height diagnostic.  It does not infer h, ice
    contact, visibility of G, or whether the body point lies inside the rink.
    """
    camera = _camera(camera25)
    pixels = _points(native_uv, 2)
    height = np.asarray(height_scene, dtype=np.float64)
    if height.ndim == 0:
        height = np.full(len(pixels), float(height))
    if height.shape != (len(pixels),):
        raise ValueError("height_scene must be scalar or have one value per pixel")
    if not np.isfinite(min_vertical_fraction) or min_vertical_fraction <= 0:
        raise ValueError("min_vertical_fraction must be finite and positive")
    direction, valid = rays(pixels, camera, inverse_tolerance_px=inverse_tolerance_px)
    vertical = direction[:, 2]
    norm = np.linalg.norm(direction, axis=1)
    valid &= np.isfinite(height) & np.isfinite(norm)
    valid &= np.abs(vertical) > min_vertical_fraction * norm
    ground = np.full((len(pixels), 3), np.nan)
    derivative = np.full((len(pixels), 3), np.nan)
    if valid.any():
        index = np.flatnonzero(valid)
        distance = (height[index] - camera[2]) / vertical[index]
        forward = np.isfinite(distance) & (distance > .05)
        valid[index] &= forward
        if forward.any():
            selected = index[forward]
            ground[selected, :2] = (camera[:2] + distance[forward, None] *
                                    direction[selected, :2])
            ground[selected, 2] = 0.
            derivative[selected, :2] = direction[selected, :2] / vertical[selected, None]
            derivative[selected, 2] = 0.
    return ground, derivative, valid


class FrozenCameraStream:
    """Exact source-PTS camera stream with an explicit version/domain binding.

    Three positional inputs retain the strict frozen-v15 baseline profile.
    A ``release_path`` selects strict release validation for new B/C inputs;
    it never disables provenance validation. ``enforce_v15=False`` remains
    a legacy synthetic-fixture profile and is labelled as such in metadata.
    """

    def __init__(self, poses_path, scene_path, source_shots_path,
                 *, release_path=None, profile: str | None = None,
                 enforce_v15: bool | None = None):
        if enforce_v15 is not None:
            legacy_profile = "v15" if enforce_v15 else "synthetic_fixture"
            if release_path is not None or (profile is not None and profile != legacy_profile):
                raise ValueError("Use a release profile instead of bypassing the v15 check")
            profile = legacy_profile
        self.profile = profile or ("release" if release_path is not None else "v15")
        if self.profile not in {"v15", "release", "synthetic_fixture"}:
            raise ValueError("Camera profile must be v15, release, or synthetic_fixture")
        if self.profile == "release" and release_path is None:
            raise ValueError("A versioned camera release is required for the release profile")
        if release_path is not None and self.profile != "release":
            raise ValueError("Explicit camera releases require the release profile")
        self.poses_path = resolve_asset(PROJECT, poses_path)
        self.scene_path = resolve_asset(PROJECT, scene_path)
        self.source_shots_path = resolve_asset(PROJECT, source_shots_path)
        self.release_path = resolve_asset(PROJECT, release_path) if release_path is not None else None
        self.release_sha256 = _sha256(self.release_path) if self.release_path else None
        self.poses_sha256 = _sha256(self.poses_path)
        self.scene_sha256 = _sha256(self.scene_path)
        self.source_shots_sha256 = _sha256(self.source_shots_path)
        if self.profile == "v15" and (self.poses_sha256 != V15_POSES_SHA256 or
                                      self.scene_sha256 != V15_SCENE_SHA256):
            raise ValueError("Inputs differ from the frozen final v15 camera/scene files")
        self.scene = json.loads(self.scene_path.read_text(encoding="utf-8-sig"))
        self.source = json.loads(self.source_shots_path.read_text(encoding="utf-8-sig"))
        self.source['source_path_original'] = self.source.get('source_path_original', self.source['source_path'])
        self.source['source_path'] = str(AssetStore(PROJECT).source(self.source['source_path']))
        coordinate = self.scene.get("coordinate_system", {})
        if (coordinate.get("handedness") != "right" or
                coordinate.get("z_axis") != "above ice" or
                coordinate.get("ice_z") != 0. or
                coordinate.get("unit") != WORLD_COORDINATE_DOMAIN or
                not isinstance(coordinate.get("id"), str) or not coordinate["id"]):
            raise ValueError("Scene must declare a right-handed, Z-up conditional ice coordinate system")
        self.world_coordinate_system = dict(coordinate)
        self.native_size_wh = _size(self.source["source_resolution"])
        self.time_base = self.source["source_time_base"]
        parsed_time_base = Fraction(self.time_base)
        if parsed_time_base <= 0:
            raise ValueError("Source time base must be positive")
        self._seconds_per_pts = float(parsed_time_base)
        self.release = None
        self.release_binding = None
        if self.release_path:
            self.release = json.loads(self.release_path.read_text(encoding="utf-8-sig"))
            binding = self.release.get("body_camera_binding", self.release)
            expected = dict(
                schema=CAMERA_RELEASE_SCHEMA,
                camera_stream_sha256=self.poses_sha256,
                scene_sha256=self.scene_sha256,
                source_shots_sha256=self.source_shots_sha256,
                source_sha256=self.source["source_sha256_from_timeline"],
                source_time_base=self.time_base,
                native_size_wh=list(self.native_size_wh),
                coordinate_domain=NATIVE_COORDINATE_DOMAIN,
                camera_parameterization=CAMERA_PARAMETERIZATION,
                world_coordinate_domain=WORLD_COORDINATE_DOMAIN,
                world_coordinate_system_id=coordinate["id"])
            if not isinstance(binding, dict):
                raise ValueError("Camera release binding must be an object")
            for field, value in expected.items():
                if binding.get(field) != value:
                    raise ValueError(f"Camera release {field} does not match the localization inputs")
            if any(not isinstance(binding.get(k), str) or not binding[k]
                   for k in ("revision", "status")):
                raise ValueError("Camera release needs an explicit revision and status")
            if self.release.get("model_sha256", self.scene_sha256) != self.scene_sha256:
                raise ValueError("Grid release model_sha256 disagrees with its body camera binding")
            group_status = binding.get("camera_group_status", {})
            if (not isinstance(group_status, dict) or any(
                    not isinstance(k, str) or not isinstance(v, str)
                    for k, v in group_status.items())):
                raise ValueError("camera_group_status must map camera identities to status strings")
            self.release_binding = dict(binding)
        scope = self.source.get("analysis_scope")
        if (not isinstance(scope, dict) or
                scope.get("schema") != "rinkspace.program-analysis-scope.v1" or
                set(scope.get("applies_to", [])) != {"rink_grid_fitting", "body_trajectory"}):
            raise ValueError("Source manifest lacks the shared grid/body analysis_scope")
        try:
            step = operator.index(scope["frame_step_pts"])
            window_start, window_end = map(operator.index, self.source["window_pts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Source PTS window and frame step must be exact integers") from exc
        if step <= 0 or window_end <= window_start or (window_end - window_start) % step:
            raise ValueError("Source window must be an aligned positive PTS range")
        self.frame_step_pts = step
        with np.load(self.poses_path, allow_pickle=False) as packet:
            required = ("pts", "camera25", "shot_index", "camera_id", "status",
                        "source_sha256", "model_sha256",
                        "official_exposure_representative_pts",
                        "source_pose_representative_pts")
            missing = set(required) - set(packet.files)
            if missing:
                raise ValueError(f"Camera packet lacks fields: {sorted(missing)}")
            raw = {name: packet[name].copy() for name in required}
        full_pts = np.arange(window_start, window_end, step, dtype=np.int64)
        if len(full_pts) != self.source["native_frame_count"]:
            raise ValueError("Archived native frame count differs from exact PTS window")
        if raw["pts"].dtype.kind not in "iu" or not np.array_equal(raw["pts"], full_pts):
            raise ValueError("Camera packet must cover exact complete source PTS")
        if any(raw[field].ndim != 0 for field in ("source_sha256", "model_sha256")):
            raise ValueError("Camera packet source and scene hashes must be scalars")
        if (str(raw["source_sha256"].item()) != self.source["source_sha256_from_timeline"] or
                str(raw["model_sha256"].item()) != self.scene_sha256):
            raise ValueError("Camera packet source or scene hash does not match")
        if self.profile == "v15" and str(raw["source_sha256"].item()) != SOURCE_SHA256:
            raise ValueError("Source is not the frozen programme video")
        cameras = np.asarray(raw["camera25"], np.float64)
        if cameras.shape != (len(full_pts), 25) or not np.isfinite(cameras).all():
            raise ValueError("Camera packet needs a finite camera25 for every source PTS")
        if (np.any(cameras[:, 2] <= 0) or
                np.any(np.abs(cameras[:, 23:25]) > 1e-12)):
            raise ValueError("Camera centres or Brown sensor tilt violate the scene convention")
        with np.errstate(over="ignore", under="ignore"):
            focal = np.exp(cameras[:, 6:8])
        if not np.isfinite(focal).all() or np.any(focal <= 0):
            raise ValueError("Camera focal lengths must be finite and positive")
        shot_indices = np.asarray(raw["shot_index"])
        identities = np.asarray(raw["camera_id"]).astype(str)
        statuses = np.asarray(raw["status"]).astype(str)
        if (shot_indices.dtype.kind not in "iu" or shot_indices.shape != full_pts.shape or
                identities.shape != full_pts.shape or statuses.shape != full_pts.shape or
                np.any(identities == "") or np.any(statuses == "")):
            raise ValueError("Camera and shot identities must cover all PTS")
        expected_shots = np.full(len(full_pts), -1, np.int64)
        for index, shot in enumerate(self.source["shots"]):
            member = (full_pts >= shot["start_pts"]) & (full_pts < shot["end_pts"])
            shot_pts = full_pts[member]
            if (not member.any() or np.any(expected_shots[member] >= 0) or
                    shot_pts[0] != shot["start_pts"] or shot_pts[-1] + step != shot["end_pts"] or
                    len(shot_pts) != shot["source_native_frame_count"] or
                    not np.all(shot_indices[member] == index) or
                    not np.all(identities[member] == shot["camera_id"])):
                raise ValueError(f"Camera identity disagrees with reviewed shot {index}")
            expected_shots[member] = index
        if np.any(expected_shots < 0):
            raise ValueError("Reviewed shots must partition the complete source PTS window")
        for name in ("official_exposure_representative_pts",
                     "source_pose_representative_pts"):
            aliases = np.asarray(raw[name])
            if aliases.dtype.kind not in "iu" or aliases.shape != full_pts.shape:
                raise ValueError(f"{name} must cover all source PTS")
            positions = np.searchsorted(full_pts, aliases)
            if (np.any(positions >= len(full_pts)) or
                    not np.array_equal(full_pts[np.minimum(positions, len(full_pts) - 1)], aliases) or
                    not np.array_equal(shot_indices[positions], shot_indices)):
                raise ValueError(f"{name} has nonexistent or cross-shot representatives")
            if name == "source_pose_representative_pts" and not np.array_equal(
                    cameras, cameras[positions]):
                raise ValueError("Source pose aliases do not share identical camera25")
        shot_selection = scope.get("shot_indices", [])
        if (not shot_selection or len(set(shot_selection)) != len(shot_selection) or
                any(not isinstance(i, int) or i < 0 or i >= len(self.source["shots"])
                    for i in shot_selection)):
            raise ValueError("analysis_scope must name distinct reviewed shot indices")
        try:
            active_start = operator.index(scope["start_pts"])
            active_end = operator.index(scope["end_pts_exclusive"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Active PTS endpoints must be exact integers") from exc
        if active_end <= active_start or (active_end - active_start) % step:
            raise ValueError("Active PTS must form an aligned positive range")
        active_pts = np.arange(active_start, active_end, step, dtype=np.int64)
        selected_pts = np.concatenate([full_pts[shot_indices == i] for i in shot_selection])
        if not np.array_equal(selected_pts, active_pts) or len(active_pts) != scope.get("frame_count"):
            raise ValueError("analysis_scope must select its exact ordered source PTS")
        excluded_scope = scope.get("excluded", [])
        excluded_arrays = []
        for entry in excluded_scope:
            try:
                excluded_start = operator.index(entry["start_pts"])
                excluded_end = operator.index(entry["end_pts_exclusive"])
            except (KeyError, TypeError) as exc:
                raise ValueError("Excluded PTS endpoints must be exact integers") from exc
            if excluded_end <= excluded_start or (excluded_end - excluded_start) % step:
                raise ValueError("Excluded PTS must form aligned positive ranges")
            excluded_pts = np.arange(excluded_start, excluded_end, step, dtype=np.int64)
            if len(excluded_pts) != entry.get("frame_count") or not entry.get("reason"):
                raise ValueError("Excluded source ranges need exact counts and an explicit reason")
            excluded_arrays.append(excluded_pts)
        excluded_pts = np.concatenate(excluded_arrays) if excluded_arrays else np.empty(0, np.int64)
        if not np.array_equal(np.sort(np.r_[active_pts, excluded_pts]), full_pts):
            raise ValueError("Active and excluded PTS must partition the original source cache")
        excluded = self.source["shots"][0]
        if self.profile == "v15" and (
                scope.get("start_pts") != FIRST_VALID_PTS or
                scope.get("end_pts_exclusive") != END_PTS_EXCLUSIVE or
                scope.get("frame_step_pts") != PTS_STEP or
                scope.get("frame_count") != VALID_FRAME_COUNT or
                scope.get("shot_indices") != list(range(1, 10)) or
                excluded["end_pts"] != FIRST_VALID_PTS or
                excluded["source_native_frame_count"] != 17 or
                len(excluded_scope) != 1 or
                excluded_scope[0].get("shot_index") != 0 or
                excluded_scope[0].get("start_pts") != int(full_pts[0]) or
                excluded_scope[0].get("end_pts_exclusive") != FIRST_VALID_PTS or
                excluded_scope[0].get("frame_count") != 17 or
                excluded_scope[0].get("reason") != "excluded_pre_programme_by_user" or
                full_pts[-1] != END_PTS_EXCLUSIVE - PTS_STEP):
            raise ValueError("Body analysis_scope differs from the user-authorized 10123 frames")
        keep = ((full_pts >= scope["start_pts"]) &
                (full_pts < scope["end_pts_exclusive"]) &
                np.isin(shot_indices, scope["shot_indices"]))
        if not np.array_equal(full_pts[keep], active_pts):
            raise ValueError("Camera stream does not select the exact active source frames")
        self.analysis_scope = scope.copy()
        self.excluded_pts = full_pts[~keep].copy()
        self.pts = full_pts[keep].copy()
        self.camera25 = cameras[keep].copy()
        self.shot_index = shot_indices[keep].copy()
        self.camera_id = identities[keep].copy()
        self.status = statuses[keep].copy()
        self.official_exposure_representative_pts = np.asarray(
            raw["official_exposure_representative_pts"], dtype=np.int64)[keep].copy()
        self.source_pose_representative_pts = np.asarray(
            raw["source_pose_representative_pts"], dtype=np.int64)[keep].copy()
        self._index = {int(stamp): i for i, stamp in enumerate(self.pts)}
        for array in (self.pts, self.camera25, self.shot_index, self.camera_id,
                      self.status, self.official_exposure_representative_pts,
                      self.source_pose_representative_pts, self.excluded_pts):
            array.flags.writeable = False

    def _at(self, source_pts: int) -> int:
        try:
            stamp = operator.index(source_pts)
        except TypeError as exc:
            raise TypeError("source_pts must be an exact integer PTS") from exc
        try:
            return self._index[stamp]
        except KeyError as exc:
            raise KeyError(f"No included source frame at PTS {stamp}") from exc

    def camera_at(self, source_pts: int) -> np.ndarray:
        """Return a copy of the physical camera for an exact included PTS."""
        return self.camera25[self._at(source_pts)].copy()

    def binding_metadata(self) -> dict:
        """Expose version, source, world axes and pixel domain for downstream outputs."""
        binding = self.release_binding or {}
        return {
            "camera_profile": self.profile,
            "camera_revision": binding.get("revision", "final_v15" if self.profile == "v15"
                                           else "synthetic_fixture"),
            "camera_release_status": binding.get("status", "frozen_v15_unaccepted"
                                                if self.profile == "v15" else "synthetic_fixture"),
            "camera_group_status": dict(binding.get("camera_group_status", {})),
            "camera_stream_sha256": self.poses_sha256,
            "scene_sha256": self.scene_sha256,
            "source_shots_sha256": self.source_shots_sha256,
            "source_sha256": self.source["source_sha256_from_timeline"],
            "release_path": str(self.release_path.resolve()) if self.release_path else None,
            "release_sha256": self.release_sha256,
            "source_time_base": self.time_base,
            "frame_step_pts": self.frame_step_pts,
            "native_size_wh": list(self.native_size_wh),
            "coordinate_domain": NATIVE_COORDINATE_DOMAIN,
            "camera_parameterization": CAMERA_PARAMETERIZATION,
            "world_coordinate_domain": WORLD_COORDINATE_DOMAIN,
            "world_coordinate_system": dict(self.world_coordinate_system),
            "metric_scale_status": "unverified_scene_units",
        }

    def metadata_at(self, source_pts: int) -> dict:
        """Expose distinct exposure and shared-camera identities without merging bodies."""
        i = self._at(source_pts)
        stamp = int(self.pts[i])
        return {
            **self.binding_metadata(),
            "source_pts": stamp,
            "source_time_base": self.time_base,
            "source_time_s": stamp * self._seconds_per_pts,
            "programme_time_s": ((stamp - int(self.source["window_pts"][0])) *
                                  self._seconds_per_pts),
            "shot_index": int(self.shot_index[i]),
            "camera_id": str(self.camera_id[i]),
            "camera_status": str(self.status[i]),
            "official_exposure_representative_pts": int(
                self.official_exposure_representative_pts[i]),
            "source_pose_representative_pts": int(self.source_pose_representative_pts[i]),
            "source_sha256": self.source["source_sha256_from_timeline"],
            "scene_sha256": self.scene_sha256,
            "poses_sha256": self.poses_sha256,
            "source_shots_sha256": self.source_shots_sha256,
        }
