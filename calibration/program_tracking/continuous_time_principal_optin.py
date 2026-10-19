"""Optional same-camera temporal principal point on the existing physical15 rig.

Programme principal drift uses the *same* cubic shot basis as residual C, but
new disjoint Vector11 controls. One independent static control belongs to each
source reference exposure. Only physical columns 8/9 (cx/cy divided by 1000)
are changed. This module changes no original camera state, UV or prior target.

The existing native source-edge bridge does not know these new node roles.
Call ``require_compatible_source_edge_backend`` before building a graph; the
existing Python physical15 adapter remains the supported path.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path

import numpy as np

from continuous_rig_camera import ContinuousRigCameraModel


SCHEMA = "rinkspace.optional-time-principal-physical15.v1"
LIVE_COLUMNS = (8, 9)
DEFAULT_AMPLITUDE_SIGMA_PX = 20.
DEFAULT_ACCELERATION_SIGMA_PX_S15 = 20.


def _state11(value, label: str) -> np.ndarray:
    value = np.asarray(value, np.float64)
    if value.shape != (11,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite Vector11")
    return value


def _shot_mapping(raw: Mapping[int, Sequence[int]]) -> dict[int, tuple[int, ...]]:
    return {int(shot): tuple(map(int, keys)) for shot, keys in raw.items()}


class TimePrincipalRigCameraModel(ContinuousRigCameraModel):
    """Physical15 rig with per-shot cx/cy spline and source-reference offsets."""

    requires_time_principal_native_edge_roles = True

    def __init__(self, *args, principal_control_keys_by_shot, **kwargs):
        super().__init__(*args, **kwargs)
        mapping = _shot_mapping(principal_control_keys_by_shot)
        if set(mapping) != set(self.shot_splines):
            raise ValueError("Every programme/reference shot needs principal controls")
        all_new = []
        for shot, spline in self.shot_splines.items():
            keys = mapping[shot]
            if len(keys) != len(spline.control_keys) or not keys:
                raise ValueError(f"Principal and C basis sizes differ for shot {shot}")
            all_new.extend(keys)
        old_dependencies = {key for pts in self.frames
                            for key in ContinuousRigCameraModel.dependencies(self, pts)}
        if len(set(all_new)) != len(all_new) or set(all_new) & old_dependencies:
            raise ValueError("Principal nodes must be unique and disjoint from the old rig")
        self.principal_control_keys_by_shot = mapping
        self._principal_for_c = {
            shot: dict(zip(self.shot_splines[shot].control_keys, mapping[shot]))
            for shot in mapping
        }

    @classmethod
    def from_rig(cls, rig: ContinuousRigCameraModel, mapping):
        if type(rig) is not ContinuousRigCameraModel:
            raise TypeError("Only the original physical15 rig is supported by this opt-in")
        return cls(rig.shot_splines, rig.frames, rig.lens_base_keys,
                   rig.lens_slope_keys, rig.lens_zoom_reference,
                   mount_keys=rig.mount_keys, offset_keys=rig.offset_keys,
                   offset_slope_keys=rig.offset_slope_keys,
                   brown_keys=rig.brown_keys,
                   brown_slope_keys=rig.brown_slope_keys,
                   principal_control_keys_by_shot=mapping)

    def _principal_pairs(self, pts: int) -> tuple[tuple[int, float], ...]:
        binding = self.frames[int(pts)]
        canonical = binding.pts if binding.spline_pts is None else binding.spline_pts
        return tuple((self._principal_for_c[binding.shot_id][c_key], weight)
                     for c_key, weight in self.shot_splines[binding.shot_id]
                     .basis_pairs(canonical))

    def dependencies(self, pts: int) -> tuple[int, ...]:
        original = super().dependencies(pts)
        return (*original, *(key for key, _ in self._principal_pairs(pts)))

    def _physical(self, pts, states, jacobian):
        original = super()._physical(pts, states, jacobian)
        physical, blocks = original if jacobian else (original, None)
        if physical.shape != (15,):
            raise ValueError("Time-principal model requires physical15 Brown4")
        for key, weight in self._principal_pairs(pts):
            node = _state11(states[key], "principal control")
            physical[8:10] += weight * node[8:10]
            if jacobian:
                block = np.zeros((15, 11), np.float64)
                block[8, 8] = weight
                block[9, 9] = weight
                blocks[key] = block
        return (physical, blocks) if jacobian else physical


def _same_shared_rig(programme: ContinuousRigCameraModel,
                     references: ContinuousRigCameraModel) -> str:
    if type(programme) is not ContinuousRigCameraModel or type(references) is not ContinuousRigCameraModel:
        raise TypeError("Prepare this opt-in from the original programme/reference rig models")
    fields = ("lens_base_keys", "lens_slope_keys", "lens_zoom_reference",
              "mount_keys", "offset_keys", "offset_slope_keys",
              "brown_keys", "brown_slope_keys")
    if any(getattr(programme, name) != getattr(references, name) for name in fields):
        raise ValueError("Programme and references must share the same rig optical nodes")
    if set(programme.frames) & set(references.frames):
        raise ValueError("Programme and reference source PTS must be disjoint")
    if set(programme.shot_splines) & set(references.shot_splines):
        raise ValueError("Programme and reference shot ids must be disjoint")
    cameras = set(programme.lens_base_keys)
    if len(cameras) != 1 or cameras != set(references.lens_base_keys):
        raise ValueError("One same-camera group is required")
    if not programme.brown_keys or not references.brown_keys:
        raise ValueError("Source factors require the current shared Brown4 physical15 rig")
    return next(iter(cameras))


def _original_dependencies(model: ContinuousRigCameraModel) -> set[int]:
    return {key for pts in model.frames for key in model.dependencies(pts)}


def prepare_time_principal_rig(
        programme_rig: ContinuousRigCameraModel,
        reference_rig: ContinuousRigCameraModel,
        reference_augmented_initial: Mapping[int, np.ndarray], *,
        reserved_keys: Sequence[int] = (),
        ) -> tuple[TimePrincipalRigCameraModel, TimePrincipalRigCameraModel,
                   dict[int, np.ndarray], dict]:
    """Allocate new nodes *after* old reference nodes; return both model views.

    ``reference_augmented_initial`` is the original programme+reference mapping.
    Pass original captured-state keys as ``reserved_keys`` so allocation cannot
    collide with an inactive external graph state.
    """
    camera_id = _same_shared_rig(programme_rig, reference_rig)
    initial = {int(key): _state11(value, "original initial state").copy()
               for key, value in reference_augmented_initial.items()}
    old_dependencies = (_original_dependencies(programme_rig)
                        | _original_dependencies(reference_rig))
    if not old_dependencies <= set(initial):
        raise ValueError("Original programme/reference graph state is incomplete")
    reserved = (set(initial) | set(map(int, reserved_keys))
                | set(programme_rig.frames) | set(reference_rig.frames))
    if not reserved:
        raise ValueError("Original graph is empty")
    next_key = max(reserved) + 1
    programme_map: dict[int, tuple[int, ...]] = {}
    reference_map: dict[int, tuple[int, ...]] = {}
    all_new: list[int] = []
    for rig, mapping in ((programme_rig, programme_map),
                         (reference_rig, reference_map)):
        for shot in sorted(rig.shot_splines):
            count = len(rig.shot_splines[shot].control_keys)
            keys = tuple(range(next_key, next_key + count))
            next_key += count
            mapping[shot] = keys
            all_new.extend(keys)
    augmented = dict(initial)
    for key in all_new:
        augmented[key] = np.zeros(11, np.float64)
    programme_model = TimePrincipalRigCameraModel.from_rig(programme_rig,
                                                           programme_map)
    reference_model = TimePrincipalRigCameraModel.from_rig(reference_rig,
                                                           reference_map)
    programme_times = {}
    for shot, spline in programme_rig.shot_splines.items():
        knots = tuple(map(float, spline.knot_pts))
        times = [float(np.mean(knots[i+1:i+4]) / 15360.)
                 for i in range(len(spline.control_keys))]
        if (len(times) != len(programme_map[shot])
                or not np.isfinite(times).all()
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError("Programme principal acceleration needs distinct knot times")
        programme_times[str(shot)] = times
    for pts in (*programme_rig.frames, *reference_rig.frames):
        old_model = (programme_rig if pts in programme_rig.frames else reference_rig)
        new_model = (programme_model if pts in programme_rig.frames else reference_model)
        if not np.array_equal(old_model.physical_state(pts, initial),
                              new_model.physical_state(pts, augmented)):
            raise RuntimeError(f"Zero principal nodes changed physical15 at PTS {pts}")
    metadata = dict(
        schema=SCHEMA, camera_id=camera_id,
        implementation_path=str(Path(__file__).resolve()),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        programme_shot_ids=sorted(programme_map),
        reference_shot_ids=sorted(reference_map),
        programme_principal_control_keys_by_shot={str(k): list(v)
                                                   for k, v in programme_map.items()},
        reference_principal_control_keys_by_shot={str(k): list(v)
                                                   for k, v in reference_map.items()},
        original_c_control_keys_by_shot={str(shot): list(spline.control_keys)
            for rig in (programme_rig, reference_rig)
            for shot, spline in rig.shot_splines.items()},
        programme_acceleration_times_seconds=programme_times,
        original_graph_key_max=max(reserved), new_key_min=min(all_new),
        new_key_max=max(all_new), new_keys=all_new,
        programme_new_node_count=sum(map(len, programme_map.values())),
        reference_new_node_count=sum(map(len, reference_map.values())),
        effective_new_parameter_count=2 * len(all_new),
        live_columns=list(LIVE_COLUMNS), state_to_pixel_scale=1000.,
        amplitude_sigma_px_default=DEFAULT_AMPLITUDE_SIGMA_PX,
        acceleration_sigma_px_s15_default=DEFAULT_ACCELERATION_SIGMA_PX_S15,
        zero_new_nodes_physical15_equal=True, original_initial_modified=False,
        first_frame_fixed=False, world_optical_center_fixed=False,
        principal_point_fixed=False, source_uv_modified=False,
        old_native_source_edge_backend="reject; or replace complete factor with Python physical15 callback",
    )
    return programme_model, reference_model, augmented, metadata


def reconstruct_time_principal_rigs(programme_rig: ContinuousRigCameraModel,
                                    reference_rig: ContinuousRigCameraModel,
                                    metadata: Mapping):
    """Rebuild the two derived models from frozen shot-to-node metadata."""
    camera_id = _same_shared_rig(programme_rig, reference_rig)
    if metadata.get("schema") != SCHEMA or metadata.get("camera_id") != camera_id:
        raise ValueError("Frozen principal model metadata or camera id changed")
    if metadata.get("implementation_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Frozen principal implementation changed")
    programme_map = _shot_mapping(metadata["programme_principal_control_keys_by_shot"])
    reference_map = _shot_mapping(metadata["reference_principal_control_keys_by_shot"])
    if set(programme_map) != set(programme_rig.shot_splines) or set(reference_map) != set(reference_rig.shot_splines):
        raise ValueError("Frozen principal shot mapping changed")
    result = (TimePrincipalRigCameraModel.from_rig(programme_rig, programme_map),
              TimePrincipalRigCameraModel.from_rig(reference_rig, reference_map))
    observed = [key for mapping in (programme_map, reference_map)
                for keys in mapping.values() for key in keys]
    if observed != list(metadata["new_keys"]):
        raise ValueError("Frozen principal node order changed")
    return result


def require_compatible_source_edge_backend(manager) -> None:
    """Prevent the old native bridge from silently ignoring new principal roles."""
    if manager is not None and not getattr(manager, "supports_time_principal_roles", False):
        raise ValueError("Old native source-edge bridge lacks time-principal node roles; use full Python factor fallback")


def add_time_principal_priors(graph, metadata: Mapping, *,
                              amplitude_sigma_px: float = DEFAULT_AMPLITUDE_SIGMA_PX,
                              acceleration_sigma_px_s15: float = DEFAULT_ACCELERATION_SIGMA_PX_S15,
                              affine_priors=None) -> dict:
    """Add soft zero-amplitude and nonuniform-knot acceleration priors.

    These are experimental regularizers, not source measurements. References
    each receive only a static amplitude prior; none is fixed to zero.
    """
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Time-principal prior metadata schema changed")
    if (not np.isfinite(amplitude_sigma_px) or amplitude_sigma_px <= 0
            or not np.isfinite(acceleration_sigma_px_s15)
            or acceleration_sigma_px_s15 <= 0):
        raise ValueError("Principal prior scales must be positive and finite")
    from fit_continuous_camera import dummy_factor, acceleration_factor

    programme_map = _shot_mapping(metadata["programme_principal_control_keys_by_shot"])
    reference_map = _shot_mapping(metadata["reference_principal_control_keys_by_shot"])
    keys = [key for mapping in (programme_map, reference_map)
            for controls in mapping.values() for key in controls]
    if len(keys) != len(set(keys)) or keys != list(metadata["new_keys"]):
        raise ValueError("Principal prior node mapping changed")
    counts = dict(amplitude_factors=0, dummy_factors=0,
                  acceleration_factors=0, reference_temporal_factors=0)
    amplitude = np.full(2, amplitude_sigma_px / 1000., np.float64)
    acceleration = np.full(2, acceleration_sigma_px_s15 / 1000., np.float64)
    select = np.eye(11)[list(LIVE_COLUMNS)]
    for key in keys:
        if affine_priors is None:
            dummy_factor(graph, key, LIVE_COLUMNS)
            graph.add_observation([key], lambda q, s=select: s @ q[0],
                                  jac_fn=lambda q, s=select: [s],
                                  residual_dim=2, sigma=amplitude)
        else:
            affine_priors.add_dummy_factor(graph, key, LIVE_COLUMNS)
            affine_priors.add_selected_prior(graph, key, LIVE_COLUMNS,
                                             np.zeros(2), amplitude)
        counts["dummy_factors"] += 1
        counts["amplitude_factors"] += 1
    for shot, controls in programme_map.items():
        times = np.asarray(metadata["programme_acceleration_times_seconds"][str(shot)],
                           np.float64)
        if len(controls) != len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Principal spline time basis changed")
        for index in range(1, len(controls)-1):
            triple = controls[index-1:index+2]
            t = times[index-1:index+2]
            if affine_priors is None:
                acceleration_factor(graph, triple, t, list(LIVE_COLUMNS), acceleration)
            else:
                affine_priors.add_acceleration_factor(graph, triple, t,
                                                       LIVE_COLUMNS, acceleration)
            counts["acceleration_factors"] += 1
    return dict(counts, amplitude_sigma_px=float(amplitude_sigma_px),
                acceleration_sigma_px_s15=float(acceleration_sigma_px_s15),
                nonuniform_acceleration_formula="fit_continuous_camera.acceleration_factor",
                priors_are_experimental_assumptions=True,
                reference_offset_hard_fixed=False,
                first_frame_hard_fixed=False)
