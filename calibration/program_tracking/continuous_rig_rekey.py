"""Pure, checked key remapping for a programme rig and its reference rig.

This changes graph-node identities only. Source PTS, external scene nodes,
state values, source camera templates, physical cameras and Jacobian columns
are preserved. It neither creates a graph nor modifies a source observation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy

import numpy as np

from continuous_camera_model import FrameBinding, ShotSpline
from continuous_reference_calibration import StaticReferenceBasis
from continuous_rig_camera import ContinuousRigCameraModel
from continuous_quadratic_rig_optin import QuadraticRigCameraModel
from continuous_time_principal_optin import TimePrincipalRigCameraModel


GENERATED_KEY_FLOOR = 1_000_000_000
SCENE_RING_KEYS = frozenset((2000, 2001, 2002, 2003))


@dataclass(frozen=True)
class RekeyedRig:
    programme: ContinuousRigCameraModel
    references: ContinuousRigCameraModel
    states: dict[int, np.ndarray]
    prior: dict[int, np.ndarray]
    metadata: dict
    key_mapping: dict[int, int]
    reference_shot_mapping: dict[int, int]
    verification: dict


def _state_map(source: Mapping[int, np.ndarray], label: str) -> dict[int, np.ndarray]:
    result = {}
    for raw_key, raw_value in source.items():
        key = int(raw_key)
        value = np.asarray(raw_value, np.float64)
        if key in result or value.shape != (11,) or not np.isfinite(value).all():
            raise ValueError(f'{label} must have unique finite Vector11 nodes')
        result[key] = value.copy()
    return result


def _remap_metadata(metadata: Mapping, key_mapping: Mapping[int, int],
                    shot_mapping: Mapping[int, int]) -> dict:
    """Replace exact structured node IDs; never rewrite prose, hashes or PTS."""
    def nodes(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                replacement = key_mapping.get(int(key), int(key)) if (
                    isinstance(key, str) and key.isdecimal()) else key
                if isinstance(key, str) and key.isdecimal():
                    replacement = str(replacement)
                if replacement in result:
                    raise ValueError('Metadata key remapping collided')
                result[replacement] = nodes(item)
            return result
        if isinstance(value, list):
            return [nodes(item) for item in value]
        if isinstance(value, tuple):
            return tuple(nodes(item) for item in value)
        if isinstance(value, int) and not isinstance(value, bool):
            return key_mapping.get(value, value)
        return copy.deepcopy(value)

    updated = nodes(dict(metadata))
    temporal = updated.get('time_principal')
    if isinstance(temporal, dict):
        temporal['reference_shot_ids'] = [shot_mapping.get(int(shot), int(shot))
                                          for shot in temporal['reference_shot_ids']]
        for field in ('reference_principal_control_keys_by_shot',
                      'original_c_control_keys_by_shot'):
            mapping = temporal[field]
            temporal[field] = {str(shot_mapping.get(int(shot), int(shot))): value
                               for shot, value in mapping.items()}
    return updated


def _clone_splines(model, keys, shots):
    result = {}
    for old_id, spline in model.shot_splines.items():
        new_id = shots.get(int(old_id), int(old_id))
        control = tuple(keys.get(key, key) for key in spline.control_keys)
        if isinstance(spline, ShotSpline):
            result[new_id] = ShotSpline(new_id, control, spline.knot_pts)
        elif isinstance(spline, StaticReferenceBasis):
            result[new_id] = StaticReferenceBasis(new_id, spline.pts, control)
        else:
            raise TypeError(f'Unsupported spline basis: {type(spline).__name__}')
    if len(result) != len(model.shot_splines):
        raise ValueError('Shot rekeying merged two spline identities')
    return result


def _clone_model(model, keys, shots, *, source_clock=None):
    splines = _clone_splines(model, keys, shots)
    frames = [FrameBinding(int(row.pts), keys.get(int(row.frame_key), int(row.frame_key)),
                           shots.get(int(row.shot_id), int(row.shot_id)),
                           row.camera_id, np.asarray(row.template25, np.float64).copy(),
                           row.spline_pts if source_clock is None else source_clock[int(row.pts)])
              for row in model.frames.values()]
    def mapping(name):
        return {group: keys.get(int(key), int(key))
                for group, key in getattr(model, name).items()}
    kwargs = dict(mount_keys=mapping('mount_keys'),
                  offset_keys=mapping('offset_keys'),
                  offset_slope_keys=mapping('offset_slope_keys'),
                  brown_keys=mapping('brown_keys'),
                  brown_slope_keys=mapping('brown_slope_keys'))
    if isinstance(model, QuadraticRigCameraModel):
        kwargs.update(lens_quadratic_keys=mapping('lens_quadratic_keys'),
                      brown_quadratic_keys=mapping('brown_quadratic_keys'))
    elif isinstance(model, TimePrincipalRigCameraModel):
        kwargs['principal_control_keys_by_shot'] = {
            shots.get(int(shot), int(shot)): tuple(keys.get(int(key), int(key))
                                               for key in control)
            for shot, control in model.principal_control_keys_by_shot.items()}
    elif type(model) is not ContinuousRigCameraModel:
        raise TypeError(f'Unsupported rig model: {type(model).__name__}')
    return type(model)(splines, frames, mapping('lens_base_keys'),
                       mapping('lens_slope_keys'), dict(model.lens_zoom_reference),
                       **kwargs)


def rekey_rig_pair(programme: ContinuousRigCameraModel,
                   references: ContinuousRigCameraModel,
                   states: Mapping[int, np.ndarray],
                   prior: Mapping[int, np.ndarray],
                   metadata: Mapping, *, key_start: int,
                   reference_shot_start: int,
                   reserved_keys=(), reserved_shot_ids=()) -> RekeyedRig:
    """Move generated nodes into caller's collision-free domain and verify all views.

    ``key_start`` and ``reference_shot_start`` are exact starts; a collision
    fails instead of silently skipping a requested identity. The returned
    maps are suitable for factor-key and provenance remapping by the caller.
    Original source frame PTS and scene ring nodes 2000..2003 remain literal.
    Programme shot identities remain literal; reference shots get explicit
    new identities so different groups can share one graph namespace.
    """
    if not isinstance(programme, ContinuousRigCameraModel) or not isinstance(
            references, ContinuousRigCameraModel):
        raise TypeError('Both inputs must be supported physical rig models')
    if type(programme) is not type(references):
        raise TypeError('Programme and reference models must have the same optical class')
    if not isinstance(metadata, Mapping):
        raise TypeError('Metadata must be a mapping')
    old_states = _state_map(states, 'states')
    old_prior = _state_map(prior, 'prior')
    if set(old_states) != set(old_prior):
        raise ValueError('State and prior keysets differ')
    all_frames = set(programme.frames) | set(references.frames)
    if len(all_frames) != len(programme.frames)+len(references.frames):
        raise ValueError('Programme and reference source PTS overlap')
    if not SCENE_RING_KEYS.isdisjoint(all_frames):
        raise ValueError('Source PTS overlaps an external scene ring node')
    old_dependencies = {int(key) for model in (programme, references)
                        for pts in model.frames for key in model.dependencies(pts)}
    if not old_dependencies <= set(old_states):
        raise ValueError('Some camera dependencies lack a supplied state/prior')
    generated = sorted(key for key in old_states
                       if key >= GENERATED_KEY_FLOOR and key not in all_frames)
    if not generated or not set(generated) <= old_dependencies:
        raise ValueError('Unknown generated node or no generated nodes to rekey')
    key_start = int(key_start)
    reference_shot_start = int(reference_shot_start)
    if key_start < GENERATED_KEY_FLOOR or reference_shot_start < 0:
        raise ValueError('Rekey destination must use a nonnegative disjoint domain')
    targets = tuple(range(key_start, key_start+len(generated)))
    preserved = (set(old_states)-set(generated)) | set(map(int, reserved_keys))
    if set(targets) & preserved:
        raise ValueError('Destination graph key domain collides with a preserved key')
    keys = dict(zip(generated, targets))
    ref_shots = sorted(map(int, references.shot_splines))
    destination_shots = tuple(range(reference_shot_start,
                                    reference_shot_start+len(ref_shots)))
    occupied_shots = set(programme.shot_splines) | set(map(int, reserved_shot_ids))
    if set(destination_shots) & occupied_shots:
        raise ValueError('Destination reference shot identity collides')
    shots = dict(zip(ref_shots, destination_shots))
    new_programme = _clone_model(programme, keys, {})
    new_references = _clone_model(references, keys, shots)
    new_states = {keys.get(key, key): value.copy() for key, value in old_states.items()}
    new_prior = {keys.get(key, key): value.copy() for key, value in old_prior.items()}
    if (len(new_states) != len(old_states) or len(new_prior) != len(old_prior)
            or any(not np.array_equal(new_states[key], old_states[key])
                   for key in SCENE_RING_KEYS if key in old_states)):
        raise ValueError('Rekeying changed or merged state payloads')
    frame_count = 0
    jacobian_blocks = 0
    for old_model, new_model in ((programme, new_programme),
                                 (references, new_references)):
        if set(old_model.frames) != set(new_model.frames):
            raise ValueError('Rekeying changed source PTS')
        for pts in old_model.frames:
            before, old_jac = old_model.physical_frame(pts, old_states)
            after, new_jac = new_model.physical_frame(pts, new_states)
            if (before.shape != (15,) or not np.array_equal(before, after)
                    or not np.array_equal(old_model.camera25(pts, old_states),
                                          new_model.camera25(pts, new_states))
                    or set(new_jac) != {keys.get(key, key) for key in old_jac}):
                raise ValueError(f'Physical camera changed while rekeying PTS {pts}')
            for old_key, block in old_jac.items():
                if not np.array_equal(block, new_jac[keys.get(old_key, old_key)]):
                    raise ValueError(f'Physical Jacobian changed at PTS {pts}:{old_key}')
                jacobian_blocks += 1
            frame_count += 1
    new_metadata = _remap_metadata(metadata, keys, shots)
    return RekeyedRig(new_programme, new_references, new_states, new_prior,
                      new_metadata, keys, shots,
                      dict(schema='rinkspace.rig-rekey-exact.v1',
                           programme_frames=len(programme.frames),
                           reference_frames=len(references.frames),
                           checked_physical15_frames=frame_count,
                           checked_jacobian_blocks=jacobian_blocks,
                           checked_camera25_frames=frame_count,
                           exact_array_equal=True,
                           preserved_source_pts=True,
                           preserved_scene_ring_keys=sorted(SCENE_RING_KEYS),
                           moved_graph_nodes=len(keys),
                           moved_reference_shots=len(shots)))
