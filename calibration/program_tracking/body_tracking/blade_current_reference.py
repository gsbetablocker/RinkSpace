"""Restore a complete body/skate reference from one current saved S05 case.

Only current case packets, the selected saved parameter/result, the frozen
camera and the existing generic body model are consumed. No prepare, parent
restore, historical pose layer, inference, fitting or candidate write occurs.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rinkspace.storage import AssetStore
from rinkspace.viewer3d.prepare import FullSurface
from rinkspace.viewer3d.open_boot_topology import open_boot_shell
from .absolute_motion_state import absolute_reference, curve_from_packet
from .blade_projection import SavedBladeProjection
from .exmr_integration import CASE_FIELD, read_case_identity
from .ice_soft import IceSoftPolicy
from .rebuild_prepare import digest
from .skate_binding import SkateBinding, SHAPE_NAMES, upper_topology
from .skate_checkpoint import IDENTITY_FIELDS, validate_checkpoint_identity
from .skate_template import SkateParameters


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def restore_current_reference(project, stage, parameter, *, seed=False):
    """Return ``(reference, camera, camera_info, saved_parameter)``.

    The default reference follows the authoritative saved final curve. Set
    ``seed=True`` only when resuming with ``saved_parameter['x'][17:]`` in
    AbsoluteMotionState: the saved archived seed and saved x then reproduce
    the final curve once. The full surface, neutral foot/collar selections,
    fixed shoes, hands, grasp documents and finite ice policy are restored
    independently of that choice. No optimizer or execution owner is created.

    The returned reference exposes ``ice_policy``, ``source_document``,
    ``observations_path``, ``current_restore_info`` and its existing world/FK
    interfaces. Callers own new-run paths, input revisions and active columns.
    Missing or conflicting current fields raise explicitly; there is no
    fallback to a historical production chain.
    """
    query = SavedBladeProjection(project, stage, parameter)
    project, stage = query.project, query.stage
    case = stage / 'common/exmr_hand_nofoot'
    saved = query.parameter
    validate_checkpoint_identity(saved)
    identity = read_case_identity(saved)
    result_path = case / 'fitted/result.npz'
    required = ('pts', 'shared_geometry', 'footwear_binding_json',
                'camera_binding_json', CASE_FIELD)
    with np.load(result_path, allow_pickle=False) as source:
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError('Current result missing fields: ' + ', '.join(missing))
        result = {key: source[key].copy() for key in required}
    np.testing.assert_array_equal(result['pts'], query.pts)
    if read_case_identity(result) != identity:
        raise ValueError('Current parameter/result case identities differ')
    boot = json.loads(str(result['footwear_binding_json']))
    for key in ('T_FB', 'main_rocker_ft', 'shape_parameters', 'foot_correspondences',
                'collar_section_in_solver'):
        if key not in boot:
            raise ValueError('Current footwear binding record missing field: ' + key)

    # Check the current independent input bundle, never old parent packages.
    current_files = {
        'case_input_sha256': case / 'input.npz',
        'compact_smplx_sha256': case / 'compact_smplx.npz',
        'initial_reference_sha256': case / 'initial.npz',
        'observations_sha256': case / 'observations.npz',
        'relation_document_sha256': stage / 'grasp_relations.json',
        'geometry_semantics_sha256': stage / 'palm_grip_geometry.json',
        'support_document_sha256': stage / 'support_and_free_shoe_regions.json',
    }
    hashes = {}
    for field, path in current_files.items():
        value = digest(path)
        if value != identity[field]:
            raise ValueError('Current input differs from saved ' + field + ': ' + str(path))
        hashes[field] = value

    reference = query.reference
    reference.geometry = np.asarray(result['shared_geometry'], dtype=float).copy()
    if reference.geometry.shape != (14,) or not np.isfinite(reference.geometry).all():
        raise ValueError('Expected current saved shared_geometry (14,)')
    # AbsoluteReference inherits optional lookups through ClipReference's
    # parent. An empty current namespace makes absent optional fields explicit.
    reference.parent = SimpleNamespace()
    reference.grasp_geometry = _json(stage / 'palm_grip_geometry.json')
    reference.grasp_relations = _json(stage / 'grasp_relations.json')
    reference.source_document = _json(stage / 'support_and_free_shoe_regions.json')
    reference.observations_path = case / 'observations.npz'
    reference.export_source_path = case / 'export_source.npz'
    reference.hashes = {key: identity[key] for key in
                        ('native_input_sha256', 'camera_prior_sha256', 'shared_s01_parameters_sha256')}
    reference.footwear_observations = dict(
        visible_rows=[], blade_rows=[], landmark_rows=[],
        provenance=dict(source_document_sha256=identity['support_document_sha256'],
                        same_frozen_source_evidence=True))
    reference.ice_policy = IceSoftPolicy(**json.loads(str(saved['ice_soft_policy_json'])))

    config = project / 'events/ankara_2026/programmes/okada_mei_sp/viewer3d_spin_281_repaired_1105.json'
    model_asset = _json(config)['model_asset']
    model_path = AssetStore(project).resolve(project / model_asset, must_exist=True)
    model_sha = digest(model_path)
    if model_sha != reference.grasp_geometry['model_sha256']:
        raise ValueError('Current generic surface model SHA differs from grasp geometry')
    surface = FullSurface(SimpleNamespace(root=case, packet=reference.p, model_path=model_path))
    # The same current CompactBody owns exact COCO, native FK and surface LBS.
    surface.body = reference.body
    binding = SkateBinding(reference, surface, query.camera_info,
                           rocker_ft=float(boot['main_rocker_ft']))
    # Constructor-generated registration is never used as a replacement fit.
    # Restore the saved frozen registration/shape before any live evaluation.
    binding.base_parameters = SkateParameters(**json.loads(str(saved['base_skate_parameters_json'])))
    binding.R0 = np.asarray(saved['R_FB_initial'], dtype=float).copy()
    binding.t0 = np.asarray(saved['t_FB_initial_solver'], dtype=float).copy()
    binding.x = np.asarray(saved['x'][:17], dtype=float).copy()
    binding._parameter_key = binding._packet = None
    geometry = binding.packet()
    sections = len(geometry['xs'])
    collar_end = int(np.searchsorted(geometry['xs'], .46 * geometry['parameters'].sole_length_m))
    binding.faces, binding.region_ids = upper_topology(sections, collar_end)
    binding.adapter_faces, _ = open_boot_shell(geometry['adapter'][:14])
    reference.binding = binding
    binding.reference = reference
    if bool(saved['collar_constraints_enabled']) != bool(boot['collar_section_in_solver']):
        raise ValueError('Current parameter/result collar policies differ')
    if bool(saved['collar_constraints_enabled']):
        binding.enable_collar_constraints()

    frozen = {key: np.asarray(saved[key]).copy() for key in IDENTITY_FIELDS}
    frozen.update(base_skate_parameters_json=np.array(json.dumps(asdict(binding.base_parameters))),
                  R_FB_initial=binding.R0, t_FB_initial_solver=binding.t0,
                  camera_binding_json=np.array(json.dumps(query.camera_info, sort_keys=True)))
    validate_checkpoint_identity(saved, frozen)
    reference.fixed_checkpoint_identity = {key: np.asarray(saved[key]).copy()
                                          for key in (*IDENTITY_FIELDS, CASE_FIELD)}
    R, t, _ = binding.parameters()
    recorded_T = np.asarray(boot['T_FB'], dtype=float)
    np.testing.assert_allclose(R, recorded_T[:, :3, :3], rtol=0, atol=2e-12)
    np.testing.assert_allclose(t * binding.ratio, recorded_T[:, :3, 3], rtol=0, atol=2e-12)
    for name in SHAPE_NAMES:
        np.testing.assert_allclose(getattr(geometry['parameters'], name) * binding.ratio,
                                   boot['shape_parameters'][name], rtol=0, atol=2e-12)
    for actual, recorded in zip(binding.foot_correspondences, boot['foot_correspondences']):
        for key, value in actual.items():
            np.testing.assert_allclose(value * binding.ratio, recorded[key], rtol=0, atol=2e-10)
    reference.current_restore_info = dict(
        stage=str(stage), result=str(result_path), model=str(model_path),
        model_sha256=model_sha, current_input_sha256=hashes,
        archived_fixed_identity_preserved=True, frozen_shoe_parameters_preserved=True,
        historical_production_replayed=False, pose_seed='saved_final',
        foot_sample_vertices=len(binding.sample_ids), collar_constraints_enabled=binding.collar_constraints_enabled)
    if seed:
        reference = absolute_reference(reference, curve_from_packet(saved, seed=True), root=case)
        reference.binding = copy.copy(binding)
        reference.binding.reference = reference
        reference.current_restore_info = dict(reference.current_restore_info, pose_seed='saved_archived_seed')
    return reference, query.camera, query.camera_info, saved
