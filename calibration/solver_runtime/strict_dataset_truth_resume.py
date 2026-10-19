"""Authenticate a new finite pixel objective without changing saved hard caps.

The saved source target is intentionally replaced. Its old pixel cost, normal
and derivatives are not equivalent to the new dataset. Original manual source
operators, physical domains, complete state and fixed seed budgets stay exact.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
from calibration.program_tracking.strict_source_fit_dataset import SCHEMA as DATASET_SCHEMA
from .fixed_seed_truth_envelope import make_fixed_seed_envelope


PRESERVED_MANIFEST_FIELDS = (
    'full_parameters', 'complete_parameter_layout', 'geometry_world_indices',
    'physical_geometry_initializer', 'camera_mount_working_constraints',
    'camera_pivot_exterior_constraints', 'physical_course_scope_source',
    'highest_direct_human_hard_truth', 'highest_user_confirmed_native_sources',
    'human_truth_support_operator', 'principal_point_engineering_prior',
    'latest9a8c_reference18_preserved', 'correlated_roll_prior',
    'roll_prior_component_sigma_degrees', 'user_trace_sha256',
    'original_ring_source_qualification', 'original_ice_pixel_correction',
    'original_measured_transition_locations', 'original_source_point_qualification',
    'additional_real_native_ice_curves', 'additional_real_native_board_edges',
    'additional_real_native_board_material_points', 'additional_real_native_board_circles',
    'independent_board_pattern_placements',
)
PRESERVED_RESTORATION_FIELDS = (
    'enabled', 'progress_policy', 'seed_engineering_scale_source',
    'fixed_seed_policy_implementation', 'step_group_bound_mode',
    'nonlinear_group_guard_mode', 'historical_group_minimum_limits_preserved',
    'original_support_bounds_in_step', 'native_endpoint_inward_branch_in_step',
    'measured_source_face_curvature_compensation', 'finite_rectangle_station_model',
    'actual_practical_band_halfspaces_retained',
)
WORLD_OPERATOR_FIELDS = ('schema', 'parameter_count', 'parameter_names',
                         'original_world66_prefix_count')
PROTECTED_PACKET_FIELDS = ('families', 'progress_policy', 'fixed_seed_envelopes',
                          'seed_engineering_scale_source',
                          'historical_group_minimum_limits_preserved')


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False, default=lambda v:
        v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else v))


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _read(reference):
    authenticate_source_reference(reference)
    return json.loads(Path(reference['path']).read_text(encoding='utf-8-sig'))


def _same_fields(old, current, names, where):
    for name in names:
        if name not in old or name not in current or _canonical(old[name]) != _canonical(current[name]):
            raise ValueError('Strict dataset revision changed protected ' + where + ': ' + name)


def _contract_occurrences(value, contract):
    if not isinstance(value, dict):
        return 0
    own = int(_canonical(value) == _canonical(contract))
    return own + sum(_contract_occurrences(child, contract)
                     for child in value.values() if isinstance(child, dict))


def _protected_digest(packet):
    protected = {key: packet.get(key) for key in PROTECTED_PACKET_FIELDS}
    arrays = {}
    for family, record in packet['families'].items():
        for field in ('limits', 'uncertainties'):
            values = np.asarray(record[field], dtype='<f8')
            if values.shape != (len(record['group_ids']),) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Original complete nonnegative cap arrays required')
            arrays[family + ':' + field] = hashlib.sha256(values.tobytes()).hexdigest()
    return dict(protected_json_sha256=hashlib.sha256(_canonical(protected).encode('utf8')).hexdigest(),
                cap_float64_sha256=arrays)


def _check_envelopes(packet):
    if packet.get('progress_policy') != 'fixed_seed_envelope':
        raise ValueError('This source-only revision requires the original fixed seed envelopes')
    envelopes = packet.get('fixed_seed_envelopes')
    if not isinstance(envelopes, dict) or set(envelopes) != set(packet['families']):
        raise ValueError('Every original cap family needs its unchanged seed envelope')
    for family, record in packet['families'].items():
        envelope = envelopes[family]
        if (envelope['group_ids'] != record['group_ids']
                or np.asarray(envelope['limits_px'], dtype='<f8').tobytes()
                   != np.asarray(record['limits'], dtype='<f8').tobytes()
                or np.asarray(envelope.get('original_recorded_engineering_uncertainties_px',
                                          envelope['declared_engineering_uncertainties_px']), dtype='<f8').tobytes()
                   != np.asarray(record['uncertainties'], dtype='<f8').tobytes()):
            raise ValueError('Saved family caps/uncertainties differ from their original seed envelope')
        make_fixed_seed_envelope(envelope['seed_violations_px'],
            envelope['declared_engineering_uncertainties_px'],
            seed_q_sha256=envelope['seed_q_sha256'], group_ids=record['group_ids'], resumed=envelope)
    scale = packet.get('seed_engineering_scale_source')
    if scale is not None:
        authenticate_source_reference(scale)


def load_for_strict_revision(path, q, manifest, factors, *, archived_loader=None,
                            source_target=None, role_target=None):
    """Return a migrated in-memory cap packet after authenticating both targets.

    The caller uses the existing runtime factors. Optional function arguments
    make the precise archived loader and target functions explicit in tests or
    embedding; production defaults to the source-owned named66 driver.
    Nothing is written and no current source residual or fit is evaluated.
    """
    if archived_loader is None or source_target is None or role_target is None:
        import run_current_named66_full_programme as driver
        archived_loader = archived_loader or driver.load_truth_restoration_state
        source_target = source_target or driver.truth_restoration_source_target
        role_target = role_target or driver.role_target_from_restored_manifest
    path = Path(path).resolve()
    original_packet = json.loads(path.read_text(encoding='utf-8-sig'))
    old = _read(original_packet['source_input_manifest_reference'])
    authenticate_source_reference(original_packet['complete_state_reference'])
    with np.load(original_packet['complete_state_reference']['path'], allow_pickle=False) as saved:
        previous_q = saved['q']
        current_q = np.asarray(q)
        if (previous_q.dtype != current_q.dtype or previous_q.shape != current_q.shape
                or previous_q.tobytes() != current_q.tobytes()
                or hashlib.sha256(current_q.tobytes()).hexdigest() != original_packet['complete_q_sha256']):
            raise ValueError('Strict dataset revision requires the exact complete saved q bytes')
    # This is the original strict authentication path, against its own target.
    # Current factors must authenticate the same manual authorities and groups.
    validated = archived_loader(path, q, old, factors)
    before = _protected_digest(original_packet)
    if _protected_digest(validated) != before:
        raise ValueError('Archived loader changed original caps or fixed seed provenance')
    _check_envelopes(original_packet)
    _same_fields(old, manifest, PRESERVED_MANIFEST_FIELDS, 'manual/physical contract')
    _same_fields(old['joint_source_truth_restoration'], manifest['joint_source_truth_restoration'],
                 PRESERVED_RESTORATION_FIELDS, 'restoration contract')
    _same_fields(old['current_geometry'], manifest['current_geometry'],
                 ('coordinate_chart', 'programme_blocks'), 'physical camera chart')
    _same_fields(old['current_geometry']['world_geometry'],
                 manifest['current_geometry']['world_geometry'], WORLD_OPERATOR_FIELDS,
                 'physical world operator')
    strict = manifest.get('strict_source_dataset')
    if (not isinstance(strict, dict)
            or strict.get('schema') != 'rinkspace.strict-reviewed-finite-programme-source.v1'
            or manifest.get('legacy_pixel_objective_replaced') is not True
            or strict.get('archived_pixel_factors_in_FIT') is not False
            or strict.get('archived_pixel_guards_evaluated') is not False
            or strict.get('archived_motion_profiles_evaluated') is not False
            or strict.get('all_pixel_families_routed_to_strict_input') is not True
            or strict.get('admitted_motion_points') != 0
            or _contract_occurrences(manifest['source_contract'], strict) != 1):
        raise ValueError('Actual complete source contract must use only the strict finite dataset route')
    finite = strict['finite_dataset_contract']
    if finite.get('schema') != 'rinkspace.strict-reviewed-finite-dataset-factor.v1':
        raise ValueError('Authenticated finite dataset factor required')
    dataset = _read(finite['dataset_reference'])
    motion = _read(strict['strict_motion_reference'])
    frozen_identity = _read(dataset['frozen_model_reference'])
    if (dataset.get('schema') != DATASET_SCHEMA
            or motion.get('schema') != 'rinkspace.strict-model-scope-original-motion-inputs.v1'
            or motion.get('unknown_or_unreviewed_default') != 'excluded'
            or dataset['source'] != motion['source']
            or finite['source_video'] != dataset['source']
            or strict['source_identity'] != motion['source']
            or finite['original_fit_points'] != dataset['fit_points']
            or finite['held_points'] != dataset['held_points']
            or finite['scalar_rows'] != 2 * dataset['fit_points']
            or finite.get('held_used_in_fit') is not False
            or any(row['after_points'] != 0 for row in motion['blocks'])):
        raise ValueError('Actual strict source dataset identity/roles/inventory changed')
    if old.get('strict_source_dataset') is None and frozen_identity['q_sha256'] != original_packet['complete_q_sha256']:
        raise ValueError('First strict revision dataset must use the declared saved baseline model')
    if (strict['parameter_count'] != manifest['full_parameters']
            or finite['parameter_count'] != manifest['full_parameters']
            or strict['source_rows'] != strict['engineering_prior_rows'] + finite['scalar_rows']):
        raise ValueError('Strict source complete chart or scalar inventory changed')
    prior = manifest['principal_point_engineering_prior']
    expected_rows = strict['source_rows'] + (0 if prior is None else int(prior['rows']))
    if manifest['source_rows'] != expected_rows:
        raise ValueError('Actual complete objective contains unaccounted source rows')
    new = deepcopy(original_packet)
    new['source_target'] = source_target(manifest)
    new['source_role_operator_target'] = role_target(manifest)
    if _protected_digest(new) != before:
        raise ValueError('Strict source migration changed a saved cap/envelope byte')
    new['authenticated_strict_dataset_source_revision'] = dict(
        schema='rinkspace.strict-finite-dataset-original-hard-cap-continuation.v1',
        original_truth_state=dict(path=str(path), sha256=file_sha(path)),
        original_source_manifest=original_packet['source_input_manifest_reference'],
        original_source_target=original_packet['source_target'],
        actual_current_source_target=new['source_target'],
        strict_dataset_reference=finite['dataset_reference'],
        strict_motion_reference=strict['strict_motion_reference'],
        complete_q_sha256=original_packet['complete_q_sha256'],
        original_cap_and_seed_envelope_identity=before,
        original_manual_physical_operators_and_authorities_exact=True,
        limits_uncertainties_group_ids_seed_envelopes_unchanged=True,
        source_loss_objective_revised=True, old_pixel_cost_preserved=False,
        old_source_J_profiles_normals_reuse_authorized=False,
        complete_fresh_source_objective_required=True,
        original_packet_and_arrays_not_rewritten=True, accuracy_accepted=False)
    return new
