"""Strict objective identity for archived-chart startup and optimizer resume."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .strict_dataset_truth_resume import _canonical, _same_fields
from .strict_model_scope_inputs import load_strict_programme_motion_inputs


def retired_motion_compaction_policy(reference, blocks):
    """Authenticate zero admitted motion before skipping its archived r/J test.

    The compactor still checks every original source/map/certificate and builds
    the complete archived chart. Its old UV operator is no longer a FIT input.
    """
    motion, binding = load_strict_programme_motion_inputs(
        reference['path'], expected_sha256=reference['sha256'])
    if set(motion) != {block.name for block in blocks}:
        raise ValueError('Strict zero-motion authority must cover every programme family')
    for block in blocks:
        values = motion[block.name]
        if len(values['a']) or len(values['b']):
            raise ValueError('Archived motion numerical check may only retire with zero admitted motion')
        if not np.array_equal(values['stamps'], np.asarray(block.bundle.stamps)):
            raise ValueError('Strict zero-motion authority changed original exposure clocks')
    return dict(schema='rinkspace.strict-retired-motion-compaction-policy.v1',
        strict_motion_reference=deepcopy(reference),
        strict_motion_review_reference=deepcopy(binding['review']),
        authenticated_admitted_motion_points=0,
        verify_value_J=False,
        archived_motion_numerical_check='NOT_APPLICABLE_RETIRED_PIXEL_OBJECTIVE',
        original_source_bytes_maps_and_finite_duplicate_certificates_required=True,
        strict_wrapper_original_index_and_clock_validation_still_required=True)


def _active_identity(contract):
    value = deepcopy(contract)
    if (value.get('schema') != 'rinkspace.strict-reviewed-finite-programme-source.v1'
            or value.get('admitted_motion_points') != 0
            or value.get('archived_pixel_factors_in_FIT') is not False
            or value.get('archived_pixel_guards_evaluated') is not False
            or value.get('archived_motion_profiles_evaluated') is not False
            or value.get('all_pixel_families_routed_to_strict_input') is not True):
        raise ValueError('Actual strict zero-motion objective required for optimizer inheritance')
    execution = value.pop('strict_dataset_execution')
    if execution.get('candidate_camera_mapping_enabled') is not False:
        raise ValueError('Optimizer identity requires the existing original camera mapping')
    finite = value['finite_dataset_contract']
    # Remove only measured execution times. All source hashes, rows, target
    # bindings, mathematical policy, chart and implementation identities stay.
    for key in ('authenticated_FIT_consumer_seconds', 'complete_factor_preparation_seconds'):
        finite.pop(key)
    receipt = finite['strict_source_read_receipt']
    receipt.pop('seconds')
    for pack in receipt['source_packs']:
        pack.pop('seconds')
    return value


def same_strict_optimizer_target(old_manifest, current_contract, *,
        old_quadratic, quadratic, old_scales, scales, protected_fields,
        full_truth_resume_required, physical_initialization_changed=False):
    """Permit damping only for the same active target, before full caps restore.

    This gate cannot authorize a source revision. The driver must subsequently
    run load_for_strict_revision, including complete q/caps/envelopes and every
    protected manual, world-chart and physical contract, before solver entry.
    """
    if not full_truth_resume_required or physical_initialization_changed:
        return False
    old_contract = old_manifest.get('strict_source_dataset')
    if not isinstance(old_contract, dict):
        return False
    if (not np.array_equal(old_quadratic, quadratic)
            or not np.array_equal(old_scales, scales)):
        return False
    try:
        if _canonical(_active_identity(old_contract)) != _canonical(_active_identity(current_contract)):
            return False
        _same_fields(old_manifest, protected_fields, tuple(protected_fields),
                     'active optimizer objective')
    except (KeyError, ValueError):
        return False
    return True
