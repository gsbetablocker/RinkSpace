"""Opt-in numerical recovery room without historical minimum ratcheting.

The actual source-support operators and physical guards remain caller-owned.
Only already violated seed groups receive their existing declared engineering
uncertainty again; seed-feasible groups retain zero extra violation.  These
limits are an experiment's fixed numerical envelope, never measured precision.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import numpy as np


POLICY = 'fixed_seed_envelope'
HISTORICAL_POLICY = 'historical_ratchet'
COMPARISON_TOLERANCE_PX = 1e-8


def load_existing_source_engineering_scales(path, *, native_authority_path,
                                           location_envelope_path):
    """Authenticate explicitly selected existing working location scales.

    A source sigma alone is never a hard tolerance. Each selected positional
    stroke must also carry the same already recorded desired location radius
    in the original engineering policy, with explicit non-measurement labels.
    Raw source packets and their FIT sigma remain untouched.
    """
    path = Path(path).resolve()
    packet = json.loads(path.read_text(encoding='utf8'))
    if (packet.get('schema') != 'rinkspace.fixed-seed-existing-engineering-location-scales.v1'
            or packet.get('engineering_scale_is_measured_precision') is not False
            or packet.get('FIT_sigma_or_source_supports_modified') is not False):
        raise ValueError('Explicit existing engineering source-scale policy required')
    def authenticated(reference, expected=None):
        actual = Path(reference['path']).resolve()
        if expected is not None and actual != Path(expected).resolve():
            raise ValueError('Engineering scale authority differs from current source options')
        if hashlib.sha256(actual.read_bytes()).hexdigest() != reference['sha256']:
            raise ValueError('Engineering scale source authority bytes changed')
        return json.loads(actual.read_text(encoding='utf8'))
    authority = authenticated(packet['native_authority'], native_authority_path)
    policy = authenticated(packet['location_envelope'], location_envelope_path)
    native = {row['id']: row for row in authority['records']}
    location = {row['id']: row for row in policy['records'] if row['family'] == 'native'}
    result = {}
    packets = {}
    for item in packet['records']:
        identity = item['id']
        if identity in result or item.get('family') != 'native' or identity not in native or identity not in location:
            raise ValueError('Every engineering override needs one unchanged native source identity')
        row = native[identity]
        if not identity.startswith('user_complete_position_'):
            raise ValueError('Only explicitly identified positional strokes may use this engineering scale policy')
        reference = row['packet_reference']
        key = (reference['path'], reference['sha256'])
        if key not in packets:
            packets[key] = authenticated(reference)
        original = packets[key]['rows'][row['original_row_index']]
        raw = json.dumps(original, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        canonical = hashlib.sha256(raw.encode('utf8')).hexdigest()
        if (canonical != row['source_row_canonical_json_sha256']
                or canonical != item['source_row_canonical_json_sha256'] or original['id'] != identity):
            raise ValueError('Existing positional source row changed')
        scales = _vector(original['source_working_sigma_before_stroke_correlation_px'],
                         name='existing source engineering location scales')
        if not len(scales) or not np.all(scales == scales[0]):
            raise ValueError('A source stroke needs one already recorded common engineering location scale')
        value = float(scales[0])
        desired = float(location[identity]['desired_minimum_signed_location_radius_px'])
        if (original.get('working_sigma_is_manual_measurement_precision') is not False
                or original.get('source_support_summary',{}).get('source_single_optical_material_branch_qualified') is not False
                or not math.isfinite(value) or value < 0. or value != desired
                or value != float(item['engineering_location_scale_px'])):
            raise ValueError('An unchanged explicit engineering location scale with non-measurement semantics is required')
        result[identity] = value
    return result, dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        engineering_scale_is_measured_precision=False, FIT_sigma_or_source_supports_modified=False,
        original_source_row_hashes_authenticated=True, native_groups=len(result))


def _vector(value, *, name, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 1 or (shape is not None and result.shape != shape)
            or not np.isfinite(result).all() or np.any(result < 0.)):
        raise ValueError('Complete finite nonnegative ' + name + ' required')
    return result.copy()


def make_fixed_seed_envelope(violations, uncertainties, *, seed_q_sha256,
                             group_ids, resumed=None):
    """Create once, or authenticate the unchanged original seed envelope.

    The caller authenticates checkpoint q/source/authority before calling.
    Resuming this policy reuses the original seed budget, even after groups
    improve; it does not add uncertainty again at each new checkpoint.
    """
    current = _vector(violations, name='actual source group violations')
    uncertainty = _vector(uncertainties, name='declared engineering uncertainties', shape=current.shape)
    ids = list(group_ids)
    if (len(ids) != len(current) or len(set(ids)) != len(ids)
            or not all(isinstance(identity, str) and identity for identity in ids)):
        raise ValueError('Every source group needs its unchanged ordered identity')
    if resumed is None:
        seed = current.copy()
        origin = seed_q_sha256
    else:
        if (resumed.get('policy') != POLICY or resumed.get('group_ids') != ids
                or resumed.get('historical_group_minimum_limits_preserved') is not False):
            raise ValueError('Explicit fixed seed envelope identity required')
        seed = _vector(resumed['seed_violations_px'], name='original seed violations', shape=current.shape)
        recorded = _vector(resumed['declared_engineering_uncertainties_px'],
                           name='original declared uncertainties', shape=current.shape)
        if not np.array_equal(recorded, uncertainty):
            raise ValueError('The declared engineering uncertainty changed on resume')
        origin = resumed['seed_q_sha256']
    if (not isinstance(origin, str) or len(origin) != 64
            or any(character not in '0123456789abcdef' for character in origin)):
        raise ValueError('Original complete seed q SHA256 required')
    protected = seed <= COMPARISON_TOLERANCE_PX
    limits = np.where(protected, 0., seed + uncertainty)
    if not np.isfinite(limits).all():
        raise ValueError('Fixed seed limits must remain finite')
    if resumed is not None:
        recorded = _vector(resumed['limits_px'], name='recorded fixed seed limits', shape=current.shape)
        if not np.array_equal(recorded, limits):
            raise ValueError('Saved envelope differs from original seed plus declared uncertainty')
        if resumed.get('seed_protected_zero_groups') != protected.tolist():
            raise ValueError('The original protected zero group set changed on resume')
    if np.any(current > limits + COMPARISON_TOLERANCE_PX):
        raise ValueError('Current checkpoint is outside its original fixed seed envelope')
    metadata = dict(policy=POLICY, seed_q_sha256=origin, group_ids=ids,
        seed_violations_px=seed.tolist(), declared_engineering_uncertainties_px=uncertainty.tolist(),
        limits_px=limits.tolist(), seed_protected_zero_groups=protected.tolist(),
        historical_group_minimum_limits_preserved=False,
        declared_uncertainty_is_measured_statistical_precision=False,
        source_pixels_supports_sigmas_and_physical_domains_unchanged=True,
        uncertainty_added_once_at_experiment_seed=True,
        restored_groups_not_ratcheted_or_latched=True, accuracy_accepted=False)
    return limits, metadata


def fixed_seed_trial_decision(previous, trial, limits, *, before_cost,
                              after_cost, default_source_cost_accepts):
    """Require the complete original cost decrease and fixed source envelope.

    Actual physical/source guards run before this callback in production.
    It grants no exemption from them and does not make a visual claim.
    """
    old = _vector(previous, name='previous actual source violations')
    new = _vector(trial, name='trial actual source violations', shape=old.shape)
    allowed = _vector(limits, name='fixed seed limits', shape=old.shape)
    before = float(before_cost)
    after = float(after_cost)
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError('Finite complete original source costs required')
    if np.any(old > allowed + COMPARISON_TOLERANCE_PX):
        raise ValueError('Parent checkpoint is outside the fixed seed envelope')
    envelope_passed = bool(np.all(new <= allowed + COMPARISON_TOLERANCE_PX))
    precision = 1e-10 * max(1., before)
    cost_passed = bool(default_source_cost_accepts and after < before - precision)
    accepted = bool(envelope_passed and cost_passed)
    return dict(accept=accepted, mode='source_objective', progress_policy=POLICY,
        rejection_reason=None if accepted else ('fixed_seed_truth_envelope_exceeded'
            if not envelope_passed else 'complete_source_cost_not_decreased'),
        report=dict(policy=POLICY, fixed_seed_envelope_passed=envelope_passed,
            original_complete_source_cost_decrease_required=True,
            default_complete_source_cost_accepts=bool(default_source_cost_accepts),
            previous_source_cost=before, trial_source_cost=after,
            full_source_cost_change=after-before, source_cost_comparison_precision=precision,
            temporarily_increased_groups=int(np.count_nonzero(new > old+COMPARISON_TOLERANCE_PX)),
            improved_groups=int(np.count_nonzero(new < old-COMPARISON_TOLERANCE_PX)),
            seed_zero_groups_protected=True, every_group_each_step_improvement_required=False,
            historical_group_minimum_limits_preserved=False,
            actual_full_physical_and_source_guards_required=True, accuracy_accepted=False))


def attach_fixed_seed_trial_fraction_domain(domain):
    """Each short SOC solve starts with the same fixed targets and physical rows.

    Preserve the actual matrix and canonical source records by identity, as in
    the existing trial-fraction interface.  Mutable lower/target arrays are new
    on every call, so temporary full-step SOC cuts cannot survive shortening.
    """
    matrix = domain['matrix']
    lower = np.asarray(domain['lower'], dtype=np.float64).copy()
    if (np.asarray(matrix).ndim != 2 or np.asarray(matrix).shape[0] != len(lower)
            or lower.ndim != 1 or not np.isfinite(lower).all()):
        raise ValueError('Complete finite original source/physical step domain required')
    responses = {}
    for family, response in domain.get('curvature_responses', {}).items():
        responses[family] = dict(response,
            dynamic_group_limits=_vector(response['dynamic_group_limits'], name='fixed family envelope'),
            dynamic_row_group_index=np.asarray(response['dynamic_row_group_index'], dtype=int).copy())
    report = dict(domain.get('report', {}))
    def at_trial_fraction(alpha):
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0. < alpha <= 1.:
            raise ValueError('Actual trial fraction must lie in (0, 1]')
        fresh = {family: dict(response,
            dynamic_group_limits=response['dynamic_group_limits'].copy(),
            dynamic_row_group_index=response['dynamic_row_group_index'].copy(),
            dynamic_group_target_basis='unchanged_fixed_experiment_seed_envelope')
            for family, response in responses.items()}
        return dict(matrix=matrix, lower=lower.copy(), curvature_responses=fresh,
            report=dict(report, trial_fraction_domain=dict(alpha=alpha, policy=POLICY,
                targets_constant_at_every_actual_trial_fraction=True,
                physical_and_seed_zero_group_faces_unchanged=True,
                accumulated_full_step_SOC_cuts_reused=False,
                no_fabricated_common_fraction=True)))
    return dict(domain, at_trial_fraction=at_trial_fraction,
                prefer_scaled_candidate_before_fraction_soc=True)
