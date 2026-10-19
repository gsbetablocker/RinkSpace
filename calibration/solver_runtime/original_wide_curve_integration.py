"""Optional original wide45 curve/exposure budgets outside component Huber2.

The original integer row expansion still owns the literal source weights and
outer factor four. This adapter only repeats authenticated sampling weights
onto those same rows; it never rescales residuals, derivatives or guards.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
from .source_loss import SourceLossWeights
from .original_board_curve_integration import isolated_anchor_masses
from rinkspace.source_sampling import compile_curve_sampling

SCHEMA = 'rinkspace.original-wide45-curve-source-integration.v1'
FAMILY = 'strong_high_wide_source'
CURVE_SCALARS = 4288
UNIQUE_SCALARS = 5808
EXPANDED_SCALARS = 23776


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False, default=lambda v:
        v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)))


def _numeric_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()


def checked_wide_curve_integration_reuse(options):
    """A declared new loss target cannot import an old normal or derivatives."""
    if getattr(options, 'original_wide_curve_integration_policy', None) is not None:
        for name in ('reuse_complete_linearization', 'reuse_source_blocks_from',
                     'reuse_wide_linearization', 'verify_source_blocks_from'):
            if getattr(options, name, None) is not None:
                raise ValueError('Original wide curve integration requires fresh source profiles/J and objective; old source cache refused')


def authenticate_wide_sampling_truth_continuation(original, current, runtime):
    """Retain source-support limits when only this explicit loss adapter is added.

    The live adapter has already authenticated its source pixels and sampling
    policy. Removing its one declared contract field must recover the entire
    old source contract exactly. No parameter, source role or guard migration
    is authorized here, and neither old costs nor old normals are reusable.
    """
    if (runtime is None or runtime.get('schema') != SCHEMA + '.runtime'
            or runtime != current.get('original_wide_curve_integration')
            or original.get('original_wide_curve_integration') is not None
            or runtime.get('original_UV_R_J_scales_masks_profiles_and_all_hard_supports_unchanged') is not True
            or runtime.get('fresh_source_J_and_profiles_required') is not True):
        raise ValueError('An explicitly constructed new wide sampling adapter is required')
    source = _plain(current['source_contract'])
    evaluation = source.get('evaluation_source_contract')
    if not isinstance(evaluation, dict) or evaluation.pop('original_wide_curve_integration', None) != _plain(runtime):
        raise ValueError('Actual wide sampling source contract differs from its live adapter')
    if source != _plain(original['source_contract']):
        raise ValueError('Source changed beyond the explicit wide sampling loss field')
    for key in ('full_parameters', 'source_rows', 'highest_direct_human_hard_truth',
                'highest_user_confirmed_native_sources', 'human_truth_support_operator',
                'camera_pivot_exterior_constraints', 'physical_course_scope_source',
                'camera_mount_working_constraints', 'principal_point_engineering_prior',
                'current_geometry', 'physical_geometry_initializer', 'original_ring_source_qualification',
                'original_ice_pixel_correction', 'original_measured_transition_locations',
                'original_source_point_qualification'):
        if key not in original or key not in current or _plain(original[key]) != _plain(current[key]):
            raise ValueError('Wide loss continuation changed a source or physical guard: ' + key)
    return dict(schema='rinkspace.authenticated-wide-sampling-loss-only-truth-continuation.v1',
        policy_reference=runtime['policy_reference'], loss_weight_identity=runtime['loss_weight_identity'],
        old_source_contract_recovered_exactly_by_removing_one_loss_field=True,
        original_parameters_source_rows_roles_and_all_guard_contracts_retained=True,
        saved_limits_uncertainties_and_fixed_seed_envelopes_reinitialized=False,
        old_costs_profiles_J_and_normals_reuse_authorized=False,
        complete_fresh_source_objective_required=True, accuracy_accepted=False)


class SourceWithOriginalWideCurveIntegration:
    def __init__(self, base, factor, policy_path, *, wide_source):
        path = Path(policy_path).resolve()
        policy = json.loads(path.read_text(encoding='utf-8-sig'))
        checked = {}
        def authenticate(reference):
            p = Path(reference['path']).resolve(); expected = reference['sha256']
            if p not in checked:
                with p.open('rb') as stream: checked[p] = hashlib.file_digest(stream, 'sha256').hexdigest()
            if checked[p] != expected: raise ValueError('Original wide integration source reference changed: ' + str(p))
            return p
        if (policy.get('schema') != SCHEMA or type(policy.get('curve_budget')) is bool
                or policy.get('curve_budget') != 1.
                or policy.get('support_basis') != 'isolated_authenticated_source_anchors'
                or 'normalization' not in policy or policy['normalization'] is not None
                or policy.get('enable_point_compression') is not False
                or policy.get('preserve_literal_source_weights') is not True
                or policy.get('preserve_global_engineering_multiplier') is not True
                or policy.get('global_engineering_multiplier') != 4.
                or policy.get('working_sigma_unchanged') is not True
                or policy.get('original_observation_indices') != list(range(45))
                or policy.get('original_UV_count') != 2144):
            raise ValueError('Explicit original wide45 isolated-anchor budget1 policy required')
        packet = factor.packet.contract
        if (policy.get('source_files') != packet['source_files']
                or policy.get('source_packet_reference') != packet['source_files']['v5_problem']
                or policy.get('source_video_sha256') != packet['source_video_sha256']
                or policy.get('source_pts') != packet['source_pts']
                or policy.get('native_pts') != packet['source_pts']
                or type(policy.get('actual_pts')) is not int or policy['actual_pts'] != packet['source_pts']
                or policy.get('time_base') != packet['source_time_base']
                or policy.get('native_image') != packet['native_image']):
            raise ValueError('Original wide integration source, exposure or native image identity differs')
        for reference in policy['source_files'].values(): authenticate(reference)
        audit = json.loads(authenticate(policy['source_support_audit_reference']).read_text(encoding='utf-8-sig'))
        if (audit.get('schema') != 'rinkspace.current-wide45-exact-original-support-audit.v1'
                or audit.get('source_packet_contract') != packet or len(audit.get('source_rows', [])) != 45):
            raise ValueError('Original wide physical identity requires its exact authenticated source-support audit')
        duplicate = policy['duplicate_anchor_policy']
        if duplicate.get('source_anchor_identity_authority') != policy['source_packet_reference']:
            raise ValueError('Original wide duplicate-anchor authority differs')
        import rinkspace.source_sampling as sampling
        compiler = Path(sampling.__file__).resolve()
        if authenticate(policy['public_compiler_reference']) != compiler:
            raise ValueError('Original wide integration must use the public source sampling compiler')
        observations = factor.observations; rows = policy.get('rows', [])
        if (len(observations) != 45 or len(rows) != 45 or len(factor.uvs) != 45
                or sum(len(uv) for uv in factor.uvs) != 2144):
            raise ValueError('Original wide45/2144 source inventory differs')
        records = []; summary = []
        for index, (original, uv, sigma, row, evidence) in enumerate(zip(
                observations, factor.uvs, factor.sigmas, rows, audit['source_rows'])):
            uv = np.asarray(uv, float); sigma = np.asarray(sigma, float); n = len(uv)
            literal = float(original.get('weight', 1.))
            if (row.get('original_v5_index') != index or row.get('id') != original['id']
                    or row.get('target') != original['target'] or row.get('points') != n
                    or row.get('original_point_indices') != list(range(n))
                    or row.get('source_UV_float64_sha256') != _numeric_sha(uv)
                    or not np.array_equal(uv, np.asarray(original['pixels_uv'], float))
                    or row.get('source_sigma_float64_sha256') != _numeric_sha(sigma)
                    or row.get('original_sigma_px') != original['sigma_px']
                    or not np.array_equal(sigma, np.broadcast_to(original['sigma_px'], (n,)))
                    or not np.array_equal(row.get('stored_sigma_px'), sigma)
                    or not np.array_equal(row.get('working_sigma_px'), sigma)
                    or not np.array_equal(row.get('prior_sampling_multiplier'), np.ones(n))
                    or row.get('sampling_kind') != ['measured'] * n
                    or row.get('literal_source_weight') != literal or row.get('global_engineering_multiplier') != 4.
                    or row.get('actual_pts') != policy['actual_pts']
                    or row.get('source_video_sha256') != packet['source_video_sha256']
                    or row.get('operator_id') != 'wide_full_Brown_native_xy_nearest_finite_curve:' + json.dumps(original['target'],sort_keys=True,separators=(',',':'))
                    or row.get('source_support_basis') != policy['support_basis']):
                raise ValueError('Original wide row UV/sigma/role/order or independent engineering weight changed: ' + original['id'])
            if (evidence.get('id') != original['id'] or evidence.get('original_v5_index') != index
                    or evidence.get('target') != original['target'] or evidence.get('points') != n
                    or evidence.get('native_uv_sha256') != row['source_UV_float64_sha256']
                    or any(row.get(k) != evidence.get(k) for k in
                           ('physical_curve_id', 'physical_identity_status', 'source_row_reference'))
                    or evidence.get('source_support_proof', {}).get('kind') != 'isolated_authenticated_original_anchor_support'
                    or evidence.get('source_support_proof', {}).get('continuous_support_claim') is not False):
                raise ValueError('Original wide curve identity or isolated support differs from authenticated evidence: ' + original['id'])
            mass = isolated_anchor_masses(uv)
            if not np.array_equal(row.get('point_integration_mass'), mass):
                raise ValueError('Original wide source support mass changed: ' + original['id'])
            record = dict(id=row['id'], split='fit', actual_pts=policy['actual_pts'],
                source_video_sha256=packet['source_video_sha256'], physical_curve_id=row['physical_curve_id'],
                operator_id=row['operator_id'], uv=uv, stored_sigma_px=sigma, working_sigma_px=sigma,
                prior_sampling_multiplier=np.ones(n), point_integration_mass=mass,
                sampling_kind=['measured'] * n, source_anchor_identity_authority=policy['source_packet_reference'])
            records.append(record); summary.append(dict(id=row['id'], original_v5_index=index,
                points=n, physical_curve_id=row['physical_curve_id'], operator_id=row['operator_id'],
                source_UV_float64_sha256=row['source_UV_float64_sha256'],
                source_sigma_float64_sha256=row['source_sigma_float64_sha256'], literal_source_weight=literal))
        plan = compile_curve_sampling(records, duplicate_anchor_policy=duplicate)
        if _plain(plan['groups']) != policy['groups']:
            raise ValueError('Original wide public physical-curve budget differs from declared policy')
        unique = np.ones(UNIQUE_SCALARS); cursor = 0
        for original, row, uv in zip(observations, rows, factor.uvs):
            weights = plan['rows'][row['id']]['point_integration_weights']; n = len(uv)
            if (not np.array_equal(row['point_integration_weights'], weights) or np.any(weights <= 0.)
                    or not np.array_equal(plan['rows'][row['id']]['working_normalization_multiplier'], np.ones(n))):
                raise ValueError('Original wide curve weights or undeclared sigma normalization changed: ' + original['id'])
            unique[cursor:cursor+2*n] = np.repeat(weights, 2); cursor += 2*n
        counts = np.rint(4 * np.asarray(factor.cost_weights)).astype(int)
        indices = np.repeat(np.arange(UNIQUE_SCALARS), counts)
        if (cursor != CURVE_SCALARS or len(factor.scales) != UNIQUE_SCALARS or wide_source.multiplier != 4
                or not np.array_equal(counts, 4 * factor.cost_weights)
                or not np.array_equal(counts, wide_source.row_counts)
                or not np.array_equal(indices, wide_source.row_indices) or len(indices) != EXPANDED_SCALARS
                or not np.array_equal(unique[CURVE_SCALARS:], np.ones(1520))):
            raise ValueError('Original wide scalar layout, integer replicas or independent multiplier changed')
        self.base = base
        self._payload = SourceLossWeights(unique[indices])
        self.original_wide_curve_integration = dict(schema=SCHEMA + '.runtime',
            policy_reference=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            source_support_audit_reference=policy['source_support_audit_reference'], source_rows=summary,
            source_files=policy['source_files'], groups=plan['groups'], physical_curve_budget=1.,
            support_basis=policy['support_basis'], unique_curve_scalar_rows=CURVE_SCALARS,
            unchanged_other_unique_scalar_rows=1520, unique_wide_scalar_rows=UNIQUE_SCALARS,
            numerical_wide_scalar_rows=EXPANDED_SCALARS, loss_weight_identity=self._payload.identity_digest.hex(),
            original_row_indices_float64_sha256=_numeric_sha(indices), working_normalization_multiplier=None,
            original_UV_R_J_scales_masks_profiles_and_all_hard_supports_unchanged=True,
            original_literal_weights_and_outer4_preserved_independently=True,
            replicas_are_numerical_quadrature_not_new_observations=True,
            original_corners_entity_and_TOP_objectives_unchanged=True,
            fresh_source_J_and_profiles_required=True, engineering_budget_not_measured_covariance=True,
            accuracy_accepted=False, implementation_sources=[dict(path=str(p.resolve()),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in
                (Path(__file__), Path(__file__).with_name('source_loss.py'), compiler)])
        self.contract = dict(base.contract, original_wide_curve_integration=self.original_wide_curve_integration)

    def __getattr__(self, name): return getattr(self.base, name)

    def apply_block(self, block):
        if block.row_ledger.get('family') != FAMILY: return block
        if (len(block.r) != EXPANDED_SCALARS or block.loss_weights is not None
                or 'original_wide_curve_integration' in block.row_ledger):
            raise ValueError('Original wide integration scalar layout changed or budget applied twice')
        from z190_streaming_source_normal import SourceRowBlock
        return SourceRowBlock(block.r, block.J, block.quadratic, block.scales,
            dict(block.row_ledger, original_wide_curve_integration=self.original_wide_curve_integration['policy_reference']), self._payload)

    def blocks(self, q, jacobian=True):
        found = False
        for block in self.base.blocks(q, jacobian):
            if block.row_ledger.get('family') == FAMILY:
                if found: raise ValueError('Original wide source family duplicated')
                found = True
            yield self.apply_block(block)
        if not found: raise ValueError('Original wide source family omitted')


def wrap_original_wide_curve_integration(base, factor, policy_path=None, *, wide_source=None):
    return base if policy_path is None else SourceWithOriginalWideCurveIntegration(
        base, factor, policy_path, wide_source=base if wide_source is None else wide_source)
