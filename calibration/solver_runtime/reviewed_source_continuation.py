"""Transport saved truth across reviewed soft-source data revisions.

The complete numerical seed, model layout and original truth envelopes are
immutable. Source installers retain responsibility for RGB/clock/row binding.
This adapter binds their exact revised inputs and normalizes only the named
source contracts back to the saved target; it never evaluates a source model.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import truth_source_location_revision as common

SCHEMA = 'rinkspace.reviewed-soft-source-same-state-truth-proof.v1'
OPTION = '--truth-restoration-reviewed-soft-source-proof'
COMPONENT_OPTIONS = {
    'original_ice_pixel_correction': '--original-ice-pixel-correction-source',
    'original_source_point_qualification': '--original-source-point-qualification-source',
    'static_motion_source_role_retirement': '--static-motion-source-role-retirement-source',
    'additional_real_native_board_circles': '--additional-board-circle-source',
    'additional_real_native_ice_curves': '--additional-ice-source',
    'additional_real_native_board_edges': '--additional-board-source',
    'additional_real_native_board_material_points': '--additional-board-material-point-source',
    'native_base_curve_integration': '--native-base-curve-integration-policy',
    'original_measured_transition_locations': '--original-measured-transition-location-source',
}
PROTECTED_OPTIONS = (common.NATIVE, common.CONFIRMED, common.GUARD, common.CAP,
                     common.ENVELOPE, '--original-wide-human-truth-authority')
ROW_FIELDS = ('id', 'pts', 'actual_pts', 'feature_id', 'curve', 'split', 'sigma_px',
    'normal', 'normal_interval_px', 'source_optical_normal_support_authorized',
    'curve_observation_kind', 'scanline_fixed_axis', 'scanline_branch',
    'source_raw_frame', 'source_path', 'source_file_sha256', 'source_frame_sha256')
MODEL_FIELDS = ('complete_parameter_layout', 'full_parameters', 'shared_indices',
    'independent_circle_material_parameter_indices', 'independent_pattern_placement_parameter_indices',
    'physical_course_scope_source', 'camera_mount_working_constraints',
    'principal_point_engineering_prior', 'human_truth_support_operator', 'actual_exposures',
    'native_presentations', 'original_motion_rows', 'original_motion_input_compaction',
    'original_source_role_retirement',
    'original_motion_far_board_role_correction', 'original_ring_source_qualification',
    'original_wide_curve_integration')
CONTRACT_MODEL_FIELDS = ('schema', 'parameter_names', 'parameter_count', 'prefix_parameter_count',
    'new_parameter_count', 'evaluation_core_parameter_count', 'logical_world_parameter_count',
    'logical_world_parameter_indices', 'circle_indices', 'active_circle_parameter_indices',
    'shared_radius_parameter_index', 'inactive_legacy_radius_parameter_index',
    'operator', 'observation_operator', 'trace_objective', 'material', 'material_operator_revision',
    'source_video_sha256', 'time_base', 'working_unit_scene_scale', 'radius_binding',
    'held_ids', 'held_observation_ids', 'held_rows_used_in_fit', 'original_packet',
    'original_absolute_packet', 'raw_sigma_used_without_sqrt_N', 'raw_sigma_px_preserved')
COUNT_FIELDS = frozenset(('source_rows', 'total_rows', 'original_source_rows',
    'additional_scalar_rows', 'scalar_rows', 'new_scalar_rows',
    'original_source_scalar_rows', 'source_scalar_rows', 'original_source_scalars_before_PP6'))
MOTION_COMPACTION_REPORT_FIELDS = ('actual_retained_r_J_check_seconds',
    'retained_r_max_scale', 'retained_J_max_scale', 'same_exposure_actual_r_max',
    'same_state_retained_local_r_max_difference', 'same_state_retained_local_J_max_difference',
    'same_state_retained_r_J_bitexact')


def motion_compaction_semantics(contract):
    """Keep exact source/CSR maps and original gates; separate fresh check reports."""
    result = deepcopy(contract)
    if result.get('schema') != 'rinkspace.authenticated-original-motion-input-compaction.v1':
        raise ValueError('Original authenticated motion compaction required')
    for row in result['blocks']:
        if 'same_state_retained_local_r_max_difference' not in row:continue
        tolerance = row['compaction_numerical_tolerance']
        if (row.get('same_state_retained_CSR_row_column_identity_bitexact') is not True
                or row.get('same_state_retained_r_J_passed_tight_compaction_numerical_tolerance') is not True
                or row.get('batch_wide_inverse_Brown_Newton_stop_preserved') is not True
                or tolerance.get('atol') != 1e-10 or tolerance.get('rtol') != 1e-12
                or tolerance.get('inverse_Brown_batch_Newton_stop') != 2e-13
                or tolerance.get('J_relative_scale') != 'each_reference_CSR_row_Euclidean_norm'
                or tolerance.get('J_difference_measure') != 'complete_CSR_row_Euclidean_norm'
                or tolerance.get('certified_operation_error_bound') is not False):
            raise ValueError('Original retained motion CSR/numerical gates changed or failed')
        values = [row[k] for k in MOTION_COMPACTION_REPORT_FIELDS if k in row and k != 'same_state_retained_r_J_bitexact']
        fraction = tolerance['J_max_error_fraction']
        if (not np.isfinite(values).all() or np.any(np.asarray(values) < 0)
                or not np.isfinite(fraction) or not 0 <= fraction <= 1
                or row['same_state_retained_local_r_max_difference'] >
                    tolerance['atol'] + tolerance['rtol'] * row['retained_r_max_scale']):
            raise ValueError('Fresh original motion compaction verification is invalid')
        for key in MOTION_COMPACTION_REPORT_FIELDS:row.pop(key, None)
        tolerance.pop('J_max_error_fraction')
    return result


def _same(a, b):
    return common._same(a, b)


def _packet_ref(contract, key):
    if key == 'native_base_curve_integration':
        return contract['policy_reference']
    if key == 'static_motion_source_role_retirement':
        return contract['source_reviewed_dynamic_validity_append']
    if 'packet_reference' in contract:return contract['packet_reference']
    return dict(path=contract['packet_path'], sha256=contract['packet_sha256'])


def _replace(value, replacements, hashes=()):
    return common._replace_bindings(value, replacements, dict(hashes))


def _budget_values(value, path=()):
    """Declared sigma/weight/mass/budget fields remain source data invariants."""
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in key.lower() for word in ('sigma', 'weight', 'budget', 'mass', 'normalization')):
                # Records added by qualification/UV revision carry their own
                # original-source sigma proof; compare packet rows separately.
                found[path + (key,)] = item
            elif key not in ('records', 'source_support', 'source_append_v3', 'source_append_v4', 'source_append_v5'):
                found.update(_budget_values(item, path + (key,)))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.update(_budget_values(item, path + (i,)))
    return found


def _validate_packet_change(old_ref, new_ref, key):
    old = common._read(common._checked(old_ref)); new = common._read(common._checked(new_ref))
    for field in ('schema', 'source_video_sha256', 'time_base', 'image_size_wh',
                  'geometry_contract', 'original_absolute_packet', 'original_packet'):
        # A native qualification's immutable parent may acquire new UV through
        # its separately authenticated existing loader chain.
        if field != 'original_packet' or key != 'additional_real_native_ice_curves':
            if not _same(old.get(field), new.get(field)):
                raise ValueError('Reviewed source revision changed identity/model: ' + key + ':' + field)
    if 'rows' in old:
        if len(old['rows']) != len(new['rows']):
            raise ValueError('Reviewed UV revision cannot add/delete physical source rows')
        from .source_point_qualification import _direct_user_annotation
        for before, after in zip(old['rows'], new['rows']):
            if any(not _same(before.get(f), after.get(f)) for f in ROW_FIELDS):
                raise ValueError('Reviewed UV revision changed source clock/curve/sigma/operator: ' + before['id'])
            if np.asarray(before['uv']).shape != np.asarray(after['uv']).shape:
                raise ValueError('Reviewed UV revision changed original query inventory')
            if (before['split'] != 'fit' or _direct_user_annotation(before)) and not _same(before['uv'], after['uv']):
                raise ValueError('Reviewed soft revision changed HELD or direct user UV')
        budgets_old, budgets_new = _budget_values(old), _budget_values(new)
        for field, value in budgets_old.items():
            if field in budgets_new and not _same(value, budgets_new[field]):
                raise ValueError('Reviewed source revision changed declared weight/budget: ' + str(field))
    elif key == 'static_motion_source_role_retirement':
        a, b = old, new
        previous = {r['current_original_prefix_bundle_index']: r for r in a['records']}
        actual = {r['current_original_prefix_bundle_index']: r for r in b['records']}
        if any(actual.get(i) != r for i, r in previous.items()):
            raise ValueError('An existing moving-foreground retirement changed')
        for row in b['records']:
            if (row.get('original_UV_sigma_weights_clock_split_preserved') is not True
                    or row.get('source_camera_or_projection_or_residual_used_for_validity') is not False
                    or not np.array_equal(row['expected_original_finite_plane'], np.zeros(4))):
                raise ValueError('Only original observed foreground validity may change')
    elif key == 'original_source_point_qualification':
        before = {r['original_packet_row_index']: r for r in old['records']}
        after = {r['original_packet_row_index']: r for r in new['records']}
        for index, row in before.items():
            other = after.get(index)
            if other is None or not set(row['excluded_point_indices']) <= set(other['excluded_point_indices']):
                raise ValueError('Source qualification restored a retired pixel without review')
    elif key == 'native_base_curve_integration':
        values_old = _budget_values(old) | _budget_values(old.get('records', []), ('records',))
        values_new = _budget_values(new) | _budget_values(new.get('records', []), ('records',))
        if (any(k not in values_new or not _same(v,values_new[k]) for k,v in values_old.items())
                or any(type(v) not in (bool,str) for k,v in values_new.items() if k not in values_old)):
            raise ValueError('Native integration revision changed declared sigma/weight/mass/budget')
    return old, new


def _source_scalar_delta(old, new, key):
    if key == 'original_source_point_qualification':
        def excluded(packet):
            return {(r['original_packet_row_index'], i) for r in packet['records'] for i in r['excluded_point_indices']}
        return -2 * len(excluded(new[1]) - excluded(old[1]))
    if key != 'additional_real_native_ice_curves':
        return 0
    from .truth_source_scanline_revision import original_qualified_packet
    from .source_point_qualification import load_source_point_qualification
    def count(reference, packet):
        original = original_qualified_packet(reference['path'], packet)
        qualifier = load_source_point_qualification(reference['path'], original,
            source_family='additional_native_ice_curves')
        return int(2 * sum(np.count_nonzero(qualifier.point_mask(r)) if qualifier else len(r['uv'])
            for r in packet['rows'] if r['split'] == 'fit'))
    return count(new[0], new[1]) - count(old[0], old[1])


def prepare_engineering_reference_rebind(source_options, output_dir):
    """Pure reference transport of the existing fixed engineering policy."""
    options = deepcopy(source_options)
    key = '--truth-restoration-seed-engineering-scale-source'
    path = options.get(key)
    if path is None:return dict(source_options=options, rebound_files=[])
    original = common._read(path); derived = deepcopy(original)
    pairs = (('native_authority', common.NATIVE), ('location_envelope', common.ENVELOPE))
    for field, option in pairs:
        common._checked(original[field])
        derived[field] = common.reference(options[option])
    if derived == original:return dict(source_options=options, rebound_files=[])
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    reference = common._write(output / 'fixed_seed_engineering_scale_reference_rebind.json', derived)
    options[key] = reference['path']
    return dict(source_options=options, rebound_files=[dict(option_key=key,
        parent_reference=common.reference(path), derived_reference=reference)])


def prepare_reviewed_source_revision(truth_state_path, state_identity_path, source_options,
        output_dir, *, evidence_references, rebound_files=(), expected_actual_contracts=()):
    """Freeze actual revised input references; source installers run at startup."""
    state = common._read(truth_state_path)
    original = common._read(common._checked(state['source_input_manifest_reference']))
    identity = common._read(state_identity_path)
    if (identity['q_sha256'] != state['complete_q_sha256']
            or identity['input_manifest_sha256'] != state['source_input_manifest_reference']['sha256']
            or identity['state_npz_sha256'] != state['complete_state_reference']['sha256']):
        raise ValueError('Saved authoritative complete state/input identity differs')
    components = []
    delta = 0
    for key, option in COMPONENT_OPTIONS.items():
        if key not in original or original[key] is None or option not in source_options:
            continue
        before = _packet_ref(original[key], key); consumer = common.reference(source_options[option]);after = consumer
        if key == 'static_motion_source_role_retirement':
            compound = common._read(consumer['path'])
            ui = original[key]['original_UI_retirement_contract']
            if compound['original_UI_retirement'] != dict(path=ui['packet_path'],sha256=ui['packet_sha256']):
                raise ValueError('Original UI motion retirement changed')
            after = compound['source_reviewed_dynamic_validity_append']
        if Path(before['path']).resolve() == Path(after['path']).resolve() and before['sha256'] == after['sha256']:
            continue
        old, new = _validate_packet_change(before, after, key)
        change = _source_scalar_delta((before, old), (after, new), key)
        delta += change
        components.append(dict(manifest_key=key, option_key=option, original_reference=before,
            revised_reference=after, consumer_input_reference=consumer,scalar_row_delta=change))
    if not components:
        raise ValueError('An actual reviewed source target change is required')
    for ref in evidence_references:
        common._checked(ref)
    for item in expected_actual_contracts:
        common._checked(item['evidence_reference'])
    # Certify protected documents now, before constructing the full source.
    inverse = [(r['revised_reference'], r['original_reference']) for r in components]
    inverse += [(r['derived_reference'], r['parent_reference']) for r in rebound_files]
    hashes = {}
    for component in components:
        old = common._read(component['original_reference']['path']); new = common._read(component['revised_reference']['path'])
        if 'rows' in old:
            for before, after in zip(old['rows'], new['rows']):
                if _same(before['uv'], after['uv']):hashes[common._row_hash(after)] = common._row_hash(before)
    for item in rebound_files:
        if item['option_key'] == common.INTEGRATION:continue  # separately verified policy/budget + exact active pairs
        before = common._read(common._checked(item['parent_reference']))
        after = common._read(common._checked(item['derived_reference']))
        if not _same(_replace(after, inverse, hashes.items()), before):
            raise ValueError('Protected reference rebind changed original observations/support/engineering scales')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    proof = dict(schema=SCHEMA, original_truth_state=common.reference(truth_state_path),
        original_state_identity=common.reference(state_identity_path),
        complete_q_sha256=state['complete_q_sha256'], original_source_target=state['source_target'],
        original_fixed_seed_envelopes_sha256=hashlib.sha256(common.canonical(state.get('fixed_seed_envelopes'))).hexdigest(),
        original_motion_compaction_semantics_sha256=hashlib.sha256(common.canonical(
            motion_compaction_semantics(original['original_motion_input_compaction']))).hexdigest(),
        components=components, expected_scalar_row_delta=delta,
        reviewed_source_evidence=list(evidence_references), rebound_files=list(rebound_files),
        expected_actual_contracts=list(expected_actual_contracts),
        protected_group_count=sum(len(r['group_ids']) for r in state['families'].values()),
        original_truth_limits_uncertainties_and_fixed_seed_envelopes_preserved=True,
        fresh_initial_profiles_normal_and_J_required=True, ensuing_iteration_acceleration_preserved=True)
    proof_hash = hashlib.sha256(common.canonical(proof)).hexdigest()
    reference = common._write(output / ('reviewed_soft_source_truth_proof_' + proof_hash[:16] + '.json'), proof)
    return dict(proof_reference=reference, source_options={**source_options, OPTION: reference['path']})


def _normalize_contract(value, replacements, counts, reference_pairs, hashes):
    token = hashlib.sha256(common.canonical(value)).hexdigest()
    if token in replacements:
        return deepcopy(replacements[token])
    if isinstance(value, list):
        return [_normalize_contract(v, replacements, counts, reference_pairs, hashes) for v in value]
    if not isinstance(value, dict):
        return hashes.get(value, value) if isinstance(value, str) else value
    result = {k: _normalize_contract(v, replacements, counts, reference_pairs, hashes) for k, v in value.items()}
    result = _replace(result, reference_pairs, hashes.items())
    for key in COUNT_FIELDS:
        if key in value and type(value[key]) is int and (key, value[key]) in counts:
            result[key] = counts[key, value[key]]
    return result


def authenticate_reviewed_source_revision(proof_path, state_path, q, manifest, factors):
    proof = common._read(proof_path); state = common._read(state_path)
    if proof.get('schema') != SCHEMA or common._checked(proof['original_truth_state']) != Path(state_path).resolve():
        raise ValueError('Explicit saved-state reviewed soft-source proof required')
    identity = common._read(common._checked(proof['original_state_identity']))
    original = common._read(common._checked(state['source_input_manifest_reference']))
    with np.load(common._checked(state['complete_state_reference']), allow_pickle=False) as arrays:
        if not np.array_equal(np.asarray(q, np.float64), arrays['q']):
            raise ValueError('Reviewed source continuation changed complete numerical seed')
    if hashlib.sha256(np.asarray(q, np.float64).tobytes()).hexdigest() != proof['complete_q_sha256'] or identity['q_sha256'] != proof['complete_q_sha256']:
        raise ValueError('Reviewed source continuation complete q identity differs')
    if state['source_target'] != common._target(original) or proof['original_source_target'] != state['source_target']:
        raise ValueError('Reviewed source continuation saved mathematical target differs')
    if hashlib.sha256(common.canonical(state.get('fixed_seed_envelopes'))).hexdigest() != proof['original_fixed_seed_envelopes_sha256']:
        raise ValueError('Original fixed seed envelope changed')
    for ref in proof['reviewed_source_evidence']:
        common._checked(ref)
    replacements = {}; inverse_refs = []; row_hashes = {}; delta = 0
    for component in proof['components']:
        key = component['manifest_key']
        if COMPONENT_OPTIONS.get(key) != component['option_key']:
            raise ValueError('Unknown reviewed source consumer')
        old_ref, new_ref = component['original_reference'], component['revised_reference']
        common._checked(component['consumer_input_reference'])
        if not _same(_packet_ref(original[key], key), old_ref) or not _same(_packet_ref(manifest[key], key), new_ref):
            raise ValueError('Actual loaded source consumer differs from the frozen reviewed input: ' + key)
        old, new = _validate_packet_change(old_ref, new_ref, key)
        change = _source_scalar_delta((old_ref, old), (new_ref, new), key)
        if change != component['scalar_row_delta']:
            raise ValueError('Reviewed source active-query layout delta differs')
        delta += change
        for field in CONTRACT_MODEL_FIELDS:
            if not _same(original[key].get(field), manifest[key].get(field)):
                raise ValueError('Reviewed source consumer changed model/operator/sigma semantics: ' + key + ':' + field)
        replacements[hashlib.sha256(common.canonical(manifest[key])).hexdigest()] = original[key]
        inverse_refs.append((new_ref, old_ref))
        if 'rows' in old:
            for before, after in zip(old['rows'], new['rows']):
                # Protected hashes may rebind metadata only, never actual UV.
                if _same(before['uv'], after['uv']):
                    row_hashes[common._row_hash(after)] = common._row_hash(before)
    for item in proof.get('expected_actual_contracts', []):
        key = item['manifest_key']
        if key not in ('original_board_curve_integration', 'original_ice_curve_integration', 'native_base_curve_integration'):
            raise ValueError('Only existing independently validated absolute sampling contracts may bind')
        expected = common._read(common._checked(item['evidence_reference']))
        for field in item['value_path']:
            expected = expected[field]
        if not _same(manifest.get(key), expected):
            raise ValueError('Actual absolute sampling contract differs from public source validation: ' + key)
        fields = ('schema','apply_to_base_native_blocks_only','direct_manual_centre_and_all_human_truth_supports_unchanged') if key == 'native_base_curve_integration' else ('schema', 'policy_reference', 'original_packet', 'source_video_sha256', 'physical_curve_budget', 'loss')
        for field in fields:
            if not _same(expected.get(field), original[key].get(field)):
                raise ValueError('Public absolute sampling revision changed original policy/budget: ' + key + ':' + field)
        replacements[hashlib.sha256(common.canonical(expected)).hexdigest()] = original[key]
    if delta != proof['expected_scalar_row_delta'] or manifest['source_rows'] != original['source_rows'] + delta:
        raise ValueError('Complete source rows differ outside the reviewed observability delta')
    for key in MODEL_FIELDS:
        if key == 'original_motion_input_compaction':
            before = motion_compaction_semantics(original[key]); after = motion_compaction_semantics(manifest[key])
            if (not _same(before, after) or proof.get('original_motion_compaction_semantics_sha256') !=
                    hashlib.sha256(common.canonical(before)).hexdigest()):
                raise ValueError('Original motion compaction source/maps/CSR/layout/gates changed')
            replacements[hashlib.sha256(common.canonical(manifest[key])).hexdigest()] = original[key]
            continue
        if not _same(original.get(key), manifest.get(key)):
            raise ValueError('Reviewed source revision changed an unapproved model/layout/input: ' + key)
    rebound = proof['rebound_files']
    for item in rebound:
        if item['option_key'] == common.INTEGRATION:continue
        common._checked(item['parent_reference']); common._checked(item['derived_reference'])
        inverse_refs.append((item['derived_reference'], item['parent_reference']))
    for item in rebound:
        if item['option_key'] == common.INTEGRATION:continue  # independently bound public active-pair contract above
        old = common._read(item['parent_reference']['path']); new = common._read(item['derived_reference']['path'])
        if not _same(_replace(new, inverse_refs, row_hashes.items()), old):
            raise ValueError('Protected source rebind changed UV/support/caps/engineering scale: ' + item['option_key'])
    # Existing absolute compaction may physically omit exactly the newly
    # source-qualified original queries; its immutable original policy stays.
    if not _same(original.get('original_absolute_input_compaction'), manifest.get('original_absolute_input_compaction')):
        before = original['original_absolute_input_compaction']; after = manifest['original_absolute_input_compaction']
        for field in ('packet_path', 'packet_sha256', 'schema'):
            if before.get(field) != after.get(field):
                raise ValueError('Original absolute compaction policy changed')
        components = {r['manifest_key']: r for r in proof['components']}
        component = components.get('original_source_point_qualification')
        if component is None:raise ValueError('Absolute compaction changed without reviewed original query qualification')
        oldq = common._read(component['original_reference']['path']); newq = common._read(component['revised_reference']['path'])
        oldsets = {r['original_packet_row_index']: set(r['excluded_point_indices']) for r in oldq['records']}
        allowed = {(r['locator']['block'], r['locator']['group_order'], r['locator']['row_order']):
            set(r['excluded_point_indices']) - oldsets.get(r['original_packet_row_index'], set()) for r in newq['records']}
        from .original_source_point_qualification import OriginalSourcePointQualification
        qualified = OriginalSourcePointQualification(component['revised_reference']['path'])
        source_rows = {(r['item']['block'], r['item']['group_order'], r['item']['row_order']): r
            for r in qualified.records}
        if len(before['blocks']) != len(after['blocks']):raise ValueError('Absolute source block inventory changed')
        expected_blocks = []; cursor = 0; deleted = 0
        for a, b in zip(before['blocks'], after['blocks']):
            if a['block'] != b['block']:raise ValueError('Absolute source block order changed')
            rows = {(r['group_order'], r['row_order']): deepcopy(r) for r in a['row_maps']}
            removed = 0
            for locator, excluded in allowed.items():
                if locator[0] != a['block'] or not excluded:continue
                selected = source_rows[locator]; original_row = selected['original']; key = locator[1:]
                n = len(original_row['uv'])
                if key in rows:
                    row = rows[key]
                    if (row['old_points'] != n or row['id'] != original_row['id']
                            or row['actual_pts'] != selected['item']['group_actual_pts']):
                        raise ValueError('Original absolute map differs from immutable qualified source row')
                    kept = row['kept_original_point_indices']
                else:
                    kept = list(range(n))
                    row = dict(group_order=key[0], row_order=key[1],
                        actual_pts=selected['item']['group_actual_pts'], id=original_row['id'], old_points=n)
                remaining = [i for i in kept if i not in excluded]
                removed += len(kept) - len(remaining)
                row['kept_original_point_indices'] = remaining
                row['retired_source_identity_point_indices'] = selected['record']['excluded_point_indices']
                rows[key] = row
            expected = deepcopy(a)
            expected['row_maps'] = [rows[k] for k in sorted(rows)]
            expected['compact_start'] = cursor
            expected['compact_rows'] -= 2 * removed
            cursor += expected['compact_rows']; deleted += removed
            expected_blocks.append(expected)
        expected = deepcopy(before)
        expected['blocks'] = expected_blocks
        expected['removed_unqualified_absolute_points'] += deleted
        expected['removed_active_scalar_rows'] += 2 * deleted
        if (deleted != sum(len(indices) for indices in allowed.values()) or not _same(expected, after)):
            raise ValueError('Absolute compaction source/maps/counts/layout changed outside exact qualified queries')
        replacements[hashlib.sha256(common.canonical(after)).hexdigest()] = before
    # Existing source wrappers append their implementation identity to an
    # otherwise complete, identical typed consumer contract. Authenticate
    # that exact variant instead of allowing individual changed fields.
    component_keys = {r['manifest_key'] for r in proof['components']}
    def bind_consumer_variants(before, after):
        if not isinstance(before, dict) or not isinstance(after, dict):return
        for key in before.keys() & after.keys():
            if key in component_keys and isinstance(after[key], dict):
                a, b = before[key], after[key]
                old_extra = a.keys() - original[key].keys()
                new_extra = b.keys() - manifest[key].keys()
                if (old_extra == new_extra == {'implementation_sources'}
                        and _same(a['implementation_sources'], b['implementation_sources'])
                        and _same({k:v for k,v in a.items() if k not in old_extra}, original[key])
                        and _same({k:v for k,v in b.items() if k not in new_extra}, manifest[key])):
                    replacements[hashlib.sha256(common.canonical(b)).hexdigest()] = a
            bind_consumer_variants(before[key], after[key])
    bind_consumer_variants(original['source_contract'], manifest['source_contract'])
    counts = {}
    allowed_deltas = {0}
    for component in proof['components']:
        allowed_deltas |= {n + component['scalar_row_delta'] for n in tuple(allowed_deltas)}
    def count_pairs(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a.keys() & b.keys():
                if key in COUNT_FIELDS and type(a[key]) is int and type(b[key]) is int and a[key] != b[key]:
                    if b[key] - a[key] not in allowed_deltas:
                        raise ValueError('Composite source scalar count changed by an unapproved amount')
                    counts[key, b[key]] = a[key]
                elif key not in COUNT_FIELDS:
                    count_pairs(a[key], b[key])
    # Replace typed source contract leaves before accepting propagated counts.
    current = _normalize_contract(manifest['source_contract'], replacements, {}, inverse_refs, row_hashes)
    count_pairs(original['source_contract'], current)
    normalized = _normalize_contract(current, {}, counts, (), {})
    if not _same(normalized, original['source_contract']):
        raise ValueError('Mathematical source contract changed outside the exact reviewed source consumers')
    for key in ('highest_direct_human_hard_truth', 'highest_user_confirmed_native_sources'):
        if not _same(_replace(manifest[key], inverse_refs, row_hashes.items()), original[key]):
            raise ValueError('Actual protected truth contract changed beyond reference rebind: ' + key)
    resumed = deepcopy(state); active = {k: f for k, f in factors.items() if f is not None}
    if set(active) != set(state['families']):
        raise ValueError('Protected truth family inventory changed')
    rebound_by_key = {r['option_key']: r['derived_reference'] for r in rebound}
    for family, factor in active.items():
        record = resumed['families'][family]
        key = common.NATIVE if family == 'native' else common.CONFIRMED if family == 'confirmed' else '--original-wide-human-truth-authority'
        expected = rebound_by_key.get(key, record['authority'])
        if not _same(factor.contract['authority'], expected):
            raise ValueError('Actual loaded hard source authority differs: ' + family)
        ids = [r['record']['id'] if family == 'wide' else r['row']['id'] for r in factor.records]
        if ids != record['group_ids'] or len(set(ids)) != len(ids):
            raise ValueError('Protected group identity/order changed')
        for field in ('limits', 'uncertainties'):
            values = np.asarray(record[field], float)
            if values.shape != (len(ids),) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Original finite truth limits required')
        record['authority'] = expected
    scale_key = '--truth-restoration-seed-engineering-scale-source'
    if scale_key in rebound_by_key:
        saved = resumed.get('seed_engineering_scale_source')
        if saved is None:raise ValueError('A continued engineering policy cannot add original seed scales')
        resumed['seed_engineering_scale_source'] = dict(saved, **rebound_by_key[scale_key])
    if sum(len(r['group_ids']) for r in resumed['families'].values()) != proof['protected_group_count']:
        raise ValueError('Protected truth group count changed')
    migration = dict(proof_reference=common.reference(proof_path), complete_q_sha256=proof['complete_q_sha256'],
        original_source_target=state['source_target'], actual_current_source_target=common._target(manifest),
        original_fixed_seed_envelopes_and_limits_preserved=True, original_truth_packet_not_rewritten=True,
        fresh_initial_profiles_normal_and_J_required=True, ensuing_iteration_acceleration_preserved=True)
    resumed['authenticated_reviewed_soft_source_revision'] = migration
    return dict(packet=resumed, migration=migration)
