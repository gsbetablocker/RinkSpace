"""Filter an existing thin fit using original exposure roles and source verdicts.

Raw/physical membership is immutable. Inspection is the physical nonfit
complement. Budgets preserve the producer's amount per eligible exposure;
this changes the objective and never certifies independent validation.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import json
import time
import numpy as np

from .io_utils import read_json, write_json, file_sha256
from .joint_metrics import PacketArrays
from .compact_identity_withhold_v2 import _source_identities

PROTECTED = {'validation_guard', 'reserved_pending_lineage', 'sealed_validation'}


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    return dict(file=str(Path(path).resolve()), sha256=file_sha256(path))


def checked(ref, base=None):
    path = Path(ref['file'])
    if not path.is_absolute():
        path = Path(base) / path
    require(file_sha256(path) == ref['sha256'], 'Changed source authority: ' + str(path))
    return path


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), 'Immutable output exists: ' + str(path))
    write_json(path, value)
    return reference(path)


def write_array(path, values, dtype):
    path = Path(path)
    require(not path.exists(), 'Immutable binary output exists: ' + str(path))
    values = np.asarray(values, dtype=dtype)
    values.tofile(path)
    return dict(**reference(path), dtype=values.dtype.str, shape=list(values.shape))


def source_verdict_rows(reader, verdict, *, require_packet_binding=True):
    packet_sha = file_sha256(reader.path)
    require(verdict['schema'] == 'rinkfit.source_identity_verdicts.v1', 'Unknown source identity verdict schema')
    if require_packet_binding:
        require(verdict['source_packet_sha256'] == packet_sha, 'Identity verdict binds another source packet')
    rows = verdict.get('rows', [])
    indices = [int(row['raw_row']) for row in rows]
    require(len(set(indices)) == len(indices), 'Duplicate source identity verdict row')
    n = int(reader.packet['row_count'])
    require(all(0 <= i < n for i in indices), 'Identity verdict outside complete source domain')
    identities, _ = _source_identities(reader.path, indices) if indices else ({}, None)
    cols = {key: reader.load(key) for key in ('uv', 'native_pts', 'shot_index', 'material_index', 'path')}
    for row in rows:
        require((set(cols) | {'raw_row', 'source_row_id', 'decision', 'uses_fitted_model_projection'}) <= set(row),
                'Incomplete source identity verdict tuple')
        i = int(row['raw_row'])
        require(identities[i] == row['source_row_id'], 'Source ID changed at verdict row ' + str(i))
        require(row['uses_fitted_model_projection'] is False, 'Model projection is not source identity evidence')
        for key in cols:
            if key in row:
                require(np.array_equal(cols[key][i], row[key]), 'Source verdict changed ' + key + ' at row ' + str(i))
        require(row['decision'] in ('confirmed', 'rejected', 'ambiguous', 'unchecked'), 'Unknown source identity decision')
    return rows


def rebind_identity_verdicts(packet, original, output):
    """Carry only exact unchanged row evidence to a new packet identity."""
    reader = PacketArrays(packet)
    verdict = read_json(original)
    source_verdict_rows(reader, verdict, require_packet_binding=False)
    for proof in verdict.get('evidence_files', []):
        checked(proof, Path(original).parent)
    current = deepcopy(verdict)
    current['source_packet_sha256'] = file_sha256(packet)
    current['rebound_from'] = reference(original)
    current['rebind_contract'] = dict(all_listed_ID_UV_PTS_shot_material_path_exact=True,
                                    old_training_mask_or_selection_reused=False,
                                    listed_source_identity_evidence_only=True,
                                    complete_source_identity_proven=False)
    # Absolute references preserve the original image/registration bytes.
    current['evidence_files'] = [reference(checked(p, Path(original).parent))
                                 for p in verdict.get('evidence_files', [])]
    return write_new(output, current)


def allowed_exposures(authority, model):
    require(authority['schema'] == 'rinkfit.joint_source_frame_protection.v1', 'Original frame roles required')
    require(authority['source_video_sha256'] == model['source_clock']['source_video_sha256'], 'Frame roles bind another source video')
    return authority['frame_roles']


def exposure_permission(shot, pts, shot_ids, roles):
    result = np.zeros(len(shot), bool)
    keys, inverse = np.unique(np.c_[shot, pts], axis=0, return_inverse=True)
    allowed = []
    for sid, pt in keys:
        role = roles.get(str(int(pt)))
        require(role is not None and role['shot_id'] == shot_ids[int(sid)], 'Unknown or cross-shot original native exposure')
        allowed.append(role['role'] not in PROTECTED)
    result[:] = np.asarray(allowed, bool)[inverse]
    return result


def component_keys(rows, arrays, canonical):
    return np.c_[arrays['shot_index'][rows], arrays['native_pts'][rows],
                 canonical[arrays['material_index'][rows]], arrays['kind'][rows]].astype('<i8')


def motion_endpoints(config, model, roles_ref, roles):
    if not config.get('static_surface_motion_v1'):
        return np.empty((0, 2), '<i8'), np.empty((0, 2), '<i8'), None
    ref = config['static_surface_motion_v1']
    motion = read_json(checked(ref))
    require(motion['frame_role_authority']['sha256'] == roles_ref['sha256'], 'Motion and pixels must consume the same original role authority')
    checked(motion['frame_role_authority'])
    require(motion['all_fitted_motion_exposures_pass_original_role_guard'] is True, 'Motion role check missing')
    table = np.fromfile(checked(motion['table']), dtype=motion['table_dtype']).reshape(motion['table_shape'])
    require(table.shape == (motion['selected_matches'], 12) and np.isfinite(table).all(), 'Invalid full motion table')
    first, second = table[:, [0, 1]].astype('<i8'), table[:, [0, 2]].astype('<i8')
    shot_ids = list(model['shots'])
    require(exposure_permission(first[:, 0], first[:, 1], shot_ids, roles).all() and
            exposure_permission(second[:, 0], second[:, 1], shot_ids, roles).all(), 'Protected original exposure in motion objective')
    return first, second, ref


def recalculate_budget(reader, model, config, fit, physical, roles_ref, roles):
    arrays = {key: reader.load(key) for key in ('shot_index', 'native_pts', 'material_index', 'kind')}
    canonical = np.arange(len(model['materials']))
    policy = deepcopy(config['engineering_support_budget_v2'])
    require(policy['source_packet_sha256'] == file_sha256(reader.path), 'Budget binds another current packet')
    require(policy['scheme'] == 'equal_native_exposure' and policy['calibrated_covariance'] is False, 'Unsupported engineering budget policy')
    rules = read_json(checked(policy['canonical_material_entities']))
    checked(rules['source_review'])
    assigned = set()
    for group in rules['groups']:
        representative = int(group['materials'][0]['material_index'])
        for item in group['materials']:
            i = int(item['material_index'])
            require(i not in assigned and model['materials'][i]['label'] == item['material_label'], 'Canonical material group changed')
            canonical[i] = representative
            assigned.add(i)
    components, inverse, counts = np.unique(component_keys(fit, arrays, canonical), axis=0,
                                            return_inverse=True, return_counts=True)
    first, second, motion_ref = motion_endpoints(config, model, roles_ref, roles)
    motion_counts = Counter(map(tuple, np.r_[first, second].tolist()))
    per_exposure_components = Counter(map(tuple, components[:, :2].tolist()))
    for key in motion_counts:
        per_exposure_components[key] += 1
    old_exposures = set(map(tuple, np.c_[arrays['shot_index'][physical], arrays['native_pts'][physical]].tolist())) | set(motion_counts)
    require(old_exposures and per_exposure_components, 'Empty engineering budget exposure domain')
    per_exposure = float(policy['total_budget']) / len(old_exposures)
    policy['total_budget'] = per_exposure * len(per_exposure_components)
    component_weights = np.asarray([per_exposure / per_exposure_components[tuple(key[:2])] / count
                                    for key, count in zip(components, counts)], dtype='<f8')
    weights = component_weights[inverse]
    endpoint_weights = np.asarray([[per_exposure / per_exposure_components[tuple(key)] / motion_counts[tuple(key)]
                                    for key in pair] for pair in zip(first, second)], dtype='<f8').reshape(-1, 2)
    combined = float(weights.sum() + endpoint_weights.sum())
    require(np.isfinite(weights).all() and np.all(weights > 0) and
            abs(combined - policy['total_budget']) <= 1e-10 * policy['total_budget'], 'Engineering budget does not conserve')
    information = dict(original_total_budget=config['engineering_support_budget_v2']['total_budget'],
                       total_budget=policy['total_budget'], old_exposures=len(old_exposures),
                       eligible_exposures=len(per_exposure_components),
                       removed_protected_exposures=len(old_exposures)-len(per_exposure_components),
                       original_per_exposure_budget=per_exposure, new_per_exposure_budget=per_exposure,
                       eligible_source_components=len(components), motion_exposures=len(motion_counts),
                       fit_pixel_budget=float(weights.sum()), motion_endpoint_budget=float(endpoint_weights.sum()),
                       total_recomputed=combined, motion=motion_ref,
                       source_sigma_or_W_changed=False, calibrated_covariance=False)
    return policy, weights, endpoint_weights, information


def prepare(*, packet, model, original_selection, frame_roles, source_identity_verdicts, config, output, append_source_proof=None, resume_existing_outputs=False):
    started = time.monotonic()
    new_json, new_array = write_new, write_array
    if resume_existing_outputs:
        def new_json(path, value):
            path = Path(path)
            if not path.exists(): return write_new(path, value)
            require(read_json(path) == value, 'Existing incomplete JSON differs from recomputed output: ' + str(path))
            return reference(path)
        def new_array(path, values, dtype):
            path = Path(path); values = np.asarray(values, dtype=dtype)
            if not path.exists(): return write_array(path, values, dtype)
            require(path.stat().st_size == values.nbytes and np.array_equal(np.fromfile(path, dtype=dtype), values.reshape(-1)),
                    'Existing incomplete array differs from recomputed output: ' + str(path))
            return dict(**reference(path), dtype=values.dtype.str, shape=list(values.shape))
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reader = PacketArrays(packet)
    packet_ref = reference(packet)
    n = int(reader.packet['row_count'])
    model = read_json(model)
    config = read_json(config)
    previous = read_json(original_selection)
    require(previous['schema'] == 'rinkfit.per_frame_representative_source_selection.v1', 'Original v1 thin selection required')
    append = None
    if append_source_proof is None:
        require(previous['source_packet'] == packet_ref and previous['source_row_count'] == n,
                'Original current v1 thin selection required')
    else:
        from .training_source_append_v2 import verify_append
        append = verify_append(append_source_proof, packet_ref, previous, reader)
        policy_sha = config['engineering_support_budget_v2']['source_packet_sha256']
        require(policy_sha in (previous['source_packet']['sha256'], packet_ref['sha256']), 'Append budget binds unrelated source')
        config['engineering_support_budget_v2']['source_packet_sha256'] = packet_ref['sha256']

    arrays = {key: reader.load(key).reshape(n) for key in ('admission', 'fit_role', 'shot_index', 'native_pts', 'material_index', 'kind')}
    physical = (arrays['admission'] == 1) & (arrays['fit_role'] == 1)
    if append is None:
        require(int(physical.sum()) == previous['eligible_rows'], 'Current physical source count differs from v1 producer')
        original_physical = physical
    else:
        import json
        from .effective_input_audit_v1 import CheckedBytes
        original_packet = json.loads(checked(previous['source_packet']).read_text(encoding='utf-8-sig'))
        store = CheckedBytes()
        oa = store.array(original_packet['arrays']['admission'], Path(previous['source_packet']['file']).parent)[0]
        ob = store.array(original_packet['arrays']['fit_role'], Path(previous['source_packet']['file']).parent)[0]
        original_physical = (oa == 1) & (ob == 1)
        require(int(original_physical.sum()) == previous['eligible_rows'], 'Original pre-append physical partition differs')

    old_fit = np.fromfile(checked(previous['fit_rows']), dtype='<u8')
    old_inspection = np.fromfile(checked(previous['inspection_rows']), dtype='<u8')
    require(len(old_fit) == previous['fit_count'] and np.all(old_fit < n) and np.all(np.diff(old_fit.astype('<i8')) > 0), 'Original fit order/count is invalid')
    old_keep = np.zeros(n, bool)
    old_keep[old_fit] = True
    require(original_physical[old_fit].all() and np.array_equal(np.flatnonzero(original_physical & ~old_keep[:len(original_physical)]), old_inspection), 'Original physical fit/inspection partition differs')
    roles_ref = reference(frame_roles)
    roles = allowed_exposures(read_json(frame_roles), model)
    role_allowed = np.zeros(n, bool)
    physical_rows = np.flatnonzero(physical)
    role_allowed[physical_rows] = exposure_permission(arrays['shot_index'][physical_rows], arrays['native_pts'][physical_rows], reader.packet['shot_ids'], roles)
    training = physical & role_allowed
    verdict_ref = reference(source_identity_verdicts)
    verdict = read_json(source_identity_verdicts)
    verdict_rows = source_verdict_rows(reader, verdict)
    for proof in verdict.get('evidence_files', []):
        checked(proof, Path(source_identity_verdicts).parent)
    rejected = sorted(int(row['raw_row']) for row in verdict_rows if row['decision'] == 'rejected')
    unresolved = [int(row['raw_row']) for row in verdict_rows if row['decision'] != 'confirmed']
    training[unresolved] = False
    canonical = np.arange(len(model['materials']))
    canonical_rules = read_json(checked(config['engineering_support_budget_v2']['canonical_material_entities']))
    for group in canonical_rules['groups']:
        for item in group['materials']:
            canonical[item['material_index']] = group['materials'][0]['material_index']
    added_fit = np.empty(0, dtype='<u8')
    if append is not None:
        from .training_source_append_v2 import appended_representatives
        added = np.arange(append[2], append[3], dtype=np.int64)
        added_fit = appended_representatives(reader, model, added[training[added]], canonical).astype('<u8')
    fit = np.r_[old_fit[training[old_fit]], added_fit].astype('<u8')
    keep = np.zeros(n, bool)
    keep[fit] = True
    inspection = np.flatnonzero(physical & ~keep)
    positive = []
    for group in reader.packet['groups']:
        if not group['rank']:
            continue
        rows = np.asarray(group['rows'], dtype='<i8')
        p, e, k = int(physical[rows].sum()), int(training[rows].sum()), int(keep[rows].sum())
        require(e in (0, p) and k in (0, e), 'Training/fit partially trims covariance group: ' + group['id'])
        positive.append(dict(id=group['id'], rank=group['rank'], rows=group['rows'], W=group['W']))
    # Every training exposure/entity/kind must still have its original thin support.
    before = np.unique(component_keys(np.flatnonzero(training), arrays, canonical), axis=0)
    after = np.unique(component_keys(fit, arrays, canonical), axis=0)
    require(np.array_equal(before, after), 'Filtering lost an eligible native entity/kind component; do not backfill')
    require(np.all(keep[training & (arrays['kind'] == 1)]), 'Filtering lost an eligible fixed material point')
    eligible_ref = new_array(out / 'training_eligible.u8', training.astype('u1'), 'u1')
    physical_ref = new_array(out / 'physical_admitted.u8', physical.astype('u1'), 'u1')
    fit_ref = new_array(out / 'fit_rows.u64', fit, '<u8')
    inspection_ref = new_array(out / 'inspection_rows.u64', inspection, '<u8')
    original_fit_ref = dict(**previous['fit_rows'], dtype='<u8', shape=[len(old_fit)])
    per_shot = []
    for sid, shot in enumerate(reader.packet['shot_ids']):
        per_shot.append(dict(shot=shot, physical=int(np.sum(physical & (arrays['shot_index'] == sid))),
                             eligible=int(np.sum(training & (arrays['shot_index'] == sid))),
                             fit=int(np.sum(arrays['shot_index'][fit] == sid)),
                             inspection=int(np.sum(arrays['shot_index'][inspection] == sid))))
    selection = dict(schema='rinkfit.per_frame_representative_source_selection.v2', source_packet=packet_ref,
                     source_row_count=n, physical_admitted_rows=int(physical.sum()), eligible_rows=int(training.sum()),
                     fit_count=len(fit), fit_cap=len(old_fit)+len(added_fit), fit_rows=fit_ref,
                     inspection_count=len(inspection), inspection_rows=inspection_ref,
                     training_eligibility=dict(schema='rinkfit.source_training_eligibility_mask.v1',
                                              source_packet_sha256=packet_ref['sha256'], row_count=n,
                                              eligible_count=int(training.sum()), mask=eligible_ref),
                     original_selection=reference(original_selection), original_fit_rows=original_fit_ref,
                     source_observations_deleted=False, source_UV_changed=False,
                     eligibility_provenance=dict(frame_roles=roles_ref, source_identity_verdicts=verdict_ref,
                                                 rejected_source_rows=rejected),
                     method=dict(operation='filter_original_representatives_by_SHA_bound_training_eligibility',
                                 no_backfill=True, selection_uses_fit_residual=False,
                                 all_physically_admitted_nonfit_rows_remain_inspection=True),
                     objective_change=dict(required=True, same_objective_AL_resume_permitted=False,
                                           independent_validation_certified=False), per_shot=per_shot)
    if append is not None:
        from .training_source_append_v2 import declaration
        appended_ref = new_array(out / 'appended_fit_rows.u64', added_fit, '<u8')
        selection['source_append_selection_v1'] = declaration(append[0], append[1], reference(original_selection), appended_ref)
        selection['method']['operation'] = 'filter_old_domain_and_select_only_authenticated_new_source_representatives'
        selection['method']['no_backfill_scope'] = 'all_original_raw_rows'
        selection['eligibility_provenance']['source_append_proof'] = append[0]
    manifest_ref = new_json(out / 'manifest.json', selection)
    policy, fit_weights, motion_weights, budget = recalculate_budget(reader, model, config, fit, physical, roles_ref, roles)
    budget.update(schema='rinkfit.actual_current_thin_budget_recalculation.v1', source_packet=packet_ref,
                  selection=manifest_ref, fit_count=len(fit), positive_rank_groups_checked=len(positive),
                  partial_positive_rank_eligibility_groups=0, eligible_component_coverage_exact=True,
                  role_excluded_admitted_rows=int(np.sum(physical & ~role_allowed)),
                  known_identity_rejected_rows=rejected,
                  known_identity_removed_fit_rows=sorted(set(rejected) & set(map(int, old_fit))),
                  fit_weights=new_array(out / 'fit_weights.f64', fit_weights, '<f8'),
                  motion_endpoint_weights=new_array(out / 'motion_endpoint_weights.f64', motion_weights, '<f8'),
                  standalone_validation_inputs=dict(physical_mask=physical_ref,
                    positive_rank_groups=new_json(out / 'positive_rank_groups.json', dict(groups=positive))),
                  source_raw_rows_retained=n, source_rows_deleted=False, thin_point_cap_increased=bool(len(added_fit)),
                  authentic_new_source_representatives=len(added_fit), old_domain_fit_backfilled=False,
                  solver_launched=False, quality_accepted=False)
    fit_locations = {int(row): i for i, row in enumerate(fit) if int(row) in {int(v['raw_row']) for v in verdict_rows}}
    budget['listed_confirmed_source_fit_weights'] = {str(row):float(fit_weights[fit_locations[row]])
        for row in fit_locations if row not in unresolved}
    budget_ref = new_json(out / 'budget_recalculation.json', budget)
    delta = dict(representative_source_selection_v1=manifest_ref, source_identity_verdicts_v1=verdict_ref,
                 engineering_support_budget_v2=policy,
                 audit_expected_source_counts={row['shot']: row['fit'] for row in per_shot})
    delta_ref = new_json(out / 'native_config_delta.json', delta)
    result = dict(schema='rinkfit.actual_current_training_input_preparation.v2', issuer=reference(__file__),
                  source_packet=packet_ref, original_selection=reference(original_selection),
                  manifest=manifest_ref, budget=budget_ref, config_delta=delta_ref,
                  raw_rows=n, physical_rows=int(physical.sum()), training_rows=int(training.sum()),
                  original_fit_rows=len(old_fit), fit_rows=len(fit), inspection_rows=len(inspection),
                  authentic_new_source_representatives=len(added_fit), old_domain_fit_backfilled=False,
                  protected_fit_rows_removed=int(np.sum(physical[old_fit] & ~role_allowed[old_fit])),
                  physically_withdrawn_fit_rows=int(np.sum(~physical[old_fit])),
                  rejected_fit_rows_removed=len(set(rejected) & set(map(int, old_fit))),
                  per_shot=per_shot, runtime_seconds=time.monotonic()-started,
                  full_original_auxiliary_state_migration_required=True,
                  inherited_fit_or_validation_lineage_reset=False,
                  solver_launched=False, quality_accepted=False)
    new_json(out / 'preparation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'model', 'original-selection', 'frame-roles', 'source-identity-verdicts', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    args = vars(parser.parse_args())
    print(json.dumps(prepare(**{key.replace('-', '_'): value for key, value in args.items()}), indent=2))


if __name__ == '__main__':
    main()
