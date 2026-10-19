"""Saved-step native curve counterfactuals, values only and stdout only."""
from collections import Counter
from pathlib import Path
import argparse
import json
import time

import joint_absolute_source_context_20261011 as ctx
from compare_complete_original_source_states_20261011 import review_metric, row_key, shot_for
from calibration.solver_runtime.source_loss import weighted_source_cost

a, np = ctx.a, ctx.np


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False, default=a.plain), flush=True)


def parameter_groups(c):
    private, installation = [], []
    for block in c.state.blocks:
        private.extend(block.start+4*np.asarray(list(block.bundle.index.values()))[:, None]+np.arange(4))
        installation.extend(np.r_[block.xyz_slots, block.optics_chart.mount_indices, block.optics_chart.indices])
    geometry = np.unique(np.r_[c.world_indices,
        int(c.arrays['aux__shared_pattern_start'])+np.arange(9),
        c.circle.active_circle_indices, c.material.active_material, c.placement.placement_indices]).astype(int)
    groups = dict(geometry=geometry, private_RF4=np.unique(private).astype(int),
                  installation=np.unique(installation).astype(int))
    for name, columns in groups.items():
        if columns.ndim != 1 or np.any(columns < 0) or np.any(columns >= len(c.q)):
            raise ValueError('Original parameter group is outside current layout: '+name)
    for left, right in [('geometry', 'private_RF4'), ('geometry', 'installation'), ('private_RF4', 'installation')]:
        if np.intersect1d(groups[left], groups[right]).size:
            raise ValueError('Physical parameter groups overlap: '+left+' / '+right)
    return groups


def installation_subgroups(c):
    return {name:np.unique(np.concatenate([np.asarray(getter(block), int)
                for block in c.state.blocks])).astype(int)
            for name,getter in (
                ('shared_PPx_PPy_k1', lambda block:block.optics_chart.indices),
                ('shared_pivotXYZ', lambda block:block.xyz_slots),
                ('shared_leverXYZ', lambda block:block.optics_chart.mount_indices))}


def values(c, value):
    result = {}
    for name, family, factor, expected in (
        ('BOARD', 'additional_real_native_board_edges', c.board, 191),
        ('ICE', 'additional_real_native_ice_curves', c.ice, 108)):
        review = factor.review(value)
        if len(review['rows']) != expected:
            raise ValueError('Complete original native review inventory changed: '+name)
        rows = {row_key(row): row for row in review['rows']}
        if len(rows) != expected:
            raise ValueError('Duplicate native review identity: '+name)
        block = factor.block(value, False)
        loss = c.integration._payloads[family]
        result[name] = dict(rows=rows, cost=weighted_source_cost(block.r, block.quadratic, loss),
                            original_objective_r=np.asarray(block.r).copy(), family=family)
    return result


def check_saved(actual, saved, label):
    maxima = {}
    for name, saved_family in [('BOARD', 'additional_native_board_edges'), ('ICE', 'additional_native_ice_curves')]:
        rows = {row_key(row): row for row in saved[saved_family]['rows']}
        current = actual[name]['rows']
        if rows.keys() != current.keys():
            raise ValueError('Saved/current full native identities differ: '+label+' '+name)
        errors = []
        for key, old in rows.items():
            metric, expected = review_metric(old)
            actual_metric, observed = review_metric(current[key])
            if actual_metric != metric or (expected is None) != (observed is None):
                raise ValueError('Saved native metric or qualification changed: '+label+' '+str(key))
            if expected is not None:
                errors.append(abs(expected-observed))
            if name == 'BOARD':
                for field in ('mean_outside_source_support_px', 'max_outside_source_support_px'):
                    errors.append(abs(old[field]-current[key][field]))
        maximum = max(errors, default=0.)
        if maximum > 1e-8:
            raise ValueError('Same-q native review value mismatch: '+str((label, name, maximum)))
        maxima[name] = maximum
    return maxima


def summarize(base, states, shots, installation_only=False):
    for shot in ([4, 6] if installation_only else range(1, 10)):
        output = {}
        for family in ('BOARD', 'ICE'):
            for split in sorted({str(row['split']) for row in base[family]['rows'].values()}):
                keys = [key for key, row in base[family]['rows'].items()
                        if str(row['split']) == split and shot_for(row['actual_pts'], shots) == shot]
                if not keys:
                    continue
                modes = {}
                for mode, value in states.items():
                    delta = []
                    inactive = 0
                    for key in keys:
                        metric, old = review_metric(base[family]['rows'][key])
                        metric2, new = review_metric(value[family]['rows'][key])
                        if metric != metric2:
                            raise ValueError('Counterfactual changed reported metric')
                        if old is None:
                            inactive += 1
                        else:
                            delta.append(new-old)
                    modes[mode] = dict(counts=dict(Counter('improved' if d < 0 else 'worsened' if d > 0 else 'unchanged'
                                                         for d in delta)), inactive_rows=inactive,
                                       largest_improvement_delta=min(delta, default=0.),
                                       largest_worsening_delta=max(delta, default=0.))
                output[family+':'+split] = modes
        emit('all9_original_same_metric_counterfactuals', shot=shot, families=output)

    for family in ('BOARD', 'ICE'):
        selected = []
        for key, row in base[family]['rows'].items():
            at = int(row['actual_pts'])
            cut4 = family == 'BOARD' and 229922816 <= at < 229945856 and 'Ad4' in row['id']
            cut6 = (shot_for(at,shots) == 6 if installation_only else
                    230506752 <= at <= 230508032 and ('Ad7' in row['id'] or 'F3' in row['id'] or 'B1' in row['id']))
            if not (cut4 or cut6):
                continue
            metric, old = review_metric(row)
            entry = dict(id=row['id'], pts=at, split=row['split'], metric=metric, before=old,
                         states={mode:review_metric(value[family]['rows'][key])[1] for mode,value in states.items()})
            if family == 'BOARD':
                entry['original_support_mean_px'] = dict(before=row['mean_outside_source_support_px'], **{
                    mode:value[family]['rows'][key]['mean_outside_source_support_px'] for mode,value in states.items()})
            selected.append(entry)
        emit('cut4_and_cut6_entity_counterfactuals', family=family, rows=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-state', type=Path, required=True)
    parser.add_argument('--installation-subgroups-only', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    after = args.after_state.resolve()
    if after.is_file():
        after = after.parent
    identity = a.read(after/'state_identity.json')
    path = after/'complete_state_UNACCEPTED.npz'
    if a.sha(path) != identity['state_npz_sha256']:
        raise ValueError('Saved complete candidate state SHA changed')
    if a.sha(after/'source_review_UNACCEPTED.json') != identity['source_review_sha256']:
        raise ValueError('Saved complete candidate review SHA changed')
    c = ctx.load_context()
    with np.load(path, allow_pickle=False) as z:
        canonical = z['q'].copy()
        if not np.array_equal(z['aux__current_evaluation_to_canonical_parameter_indices'],
                              c.state.canonical_parameter_layout.evaluation_to_canonical):
            raise ValueError('Canonical/evaluation source parameter layout changed')
    fullq = c.state.evaluation_q(canonical)
    if fullq.shape != c.q.shape:
        raise ValueError('Complete source parameter count changed')
    groups = parameter_groups(c)
    evaluated_groups = installation_subgroups(c) if args.installation_subgroups_only else groups
    if args.installation_subgroups_only:
        if not np.array_equal(np.unique(np.concatenate(list(evaluated_groups.values()))), groups['installation']):
            raise ValueError('Installation subgroups do not cover the exact original camera columns')
    states = {}
    baseline = values(c, c.q)
    baseline_error = check_saved(baseline, a.read(c.directory/'source_review_UNACCEPTED.json'), 'q8')
    states['full'] = values(c, fullq)
    full_error = check_saved(states['full'], a.read(after/'source_review_UNACCEPTED.json'), 'candidate')
    saved_costs = {r['original_row_ledger']['family']:r['normalized_cost']
                  for r in a.read(after.parent.parent/'actual_trial_family_cost_pass_0001.json')['families']}
    for name, value in states['full'].items():
        old_saved = c.ledgers[value['family']][0]['normalized_cost']
        if abs(baseline[name]['cost']-old_saved) > 2e-8 or abs(value['cost']-saved_costs[value['family']]) > 2e-8:
            raise ValueError('Original native family cost not reproduced: '+name)
    emit('baseline_and_candidate_aligned', baseline_max_metric_error=baseline_error,
         candidate_max_metric_error=full_error, complete_native_rows=dict(BOARD=191, ICE=108),
         group_column_counts={name:len(columns) for name,columns in evaluated_groups.items()}, context_seconds=c.load_seconds)
    for name, columns in evaluated_groups.items():
        hybrid = c.q.copy()
        hybrid[columns] = fullq[columns]
        states[name] = values(c, hybrid)
        emit('counterfactual_group_evaluated', group=name, seconds=time.perf_counter()-started)
    union = np.unique(np.concatenate(list(groups.values())))
    merged = c.q.copy()
    merged[union] = fullq[union]
    union_errors = {}
    for name, factor in [('BOARD', c.board), ('ICE', c.ice)]:
        current = factor._residual(fullq, factor.rows+factor.held)
        reconstructed = factor._residual(merged, factor.rows+factor.held)
        union_errors[name] = float(np.max(abs(current-reconstructed), initial=0.))
        if not np.array_equal(current, reconstructed):
            raise ValueError('Three groups omitted an actual native dependency: '+str((name,union_errors[name])))
    ignored_changed = np.setdiff1d(np.flatnonzero(fullq != c.q), union)
    emit('three_groups_explain_complete_state_source_effect', exact_all_FIT_and_HELD_residual_union_error=union_errors,
         ignored_changed_evaluation_columns=ignored_changed.tolist(),
         original_native_family_costs={name:dict(before=baseline[name]['cost'], **{
             mode:value[name]['cost'] for mode,value in states.items()}) for name in ('BOARD','ICE')},
         single_group_metric_changes_not_assumed_additive=True)
    shots = a.read(a.ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10]
    summarize(baseline, states, shots, args.installation_subgroups_only)
    emit('complete', seconds=time.perf_counter()-started, source_images_decoded=0, source_images_written=0,
         source_J_or_fit=False, report_files_written=0,
         counterfactuals_are_conditional_on_other_saved_parameter_groups=True,
         whole_source_guards_or_feasible_improvement_not_certified=True)


if __name__ == '__main__':
    main()
