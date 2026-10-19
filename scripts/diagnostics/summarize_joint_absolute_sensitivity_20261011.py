"""Summarize saved projections only; no source loading, derivatives or candidates."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
REPORT = H / 'joint_absolute_q8a15_local_sensitivity_20261011.json'


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    d = json.loads(REPORT.read_text(encoding='utf8'))
    array_path = Path(d['projection_arrays']['path'])
    assert reference(array_path)['sha256'] == d['projection_arrays']['sha256']
    with np.load(array_path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    gs = d['source_group_residual_attribution']
    indices = arrays['observation_only_original_row_indices']
    obs_scales = arrays['original_scalar_scales'][indices]
    out = dict(schema='rinkspace.saved-joint-absolute-interpretation.v1',
        report=reference(REPORT), arrays=reference(array_path), script=reference(Path(__file__)),
        scope=d['scope'], complete_q_sha256=d['complete_q_sha256'],
        no_source_load_no_J_no_dq_no_candidate_no_fit_no_images=True,
        parameter_partition_counts=d['free_parameter_partition_counts'],
        final_runtime_seconds=d['seconds'])
    views = {}
    for view, summary_key in [('actual_observation_only','actual_observation_only_relaxation'),
                              ('rawtyped_observation_only','supportless_typed_reference_observation_only_relaxation')]:
        summary = d[summary_key]
        group_index = arrays[view + '__row_group_index']
        w = arrays[view + '__nominal_IRLS_row_weights']
        r = arrays[view + '__original_residual']
        rows = []
        for gi, g in enumerate(gs):
            mask = group_index == gi
            if not np.any(mask):
                continue
            record = {k: g[k] for k in ('group_index','family','actual_pts','id','physical_group','source_kind')}
            record.update(scalar_rows=int(mask.sum()),
                weighted_original_energy=float(np.sum(arrays[view+'__weighted_residual'][mask]**2)),
                actual_nonlinear_cost=g['actual_original_source_cost'],
                source_reference_pixel_component_abs_p95=(float(np.quantile(abs(r[mask]*obs_scales[mask]),.95))
                    if g['source_kind']!='global_tail' else None))
            for h in ('first','second'):
                for component in ('camera_only_remaining','geometry_only_remaining','joint_remaining'):
                    v = arrays[view+'__'+h+'_'+component][mask]
                    record[h+'_'+component+'_energy'] = float(v@v)
                joint = arrays[view+'__'+h+'_joint_remaining'][mask]
                pixel = np.divide(joint,np.sqrt(w[mask]),out=np.zeros_like(joint),where=w[mask]>0)*obs_scales[mask]
                record[h+'_joint_local_pixel_component_abs_p95']=(float(np.quantile(abs(pixel),.95))
                    if g['source_kind']!='global_tail' else None)
            v = arrays[view+'__first_joint_remaining'][mask]-arrays[view+'__second_joint_remaining'][mask]
            record['joint_two_stencil_difference_norm'] = float(np.linalg.norm(v))
            rows.append(record)
        families = defaultdict(lambda:defaultdict(float))
        for g in rows:
            for k in ('scalar_rows','weighted_original_energy','first_camera_only_remaining_energy',
                      'first_geometry_only_remaining_energy','first_joint_remaining_energy','second_joint_remaining_energy'):
                families[g['family']][k] += g[k]
        target = [g for g in rows if g['source_kind']!='wide' and g['family']!='strong_high_wide_source' and
                  (str(g['physical_group']) in ('B2','Ad24','Ad25','Ad26','Ad4') or str(g['physical_group']).startswith('F2'))]
        totals = defaultdict(lambda:defaultdict(float))
        for g in target:
            # The centre secondary is a correlated objective, not an independent observation.
            key = (g['physical_group'],g['source_kind'])
            for k in ('scalar_rows','weighted_original_energy','first_camera_only_remaining_energy',
                      'first_geometry_only_remaining_energy','first_joint_remaining_energy'):
                totals[str(key)][k] += g[k]
        outcomes = {h:{k:v for k,v in values.items() if not isinstance(v,list)}
            for h,values in summary['outcomes'].items()}
        spaces = {key:{k:value for k,value in space.items() if k in ('shape','machine_rank_first','machine_rank_second',
            'machine_threshold_first','machine_threshold_second','two_stencil_operator_difference_spectral_norm',
            'empirical_combination_agreement_rank_first','empirical_combination_agreement_rank_second',
            'selected_projector_change_spectral_norm','combination_agreement_arithmetically_ambiguous_modes')}
            for key,space in summary['spaces'].items()}
        views[view] = dict(outcomes=outcomes,spaces=spaces,
            residual_attribution_two_stencil_stability=summary['residual_attribution_two_stencil_stability'],
            families=dict(families), target_physical_group_totals=dict(totals), target_exposures=target,
            largest_joint_remainder_groups=sorted(rows,key=lambda g:g['first_joint_remaining_energy'],reverse=True)[:15],
            largest_programme_joint_remainder_groups=sorted([g for g in rows if not g['family'].startswith('strong_high')
                and g['source_kind']!='global_tail'],key=lambda g:g['first_joint_remaining_energy'],reverse=True)[:15])
    out['views'] = views
    extension=[]
    for receipt in d['derivative_receipts']:
        for f in receipt.get('factors',[]):
            if 'world_dependency_extension' not in f:
                continue
            ex=f['world_dependency_extension']
            extension.append(dict(family=f['family'],step_multiplier=receipt['step_multiplier'],
                original_declared_world_columns=ex['original_declared_world_columns'],
                tested_additional_world_columns=ex['diagnostic_added_world_columns'],
                maximum_additional_raw_column_norm=max(ex['added_column_raw_l2_norm'],default=0.),
                maximum_additional_actual_column_norm=max(ex['added_column_actual_l2_norm'],default=0.),
                added_nonzero_raw_world_columns=ex['added_nonzero_raw_world_columns']))
    out['native_declared_dependency_crosscheck']=extension
    out['interpretation_limits']=[
        'Camera-only and geometry-only projections are separate globally coupled counterfactual column spaces; their explained energies cannot be added.',
        'Joint projection may move residual into another entity or exposure. Its row-wise remainder is not a newly evaluated model or an observed deterioration.',
        'Observation-only excludes all 38 external engineering roll rows and 3 installation bias rows. Finite supports and unknown pixel correspondences are not surveyed world anchors.',
        'The aggregate remainder is dominated by the independent wide source. It cannot be used as evidence that the programme 20–45 px examples lack local response capacity.',
        'Two-stencil agreement is empirical. Fixed internal public derivatives and bias shared by both stencils are not tested by the h comparison.',
        'Native ice/board use frozen nominal segment/clamp and original support derivatives; original programme geometry and material/circle interfaces retain their public value-FD branch semantics.',
        'The original source priors, guards, motion and other programme exposures are absent from this relaxed column-space comparison. A small remainder does not prove a finite legal change or full-objective improvement.',
        'A stable large joint remainder with all affecting DOFs admitted is an obstruction for this current first-order response on these rows, not proof of finite nonlinear impossibility or its physical cause.'
    ]
    out['fixed_public_FD_not_varied_between_stencils']=dict(
        external_main_mount_indices=[34195,34196,34197], public_halfwidth=1e-7,
        some_internal_pattern_nearest_foot_camera_derivative_halfwidth=1e-5,
        scope='These original internal public finite differences are reused at both multiplier labels; analytic derivatives are separately identical by construction.',
        no_blanket_claim_all_columns_two_scale_validated=True)
    out['source_role_metadata_clarification']=dict(
        original_saved_generic_roles_do_not_control_numeric_operator=True,
        measured_transition_midpoint_native_Ad4_group_indices=list(range(67,73)),
        semantics='The six LOW Ad4 native normal intervals retain RGB transition footprint metadata. The actual native measured_transition_midpoint operator does not turn that interval into a dead zone; raw and actual are equal. Their original source penalty and numerical derivatives were preserved.',
        raw_reference_semantics='Supportless typed reference where the original source supplies such a reference; search-only external supports remain supported and do not acquire an invented centre.')
    target=H/'joint_absolute_q8a15_local_interpretation_20261011.json'
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(output=reference(target),bytes=target.stat().st_size,
        actual_family_remainders=views['actual_observation_only']['families'],
        target_totals=views['actual_observation_only']['target_physical_group_totals']),ensure_ascii=False))


if __name__=='__main__':
    main()
