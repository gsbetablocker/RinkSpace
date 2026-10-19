"""Check archived all86 LP witnesses without reconstructing A or running LP.

This verifies saved original-unit row evaluations and the reversible return
mapping. The assembly/capture receipt, rather than this script, certifies their
evaluation against every original matrix row. No new optimality is claimed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BOOT=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q276_vertical_print_station_one_cycle_20261011.bootstrap')
DEST=BOOT/'q8a15_complete_truth_phase_I_readonly'/'station_tangent_cone_repaired'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def authenticate(reference):
    assert sha(reference['path'])==reference['sha256'],reference['path']
    return Path(reference['path'])


def main():
    capture_path=DEST/'complete_original_phase_I_primal_witness.json'
    capture=load(capture_path)
    assembly=load(authenticate(capture['assembly']))
    domain_path=authenticate(capture['unchanged_original_domain_arrays'])
    witness_path=authenticate(capture['actual_primal_dual_and_return_mapping'])
    sources=[ROOT/'calibration/solver_runtime'/p for p in (
        'finite_rectangle_station_faces.py','finite_rectangle_projection_fd.py',
        'finite_rectangle_station_step_constraints.py',
        'legacy/structured_programme_source_candidate/feasibility_priority_restoration.py',
        'priority_feasibility_witness_runtime.py',
        'legacy/structured_programme_source_candidate/source_step_curvature_tightening.py',
        'legacy/run_current_named66_full_programme.py','source_manifest.json')]
    frozen=[ref(p)for p in sources]
    with np.load(domain_path,allow_pickle=False)as dz,np.load(witness_path,allow_pickle=False)as wz:
        d={k:dz[k].copy()for k in dz.files};w={k:wz[k].copy()for k in wz.files}
    assert not any(k.startswith('A_')for k in d),'No second matrix should be persisted'
    v=d['current_group_violations'];n=len(d['lower']);rows=len(d['original_full_rhs'])
    assert len(v)==86 and rows==127831 and n==285
    assert len(assembly['all86_groups'])==86
    x=w['conic_normalized_primal'];primary=w['primary_normalized_primal']
    active=d['exact_nonconstant_row_mask'];norm=d['reversible_row_norm']
    assert x.shape==primary.shape==(n+1,)
    assert np.isfinite(x).all()and np.isfinite(primary).all()
    assert w['conic_normalized_dual'].shape==(int(active.sum())+2,)
    assert np.isfinite(w['conic_normalized_dual']).all()
    fraction=float(w['certified_remaining_fraction']);target=float(w['actual_driver_midpoint_fraction'])
    assert fraction==float(x[-1])and target==.5*(1.+fraction)
    assert 0.<=fraction<1.-64*np.finfo(float).eps
    for key in ('selected_canonical_columns','selected_evaluation_columns',
                'reversible_column_scale','reversible_row_norm','exact_nonconstant_row_mask'):
        np.testing.assert_array_equal(w[key],d[key],err_msg=key)
    np.testing.assert_array_equal(w['original_unit_direction'],d['reversible_column_scale']*x[:-1])
    slack=w['all_original_unit_slack'];midpoint=w['all_original_unit_midpoint_slack']
    assert slack.shape==midpoint.shape==(rows,)
    assert np.isfinite(slack).all()and np.isfinite(midpoint).all()
    assert np.min(slack[active]/norm[active])>=-1e-9 and np.all(slack[~active]>=0.)
    assert np.min(midpoint[active]/norm[active])>=-1e-9 and np.all(midpoint[~active]>=0.)
    # A floating arithmetic identity, not an added pixel/support tolerance.
    expected=d['original_full_expansion']*(target-fraction)
    error=np.abs(midpoint-slack-expected)
    arithmetic_scale=np.abs(midpoint)+np.abs(slack)+np.abs(expected)+np.abs(d['original_full_rhs'])+1.
    assert np.all(error<=64*np.finfo(float).eps*arithmetic_scale)
    owned=d['row_group_index'];coeff=np.zeros(len(owned));mask=owned>=0
    coeff[mask]=np.where(v[owned[mask]]>1e-8,v[owned[mask]],0.)
    np.testing.assert_array_equal(w['actual_contracted_support_lower'],d['original_constraint_lower']-target*coeff)
    np.testing.assert_array_equal(coeff,d['original_full_expansion'][:len(owned)])
    nominal_deficit=d['original_full_rhs']-d['original_full_expansion']
    assert np.all(nominal_deficit<=0.)
    primary_check=capture['original_contract_report']['phase_I_primary_backend_complete_primal_check']
    assert primary_check['backend_success']and not primary_check['passed']
    assert primary_check['original_all_face_normalized_minimum_slack'] < -1e-9
    assert capture['complete_original_primal_witness_certified']
    assert not capture['LP_dual_optimality_certified']and not capture['bottleneck_dual_attribution_claimed']
    assert frozen==[ref(p)for p in sources]
    report=dict(schema='rinkspace.archived-all86-primal-return-mapping-final-check.v1',
        script=ref(__file__),actual_capture=ref(capture_path),original_domain_arrays=ref(domain_path),
        actual_candidate_vectors_and_saved_original_row_evaluations=ref(witness_path),final_production_sources=frozen,
        actual_LP_capture_precedes_final_midpoint_and_shape_only_checks=True,
        final_checks_do_not_change_candidate_generation_or_original_LP=True,
        all86_groups=86,original_full_LP_rows=rows,original_shared_columns=n,
        archived_complete_matrix_identity=capture['original_matrix_identity'],
        all_archived_original_row_primal_checks_passed=True,
        zero_step_t1_all_original_rows_feasible=True,zero_step_t1_maximum_original_unit_deficit=float(nominal_deficit.max()),
        original_normalized_primal_minimum=float(np.min(slack[active]/norm[active])),
        original_normalized_midpoint_minimum=float(np.min(midpoint[active]/norm[active])),
        original_unit_primal_minimum=float(slack.min()),original_unit_midpoint_minimum=float(midpoint.min()),
        exact_constant_original_rows=int((~active).sum()),physical_prefix_rows=assembly['physical_prefix_rows'],
        finite_physical_box_rows=rows-len(owned),
        original_reversible_direction_refill_bitexact=True,actual_returned_support_targets_bitexact=True,
        midpoint_slack_arithmetic_identity_maximum_error=float(error.max()),
        common_fraction_feasible_witness=fraction,actual_driver_midpoint_fraction=target,
        primary_backend_success_is_false_primal_success=True,primary_original_primal_check=primary_check,
        LP_optimality_or_dual_bottleneck_attribution_claimed=False,
        source_QP_full_KKT_and_original_nonlinear_guards_still_required=True,
        limitation='Original row evaluations were produced by the authenticated actual all86 assembly/capture. This value-level check does not reconstruct A, recompute J, solve LP or establish nonlinear truth improvement.',
        final_related_unit_tests=[dict(path='tests/'+name,count=count,fresh_process=True)for name,count in (
            ('test_original_phase_I_primary_primal_gate.py',8),('test_priority_feasibility_witness_runtime.py',7),
            ('test_full_priority_clarabel_runtime.py',1),('test_full_priority_LP_backend_runtime.py',1),
            ('test_separated_priority_LP_runtime.py',5))],
        new_LP_or_fit_or_full_source_J_H_g_or_source_images=False,accepted_q_unchanged=True)
    out=DEST/'final_saved_primal_mapping_verification.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=ref(out),fraction=fraction,midpoint=target,all_original_rows_primal_passed=True)))


if __name__=='__main__':main()
