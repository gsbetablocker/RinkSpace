"""Replay the real two production witnesses through the updated SOC control.

No new QP, source owner, J or graph is constructed. All witness directions and
actual carriers come from the production capture with original full KKT.
"""
from pathlib import Path
from types import SimpleNamespace
import json,sys
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime import optional_soc_runtime as control
    import z190_joint_source_solver as joint
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    root=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
    run=root/'okada_actual0758_SOC_production_face_capture_20261011_0840'
    path=Path(str(run)+'.bootstrap')/'actual_SOC_complete_witness_capture'
    history=root/'okada_review86_samea454_actual672_roles_SOCcontinue_20261011_0758/saved_normal_SOC_single_face_replay'
    records=[json.loads((path/f'witness_{i}.json').read_text(encoding='utf8'))for i in range(2)]
    with np.load(path/'actual_production_step_faces.npz',allow_pickle=False)as data,np.load(history/'actual_current_original_carrier_maps.npz',allow_pickle=False)as previous:
        A=data['matrix'];lower=data['lower'];columns=data['selected'];q=data['q'];responses={}
        for family in ['native','wide','confirmed']:
            key=family+'_';r={name:data[key+name] for name in ['nominal_carrier','row_start','row_stop']}
            for label in ['nominal_carrier','row_start','row_stop']:
                np.testing.assert_array_equal(r[label],previous[key+label])
            r['row_start']=int(r['row_start']);r['row_stop']=int(r['row_stop'])
            if family=='native':
                r['row_carrier_projection']=csr_matrix((data[key+'projection_data'],data[key+'projection_indices'],data[key+'projection_indptr']),shape=tuple(data[key+'projection_shape']))
            else:r['retained_face_mask']=previous[key+'retained_face_mask']
            responses[family]=r
    steps=[];actuals=[]
    for i in range(2):
        with np.load(path/f'witness_{i}.npz',allow_pickle=False)as z:
            np.testing.assert_array_equal(q,z['q']);steps.append(z['step'].copy());diagonal=z['normal_diagonal'].copy()
        with np.load(path/f'witness_{i}_actual_carriers.npz',allow_pickle=False)as z:actuals.append({k:z[k].copy()for k in z.files})
    with np.load(run/'first_complete_source_normal_numeric.npz',allow_pickle=False)as z:
        H=csc_matrix((z['normal_data'],z['normal_indices'],z['normal_indptr']),shape=tuple(z['normal_shape']));g=z['gradient']
    witnesses=[SimpleNamespace(step=steps[i],report=records[i]['linear_report'])for i in range(2)]
    assert all(control._previous_full_KKT_verified(w)for w in witnesses)
    for i in range(2):
        m=control.original_nonlinear_step_face_feasibility(A,lower,steps[i][columns],responses,actuals[i])
        assert m==records[i]['merit']
    control._install_original_nonlinear_face_observer()
    producer=joint._sequential_measured_shared_source_curvature
    while hasattr(producer,'_original_SOC_producer'):producer=producer._original_SOC_producer
    options=dict(global_step_constraint_lower=lower.copy());measured=0;solved=0
    def tightening(value,step,selected,domain):
        nonlocal measured
        assert measured<2
        np.testing.assert_array_equal(step,steps[measured]);actual=actuals[measured];measured+=1
        return source.measured_outward_step_tightening(A,lower,step[selected],responses,actual)
    def solve(*args,**kwargs):
        nonlocal solved
        assert solved==0;solved+=1
        return witnesses[1]
    final=control._with_outer_nonlinear_progress(producer,q,witnesses[0],columns,
        {'lower':lower},options,H,g,diagonal,tightening,solve)
    report=final.report['sequential_measured_shared_source_curvature_tightening']
    assert final is witnesses[1] and solved==1 and measured==2
    assert report['stop_reason']=='optional_actual_original_nonlinear_feasibility_insufficient_model_reduction'
    assert control._previous_full_KKT_verified(final)
    result=dict(passed=True,original_production_full_q_sha256=records[0]['full_q_sha256'],
        retained_full_step_sha256=records[1]['full_step_sha256'],
        original_full_KKT_preserved=True,current_better_SOC_direction_retained=True,
        original_actual_merits_byteexact=True,measurement_count=measured,additional_QP_or_J_or_source_calls=0,
        report=report,all_original_nonlinear_trial_acceptance_still_required=True,
        source_cost_or_physical_acceptance_not_claimed=True)
    out=path/'actual_original_two_witness_SOC_sufficient_progress_control_replay.json'
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({'passed':True,'output':str(out),'retained_step':records[1]['full_step_sha256']}))


if __name__=='__main__':main()
