"""Solve the identical saved real SOC with all changed original faces seeded."""
from pathlib import Path
import os,sys,time,json,hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
import numpy as np
from scipy.sparse import csc_matrix,diags
from scipy.sparse.linalg import splu
import z190_global_schur_linear_step as qp
from calibration.solver_runtime.optional_soc_runtime import install_optional_SOC_numerical_failure_route,_optional_soc,OptionalSOCNumericalFailure
BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
RUN=BASE/'okada_review86_TOP_RGB36_fixed285_20261011_034757'
PARENT=RUN/'saved_normal_complete_SOC_replay_after_public_polish'
OUT=PARENT/'all_strict_tightened_faces_seed_replay'
Q='75719b65139fa662366146c4517312319bdad9a88e6f2c25704966f550cb1eee'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,
    default=lambda e:e.tolist()if isinstance(e,np.ndarray)else e.item()if isinstance(e,np.generic)else str(e)),encoding='utf8')
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def emit(**v):print(json.dumps(v,ensure_ascii=False,allow_nan=False),flush=True)
def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=False)
    source=PARENT/('optional_SOC_reduced_'+Q+'.npz')
    with np.load(source,allow_pickle=False)as z:args={k:z[k].copy()for k in z.files}
    with np.load(PARENT/'actual_required_reduced_QP.npz',allow_pickle=False)as z:
        changed=np.flatnonzero(args['constraint_lower']>z['constraint_lower'])
        assert np.all(args['constraint_lower']>=z['constraint_lower'])
        for key in ('matrix','gradient','lower','upper','constraint_matrix'):np.testing.assert_array_equal(args[key],z[key])
    parent_report=read(PARENT/'actual_required_complete_step_report.json')
    kkt=parent_report['shared_linear_domain_step'];selected=np.asarray(parent_report['global_parameter_indices'],int)
    finite_lo=np.flatnonzero(np.isfinite(args['lower']));finite_hi=np.flatnonzero(np.isfinite(args['upper']))
    warm=list(kkt['active_linear_constraint_rows'])
    for column,side in zip(parent_report['shared_parameter_bound_step']['active_original_parameter_indices'],kkt['active_bound_sides']):
        position=int(np.flatnonzero(selected==column)[0]);finite=finite_lo if side==-1 else finite_hi
        warm.append(len(args['constraint_lower'])+(0 if side==-1 else len(finite_lo))+int(np.flatnonzero(finite==position)[0]))
    qp._linearly_constrained_shared_step._last_verified_separation_faces=dict(
        shape=(len(args['constraint_lower'])+len(finite_lo)+len(finite_hi),len(args['gradient'])),
        scope=os.environ.get('RINKSPACE_SOLVER_DATA_ROOT'),rows=tuple(warm))
    install_optional_SOC_numerical_failure_route()
    token=_optional_soc.set(dict(original_q_sha256=Q,previous_direction_sha256='from verified complete parent',
        strictly_tighter_measured_SOC_faces=len(changed),strictly_tightened_original_faces=changed))
    report=dict(schema='rinkspace.saved-real-SOC-all-strict-tightened-original-faces-seed-replay.v1',
        original_SOC_problem_reference=ref(source),strictly_tightened_original_face_indices=changed.tolist(),
        source_H_g_original_A_targets_bounds_unchanged=True,source_J_or_graph_recomputed=False)
    try:
        x,_,result=qp._linearly_constrained_shared_step(args['matrix'],args['gradient'],args['lower'],args['upper'],
            args['constraint_matrix'],args['constraint_lower'])
        report.update(QP_passed=True,original_full_KKT=result)
        with np.load(RUN/'first_complete_source_normal_numeric.npz',allow_pickle=False)as z:
            q=z['q'].copy();normal=csc_matrix((z['normal_data'].copy(),z['normal_indices'].copy(),z['normal_indptr'].copy()),shape=tuple(z['normal_shape']))
            gradient=z['gradient'].copy();fixed=json.loads(str(z['report_json']))['fixed_parameter_indices']
        history=read(RUN/'solver/iterations.json')[0];damping=history['damping'];diagonal=np.maximum(normal.diagonal(),1.)
        remaining=np.flatnonzero(~np.isin(np.arange(len(q)),selected));viewlm=damping*diagonal.copy();viewlm[selected]=0.
        regularized=normal+diags(viewlm+1e-8*diagonal,format='csc')
        factor=splu(regularized[remaining,:][:,remaining],permc_spec='MMD_AT_PLUS_A')
        views=-factor.solve(gradient[remaining]+regularized[remaining,:][:,selected]@x)
        full=np.empty(len(q));full[selected]=x;full[remaining]=views
        np.testing.assert_array_equal(full[fixed],0.)
        with np.load(PARENT/'actual_required_complete_step_and_trial.npz',allow_pickle=False)as z:
            mapping=z['canonical_to_evaluation'];evalparent=z['evaluation_parent_q']
        evalfull=np.empty_like(full);evalfull[mapping]=full
        np.savez(OUT/'verified_SOC_full_complete_step_and_trial.npz',parent_q=q,step=full,full_trial_q=q+full,
            evaluation_parent_q=evalparent,evaluation_trial_q=evalparent+evalfull,canonical_to_evaluation=mapping,
            shared_indices=selected,shared_step=x)
        emit(stage='saved_actual_SOC_original_full_KKT_PASS',rows=result['complete_original_primal']['all_original_rows_checked'],
             rounds=len(result['verified_full_constraint_separation']['rounds']),seconds=time.perf_counter()-started)
    except OptionalSOCNumericalFailure as error:
        report.update(QP_passed=False,optional_failure=error.optional_soc_report)
        emit(stage='saved_actual_SOC_still_retreats',reason=error.optional_soc_report['restricted_failure_reason'],
            seconds=time.perf_counter()-started)
    finally:_optional_soc.reset(token)
    report['seconds']=time.perf_counter()-started;save(OUT/'report.json',report)
    emit(stage='saved_SOC_exact_target_tightened_seed_replay_finished',report=ref(OUT/'report.json'))
if __name__=='__main__':main()
