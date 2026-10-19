"""Saved real QP replay with backend-stage evidence, no source construction."""
from pathlib import Path
import os, sys, time, json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
import numpy as np
from scipy.sparse import save_npz
import clarabel
import z190_global_schur_linear_step as qp

DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
CAP=DATA/'calibration/runs/solver_acceleration_20261003/required_QP_capture_q757_20261011_0100/required_reduced_QP'
regularized='--regularized' in sys.argv
label='regularized' if regularized else 'original_settings'
exclude_declared_fixed='--exclude-declared-fixed' in sys.argv
if exclude_declared_fixed:label+='__declared_fixed_free_chart'
OUT=CAP/f'constant_axes_restricted_trace_{label}'
OUT.mkdir(exist_ok=True)
events=[]
def emit(**value):
    events.append(value)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
    (OUT/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
class DiagnosticStop(BaseException): pass
real_solver=clarabel.DefaultSolver
call_count=0
class TracedSolver:
    def __init__(self,P,q,A,b,cones,settings):
        global call_count
        call_count+=1; self.ordinal=call_count
        self.rows=A.shape[0]
        if self.rows==full_original_row_count:
            emit(stage='full_IPM_fallback_not_repeated',rows=self.rows,variables=A.shape[1])
            raise DiagnosticStop()
        save_npz(OUT/f'call{self.ordinal:03d}_P.npz',P)
        save_npz(OUT/f'call{self.ordinal:03d}_A.npz',A)
        np.savez(OUT/f'call{self.ordinal:03d}_q_b.npz',q=q,b=b)
        if regularized:
            default=clarabel.DefaultSettings()
            settings.static_regularization_enable=default.static_regularization_enable
            settings.dynamic_regularization_enable=default.dynamic_regularization_enable
        emit(stage='restricted_backend_begin',ordinal=self.ordinal,rows=self.rows,variables=A.shape[1],
             static_regularization=settings.static_regularization_enable,
             dynamic_regularization=settings.dynamic_regularization_enable)
        self.solver=real_solver(P,q,A,b,cones,settings)
    def solve(self):
        started=time.perf_counter(); result=self.solver.solve()
        np.savez(OUT/f'call{self.ordinal:03d}_x_z.npz',x=result.x,z=result.z)
        emit(stage='restricted_backend_end',ordinal=self.ordinal,rows=self.rows,
             status=str(result.status),iterations=int(result.iterations),seconds=time.perf_counter()-started,
             primal_residual=float(result.r_prim),dual_residual=float(result.r_dual))
        return result
clarabel.DefaultSolver=TracedSolver
seen=set()
def trace(frame,event,arg):
    if frame.f_code.co_name=='separated_affine_step':
        if event=='line' and 'entry' in frame.f_locals:
            entry=frame.f_locals['entry']
            if 'complete_original_primal' in entry:
                key=(id(entry),entry.get('unselected_violations'))
                if key not in seen:
                    seen.add(key);emit(stage='restricted_original_row_gate',**entry)
        if event=='return':
            data=frame.f_locals
            if 'candidate' in data:
                selected=data['selected'];bad=selected[data['primal_bad'][selected]]
                np.savez(OUT/'separation_last_candidate.npz',candidate=data['candidate'],selected=selected,
                         candidate_slack=data['candidate_slack'],candidate_dual=data['candidate_dual'],
                         original_primal_tolerance=data['primal_tolerance'])
                emit(stage='restricted_selected_row_failure_detail',bad_selected_rows=bad.tolist(),
                     slacks=data['candidate_slack'][bad].tolist(),tolerances=data['primal_tolerance'][bad].tolist())
            emit(stage='separation_return',report=frame.f_locals.get('separation_report',{}))
        return trace
    return None
with np.load(CAP/'all_truth_constant_axes_contracted_problem.npz',allow_pickle=False) as z:
    H=z['matrix'];g=z['gradient'];lo=z['lower'];hi=z['upper'];A=z['constraint_matrix'];b=z['constraint_lower']
    columns=z['global_parameter_indices']
original_H=H;original_g=g;original_A=A;original_lo=lo;original_hi=hi
free=np.ones(len(g),bool)
if exclude_declared_fixed:
    fixed=int(np.flatnonzero(columns==2)[0]);free[fixed]=False
    assert np.count_nonzero(A[:,fixed])==0 and np.count_nonzero(H[fixed,free])==0
    assert np.count_nonzero(H[free,fixed])==0 and g[fixed]==0
    assert np.isneginf(lo[fixed]) and np.isposinf(hi[fixed])
    H=H[np.ix_(free,free)];g=g[free];lo=lo[free];hi=hi[free];A=A[:,free]
    emit(stage='declared_fixed_shared_tangent',requested_variables=len(free),effective_variables=int(free.sum()),
         removed_canonical_indices=[2],original_A_fixed_column_exact_zero=True,
         existing_fixed_H_identity_zero_gradient_reused=True,original_targets_and_free_LM_unchanged=True)
full_original_row_count=len(b)+int(np.isfinite(lo).sum())+int(np.isfinite(hi).sum())
started=time.perf_counter();sys.settrace(trace)
try:
    x,held,report=qp._linearly_constrained_shared_step(H,g,lo,hi,A,b)
    sys.settrace(None)
    if exclude_declared_fixed:
        lifted=np.zeros(len(free));lifted[free]=x
        from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal
        pieces=[original_A];targets=[b]
        if np.isfinite(original_lo).any():
            pieces.append(np.eye(len(free))[np.isfinite(original_lo)]);targets.append(original_lo[np.isfinite(original_lo)])
        if np.isfinite(original_hi).any():
            pieces.append(-np.eye(len(free))[np.isfinite(original_hi)]);targets.append(-original_hi[np.isfinite(original_hi)])
        _,_,_,primal=original_halfspace_primal(np.vstack(pieces),np.concatenate(targets),lifted)
        assert primal['accepted'] and lifted[~free].tolist()==[0.]
        report['declared_fixed_shared_chart']=dict(requested_variables=len(free),effective_variables=int(free.sum()),
            removed_canonical_indices=[2],exact_fixed_lift_values=lifted[~free].tolist(),
            complete_original_lifted_primal=primal,
            fixed_stationarity_exact=float((original_H@lifted+original_g)[~free].max()),
            new_physical_or_source_constraints_added=False)
        np.savez(OUT/'verified_direction.npz',step=lifted,free_step=x,free_canonical_indices=columns[free],
            requested_canonical_indices=columns)
    else:np.savez(OUT/'verified_direction.npz',step=x)
    (OUT/'verified_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    emit(stage='complete_QP_verified',seconds=time.perf_counter()-started,report=report)
except DiagnosticStop:
    sys.settrace(None);emit(stage='diagnostic_complete_before_full_fallback',seconds=time.perf_counter()-started)
except Exception as error:
    sys.settrace(None);emit(stage='diagnostic_failure',seconds=time.perf_counter()-started,error_type=type(error).__name__,error=str(error))
    raise
