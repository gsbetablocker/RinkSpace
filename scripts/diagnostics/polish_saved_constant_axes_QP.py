"""Refine the saved restricted QP, certifying all original half-spaces."""
from pathlib import Path
import os,sys,time,json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.solver_runtime import install
install()
import numpy as np
from scipy.linalg import solve_triangular,qr
from scipy.sparse import csc_matrix,load_npz
import clarabel
from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
CAP=DATA/'calibration/runs/solver_acceleration_20261003/required_QP_capture_q757_20261011_0100/required_reduced_QP'
C=CAP/'constant_axes_restricted_trace_original_settings'
OUT=CAP/'constant_axes_saved_restricted_active_KKT_polish'
OUT.mkdir(exist_ok=True)
def emit(**value):
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
with np.load(CAP/'all_truth_constant_axes_contracted_problem.npz',allow_pickle=False) as z:
    H=z['matrix'];g=z['gradient'];lo=z['lower'];hi=z['upper'];A=z['constraint_matrix'];b=z['constraint_lower']
eye=np.eye(len(g));fl=np.flatnonzero(np.isfinite(lo));fu=np.flatnonzero(np.isfinite(hi))
rows=np.vstack((A,eye[fl],-eye[fu]));targets=np.r_[b,lo[fl],-hi[fu]]
objective_scale=max(1.,float(np.min(np.diag(H))))
D=np.sqrt(objective_scale)/np.sqrt(np.diag(H));L=np.linalg.cholesky(H*D[:,None]*D[None,:]/objective_scale)
white=solve_triangular(L,(rows*D).T,lower=True,check_finite=False).T
scale=np.sqrt(np.einsum('ij,ij->i',white,white));B=white/scale[:,None];rhs=targets/scale
q=solve_triangular(L,D*g/objective_scale,lower=True,check_finite=False)
def original(y,z,selected):
    x=D*solve_triangular(L.T,y,lower=False,check_finite=False)
    dual=np.zeros(len(rows));dual[selected]=z*objective_scale/scale[selected]
    slack,tol,bad,primal=original_halfspace_primal(rows,targets,x)
    stat=H@x+g-rows.T@dual
    comp=slack*dual
    st=1e-9*max(1.,float(np.max(abs(g))),float(np.max(abs(H@x))))
    ct=1e-9*max(1.,abs(float(x@H@x)),abs(float(g@x)),abs(float(targets@dual)))
    return x,dual,slack,tol,bad,dict(primal=primal,maximum_stationarity=float(np.max(abs(stat))),
        stationarity_tolerance=st,minimum_original_dual=float(dual.min()),
        complementarity_absolute_sum=float(np.sum(abs(comp))),complementarity_tolerance=ct,
        accepted=bool(primal['accepted'] and np.max(abs(stat))<=st and dual.min()>=-1e-11 and np.sum(abs(comp))<=ct))
# Recover the exact original selected index set using the saved backend steps.
violated=np.flatnonzero(rhs>0.);initial=violated[np.argsort(rhs[violated])[-min(len(g),len(violated)):]]
selected=np.union1d(initial,np.arange(len(b),len(rows)))
for ordinal in range(1,14):
    saved=np.load(C/f'call{ordinal:03d}_x_z.npz');y=saved['x'];z=saved['z']
    expected=load_npz(C/f'call{ordinal:03d}_A.npz').toarray()
    assert np.array_equal(expected,-B[selected])
    if ordinal<13:
        x,dual,slack,tol,bad,report=original(y,z,selected)
        mask=bad.copy();mask[selected]=False;violated=np.flatnonzero(mask)
        added=violated[np.argsort(slack[violated]/(scale[violated]/np.sqrt(objective_scale)))[:min(len(g)//2,len(violated))]]
        selected=np.union1d(selected,added)
np.savez(OUT/'saved_round13_original_candidate.npz',selected=selected,y=y,z=z)
np.savez(OUT/'saved_round13_physical_candidate.npz',selected=selected,step=original(y,z,selected)[0],
         original_rows=rows[selected],original_targets=targets[selected])
emit(stage='saved_round13_recovered',selected_rows=len(selected),original=original(y,z,selected)[-1])
def polish(Bs,rs,y,z):
    # IPM duals propose an equality basis; every original inequality still gates
    # the candidate. This threshold changes work only, never accepted geometry.
    working=np.flatnonzero(z>np.sqrt(np.finfo(float).eps)*max(1.,float(z.max())))
    history=[];seen=set()
    while len(working):
        key=tuple(working.tolist())
        if key in seen:
            emit(stage='refinement_cycle',active_rows=len(working),history=history)
            raise RuntimeError('Active equality refinement cycled')
        seen.add(key)
        W=Bs[working]
        Q,R=qr(W.T,mode='economic')
        v=solve_triangular(R.T,rs[working]+W@q,lower=True)
        candidate=-q+Q@v
        for iteration in range(3):
            residual=np.asarray(np.asarray(rs[working],np.longdouble)-np.asarray(W,np.longdouble)@np.asarray(candidate,np.longdouble),float)
            candidate+=Q@solve_triangular(R.T,residual,lower=True)
        multiplier=solve_triangular(R,Q.T@(candidate+q),lower=False)
        if multiplier.min() < -1e-11:
            remove=int(np.argmin(multiplier));history.append(dict(removed=int(working[remove]),dual=float(multiplier[remove])))
            working=np.delete(working,remove);continue
        slack=Bs@candidate-rs
        error=64*np.finfo(float).eps*(np.abs(Bs)@np.abs(candidate)+np.abs(rs)+1.)
        violated=np.flatnonzero(slack < -error)
        if len(violated):
            available=np.setdiff1d(violated,working)
            if not len(available):raise RuntimeError('Equality refinement failed original restricted faces')
            add=int(available[np.argmin(slack[available])]);history.append(dict(added=add,slack=float(slack[add])))
            working=np.union1d(working,[add]);continue
        dual=np.zeros(len(rs));dual[working]=multiplier
        return candidate,dual,dict(active_rows=len(working),history=history,
            maximum_active_equalities_error=float(np.max(abs(slack[working]))),
            maximum_white_stationarity=float(np.max(abs(candidate+q-Bs.T@dual))))
    return -q,np.zeros(len(rs)),dict(active_rows=0,history=history)
settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
settings.static_regularization_enable=False;settings.dynamic_regularization_enable=False
for name in ('tol_feas','tol_gap_abs','tol_gap_rel','tol_infeas_abs','tol_infeas_rel'):setattr(settings,name,min(1e-12,np.finfo(float).eps**.875))
started=time.perf_counter();round_index=13
while True:
    before=original(y,z,selected)[-1];t=time.perf_counter()
    y,z,p=polish(B[selected],rhs[selected],y,z)
    x,dual,slack,tol,bad,report=original(y,z,selected)
    emit(stage='active_KKT_refined',round=round_index,selected_rows=len(selected),seconds=time.perf_counter()-t,
        refinement=p,before=before,after=report,selected_bad=int(bad[selected].sum()))
    np.savez(OUT/'latest_direction.npz',step=x,original_dual=dual,selected=selected,y=y,z=z)
    if bad[selected].any():raise RuntimeError('Refined selected original face remains infeasible')
    if report['accepted']:
        (OUT/'verified_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        emit(stage='complete_original_QP_verified',seconds=time.perf_counter()-started,report=report);break
    violated=np.flatnonzero(bad)
    if not len(violated):raise RuntimeError('Primal passed but refined original full KKT failed')
    added=violated[np.argsort(slack[violated]/scale[violated])[:min(len(g)//2,len(violated))]]
    selected=np.union1d(selected,added);round_index+=1
    t=time.perf_counter()
    result=clarabel.DefaultSolver(csc_matrix(np.eye(len(g))),q,csc_matrix(-B[selected]),-rhs[selected],
        [clarabel.NonnegativeConeT(len(selected))],settings).solve()
    emit(stage='next_restricted_backend',round=round_index,rows=len(selected),seconds=time.perf_counter()-t,
        status=str(result.status),iterations=int(result.iterations))
    if str(result.status)not in ('Solved','AlmostSolved'):raise RuntimeError('Restricted solver did not provide a refineable candidate')
    y=np.asarray(result.x);z=np.asarray(result.z)
