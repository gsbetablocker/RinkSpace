"""Audit and replay the captured real reduced QP, without source/J evaluation."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from scipy.linalg import solve_triangular
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from calibration.solver_runtime import install
install()
import z190_global_schur_linear_step as qp
from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal

directory=Path(sys.argv[1]).resolve()
with np.load(directory/'original_problem.npz',allow_pickle=False) as packet:
    H=packet['matrix'];g=packet['gradient'];lo=packet['lower'];hi=packet['upper']
    A=packet['constraint_matrix'];b=packet['constraint_lower']
    priority_A=packet['priority_constraint_matrix'];original_b=packet['priority_original_constraint_lower']
    ownership=packet['priority_row_group_index'];violations=packet['priority_current_group_violations']
lower=np.flatnonzero(np.isfinite(lo));upper=np.flatnonzero(np.isfinite(hi));eye=np.eye(len(g))
rows=np.vstack((A,eye[lower],-eye[upper]));targets=np.r_[b,lo[lower],-hi[upper]]
objective=max(1.,float(np.min(np.diag(H))));D=np.sqrt(objective)/np.sqrt(np.diag(H))
L=np.linalg.cholesky(H*D[:,None]*D[None,:]/objective)
white=solve_triangular(L,(rows*D).T,lower=True,check_finite=False).T
row_scale=np.linalg.norm(white,axis=1)/np.sqrt(objective)
center=D*np.linalg.solve(H*D[:,None]*D[None,:]/objective,-g*D/objective)
rhs=(targets-rows@center)/row_scale;worst=int(np.argmax(abs(rhs)))
report=dict(schema='rinkspace.actual-required-reduced-QP-primal-replay.v1',
    input_sha256=hashlib.sha256((directory/'original_problem.npz').read_bytes()).hexdigest(),
    source_QP_module_sha256=hashlib.sha256(Path(qp.__file__).read_bytes()).hexdigest(),
    original_A_shape=list(A.shape),original_bounds=int(len(targets)-len(b)),
    legacy_global_tolerance=float(1e-11*max(1.,np.max(abs(rhs)))),
    legacy_global_primal_threshold=float(1e-10*max(1.,np.max(abs(rhs)))),
    unconstrained_center_norm=float(np.linalg.norm(center)),
    maximum_rhs_original_row=worst,maximum_rhs=float(rhs[worst]),
    maximum_rhs_target=float(targets[worst]),maximum_rhs_row_scale=float(row_scale[worst]),
    maximum_rhs_original_A_center=float(rows[worst]@center),
    maximum_rhs_original_A_norm=float(np.linalg.norm(rows[worst])),
    maximum_rhs_group=None if worst>=len(b) else int(ownership[worst]),
    maximum_rhs_group_violation=None if worst>=len(b) or ownership[worst]<0 else float(violations[ownership[worst]]))
start=time.perf_counter()
try:
    step,held,linear=qp._linearly_constrained_shared_step(H,g,lo,hi,A,b)
    slack,tolerance,bad,primal=original_halfspace_primal(rows,targets,step)
    np.savez(directory/(sys.argv[2]+'_direction.npz'),step=step,slack=slack,
        original_primal_tolerance=tolerance,violated_original_rows=np.flatnonzero(bad))
    report.update(seconds=time.perf_counter()-start,returned_direction=True,
        complete_original_primal=primal,linear_report=linear,
        maximum_primal_error_group=int(ownership[primal['worst_original_row']])
            if primal['worst_original_row'] is not None and primal['worst_original_row']<len(b) else None)
except Exception as exc:
    report.update(seconds=time.perf_counter()-start,returned_direction=False,
        error_type=type(exc).__name__,error=str(exc))
(directory/(sys.argv[2]+'_replay.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in report.items()if k!='linear_report'},indent=2),flush=True)
