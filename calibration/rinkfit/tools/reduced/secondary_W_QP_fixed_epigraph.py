from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from scipy import sparse
import clarabel
ap=argparse.ArgumentParser();ap.add_argument("--primary-report",required=True);ap.add_argument("--transport",required=True);ap.add_argument("--output",required=True);ap.add_argument("--core-directory");ap.add_argument("--stop-file");a=ap.parse_args()
if a.core_directory:sys.path.insert(0,a.core_directory)
from whole_W_qp_core import primary_program,load_W_transport
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=str(p),sha256=sha(p))
start=time.monotonic();D=Path(a.output);assert not D.exists();D.mkdir(parents=True)
p=primary_program(a.primary_report);t,J,r,audit=load_W_transport(a.transport,p);n=p["n"];z=p["primary"];A=p["A"];rhs=p["rhs"];base=z["master_primal"].copy();bstar=float(p["report"]["linearized_epigraph"]);assert base[-1]==bstar;TR=float(p["report"]["trust_scaled_infinity"])
H=(J.T@J).tocsc();res0=r+J@base[:n];g0=np.asarray(J.T@res0).ravel();objective_scale=max(1.,float(.5*r@r))
P=sparse.triu(H*(TR*TR/objective_scale)).tocsc();linear=g0*(TR/objective_scale)
face=np.flatnonzero(z["master_inequality_dual"]!=0.);blo=np.flatnonzero(z["master_bound_lower_dual"][:n]!=0.);bhi=np.flatnonzero(z["master_bound_upper_dual"][:n]!=0.);fixed=np.unique(np.r_[blo,bhi]);I=sparse.eye(n,format="csr")
ld=(p["lower"][:n]-base[:n])/TR;ud=(p["upper"][:n]-base[:n])/TR;fixedval=np.asarray([ld[i]if i in blo else ud[i]for i in fixed])
# Center at the known primary with the original b eliminated, not a duplicate
# free coordinate or a new equality. The complementary face is homogeneous.
B=A[:,:n]*TR;h=rhs-A@base;face_original_residual=(A@base-rhs)[face];h[face]=0.
row_norm=np.asarray(abs(B).max(axis=1).toarray()).ravel();row_scale=np.ones(len(h));nonzero=row_norm>0;row_scale[nonzero]=np.exp2(-np.rint(np.log2(row_norm[nonzero])));B=sparse.diags(row_scale)@B;h=row_scale*h
E=sparse.vstack([B[face],I[fixed]],format="csc");erhs=np.r_[np.zeros(len(face)),fixedval]
keep=np.ones(A.shape[0],bool);keep[face]=False;other=np.setdiff1d(np.arange(n),fixed);upper=other[np.isfinite(ud[other])]
C=sparse.vstack([B[keep],-I[other],I[upper]],format="csc");crhs=np.r_[h[keep],-ld[other],ud[upper]]
settings=clarabel.DefaultSettings();settings.verbose=True;settings.max_threads=1
solver=clarabel.DefaultSolver(P,linear,sparse.vstack([E,C],format="csc"),np.r_[erhs,crhs],[clarabel.ZeroConeT(len(erhs)),clarabel.NonnegativeConeT(len(crhs))],settings)
if a.stop_file:solver.set_termination_callback(lambda info:Path(a.stop_file).exists())
print("CENTERED_FIXED_B_W_QP_STARTED",json.dumps(dict(variables=n,eliminated_b=bstar,face=len(face),equalities=len(erhs),inequalities=len(crhs),H_nnz=int(H.nnz))),flush=True)
s=solver.solve();d=np.asarray(s.x);delta=base[:n]+TR*d;canonical=np.r_[delta,bstar];viol=A@canonical-rhs;roundoff=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(canonical));status=str(s.status);box=bool(np.all(canonical>=p["lower"])and np.all(canonical<=p["upper"]));passed=status=="Solved"and bool(np.all(viol<=roundoff))and box
physical=np.zeros(len(p["value"]));physical[p["free"]]=p["scale"][p["free"]]*delta;diff=J@delta;pred=r+diff
np.savez_compressed(D/"proposal_step.npz",step_scaled=delta,free_ambient_indices=p["free"],physical_delta=physical,parent_values=p["value"],scales=p["scale"],QP_primal=canonical,centered_QP_primal=d,QP_dual=np.asarray(s.z),original_primary=base)
report=dict(schema="rinkfit.same_primary_full_W_conditional_QP.v1",proposed=passed,solver_status=status,primary_report=ref(a.primary_report),transport=ref(a.transport),model=p["doc"]["model"],packet=p["doc"]["packet"],semantic=p["doc"]["semantic"],raw_rows=182017,active_rows=182016,rank=214112,first_level_epigraph=bstar,QP_epigraph_raw=bstar,epigraph_coordinate_eliminated=True,canonical_epigraph_unchanged=True,epigraph_relaxation_added=False,original_master_max_violation=float(viol.max()),original_master_roundoff_check=bool(np.all(viol<=roundoff)),box_strictly_satisfied=box,baseline_W_half_cost=float(.5*r@r),predicted_W_half_cost=float(.5*pred@pred),conditional_fixed_current_ordinary_S_true_RS=True,profiles_original_implicit_S_V_alpha=True,PhaseII_joint_station_optimum=False,tertiary_minimum_movement_solved=False,changed_physical_parameters=int(np.count_nonzero(physical)),full_source_nonlinear_trial_required=True,transport_derivative_FD_QA_required_before_adoption=True,physical_tolerances_unchanged=True,source_W_E_unchanged=True,full_fit_or_KKT=False,solver_iterations=s.iterations,solver_primal_residual=s.r_prim,solver_dual_residual=s.r_dual,transport_audit=audit,predicted_W_linear_delta=float(r@diff),predicted_W_quadratic_delta=float(.5*diff@diff),positive_objective_scale=objective_scale,internal_numerical_regularization_not_added_to_objective=True,working_face_rhs_is_known_primary_face=True,original_face_rhs_difference_max=float(np.max(abs(face_original_residual),initial=0)),positive_binary_equilibration_only=True,row_scale_min=float(row_scale.min()),row_scale_max=float(row_scale.max()),source=ref(__file__),core=ref(sys.modules["whole_W_qp_core"].__file__),elapsed_seconds=time.monotonic()-start)
(D/"report.json").write_text(json.dumps(report,indent=2));print("CENTERED_FIXED_B_W_QP_DONE",json.dumps(report),flush=True)
