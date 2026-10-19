import sys
from pathlib import Path
import argparse,json,time,numpy as np
from scipy import sparse
from scipy.optimize import linprog
from whole_W_qp_core import read,ref,verify,primary_program,load_W_transport
ap=argparse.ArgumentParser();ap.add_argument("--qp-report",required=True);ap.add_argument("--output",required=True);a=ap.parse_args();started=time.monotonic();Q=Path(a.qp_report);q=read(Q);assert q["solver_status"]=="Solved";D=Path(a.output);assert not D.exists();D.mkdir(parents=True)
p=primary_program(verify(q["primary_report"]));t,J,residual,transport_audit=load_W_transport(verify(q["transport"]),p);J.eliminate_zeros();n=p["n"];A=p["A"];rhs=p["rhs"];z=p["primary"];raw=np.load(Q.parent/"proposal_step.npz");wanted=raw["step_scaled"];bstar=float(p["report"]["linearized_epigraph"]);base=z["master_primal"].copy();assert base[n]==bstar;TR=float(p["report"]["trust_scaled_infinity"])
face=np.flatnonzero(z["master_inequality_dual"]!=0.);blo=np.flatnonzero(z["master_bound_lower_dual"][:n]!=0.);bhi=np.flatnonzero(z["master_bound_upper_dual"][:n]!=0.)
zero=np.flatnonzero((np.diff(J.indptr)==0)&(np.asarray(abs(A[:,:n]).sum(axis=0)).ravel()==0))
# The known primary is the affine origin. b is exactly the original primary
# scalar, not an independently rounded duplicate equality coordinate.
canonical_wanted=np.r_[wanted,bstar]
original_row_roundoff=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(canonical_wanted))
row_scale=np.exp2(np.maximum(0.,np.ceil(np.log2((32*1e-10)/original_row_roundoff))))
assert np.isfinite(row_scale).all() and np.all(row_scale>0)
H=sparse.diags(row_scale)@(A[:,:n]*TR);h=row_scale*(rhs-A@base);original_face_residual=(A@base-rhs)[face];h[face]=0.
I=sparse.eye(n,format="csr");desired=(wanted-base[:n])/TR
AA=sparse.vstack([sparse.hstack([H,sparse.csr_matrix((len(rhs),n))]),sparse.hstack([I,-I]),sparse.hstack([-I,-I])],format="csr");bb=np.r_[h,desired,-desired];EE=sparse.hstack([H[face],sparse.csr_matrix((len(face),n))],format="csr");ee=np.zeros(len(face))
ld=(p["lower"][:n]-base[:n])/TR;ud=(p["upper"][:n]-base[:n])/TR;bounds=list(zip(ld,ud))+[(0.,None)]*n
for i in blo:bounds[i]=(0.,0.)
for i in bhi:bounds[i]=(0.,0.)
for i in zero:bounds[i]=(-base[i]/TR,-base[i]/TR)
rr=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=AA,b_ub=bb,A_eq=EE,b_eq=ee,bounds=bounds,method="highs",options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
if not rr.success:
 (D/"report.json").write_text(json.dumps(dict(proposed=False,reason=rr.message,no_bare_primary_fallback=True,source=ref(__file__)),indent=2));raise SystemExit(5)
delta=base[:n]+TR*rr.x[:n];native_bound_representations=[]
for i in range(n):
 if rr.x[i]==ld[i]:delta[i]=p["lower"][i];native_bound_representations.append([i,"lower"])
 elif rr.x[i]==ud[i]:delta[i]=p["upper"][i];native_bound_representations.append([i,"upper"])
for i in blo:delta[i]=p["lower"][i]
for i in bhi:delta[i]=p["upper"][i]
for i in zero:delta[i]=0.
canonical=np.r_[delta,bstar];viol=A@canonical-rhs;eps=np.finfo(float).eps;error=64*eps*(1+abs(rhs)+abs(A)@abs(canonical));box=bool(np.all(canonical>=p["lower"])and np.all(canonical<=p["upper"]));master=bool(np.all(viol<=error));physical=np.zeros(len(p["value"]));physical[p["free"]]=p["scale"][p["free"]]*delta
np.savez_compressed(D/"proposal_step.npz",step_scaled=delta,free_ambient_indices=p["free"],physical_delta=physical,parent_values=p["value"],scales=p["scale"],centered_primal=rr.x,source_QP_primal=raw["QP_primal"],original_primary=base,original_face_rhs=rhs[face],internal_face_rhs=(A@base)[face])
diff=J@delta;predW=float(.5*(residual+diff)@(residual+diff));proposed=master and box and bool(np.all(delta[zero]==0));report=dict(schema="rinkfit.same_primary_W_QP_primal_feasibility_recovery.v2",proposed=proposed,QP_report=ref(Q),model=p["doc"]["model"],packet=p["doc"]["packet"],semantic=p["doc"]["semantic"],linearization=p["report"]["linearization"],original_q_max=p["report"]["original_q_max"],linearized_epigraph=bstar,primary_epigraph_unchanged=True,epigraph_relaxation_added=False,original_master_max_violation=float(viol.max()),original_master_roundoff_check=master,box_strictly_satisfied=box,box_max_lower_violation=float(np.max(p["lower"]-canonical)),box_max_upper_violation=float(np.max(canonical-p["upper"])),QP_raw_preserved=True,repair_objective="minimum scaled L1 distance to QP in original primary complementary face; centered at known primary",repair_is_not_a_complete_W_optimality_certificate=True,delta_to_QP_scaled_l1=float(np.sum(abs(delta-wanted))),delta_to_QP_scaled_inf=float(np.max(abs(delta-wanted))),original_primary_A_rhs_bounds_unchanged=True,internal_face_rhs_is_original_E_times_known_primary=True,original_face_rhs_difference_max=float(np.max(abs(original_face_residual),initial=0)),native_active_box_coordinate_representations=native_bound_representations,zero_W_and_master_columns=zero.tolist(),zero_null_column_nonzero_steps=int(np.count_nonzero(delta[zero])),predicted_W_linear_delta=float(residual@diff),predicted_W_quadratic_delta=float(.5*diff@diff),repaired_predicted_W_half_cost=predW,raw_QP_predicted_W_half_cost=q["predicted_W_half_cost"],predicted_W_repair_change=predW-q["predicted_W_half_cost"],baseline_W_half_cost=q["baseline_W_half_cost"],changed_physical_parameters=int(np.count_nonzero(physical)),conditional_fixed_current_ordinary_S_true_RS=True,PhaseII_joint_station_optimum=False,source_W_E_unchanged=True,physical_tolerances_unchanged=True,full_source_actual_trial_required=True,full_fit_or_KKT=False,solver_message=rr.message,positive_binary_row_scaling_only=True,row_scale_min=float(row_scale.min()),row_scale_max=float(row_scale.max()),original_row_roundoff_unchanged=True,source=ref(__file__),core=ref(Path(sys.modules["whole_W_qp_core"].__file__)),elapsed_seconds=time.monotonic()-started)
(D/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k]for k in["proposed","original_master_max_violation","box_strictly_satisfied","delta_to_QP_scaled_inf","original_face_rhs_difference_max","repaired_predicted_W_half_cost"]}),flush=True)
if not proposed:raise SystemExit(6)
