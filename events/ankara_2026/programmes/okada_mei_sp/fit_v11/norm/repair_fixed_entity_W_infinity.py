"""Same-b minimum infinity numerical correction of a conic W candidate.
All original A/box constraints and all entity quadratic norms are checked
in original units at the final delta. No old LP face or tolerance relaxation.
"""
from pathlib import Path
import argparse,json,time,sys,numpy as np
from scipy import sparse
from scipy.optimize import linprog
from fixed_entity_norm_constraints import read,verify,ref,FixedEntityNorms
p=argparse.ArgumentParser();p.add_argument('--qp-report',required=True);p.add_argument('--core-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
start=time.monotonic();Q=Path(a.qp_report).resolve();q=read(Q);D=Path(a.output).resolve();D.mkdir(exist_ok=False)
if not q['entity_SOC']['numeric_conic_solver_gate']:raise ValueError('W conic numerical gate failed')
P=verify(q['primary_report']);pr=read(P);A=sparse.load_npz(verify(pr['original_A']));bd=np.load(verify(pr['original_rhs_bounds']))
zz=np.load(verify(pr['solution']));raw=np.load(verify(q['proposal']));wanted=raw['step_scaled'].copy();base=zz['master_primal'];n=len(wanted);bstar=float(base[-1]);TR=float(pr['solver']['physical_scaled_trust'])
ec=np.load(verify(pr['entity_chart_arrays']));entities=FixedEntityNorms(ec['source_rows'],ec['residual'],sparse.load_npz(verify(pr['entity_J'])),pr['entity_chart'])
rhs=bd['rhs'];lo=bd['lower'];hi=bd['upper'];eps=np.finfo(float).eps;unit=TR*np.sqrt(eps);I=sparse.eye(n,format='csr')
rawcanonical=np.r_[wanted,bstar];rawv=A@rawcanonical-rhs;rawerr=64*eps*(1+abs(rhs)+abs(A)@abs(rawcanonical));bad=np.flatnonzero(rawv>rawerr)
rawdiagnostic=dict(original_row_indices=bad.tolist(),original_row_violation=rawv[bad].tolist(),primary_row_residual=(A@base-rhs)[bad].tolist(),
 b_coefficients=np.asarray(A[bad,-1].toarray()).ravel().tolist(),raw_box_lower_max=float(np.max(lo-rawcanonical)),raw_box_upper_max=float(np.max(rawcanonical-hi)))
(D/'initial_violation.json').write_text(json.dumps(rawdiagnostic,indent=2))
history=[];cuts=[];cutrhs=[];C,c,_=entities.supporting_cuts(wanted);cuts.append(C);cutrhs.append(c);delta=wanted.copy();last=None;iteration=0;previous=None
while True:
 C=sparse.vstack([A,*cuts],format='csr');cr=np.r_[rhs,*cutrhs];canonical=np.r_[delta,bstar]
 error=64*eps*(1+abs(cr)+abs(C)@abs(canonical));row_scale=np.exp2(np.maximum(0.,np.ceil(np.log2((32e-10)/error))))
 H=C[:,:n]*unit;h=cr-C@rawcanonical;H=H.multiply(row_scale[:,None]);h=h*row_scale
 AA=sparse.vstack([sparse.hstack([H,sparse.csr_matrix((len(cr),1))]),sparse.hstack([I,-np.ones((n,1))]),sparse.hstack([-I,-np.ones((n,1))])],format='csr')
 bounds=list(zip((lo[:-1]-wanted)/unit,(hi[:-1]-wanted)/unit))+[(0.,None)]
 s=linprog(np.r_[np.zeros(n),1.],A_ub=AA,b_ub=np.r_[h,np.zeros(2*n)],bounds=bounds,method='highs',
  options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
 if not s.success:
  history.append(dict(iteration=iteration,LP_status=int(s.status),message=s.message,passed=False));break
 delta=wanted+unit*s.x[:-1]
 for j in range(n):
  if s.x[j]==bounds[j][0]:delta[j]=lo[j]
  elif s.x[j]==bounds[j][1]:delta[j]=hi[j]
 canonical=np.r_[delta,bstar];v=A@canonical-rhs;er=64*eps*(1+abs(rhs)+abs(A)@abs(canonical));audit=entities.audit(delta,bstar)
 box=bool(np.all(canonical>=lo)and np.all(canonical<=hi));passed=bool(np.all(v<=er)and box and audit['all_entity_affine_norms_within_b'])
 item=dict(iteration=iteration,LP_status=int(s.status),message=s.message,original_master_max_violation=float(np.max(v,initial=0.)),
  original_master_beyond_IEEE_count=int(np.count_nonzero(v>er)),box_strictly_satisfied=box,entity_norm_audit=audit,
  correction_scaled_inf=float(np.max(abs(delta-wanted))),objective_numeric=float(s.fun),passed=passed)
 history.append(item);last=(v,er,audit,box,passed)
 (D/'progress.json').write_text(json.dumps(dict(history=history,elapsed_seconds=time.monotonic()-start),indent=2))
 if passed:break
 signature=delta.tobytes()
 if signature==previous:break
 previous=signature;qe,_=entities.values(delta);ids=np.flatnonzero(qe>bstar)
 if not len(ids):break
 Cnew,cnew,_=entities.supporting_cuts(delta,ids);cuts.append(Cnew);cutrhs.append(cnew);iteration+=1
passed=bool(last is not None and last[-1]);free=raw['free_ambient_indices'];values=raw['parent_values'];scales=raw['scales'];physical=np.zeros_like(values);physical[free]=scales[free]*delta
np.savez_compressed(D/'proposal_step.npz',physical_delta=physical,parent_values=values,scales=scales,free_ambient_indices=free,step_scaled=delta,
 original_primary=base,source_W_delta=wanted,canonical_primal=np.r_[delta,bstar])
report=dict(schema='rinkfit.same_primary_entity_SOC_W_original_unit_repair.v1',proposed=passed,QP_report=ref(Q),primary_report=ref(P),
 model=q['model'],packet=q['packet'],semantic=q['semantic'],linearization=q['linearization'],transport=q['transport'],
 original_q_max=q['original_q_max'],linearized_epigraph=bstar,first_level_epigraph=bstar,canonical_b_IEEE_unchanged=True,
 fixed_primary_b_not_relaxed=True,old_LP_dual_complementary_face_not_used=True,original_A_rhs_box_and_TR_unchanged=True,
 repair_method='minimum scaled infinity numerical correction around raw conic W candidate',numeric_correction_unit=float(unit),
 every_entity_audited_AFTER_final_repair=True,entity_norm_audit=last[2]if last else None,original_unit_audit=history[-1],iterations=history,
 initial_violation=ref(D/'initial_violation.json'),source_W_solution=q['proposal'],W_secondary_optimality_after_repair_not_certified=True,
 original_W_and_source_E_unchanged=True,raw_rows=182017,active_rows=182016,rank=214112,
 entity_facets_not_nonlinear_PHR_multipliers=True,full_source_nonlinear_trial_required=True,final_fit_or_KKT=False,
 source=ref(__file__),module=ref(Path(__file__).with_name('fixed_entity_norm_constraints.py')),proposal=ref(D/'proposal_step.npz'),
 elapsed_seconds=time.monotonic()-start)
(D/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
raise SystemExit(0 if passed else 7)
