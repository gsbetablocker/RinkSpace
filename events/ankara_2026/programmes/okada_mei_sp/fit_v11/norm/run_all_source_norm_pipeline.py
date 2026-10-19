"""All-source affine-norm constraint generation, original conditional W.
This is an explicit saved-witness proposal chart. It does not ask unknown
qbar envelope gradients to be zero or stationary. Every raw source remains
in the final all-active affine audit and later nonlinear full-shot trial.
"""
from pathlib import Path
import argparse,json,sys,time,subprocess,numpy as np
from scipy import sparse
from fixed_entity_norm_constraints import read,verify,ref,array,solve_entity_primary_soc,solve_entity_W_secondary
from source_affine_norm_constraints import load_all_source_affine_norms
p=argparse.ArgumentParser();p.add_argument('--linearization',required=True);p.add_argument('--transport',required=True)
p.add_argument('--profile-domains',required=True);p.add_argument('--core-directory',required=True);p.add_argument('--trust',type=float,required=True)
p.add_argument('--output',required=True);p.add_argument('--seed-source-audit');p.add_argument('--stop-file')
p.add_argument('--allow-declared-inexact-primary',action='store_true');p.add_argument('--reuse-primary-report')
p.add_argument('--allow-declared-inexact-W',action='store_true');p.add_argument('--reuse-W-report')
p.add_argument('--allow-phase-I-budget-priority-W',dest='allow_budget_priority_W',action='store_true');p.add_argument('--allow-inexact-epigraph-upper',action='store_true');p.add_argument('--seed-source-membership');a=p.parse_args()
start=time.monotonic();D=Path(a.output).resolve();D.mkdir(exist_ok=False);L=Path(a.linearization).resolve();d=read(L);T=Path(a.transport).resolve();core=Path(a.core_directory).resolve();sys.path.insert(0,str(core))
modelpath=verify(d['model']);m=read(modelpath);blocks=m['blocks'];sizes=np.array([len(b['values'])for b in blocks]);offset=np.r_[0,np.cumsum(sizes)]
value=np.concatenate([b['values']for b in blocks]);scale=np.concatenate([b['scales']for b in blocks]);lo=np.concatenate([b['lower']for b in blocks]);hi=np.concatenate([b['upper']for b in blocks])
fixed=np.zeros(len(value),bool)
for k,b in enumerate(blocks):
 if b.get('constant',False):fixed[offset[k]:offset[k+1]]=True
 for c in b.get('fixed_indices',[]):fixed[offset[k]+c]=True
if not np.isfinite(np.r_[value,scale,lo,hi]).all()or np.any(scale<=0)or np.any(value<lo)or np.any(value>hi):raise ValueError('Original physical chart')
free=np.flatnonzero(~fixed);n=len(free);mapping=np.full(len(value),-1,np.int64);mapping[free]=np.arange(n)
def csr(c,rows,base=L.parent):
 ptr=array(c['row_offsets'],base).astype(np.int64);bi=array(c['block_id'],base).astype(np.int64);co=array(c['component'],base).astype(np.int64);v=array(c['values'],base)
 if len(ptr)!=rows+1 or ptr[0]!=0 or ptr[-1]!=len(v)or np.any(np.diff(ptr)<0)or np.any(bi<0)or np.any(bi>=len(blocks))or np.any(co<0)or np.any(co>=sizes[bi]):raise ValueError('CSR layout')
 amb=offset[bi]+co;keep=~fixed[amb];ri=np.repeat(np.arange(rows),np.diff(ptr))
 return sparse.csr_matrix((v[keep]*scale[amb[keep]],(ri[keep],mapping[amb[keep]])),shape=(rows,n))
selected=array(d['source_rows'],L.parent).astype(np.int64);q=array(d['q'],L.parent);status=array(d['status'],L.parent);mode=array(d['mode'],L.parent)
if len(np.unique(selected))!=len(selected)or 25537 in selected:raise ValueError('Selected source ledger')
full=load_all_source_affine_norms(L,T,offset,scale,free);baseq,_=full.values(np.zeros(n));qmax=float(baseq.max())
verified=np.flatnonzero(status==0);Q=csr(d['qbar_CSR'],len(selected));G=csr(d['scene_CSR'],len(d['scene_specs']));g=array(d['scene_values'],L.parent)
A=sparse.vstack([sparse.hstack([Q[verified],-np.ones((len(verified),1))]),sparse.hstack([G,sparse.csr_matrix((len(g),1))])],format='csr');rhs=np.r_[-q[verified],-g]
from profile_domain_constraints import load_profile_domains
H,hr,profile_mapping=load_profile_domains(L,a.profile_domains,offset,scale,free);A=sparse.vstack([A,sparse.hstack([H,sparse.csr_matrix((len(hr),1))])],format='csr');rhs=np.r_[rhs,hr]
from temporary_visibility_chart import select_visibility_chart_risks,load_visibility_chart_guards
risk_rows,risk=select_visibility_chart_risks(L,T,offset,scale,free,a.trust)
visibility_mapping=None
if len(risk_rows):
 V,vr,visibility_mapping=load_visibility_chart_guards(L,T,offset,scale,free,risk_rows);A=sparse.vstack([A,sparse.hstack([V,sparse.csr_matrix((len(vr),1))])],format='csr');rhs=np.r_[rhs,vr]
unselected=np.setdiff1d(full.rows,selected);floor=float(baseq[np.searchsorted(full.rows,unselected)].max())if len(unselected)else 0.
# The former unmodeled-q constant floor is an algorithmic substitute for
# absent pixel models. All-active affine norm separation replaces it here;
# retaining that obsolete floor would forbid genuine source improvement.
lower=np.r_[np.maximum(-a.trust,(lo[free]-value[free])/scale[free]),0.];upper=np.r_[np.minimum(a.trust,(hi[free]-value[free])/scale[free]),np.inf]
unknown=selected[status!=0];mandatory_unknown=selected[(status!=0)&(q>=.9*qmax)]
coverage=dict(raw_rows=182017,active_rows=182016,linearized_selected_rows=len(selected),verified_qbar_rows=len(verified),selected_unknown_rows=unknown.tolist(),
 old_envelope_guard_would_reject=bool(len(mandatory_unknown)),mandatory_unknown_rows=mandatory_unknown.tolist(),missing_qbar_gradient_not_treated_as_zero=True,
 unknown_sources_covered_by_defined_fixed_saved_S_true_RS_UV_norm=True,all_source_affine_norms_required_before_proposal=True,
 final_joint_stationarity_or_KKT_claim=False,ordinary_station_remains_a_free_unknown_in_final_problem=True,
 historical_unmodeled_q_constant_floor=floor,historical_unmodeled_floor_replaced_by_all_active_norm_separation=True)
sparse.save_npz(D/'original_A.npz',A);np.savez_compressed(D/'original_rhs_bounds.npz',rhs=rhs,lower=lower,upper=upper)
ctx=dict(report={'linearization':ref(L)},doc=d,model=m,model_path=modelpath,free=free,n=n,csr=csr)
from whole_W_qp_core import load_W_transport
t,JW,rW,Winfo=load_W_transport(T,ctx)
cone_rows=set(map(int,full.rows[full.kind==1]));cone_rows.update(map(int,mandatory_unknown));seed=None
if a.seed_source_audit:
 seed=ref(a.seed_source_audit);sd=read(verify(seed));cone_rows.update(map(int,sd['audit']['violating_source_rows']))
membership_ref=None
if a.seed_source_membership:
 membership_ref=ref(a.seed_source_membership);membership_path=verify(membership_ref);membership=read(membership_path)
 if membership['schema']!='rinkfit.source_norm_constraint_membership.v1'or not membership['membership_only_not_new_state_violation_or_derivative']:raise ValueError('Norm membership scope')
 for key in ['packet','semantic']:
  verify(membership[key]);assert membership[key]['sha256']==d[key]['sha256'],'Membership different source authority'
 member_rows=array(membership['source_rows'],membership_path.parent).astype(np.int64)
 if len(np.unique(member_rows))!=len(member_rows)or not np.isin(member_rows,full.rows).all():raise ValueError('Invalid active source membership')
 cone_rows.update(map(int,member_rows))
history=[];iteration=0;final=None
while True:
 if a.stop_file and Path(a.stop_file).exists():break
 I=D/('generation_%03d'%iteration);I.mkdir();rows=np.array(sorted(cone_rows),dtype='<u8');norms=full.subset(rows)
 print(json.dumps(dict(stage='primary_started',generation=iteration,norm_cones=len(rows),original_linear_rows=A.shape[0],elapsed_seconds=time.monotonic()-start)),flush=True)
 if iteration==0 and a.reuse_primary_report:
  RP=Path(a.reuse_primary_report).resolve();rp=read(RP)
  for key,actual in [('linearization',ref(L)),('transport',ref(T)),('profile_domains',ref(a.profile_domains))]:
   verify(rp[key]);assert rp[key]['sha256']==actual['sha256'],'Reused primary different '+key
  ra=sparse.load_npz(verify(rp['original_A']));rb=np.load(verify(rp['original_rhs_bounds']));rc=np.load(verify(rp['entity_chart_arrays']));rj=sparse.load_npz(verify(rp['entity_J']))
  assert ra.shape==A.shape and np.array_equal(ra.indptr,A.indptr)and np.array_equal(ra.indices,A.indices)and np.array_equal(ra.data,A.data)
  for key,actual in [('rhs',rhs),('lower',lower),('upper',upper)]:assert rb[key].tobytes()==actual.tobytes()
  assert rc['source_rows'].tobytes()==rows.tobytes()and rc['residual'].tobytes()==norms.residual.tobytes()
  assert rj.shape==norms.J.shape and np.array_equal(rj.indptr,norms.J.indptr)and np.array_equal(rj.indices,norms.J.indices)and np.array_equal(rj.data,norms.J.data)
  rz=np.load(verify(rp['solution']));sol={k:rz[k]for k in rz.files if k not in ['free_ambient_indices','parent_values','scales']};sol['report']=rp['solver']
 else:sol=solve_entity_primary_soc(A,rhs,lower,upper,norms,a.trust,qref=max(1.,qmax),threads=1)
 working_inexact=bool(a.allow_declared_inexact_primary and not sol['report']['master_ready_for_W_secondary']and sol['report']['strict_original_primal_gate']and float(sol['master_primal'][-1])<qmax)
 np.savez_compressed(I/'entity_primary_solution.npz',**{k:v for k,v in sol.items()if isinstance(v,np.ndarray)},free_ambient_indices=free,parent_values=value,scales=scale)
 sparse.save_npz(I/'entity_J.npz',norms.J);np.savez_compressed(I/'entity_chart.npz',source_rows=rows,residual=norms.residual)
 primary=dict(schema='rinkfit.fixed_entity_primary_SOC_actual_driver.v1',ready_for_W_secondary=bool(sol['report']['master_ready_for_W_secondary']or working_inexact),physical_proposal_authorized=False,
  linearization=ref(L),transport=ref(T),profile_domains=ref(a.profile_domains),original_master_qmax=qmax,original_A=ref(D/'original_A.npz'),original_rhs_bounds=ref(D/'original_rhs_bounds.npz'),
  entity_chart=norms.metadata,entity_J=ref(I/'entity_J.npz'),entity_chart_arrays=ref(I/'entity_chart.npz'),solution=ref(I/'entity_primary_solution.npz'),solver=sol['report'],
  full_source_norm_generation=True,all_physical_entities=1692,temporary_ordinary_witness_norm_count=int(np.count_nonzero(full.kind[np.searchsorted(full.rows,rows)]==0)),
  source=ref(__file__),module=ref(Path(__file__).with_name('source_affine_norm_constraints.py')),coverage=coverage,total_seconds=time.monotonic()-start,
  primary_working_target_declared_inexact=working_inexact,primary_optimality_verified=bool(sol['report']['numeric_conic_solver_gate']),
  ready_for_W_is_primal_working_target_not_solver_status_override=working_inexact,PHR_multiplier_update_permitted=False,
  reused_primary_report=ref(a.reuse_primary_report)if iteration==0 and a.reuse_primary_report else None)
 (I/'primary_report.json').write_text(json.dumps(primary,indent=2))
 if not primary['ready_for_W_secondary']:
  history.append(dict(generation=iteration,primary=ref(I/'primary_report.json'),passed=False,reason='primary_original_numeric_or_primal_gate'));break
 print(json.dumps(dict(stage='same_b_W_started',generation=iteration,primary_b=float(sol['master_primal'][-1]),elapsed_seconds=time.monotonic()-start)),flush=True)
 if iteration==0 and a.reuse_W_report:
  RW=Path(a.reuse_W_report).resolve();rw=read(RW);rwp=read(verify(rw['primary_report']));rwz=np.load(verify(rwp['solution']))
  for key,actual in [('linearization',ref(L)),('transport',ref(T))]:verify(rw[key]);assert rw[key]['sha256']==actual['sha256']
  assert rwz['master_primal'].tobytes()==sol['master_primal'].tobytes()
  rx=np.load(verify(rw['proposal']));w={k:rx[k]for k in rx.files if k not in ['physical_delta','free_ambient_indices','parent_values','scales']};w['report']=rw['entity_SOC']
 else:w=solve_entity_W_secondary(A,rhs,lower,upper,norms,sol['master_primal'],JW,rW,a.trust,threads=1)
 working_W=bool(a.allow_declared_inexact_W and not w['report']['numeric_conic_solver_gate']and w['report']['original_master_beyond_IEEE_count']==0 and w['report']['original_box_strictly_satisfied']
  and w['report']['predicted_W_half_cost']<(w['report']['primary_W_half_cost']if a.allow_budget_priority_W else min(w['report']['baseline_W_half_cost'],w['report']['primary_W_half_cost'])))
 delta=w['step_scaled'];physical=np.zeros(len(value));physical[free]=scale[free]*delta
 np.savez_compressed(I/'proposal_step.npz',**{k:v for k,v in w.items()if isinstance(v,np.ndarray)},physical_delta=physical,free_ambient_indices=free,parent_values=value,scales=scale)
 wr=dict(schema='rinkfit.same_primary_full_W_entity_SOC_proposal.v1',proposed=w['report']['proposed'],primary_report=ref(I/'primary_report.json'),transport=ref(T),linearization=ref(L),
  model=d['model'],packet=d['packet'],semantic=d['semantic'],original_q_max=qmax,linearized_epigraph=sol['master_primal'][-1],first_level_epigraph=sol['master_primal'][-1],
  entity_SOC=w['report'],baseline_W_half_cost=w['report']['baseline_W_half_cost'],predicted_W_half_cost=w['report']['predicted_W_half_cost'],proposal=ref(I/'proposal_step.npz'),
  all_norm_cones_are_proposal_charts_not_permanent_source_station_facts=True,source=ref(__file__),elapsed_seconds=w['report']['elapsed_seconds'])
 (I/'W_report.json').write_text(json.dumps(wr,indent=2))
 if a.allow_inexact_epigraph_upper and a.allow_declared_inexact_W:
  # A free-b Phase-I working upper is not a proven optimal b*. Preserve
  # physical delta and all original W; derive and explicitly audit its
  # smallest canonical full-source budget upper before accepting a trial.
  qa,_=full.values(delta);old_work=float(sol['master_primal'][-1]);bcol=np.asarray(A[:,-1].toarray()).ravel()
  assert np.all(bcol<=0),'An epigraph upper-side constraint forbids this canonicalization'
  needed=max(old_work,float(qa.max()));bm=bcol<0;linear_physical=A[:,:-1]@delta
  if np.any(bm):needed=max(needed,float(np.max((linear_physical[bm]-rhs[bm])/(-bcol[bm]))))
  target=float(np.nextafter(needed,np.inf));native=np.r_[delta,target]
  vv=A@native-rhs;ee=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(native));strict_box=bool(np.all(native>=lower)and np.all(native<=upper))
  diff=JW@delta;Wcost=float(.5*(rW+diff)@(rW+diff));all_audit=full.audit(delta,target)
  legal=bool(target<qmax and np.all(vv<=ee)and strict_box and all_audit['all_source_affine_norms_within_b']and Wcost<(w['report']['primary_W_half_cost']if a.allow_budget_priority_W else float(.5*rW@rW)))
  attempt=dict(working_target_b=target,old_conic_working_target_b=old_work,parent_affine_qmax=qmax,W_half_cost=Wcost,
   original_A_beyond_IEEE=int(np.count_nonzero(vv>ee)),strict_box=strict_box,all_source_norms=all_audit,passed=legal)
  (I/'working_epigraph_upper_attempt.json').write_text(json.dumps(attempt,indent=2))
  if legal:
   history.append(dict(generation=iteration,norm_cones=len(rows),primary=ref(I/'primary_report.json'),W=ref(I/'W_report.json'),
    canonical_working_upper_audit=ref(I/'working_epigraph_upper_attempt.json'),all_source_affine_violation_count=0,
    working_target_b=target,primary_solver_status=sol['report']['solver_status'],W_solver_status=w['report']['solver_status'],
    statuses_not_changed_to_convergence=True,elapsed_seconds=time.monotonic()-start))
   raw_arrays=dict(np.load(verify(wr['proposal'])));raw_arrays['canonical_primal']=native
   np.savez_compressed(I/'canonical_working_proposal.npz',**raw_arrays)
   final=dict(wr);final.update(schema='rinkfit.all_source_affine_norm_generated_W_proposal.v1',proposed=True,
    source_norm_chart=full.metadata,coverage=coverage,all_source_affine_audit=all_audit,full_source_norm_generation=True,all_physical_entities_kept=1692,
    linearized_epigraph=target,first_level_epigraph=target,working_target_b=target,original_conic_working_target_b=old_work,source_proposal_report=ref(I/'W_report.json'),numeric_solver_status_not_overridden=True,
    PhaseI_budget_priority_secondary_policy=bool(a.allow_budget_priority_W),W_parent_descent_not_a_PhaseI_feasibility_contract=bool(a.allow_budget_priority_W),
    explicit_inexact_epigraph_upper_canonicalization=True,working_target_budget_coordinate_changed=True,
    canonical_b_IEEE_unchanged=False,canonical_primary_b_IEEE_unchanged=False,primary_epigraph_unchanged=False,fixed_primary_b_not_relaxed=False,
    engineering_budget_E_changed=False,original_phaseI_b_is_free=True,final_phaseII_requires_original_b_le_1=True,
    primary_working_target_declared_inexact=True,primary_optimality_verified=False,W_secondary_declared_inexact=not w['report']['numeric_conic_solver_gate'],
    W_secondary_optimality_verified=False,PHR_multiplier_update_permitted=False,raw_solver_status_and_numeric_gate_not_overridden=True,
    original_unit_audit=dict(original_master_max_violation=float(np.max(vv,initial=0.)),original_master_beyond_IEEE_count=0,box_strictly_satisfied=True,passed=True),
    baseline_W_half_cost=float(.5*rW@rW),predicted_W_linear_delta=float(rW@diff),predicted_W_quadratic_delta=float(.5*diff@diff),predicted_W_half_cost=Wcost,
    entity_norm_audit=full.subset(full.rows[full.kind==1]).audit(delta,target),every_entity_audited_AFTER_final_delta=True,
    proposal=ref(I/'canonical_working_proposal.npz'),original_W_conic_report=ref(I/'W_report.json'),generation_history=history,
    original_profile_domain_mapping=profile_mapping,original_temporary_visibility_mapping=visibility_mapping,
    seed_source_membership=membership_ref,source=ref(__file__),elapsed_seconds=time.monotonic()-start,
    full_source_nonlinear_trial_required=True,final_fit_or_KKT=False)
   break
 # Full-source separation precedes the numerical W gate. A rejected W
 # iterate can identify omitted source constraints, never authorize a step.
 raw_audit=full.audit(delta,float(sol['master_primal'][-1]));(I/'raw_W_full_source_audit.json').write_text(json.dumps(raw_audit,indent=2))
 raw_new=set(map(int,raw_audit['violating_source_rows']))-cone_rows
 if raw_new:
  item=dict(generation=iteration,norm_cones=len(rows),primary=ref(I/'primary_report.json'),W=ref(I/'W_report.json'),raw_W_full_source_audit=ref(I/'raw_W_full_source_audit.json'),new_source_constraints=len(raw_new),raw_W_status_not_overridden=True,no_physical_proposal_authorized=True,elapsed_seconds=time.monotonic()-start)
  history.append(item);(D/'progress.json').write_text(json.dumps(dict(history=history,coverage=coverage),indent=2));print(json.dumps(dict(stage='full_source_constraints_added_before_W_gate',**item)),flush=True)
  cone_rows.update(raw_new);iteration+=1;continue
 if not w['report']['numeric_conic_solver_gate']and not working_W:
  history.append(dict(generation=iteration,primary=ref(I/'primary_report.json'),W=ref(I/'W_report.json'),passed=False,reason='W_numeric_gate'));break
 repair=I/'repair'
 if working_W:
  # A declared-inexact W iterate is eligible only when its original-unit
  # primal is already strict. Do not lie to the old repair numerical gate.
  repair.mkdir();raw_delta=delta.copy();ray_history=[]
  if not w['report']['entity_norm_audit']['all_entity_affine_norms_within_b']:
   anchor=sol['master_primal'][:-1];budget=float(sol['master_primal'][-1]);ray_ok=False
   for k in range(52,-1,-1):
    tau=1.-2.**(-k);candidate=anchor+tau*(raw_delta-anchor);native=np.r_[candidate,budget]
    violation=A@native-rhs;error=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(native));na=norms.audit(candidate,budget)
    box=bool(np.all(native>=lower)and np.all(native<=upper));diff=JW@candidate;cost=float(.5*(rW+diff)@(rW+diff))
    ray_ok=bool(np.all(violation<=error)and box and na['all_entity_affine_norms_within_b']and cost<(w['report']['primary_W_half_cost']if a.allow_budget_priority_W else w['report']['baseline_W_half_cost']))
    ray_history.append(dict(fraction=tau,original_A_beyond_IEEE=int(np.count_nonzero(violation>error)),strict_box=box,source_norm_audit=na,predicted_W=cost,passed=ray_ok))
    if ray_ok:delta=candidate;break
   (repair/'inexact_ray_audit.json').write_text(json.dumps(ray_history,indent=2))
   if not ray_ok:
    history.append(dict(generation=iteration,W=ref(I/'W_report.json'),passed=False,reason='inexact_W_primal_repair_failed'));break
  else:tau=1.
  diff=JW@delta;physical=np.zeros(len(value));physical[free]=scale[free]*delta
  np.savez_compressed(repair/'proposal_step.npz',physical_delta=physical,free_ambient_indices=free,parent_values=value,scales=scale,step_scaled=delta,
   canonical_primal=np.r_[delta,sol['master_primal'][-1]],source_W_delta=raw_delta)
  repaired=dict(wr);repaired.update(proposed=True,W_secondary_declared_inexact=True,
   numeric_solver_status_not_overridden=True,PHR_multiplier_update_permitted=False,
   baseline_W_half_cost=float(.5*rW@rW),predicted_W_linear_delta=float(rW@diff),predicted_W_quadratic_delta=float(.5*diff@diff),
   predicted_W_half_cost=float(.5*(rW+diff)@(rW+diff)),entity_norm_audit=norms.audit(delta,float(sol['master_primal'][-1])),
   original_unit_audit=dict(original_master_beyond_IEEE_count=0,box_strictly_satisfied=True,passed=True),
   every_entity_audited_AFTER_final_delta=True,final_fit_or_KKT=False,QP_report=ref(I/'W_report.json'),
   raw_W_original_numeric_and_primal_status_unchanged=True,feasible_convex_fraction_of_W=float(tau),proposal=ref(repair/'proposal_step.npz'))
  (repair/'report.json').write_text(json.dumps(repaired,indent=2));rz={'step_scaled':delta}
 else:
  cmd=[sys.executable,str(Path(__file__).with_name('run_fixed_entity_W_repair.py')),'--qp-report',str(I/'W_report.json'),'--core-directory',str(core),'--output',str(repair)]
  child=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8');(I/'repair_stdout.log').write_text(child.stdout);(I/'repair_stderr.log').write_text(child.stderr)
  if child.returncode:
   history.append(dict(generation=iteration,W=ref(I/'W_report.json'),passed=False,reason='strict_repair_failed',exit_code=child.returncode));break
  repaired=read(repair/'report.json');rz=np.load(verify(repaired['proposal']))
 audit=full.audit(rz['step_scaled'],repaired['linearized_epigraph'])
 (I/'full_source_audit.json').write_text(json.dumps(audit,indent=2));item=dict(generation=iteration,norm_cones=len(rows),primary=ref(I/'primary_report.json'),W=ref(I/'W_report.json'),
  repaired_proposal=ref(repair/'report.json'),all_source_audit=ref(I/'full_source_audit.json'),violating_count=audit['violating_source_count'],maximum_affine_q=audit['maximum_affine_q'],
  budget=float(repaired['linearized_epigraph']),elapsed_seconds=time.monotonic()-start)
 history.append(item);(D/'progress.json').write_text(json.dumps(dict(history=history,coverage=coverage),indent=2));print(json.dumps(item),flush=True)
 if audit['all_source_affine_norms_within_b']:
  final=dict(repaired);final.update(schema='rinkfit.all_source_affine_norm_generated_W_proposal.v1',all_source_affine_audit=audit,full_source_norm_generation=True,
   all_physical_entities_kept=1692,source_norm_chart=full.metadata,coverage=coverage,generation_history=history,seed_source_audit=seed,
   original_profile_domain_mapping=profile_mapping,original_temporary_visibility_mapping=visibility_mapping,source=ref(__file__),elapsed_seconds=time.monotonic()-start,
   primary_working_target_declared_inexact=working_inexact,primary_optimality_verified=bool(sol['report']['numeric_conic_solver_gate']),
   W_secondary_declared_inexact=working_W,W_secondary_optimality_verified=bool(w['report']['numeric_conic_solver_gate']),
   working_target_b=float(sol['master_primal'][-1]),PhaseI_budget_priority_secondary_policy=bool(a.allow_budget_priority_W),W_parent_descent_not_a_PhaseI_feasibility_contract=bool(a.allow_budget_priority_W),PHR_multiplier_update_permitted=False)
  break
 new=set(map(int,audit['violating_source_rows']))-cone_rows
 if not new:
  history[-1]['failed_closed_reason']='already_constrained_norm_failed_full_source_audit';break
 cone_rows.update(new);iteration+=1
report=final or dict(schema='rinkfit.all_source_affine_norm_generated_W_proposal.v1',proposed=False,coverage=coverage,generation_history=history,source=ref(__file__),elapsed_seconds=time.monotonic()-start)
(D/'final_membership.u64').write_bytes(np.asarray(sorted(cone_rows),dtype='<u8').tobytes())
member=dict(schema='rinkfit.source_norm_constraint_membership.v1',packet=d['packet'],semantic=d['semantic'],
 source_rows=dict(file='final_membership.u64',dtype='<u8',shape=[len(cone_rows)],sha256=ref(D/'final_membership.u64')['sha256']),
 membership_only_not_new_state_violation_or_derivative=True,origin_state=ref(L))
(D/'source_membership.json').write_text(json.dumps(member,indent=2));report['next_state_source_membership']=ref(D/'source_membership.json')
(D/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(report=ref(D/'report.json'),proposed=bool(report.get('proposed')),elapsed_seconds=time.monotonic()-start)),flush=True)
raise SystemExit(0 if final else 8)
