"""Finite/visible partial minimization for the original all-source Phase-I budget.
Data prototype: no nonlinear physical line search and no KKT promotion.
"""
from pathlib import Path
import json,time,hashlib,numpy as np
from scipy import sparse
from scipy.optimize import linprog
from fixed_entity_norm_constraints import read,verify,ref,array
from source_affine_norm_constraints import load_all_source_affine_norms

class FiniteVisiblePrimary:
 def __init__(self,qbar,transport,profile_domains,trust,*,station_TR=None):
  self.L=Path(qbar).resolve();self.T=Path(transport).resolve();self.d=read(self.L);t=read(self.T);self.tr=t;self.trust=float(trust)
  self.model=read(verify(self.d["model"]));blocks=self.model["blocks"];sizes=np.array([len(b["values"])for b in blocks]);self.offset=np.r_[0,np.cumsum(sizes)]
  value=np.concatenate([b["values"]for b in blocks]);scale=np.concatenate([b["scales"]for b in blocks]);lo=np.concatenate([b["lower"]for b in blocks]);hi=np.concatenate([b["upper"]for b in blocks]);fixed=np.zeros(len(value),bool)
  for k,b in enumerate(blocks):
   if b.get("constant",False):fixed[self.offset[k]:self.offset[k+1]]=True
   for c in b.get("fixed_indices",[]):fixed[self.offset[k]+c]=True
  if not np.isfinite(np.r_[value,scale,lo,hi]).all()or np.any(scale<=0)or np.any(value<lo)or np.any(value>hi)or not np.isfinite(trust)or trust<=0:raise ValueError("Original physical box/gauge/scale")
  self.free=np.flatnonzero(~fixed);self.n=len(self.free);self.scale=scale;self.value=value;mapping=np.full(len(value),-1,int);mapping[self.free]=np.arange(self.n)
  self.full=load_all_source_affine_norms(self.L,self.T,self.offset,scale,self.free)
  def csr(c,nrows):
   ptr=array(c["row_offsets"],self.L.parent).astype(np.int64);bi=array(c["block_id"],self.L.parent).astype(int);co=array(c["component"],self.L.parent).astype(int);v=array(c["values"],self.L.parent)
   if len(ptr)!=nrows+1 or ptr[-1]!=len(v)or np.any(bi<0)or np.any(bi>=len(sizes))or np.any(co<0)or np.any(co>=sizes[bi]):raise ValueError("Original CSR layout")
   amb=self.offset[bi]+co;keep=~fixed[amb];ri=np.repeat(np.arange(nrows),np.diff(ptr))
   return sparse.csr_matrix((v[keep]*scale[amb[keep]],(ri[keep],mapping[amb[keep]])),shape=(nrows,self.n))
  selected=array(self.d["source_rows"],self.L.parent).astype(int);status=array(self.d["status"],self.L.parent);q=array(self.d["q"],self.L.parent)
  verified=np.flatnonzero(status==0);Q=csr(self.d["qbar_CSR"],len(selected));G=csr(self.d["scene_CSR"],len(self.d["scene_specs"]));g=array(self.d["scene_values"],self.L.parent)
  from profile_domain_constraints import load_profile_domains
  H,hr,self.profile_mapping=load_profile_domains(self.L,profile_domains,self.offset,scale,self.free)
  # Only reliable qbar supports, physical scene and exact original profile domains here.
  self.baseA=sparse.vstack([sparse.hstack([Q[verified],-np.ones((len(verified),1))]),sparse.hstack([G,sparse.csr_matrix((len(g),1))]),sparse.hstack([H,sparse.csr_matrix((len(hr),1))])],format="csr");self.baserhs=np.r_[-q[verified],-g,hr]
  self.preserved_counts=dict(verified_qbar=len(verified),scene=len(g),original_profile_domain=len(hr));self.qbar_verified_source_rows=selected[verified]
  self.lower=np.r_[np.maximum(-trust,(lo[self.free]-value[self.free])/scale[self.free]),0.];self.upper=np.r_[np.minimum(trust,(hi[self.free]-value[self.free])/scale[self.free]),np.inf]
  self.cpP=verify(t["checkpoint"]);self.cp=read(self.cpP);S=array(self.cp["row_state_files"]["station"],self.cpP.parent);branch=array(self.cp["row_state_files"]["branch"],self.cpP.parent)
  SP=verify(t["semantic"]);PK=verify(t["packet"]);pk=read(PK);self.N=int(pk["observations"]);self.E=array(pk["arrays"]["engineering_budget_px"],PK.parent)
  from rinkfit_start.semantic_resolution import load_semantics
  _,semantic_arrays,_,verified_semantics=load_semantics(SP,self.N,expected_sha256=t["semantic"]["sha256"]);self.kind=semantic_arrays["kind"];self.normal=semantic_arrays["profile_normal"];mi=np.asarray(verified_semantics["resolved_material_index"],dtype=int)
  self.active=np.asarray(verified_semantics["source_exclusion"]["active_mask"],dtype=bool);self.rows=np.flatnonzero(self.active);self.idx=np.full(self.N,-1,int);self.idx[self.rows]=np.arange(len(self.rows))
  if not np.array_equal(self.rows,self.full.rows)or self.cp["raw_rows"]!=self.N or self.cp["active_rows"]!=len(self.rows):raise ValueError("Whole verified source scope")
  self.row0=array(t["arrays"]["true_rs_row"],self.T.parent);self.tangent=array(t["arrays"]["station_UV_right"],self.T.parent).reshape(self.N,2)
  tl=array(t["arrays"]["station_delta_lower_right"],self.T.parent);tu=array(t["arrays"]["station_delta_upper_right"],self.T.parent);stat=array(t["arrays"]["status"],self.T.parent)
  smooth=self.active&(self.kind==0)&(stat==0)&(S>1e-10)&(S<1-1e-10)
  for row in np.flatnonzero(smooth):
   path=self.model["materials"][mi[row]]["paths"][int(branch[row])]
   if path["kind"]==2 and abs(S[row]*(len(path["points"])-1)-round(S[row]*(len(path["points"])-1)))<=1e-7:smooth[row]=False
  self.released=np.flatnonzero(smooth);self.Ls=np.zeros(self.N);self.Us=np.zeros(self.N)
  self.Ls[smooth]=tl[smooth];self.Us[smooth]=tu[smooth]
  if station_TR is not None:
   rad=np.asarray(station_TR,float)
   if rad.ndim==0:rad=np.full(self.N,float(rad))
   if rad.shape!=(self.N,)or np.any(rad[self.active]<0)or np.any(~np.isfinite(rad[self.active])):raise ValueError("Explicit current-model numerical S trust")
   self.Ls[smooth]=np.maximum(self.Ls[smooth],-rad[smooth]);self.Us[smooth]=np.minimum(self.Us[smooth],rad[smooth])
  if np.any(self.Ls[self.active]>0)or np.any(self.Us[self.active]<0):raise ValueError("Original saved S outside finite/TR domain")
  self.sensorrows=np.flatnonzero(self.active&(self.kind!=2))
  vnorm=self.full.J[2*self.idx[self.sensorrows]+1];self.Jv=sparse.diags(self.E[self.sensorrows])@vnorm
  tv=self.tangent[self.sensorrows,1];ls=self.Ls[self.sensorrows];us=self.Us[self.sensorrows]
  maximum=np.maximum(tv*ls,tv*us);minimum=np.minimum(tv*ls,tv*us)
  self.reachA=sparse.vstack([self.Jv,-self.Jv],format="csr")
  self.reachrhs=np.r_[1079-self.row0[self.sensorrows]-minimum,self.row0[self.sensorrows]+maximum]
  self.all_reachA=self.reachA;self.all_reachrhs=self.reachrhs.copy()
  # Strict box-wide linear upper enclosure. This proves omitted numerical
  # sensor rows redundant on the ORIGINAL physical box intersected with TR.
  pos=self.all_reachA.maximum(0);neg=self.all_reachA.minimum(0)
  estimate=np.asarray(pos@self.upper[:-1]+neg@self.lower[:-1]).ravel()
  magnitude=np.asarray(abs(self.all_reachA)@np.maximum(abs(self.lower[:-1]),abs(self.upper[:-1]))).ravel()
  terms=np.diff(self.all_reachA.indptr).astype(float);ge=(2*terms+8)*np.finfo(float).eps;gamma=ge/(1-ge)
  self.reach_upper_estimate=estimate;self.reach_magnitude=magnitude
  self.reach_upper_enclosure=np.nextafter(estimate+gamma*magnitude/(1-gamma),np.inf)
  self.reach_keep=np.flatnonzero(self.reach_upper_enclosure>self.all_reachrhs);self.reach_removed=np.flatnonzero(self.reach_upper_enclosure<=self.all_reachrhs)
  self.reachA=self.all_reachA[self.reach_keep].tocsr();self.reachrhs=self.all_reachrhs[self.reach_keep]
  self.A=sparse.vstack([self.baseA,sparse.hstack([self.reachA,sparse.csr_matrix((self.reachA.shape[0],1))])],format="csr");self.rhs=np.r_[self.baserhs,self.reachrhs]
  self.parent_b=float(self.cp["b"])
 def separate(self,delta):
  delta=np.asarray(delta,float)
  if delta.shape!=(self.n,)or not np.isfinite(delta).all():raise ValueError("Original physical chart vector")
  q,r=self.full.values(delta);dr=(self.full.J@delta).reshape(-1,2);G=2*r;intercept=q-np.einsum("ij,ij->i",G,dr)
  ds=np.zeros(self.N);feasible=np.ones(self.N,bool);active=np.full(self.N,0,np.uint8);rn=self.idx[self.released];rows=self.released;E=self.E[rows];t=self.tangent[rows];tn=t/E[:,None];a=r[rn];tt=np.einsum("ij,ij->i",tn,tn);at=np.einsum("ij,ij->i",a,tn);desired=np.divide(-at,tt,out=np.zeros_like(at),where=tt>0)
  y=self.row0[rows]+dr[rn,1]*E;tv=t[:,1];L=self.Ls[rows];U=self.Us[rows];vl=np.full(len(rows),-np.inf);vu=np.full(len(rows),np.inf);nonzero=tv!=0
  low=np.divide(-y,tv,out=np.zeros_like(y),where=nonzero);high=np.divide(1079-y,tv,out=np.zeros_like(y),where=nonzero)
  vl[nonzero]=np.minimum(low[nonzero],high[nonzero]);vu[nonzero]=np.maximum(low[nonzero],high[nonzero]);L2=np.maximum(L,vl);U2=np.minimum(U,vu);ok=(L2<=U2)&(nonzero|((y>=0)&(y<=1079)));x=np.minimum(np.maximum(desired,L2),U2)
  rr=a+tn*x[:,None];qx=2*np.einsum("ij,ij->i",rr,tn);lo=(desired<L2)|((L2==U2)&(qx>=0));up=(desired>U2)|((L2==U2)&(qx<0));grad=2*rr
  # Envelope contribution from moving visibility bounds. If S is fixed on
  # its original cell/TR bound, ds/dphysical=0. At a sensor bound ds/dV=-1/tv.
  lovis=lo&(vl>L);upvis=up&(vu<U);vis=(lovis|upvis)&ok
  grad[vis,1]-=qx[vis]*E[vis]/tv[vis]
  qnew=np.einsum("ij,ij->i",rr,rr);inter=qnew-np.einsum("ij,ij->i",grad,dr[rn])
  q[rn[ok]]=qnew[ok];G[rn[ok]]=grad[ok];intercept[rn[ok]]=inter[ok];ds[rows[ok]]=x[ok];feasible[rows]=ok
  active[rows[lo&ok]]=1;active[rows[up&ok]]=2;active[rows[vis]]=3
  dV=dr[self.idx[self.sensorrows],1]*self.E[self.sensorrows];sensor=self.row0[self.sensorrows]+dV+self.tangent[self.sensorrows,1]*ds[self.sensorrows]
  # Recovery may need an explicitly recorded inward IEEE step at a sensor edge.
  # No tolerance is used to change feasibility; invalid recovery remains false.
  recovered=(sensor>=0)&(sensor<=1079);feasible[self.sensorrows]&=recovered
  return dict(q=q,gradient=G,intercept=intercept,deltaS=ds,feasible=feasible,active_status=active,normalized_deltaUV=dr,sensor_rows=self.sensorrows,sensor_values=sensor,all_source_budget_max=float(q.max()),infeasible_source_rows=self.rows[~feasible[self.rows]])
 def support(self,sep,indices):
  idx=np.asarray(indices,int);rows=sparse.csr_matrix((len(idx),self.n))
  rows=sparse.diags(sep["gradient"][idx,0])@self.full.J[2*idx]+sparse.diags(sep["gradient"][idx,1])@self.full.J[2*idx+1]
  return sparse.hstack([rows,-np.ones((len(idx),1))],format="csr"),-sep["intercept"][idx]
 def original_linear_audit(self,x):
  v=np.r_[self.baseA@x-self.baserhs,self.all_reachA@x[:-1]-self.all_reachrhs];allow=np.r_[64*np.finfo(float).eps*(1+abs(self.baserhs)+abs(self.baseA)@abs(x)),64*np.finfo(float).eps*(1+abs(self.all_reachrhs)+abs(self.all_reachA)@abs(x[:-1]))]
  return dict(beyond_IEEE=int(np.count_nonzero(v>allow)),maximum_violation=float(np.max(v,initial=0)),strict_box=bool(np.all(x>=self.lower)&np.all(x<=self.upper)))
 def recover_seed_audit(self,delta,sep,b):
  ds=sep["deltaS"];nr=self.idx[self.rows];r=self.full.residual+(self.full.J@delta).reshape(-1,2);r[nr[self.kind[self.rows]==0]]+=self.tangent[self.rows[self.kind[self.rows]==0]]/self.E[self.rows[self.kind[self.rows]==0],None]*ds[self.rows[self.kind[self.rows]==0],None]
  q=np.einsum("ij,ij->i",r,r)
  return dict(all_active_rows=len(self.rows),maximum_q=float(q.max()),q_exceeding_b=int(np.count_nonzero(q>b)),all_source_norms_with_same_recovered_S=bool(np.all(q<=b)),strict_finite_S=bool(np.all(ds[self.rows]>=self.Ls[self.rows])and np.all(ds[self.rows]<=self.Us[self.rows])),sensor_source_rows_with_exact_affine_domain_failure=sep["infeasible_source_rows"].tolist(),all_source_affine_sensor_domains_passed=bool(len(sep["infeasible_source_rows"])==0),secondary_W_not_solved=True,nonlinear_acceptance=False)
 def run(self,output,*,stop_file=None):
  D=Path(output).resolve();D.mkdir(exist_ok=False);start=time.perf_counter();history=[];seed=self.separate(np.zeros(self.n))
  fixedq,_=self.full.values(np.zeros(self.n));seedrows=np.unique(np.r_[self.qbar_verified_source_rows,self.rows[self.kind[self.rows]==1],self.rows[fixedq>=.25*float(fixedq.max())]])
  seedidx=self.idx[seedrows];C,cr=self.support(seed,seedidx)
  np.savez_compressed(D/"sensor_redundancy_proof.npz",physical_lower=self.lower[:-1],physical_upper=self.upper[:-1],sensor_source_rows=self.sensorrows,reach_rhs=self.all_reachrhs,box_linear_upper_estimate=self.reach_upper_estimate,absolute_dot_magnitude=self.reach_magnitude,box_linear_upper_enclosure=self.reach_upper_enclosure,kept_row_indices=self.reach_keep,omitted_proven_redundant_row_indices=self.reach_removed,initial_budget_seed_source_rows=seedrows)
  (D/"assembly.json").write_text(json.dumps(dict(schema="rinkfit.finite_station_primary_box_redundancy_and_budget_seed.v1",source=ref(__file__),transport=ref(self.T),qbar=ref(self.L),checkpoint=self.tr["checkpoint"],proof_arrays=ref(D/"sensor_redundancy_proof.npz"),all_sensor_rows=self.all_reachA.shape[0],kept_sensor_rows=len(self.reach_keep),omitted_sensor_rows=len(self.reach_removed),every_omitted_row_box_upper_le_rhs=True,ordering="all ordinary/entity source rows sorted: upper rows first, lower rows second",box_is_original_physical_bounds_intersect_TR=True,base_preserved_rows=self.preserved_counts,initial_budget_seed_source_rows=len(seedrows),near_active_threshold_is_seed_only=.25,all_active_source_separation_required_before_working_target=True),indent=2))
  print("REDUCED_ASSEMBLY",json.dumps(dict(sensor_all=self.all_reachA.shape[0],kept_sensor=len(self.reach_keep),omitted_sensor=len(self.reach_removed),budget_seed=len(seedrows),base_rows=self.baseA.shape[0],columns=self.n+1)),flush=True)
  objective=np.r_[np.zeros(self.n),1.];iteration=0;final=None
  while True:
   if stop_file and Path(stop_file).exists():break
   I=D/("generation_%03d"%iteration);I.mkdir();AA=sparse.vstack([self.A,C],format="csr");rr=np.r_[self.rhs,cr]
   print(json.dumps(dict(stage="LP_started",iteration=iteration,shape=AA.shape,nnz=AA.nnz,elapsed_seconds=time.perf_counter()-start)),flush=True)
   # No solver tolerance is weakened. This is an explicitly inexact primary
   # proposal when the independently recovered primal passes before optimality.
   sol=linprog(objective,A_ub=AA,b_ub=rr,bounds=list(zip(self.lower,self.upper)),method="highs",options={"presolve":True,"threads":1})
   if sol.x is None:history.append(dict(iteration=iteration,status=int(sol.status),message=str(sol.message)));break
   delta=sol.x[:-1];sep=self.separate(delta);bLP=float(sol.x[-1]);bcol=np.asarray(self.A[:,-1].toarray()).ravel();mask=bcol<0;requiredA=float(np.max(((self.A[:,:-1]@delta)[mask]-self.rhs[mask])/(-bcol[mask]),initial=0));work=float(np.nextafter(max(bLP,sep["all_source_budget_max"],requiredA),np.inf));x=np.r_[delta,work];audit=self.original_linear_audit(x);seed_audit=self.recover_seed_audit(delta,sep,work)
   direction=np.zeros(len(self.value));direction[self.free]=self.scale[self.free]*delta
   np.savez_compressed(I/"proposal.npz",parent_values=self.value,scales=self.scale,free_ambient_indices=self.free,step_scaled=delta,physical_delta=direction,canonical_primal=x,station_delta=sep["deltaS"],all_source_q=sep["q"])
   item=dict(iteration=iteration,LP_status=int(sol.status),LP_message=str(sol.message),LP_objective_b=bLP,recovered_working_b=work,parent_b=self.parent_b,budget_reduction=self.parent_b-work,source_norm_violations_at_LP_b=int(np.count_nonzero(sep["q"]>bLP)),all_source_budget_max=sep["all_source_budget_max"],original_linear_audit=audit,source_recovery_audit=seed_audit,proposal=ref(I/"proposal.npz"),elapsed_seconds=time.perf_counter()-start)
   history.append(item);(I/"report.json").write_text(json.dumps(item,indent=2));print(json.dumps(item),flush=True)
   ready=work<self.parent_b and audit["beyond_IEEE"]==0 and audit["strict_box"]and seed_audit["all_source_norms_with_same_recovered_S"]and seed_audit["strict_finite_S"]and seed_audit["all_source_affine_sensor_domains_passed"]
   if ready:
    seedP=D/"station_delta.f64";sep["deltaS"].astype("<f8").tofile(seedP);descriptor=dict(file=seedP.name,dtype="<f8",shape=[self.N],sha256=sha(seedP))
    seed_manifest=dict(schema="rinkfit.station_seed_delta.v1",transport=ref(self.T),checkpoint=self.tr["checkpoint"],station_delta=descriptor,units="original station S",all_active_sources_kept=True,only_verified_smooth_ordinary_released=True,source_error_W_E_unchanged=True,no_nonlinear_feasibility_or_KKT=True)
    (D/"station_seed_manifest.json").write_text(json.dumps(seed_manifest,indent=2));selectionP=D/"selected_source_rows.u64";secondary_selection=np.intersect1d(self.released,self.qbar_verified_source_rows);secondary_selection.astype("<u8").tofile(selectionP)
    (D/"station_selection_manifest.json").write_text(json.dumps(dict(schema="rinkfit.station_column_selection.v1",transport=ref(self.T),checkpoint=self.tr["checkpoint"],source_rows=dict(file=selectionP.name,dtype="<u8",shape=[len(secondary_selection)],sha256=sha(selectionP)),units="original station S"),indent=2))
    final=dict(item,ready_for_conditional_W=True,station_seed=ref(D/"station_seed_manifest.json"),station_selection=ref(D/"station_selection_manifest.json"));break
   # Retain all previous supports and add every violating source without a cap.
   bad=np.flatnonzero((sep["q"]>bLP)&sep["feasible"][self.rows]);newC,newr=self.support(sep,bad)
   if len(bad)==0:
    history[-1]["stopped_reason"]="No budget support remains to add; strict domain/box failure retained";break
   C=sparse.vstack([C,newC],format="csr");cr=np.r_[cr,newr];iteration+=1
  report=dict(schema="rinkfit.finite_visible_primary_OA_prototype.r2",linearization=ref(self.L),transport=ref(self.T),checkpoint=self.tr["checkpoint"],model=self.tr["model"],preserved_original_rows=self.preserved_counts,temporary_fixed_station_visibility_guards_replaced=True,replacement="Original finite-S reachability for every ordinary/entity row; profile S/V/alpha original 3x implicit domains unchanged.",reachability_rows=self.all_reachA.shape[0],LP_kept_reachability_rows=self.reachA.shape[0],box_proven_redundant_reachability_rows=len(self.reach_removed),released_smooth_ordinary_stations=len(self.released),primary_coordinates_with_b=self.n+1,all_active_sources=len(self.rows),original_W_not_changed_and_not_optimized=True,S_TR_scope="Original finite cell and sensor only; no old curvature radius reused. Numerical current-model curvature trust remains required before a nonlinear proposal.",history=history,final=final,ready_for_conditional_W=final is not None,declared_inexact_working_target=True,optimal_b_or_KKT_claim=False,no_nonlinear_physical_Solve_or_line_search=True,elapsed_seconds=time.perf_counter()-start,source=ref(__file__))
  (D/"report.json").write_text(json.dumps(report,indent=2));return report
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
