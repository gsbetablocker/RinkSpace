"""Adopt one already accepted external full-source physical step; never Solve.
V5 binds both original same-state beforeq evaluators exactly; floating-point distance is not identity.
The complete state writer is extracted verbatim from the pinned frozen driver.
This is an explicit algorithm continuation, never an old-code exact resume.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,shutil,sys,time
import numpy as np
N=182017
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb")as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def atomic(p,x):
 q=p.with_name(p.name+".tmp");q.write_text(json.dumps(x,indent=2),encoding="utf-8");os.replace(q,p)
def verify(d,base=None):
 p=Path(d.get("path",d.get("file")))
 if not p.is_absolute():p=Path(base)/p
 p=p.resolve()
 if sha(p)!=d["sha256"]:raise ValueError("Immutable descriptor changed: "+str(p))
 if "dtype"in d and "shape"in d:
  if p.stat().st_size!=int(np.prod(d["shape"]))*np.dtype(d["dtype"]).itemsize:raise ValueError("Array bytes: "+str(p))
 return p
def asset(base,d):
 p=verify(d,base);return np.fromfile(p,d["dtype"]).reshape(d["shape"])
def verify_refs(x,base,skip_keys=()):
 if isinstance(x,dict):
  if "sha256"in x and ("path"in x or "file"in x):verify(x,base)
  for k,v in x.items():
   if k not in skip_keys:verify_refs(v,base,skip_keys)
 elif isinstance(x,list):
  for v in x:verify_refs(v,base,skip_keys)
def verify_frozen(f,base):
 for c in f["code"]:verify(c["frozen"],base)
 for c in f["python_modules"]:verify(c,base)
 for h in f["helpers"].values():
  verify(h["executable"],base);verify(h["recipe"],base)
  for c in h.get("source_snapshot",[]):verify(c,base)
 for c in f.get("runtime_libraries",[]):verify(c,base)
def audit_same_state_beforeq(w,parent,paths,spec,ip):
 assert "before_value_report"in spec,"Explicit frozen same-state value authority required"
 T=verify(w["transport"]);t=read(T);C=verify(t["checkpoint"]);cp=read(C)
 A=verify(spec["before_value_report"],ip.parent);a=read(A);V=verify(a["exact_value_state"]);v=read(V)
 assert a["schema"]=="rinkfit.exact_Jet_refined_complete_inner_seed.v1"
 for k in ["model","packet","semantic"]:
  assert t[k]["sha256"]==w[k]["sha256"]==v[k]["sha256"],"Beforeq authority mismatch: "+k
 assert t["model"]["sha256"]==cp["model"]["sha256"]==parent["model"]["sha256"]==a["files"]["model.json"]
 assert a["linearization"]["sha256"]==w["linearization"]["sha256"]==v["qbar_linearization"]["sha256"]
 for k,d in parent["row_state_files"].items():
  verify(d,paths["parent_checkpoint"].parent);verify(cp["row_state_files"][k],C.parent)
  vk="path"if k=="branch"else k;verify(v["arrays"][vk],V.parent)
  assert d["sha256"]==cp["row_state_files"][k]["sha256"]==v["arrays"][vk]["sha256"]==a["files"][d["file"]],"Beforeq complete auxiliary mismatch: "+k
 for name,digest in a["files"].items():assert Path(name).name==name and sha(A.parent/name)==digest,"Beforeq authority file mismatch"
 P=verify(w["packet"]);pk=read(P);S=verify(w["semantic"]);sm=read(S)
 er=asset(T.parent,t["arrays"]["error_uv"]).reshape(N,2);E=asset(P.parent,pk["arrays"]["engineering_budget_px"])
 active=asset(S.parent,sm["active_source_mask"]).astype(bool);kind=asset(S.parent,sm["arrays"]["kind"]);normal=asset(S.parent,sm["arrays"]["profile_normal"])
 assert np.flatnonzero(~active).tolist()==[25537]and active.sum()==182016
 assert np.array_equal(asset(V.parent,v["arrays"]["active_mask"]).astype(bool),active)
 assert v["all_active_strict_valid"]and np.all(asset(V.parent,v["arrays"]["strict_valid"])[active])
 r=np.zeros((N,2));ordinary=active&(kind!=2);profiles=active&(kind==2);r[ordinary]=er[ordinary]/E[ordinary,None]
 r[profiles,0]=np.einsum("ij,ij->i",normal[profiles],er[profiles])/E[profiles]
 qzero=np.einsum("ij,ij->i",r,r);cppq=asset(V.parent,v["arrays"]["q"]);ids=np.flatnonzero(active)
 worst=int(ids[np.argmax(qzero[active])]);cppworst=int(ids[np.argmax(cppq[active])]);maximum=float(qzero[worst]);cppmax=float(cppq[cppworst]);reported=float(w["original_q_max"]);cpb=float(parent["b"])
 assert maximum.hex()==reported.hex(),"Beforeq report not exact wholeUV zero-delta value"
 assert cppmax.hex()==cpb.hex()==float(parent["metrics"]["max_q"]).hex()==float(a["best"]["max_q"]).hex(),"Parent b not exact frozen native value"
 assert cppworst==int(parent["metrics"]["maximum_source_row"])==int(a["best"]["maximum_source_row"])
 assert float(w["linearized_epigraph"])<min(cpb,reported),"Working b must descend under both original evaluators"
 return dict(schema="rinkfit.same_state_separately_exact_beforeq_evaluators.v2",wholeUV=ref(T),wholeUV_checkpoint=ref(C),native_value_authority=ref(A),native_value_state=ref(V),model=parent["model"],all_four_aux_SHA_equal=True,raw_rows=N,active_rows=182016,wholeUV_worst_source_row=worst,native_worst_source_row=cppworst,zero_delta_max_q=maximum,proposal_original_q_max=reported,parent_checkpoint_b=cpb,native_evaluator_qmax=cppmax,cross_evaluator_difference=reported-cpb,source_identity_uses_exact_model_auxiliary_packet_semantic_not_ULP_distance=True,working_b_strictly_below_both=True,original_gate_or_budget_not_modified=True)

def validate_context(ip,include_line=True):
 spec=read(ip);assert spec["schema"]=="rinkfit.external_accepted_continuous_SLP_adoption_input.v1"
 paths={k:verify(spec[k],ip.parent)for k in ["parent_checkpoint","primary_report","W_report","frozen_line_bundle"]}
 parent=read(paths["parent_checkpoint"]);assert parent["schema"]=="rinkfit.complete_continuous_reduced_SLP_checkpoint.v1"
 body=dict(parent);token=body.pop("state_token");assert identity(body)==token,"Parent state token"
 fp=verify(parent["frozen_run"]);frozen=read(fp);verify_frozen(frozen,fp.parent)
 base_spec=read(verify(frozen["input"]));pr=read(paths["primary_report"]);w=read(paths["W_report"]);bundle=read(paths["frozen_line_bundle"])
 if bundle["parent_frozen_run"]["sha256"]!=parent["frozen_run"]["sha256"]:
  ancestor=parent;visited=set();found=False
  while ancestor:
   if ancestor["frozen_run"]["sha256"]==bundle["parent_frozen_run"]["sha256"]:found=True;break
   ancestor_ref=ancestor.get("parent_checkpoint")
   if not ancestor_ref:break
   ap=verify(ancestor_ref)
   if str(ap)in visited:raise ValueError("Cyclic checkpoint lineage")
   visited.add(str(ap));ancestor=read(ap)
  assert found,"Line bundle is not from this checkpoint lineage"

 verify_refs(bundle,paths["frozen_line_bundle"].parent)
 for item,name in [(pr,"primary_report"),(w,"W_report")]:verify_refs(item,paths[name].parent,("entity_chart",))
 assert pr["schema"]=="rinkfit.fixed_entity_primary_SOC_actual_driver.v1" and pr["ready_for_W_secondary"]
 assert pr["solver"]["strict_original_primal_gate"]
 allsource=w["schema"]=="rinkfit.all_source_affine_norm_generated_W_proposal.v1"
 primary_numeric=bool(pr["solver"]["numeric_conic_solver_gate"])
 W_numeric=bool(w.get("W_secondary_optimality_verified",w.get("entity_SOC",{}).get("numeric_conic_solver_gate",False)))
 if not primary_numeric:
  assert allsource and pr.get("primary_working_target_declared_inexact") and pr.get("primary_optimality_verified")is False
  assert pr.get("ready_for_W_is_primal_working_target_not_solver_status_override") and pr.get("PHR_multiplier_update_permitted")is False
  assert (w.get("primary_working_target_declared_inexact")or w.get("explicit_inexact_epigraph_upper_canonicalization"))and w.get("primary_optimality_verified")is False and w.get("PHR_multiplier_update_permitted")is False
 assert w["schema"]in ["rinkfit.same_primary_full_W_entity_SOC_proposal.v1","rinkfit.same_primary_entity_SOC_W_original_unit_repair.v1","rinkfit.all_source_affine_norm_generated_W_proposal.v1"]
 assert w["proposed"],"W proposal is not finally feasible"
 assert w["primary_report"]["sha256"]==sha(paths["primary_report"])
 rebased=bool(w.get("explicit_inexact_epigraph_upper_canonicalization"))
 budget_audit=None
 if rebased:
  from scipy import sparse
  assert allsource and w["working_target_budget_coordinate_changed"]and w["canonical_b_IEEE_unchanged"]is False and w["canonical_primary_b_IEEE_unchanged"]is False
  assert w["primary_epigraph_unchanged"]is False and w["fixed_primary_b_not_relaxed"]is False and w["engineering_budget_E_changed"]is False and w["final_phaseII_requires_original_b_le_1"]
  sourcep=verify(w["source_proposal_report"]);sourcew=read(sourcep);oldnpz=dict(np.load(verify(sourcew["proposal"])));newnpz=dict(np.load(verify(w["proposal"])))
  assert set(newnpz)-set(oldnpz)<=set(["canonical_primal"])
  for key in oldnpz:
   if key!="canonical_primal":assert oldnpz[key].dtype==newnpz[key].dtype and oldnpz[key].shape==newnpz[key].shape and oldnpz[key].tobytes()==newnpz[key].tobytes(),"Physical/W arrays drifted under b rebase: "+key
  assert float(sourcew["linearized_epigraph"]).hex()==float(w["original_conic_working_target_b"]).hex()==float(pr["solver"]["canonical_primary_b"]).hex()
  AA=sparse.load_npz(verify(pr["original_A"])).tocsr();bounds=dict(np.load(verify(pr["original_rhs_bounds"])));delta=newnpz["step_scaled"];bcol=np.asarray(AA[:,-1].toarray()).ravel();assert not np.any(bcol>0)
  pd=AA[:,:-1]@delta;mask=bcol<0;requiredA=float(np.max((pd[mask]-bounds["rhs"][mask])/(-bcol[mask]),initial=-np.inf))
  required=max(float(sourcew["linearized_epigraph"]),float(w["all_source_affine_audit"]["maximum_affine_q"]),requiredA);expected=float(np.nextafter(required,np.inf));assert expected.hex()==float(w["linearized_epigraph"]).hex(),"Working b is not the prescribed nextafter upper"
  native=np.r_[delta,expected];assert newnpz["canonical_primal"].tobytes()==native.tobytes()
  violation=AA@native-bounds["rhs"];allow=64*np.finfo(float).eps*(1+abs(bounds["rhs"])+abs(AA)@abs(native))
  assert np.all(violation<=allow)and np.all(native>=bounds["lower"])and np.all(native<=bounds["upper"])and expected<parent["b"]
  budget_audit=dict(source_proposal_report=ref(sourcep),original_working_b=float(sourcew["linearized_epigraph"]),full_source_affine_qmax=float(w["all_source_affine_audit"]["maximum_affine_q"]),original_A_required_b=requiredA,exact_nextafter_working_b=expected,physical_and_W_arrays_IEEE_unchanged=True,original_A_beyond_IEEE=int(np.count_nonzero(violation>allow)),strict_original_box=True,original_A=pr["original_A"],original_rhs_bounds=pr["original_rhs_bounds"])
 else:assert float(w["linearized_epigraph"]).hex()==float(pr["solver"]["canonical_primary_b"]).hex(),"Primary b changed without explicit rebase"

 beforeq_audit=None
 if allsource:beforeq_audit=audit_same_state_beforeq(w,parent,paths,spec,ip)
 else:assert float(w["original_q_max"]).hex()==float(parent["b"]).hex(),"Beforeq mismatch outside audited all-source evaluator scope"

 for k in ["packet","semantic"]:assert w[k]["sha256"]==frozen["immutable_target"][k]["sha256"]
 assert w["model"]["sha256"]==parent["model"]["sha256"]
 assert w.get("raw_rows",w.get("source_norm_chart",{}).get("raw_rows"))==N and w.get("active_rows",w.get("source_norm_chart",{}).get("active_rows"))==182016 and w.get("rank",w.get("source_norm_chart",{}).get("original_W_rank"))==214112
 scope=dict(same_state_beforeq_arithmetic_bridge=beforeq_audit,explicit_working_budget_rebase=budget_audit,all_source_norm_scope=allsource,primary_numerical_optimality_verified=primary_numeric,W_numerical_optimality_verified=W_numeric,primary_target_role="verified_affine_primary"if primary_numeric else"declared_inexact_feasible_working_upper_bound",old_multipliers_are_not_updated=True)
 if allsource:
  audit=w["all_source_affine_audit"]
  assert audit["all_source_affine_norms_within_b"]and audit["violating_source_count"]==0 and audit["violating_source_rows"]==[]
  assert audit["active_rows"]==182016 and audit["ordinary_count"]==179363 and audit["entity_count"]==1692 and audit["profile_count"]==961
  assert audit["ordinary_fixed_S_is_temporary_numerical_witness"]and audit["profile_original_normal_1D"]and audit["nonlinear_physical_acceptance"]is False
  assert w.get("all_physical_entities_kept",pr["all_physical_entities"])==1692 and w.get("full_source_norm_generation",pr["full_source_norm_generation"])and w["PHR_multiplier_update_permitted"]is False
  chart=w["source_norm_chart"];assert chart["active_rows"]==182016 and chart["ordinary_entity_native_2D_count"]==181055 and chart["genuine_profile_normal_1D_count"]==961
  assert chart["ordinary_station_fixed_only_for_this_numerical_proposal"]and chart["unselected_source_not_deleted"]and chart["final_PhaseII_joint_station_claim"]is False
  assert chart["model"]["sha256"]==parent["model"]["sha256"]and chart["transport"]["sha256"]==w["transport"]["sha256"]
  assert float(w["working_target_b"]).hex()==float(w["linearized_epigraph"]).hex() and w["working_target_b"]<parent["b"]
  gate=w["original_unit_audit"];assert gate["passed"]and gate["box_strictly_satisfied"]and gate["original_master_beyond_IEEE_count"]==0
  coverage=w.get("coverage",pr["coverage"])
  assert coverage["unknown_sources_covered_by_defined_fixed_saved_S_true_RS_UV_norm"]and coverage["all_source_affine_norms_required_before_proposal"]
  verify(pr["profile_domains"]);verify(pr["original_A"]);verify(pr["original_rhs_bounds"])

  if not W_numeric:
   assert w["W_secondary_declared_inexact"]and w["W_secondary_optimality_verified"]is False and w["numeric_solver_status_not_overridden"]
   wg=w["entity_SOC"];assert wg["numeric_conic_solver_gate"]is False
   assert w["original_unit_audit"]["passed"]and w["original_unit_audit"]["box_strictly_satisfied"]and w["original_unit_audit"]["original_master_beyond_IEEE_count"]==0
   if w.get("PhaseI_budget_priority_secondary_policy")is True:
    assert not primary_numeric and w["primary_optimality_verified"]is False and w["PHR_multiplier_update_permitted"]is False
    assert w["working_target_b"]<parent["b"] and w["predicted_W_half_cost"]<wg["primary_W_half_cost"]
    scope.update(PhaseI_budget_priority_secondary_policy=True,secondary_W_strictly_below_same_working_b_primary=True,predicted_W_below_parent=bool(w["predicted_W_half_cost"]<w["baseline_W_half_cost"]),statistical_improvement_required_before_PhaseI_proposal=False,primary_or_secondary_optimality_claimed=False)
   else:
    assert w["predicted_W_half_cost"]<min(w["baseline_W_half_cost"],wg["primary_W_half_cost"])
  scope.update(all_active_rows=182016,ordinary_entity_native_2D=181055,profile_original_normal_1D=961,all_source_affine_audit=audit,coverage=coverage,original_profile_domains=pr["profile_domains"],original_A=pr["original_A"],original_rhs_bounds=pr["original_rhs_bounds"],working_target_b=w["working_target_b"],working_target_is_not_optimal_bstar=not primary_numeric,full_W_secondary_is_not_claimed_optimal=not W_numeric,PHR_multiplier_update_permitted=False)
 elif w["schema"]=="rinkfit.same_primary_full_W_entity_SOC_proposal.v1":
  gate=w["entity_SOC"];assert gate["strict_original_primal_gate"]and gate["numeric_conic_solver_gate"]and gate["canonical_primary_b_IEEE_unchanged"]
  assert gate["entity_norm_audit"]["entities"]==1692 and gate["entity_norm_audit"]["all_entity_affine_norms_within_b"]
 else:
  gate=w["original_unit_audit"];assert w["canonical_b_IEEE_unchanged"]and gate.get("accepted",gate.get("passed",False)) and gate["box_strictly_satisfied"]and gate["original_master_beyond_IEEE_count"]==0
  assert w["entity_norm_audit"]["entities"]==1692 and w["entity_norm_audit"]["all_entity_affine_norms_within_b"]
 transport=read(verify(w["transport"]));t_cp=read(verify(transport["checkpoint"]))
 assert t_cp["model"]["sha256"]==parent["model"]["sha256"]
 for k,d in parent["row_state_files"].items():
  verify(d,paths["parent_checkpoint"].parent);assert d["sha256"]==t_cp["row_state_files"][k]["sha256"],"Transport differs from complete parent: "+k
 parentmodel=read(verify(parent["model"]));proposal=verify(w["proposal"]);direction=dict(np.load(proposal))
 parentvalues=np.concatenate([b["values"]for b in parentmodel["blocks"]]);scales=np.concatenate([b["scales"]for b in parentmodel["blocks"]])
 assert np.array_equal(direction["parent_values"],parentvalues)and np.array_equal(direction["scales"],scales)
 assert direction["physical_delta"].shape==parentvalues.shape and np.isfinite(direction["physical_delta"]).all()
 assert float(w.get("trust_scaled_infinity",pr["solver"]["physical_scaled_trust"])).hex()==float(parent["control"]["trust"]).hex()
 wfinal=None
 if spec.get("W_final_audit_report"):
  finalpath=verify(spec["W_final_audit_report"],ip.parent);paths["W_final_audit_report"]=finalpath;wfinal=read(finalpath);verify_refs(wfinal,finalpath.parent,("entity_chart",))
  assert wfinal["proposed"]and wfinal["postrepair_all_original_unit_checks_passed"]and wfinal["postrepair_W_coefficients_same_delta"]
  assert wfinal["proposal"]["sha256"]==w["proposal"]["sha256"]and float(wfinal["linearized_epigraph"]).hex()==float(w["linearized_epigraph"]).hex()
  for key in ["model","packet","semantic","primary_report","transport","linearization"]:assert wfinal[key]["sha256"]==w[key]["sha256"],"Final W audit belongs to different "+key
  assert wfinal["repair_report"]["sha256"]==sha(paths["W_report"])
 if not include_line:return dict(spec=spec,parent=parent,primary=pr,W=w,W_final=wfinal,norm_scope=scope,paths=paths,frozen=frozen,frozen_path=fp,base_spec=base_spec,bundle=bundle)
 lp=verify(spec["actual_line_report"],ip.parent);lr=read(lp);paths["actual_line_report"]=lp
 assert lr["schema"]=="rinkfit.reduced_physical_actual_line_search.v1"
 assert lr["proposal"]["sha256"]==sha(paths["W_report"])and lr["semantic"]["sha256"]==w["semantic"]["sha256"]
 assert lr["raw_rows"]==N and lr["active_rows"]==182016 and lr["source_W_E_unchanged"]and lr["physical_model_DOF_unchanged"]
 baseline=verify(lr["baseline"]);baseseed=baseline.parent
 for k,name in [("station","station.f64"),("branch","path.u32"),("row","initial_row.f64"),("alpha","initial_alpha.f64")]:
  assert sha(baseseed/name)==parent["row_state_files"][k]["sha256"],"Actual line used different parent "+k
 assert sha(baseseed/"model.json")==parent["model"]["sha256"]
 assert float(lr["baseline"]["max_q"]).hex()==float(parent["b"]).hex()
 accepted=lr.get("accepted_candidate");assert accepted and accepted["strict_source_scene_query_numerical_gates_passed"],"No accepted original full-source candidate"
 assert 0<float(accepted["fraction"])<=1 and accepted["whole_static_scene_passed"]
 seedreport=verify(accepted["report"]);sr=read(seedreport);seed=seedreport.parent
 assert sr["raw_rows"]==N and sr["active_rows"]==182016 and sr["statistics_rank"]==214112 and sr["partition_exactly_once"]
 assert sr["best"]==accepted["best"]and sr["whole_static_scene"]["passed"]
 metrics=sr["best"];assert metrics["valid"]and metrics["active_RS_valid_rows"]==182016 and metrics["query_numerical_roots"]==961 and metrics["max_query_equality_px"]<=1e-6
 assert metrics["max_q"]<parent["b"]and np.isfinite(metrics["whole_original_W_half_cost"])
 for name,digest in sr["files"].items():
  assert Path(name).name==name and sha(seed/name)==digest,"Accepted state file mismatch: "+name
 for name in ["model.json","station.f64","path.u32","initial_row.f64","initial_alpha.f64","candidate_predictions.npz"]:assert name in sr["files"]
 newmodel=read(seed/"model.json");assert len(newmodel["blocks"])==1376
 def template(m):
  z=copy.deepcopy(m)
  for b in z["blocks"]:b.pop("values",None)
  for k in ["reduced_initializer_provenance","reduced_physical_proposal","ad6_finite_range_candidate","finite_line_SOC_provenance"]:z.pop(k,None)
  return identity(z)
 assert template(parentmodel)==template(newmodel),"Physical hypothesis/finite paths/scales changed"
 newvalues=np.concatenate([b["values"]for b in newmodel["blocks"]]);expected=parentvalues+float(accepted["fraction"])*direction["physical_delta"]
 unplanned=np.flatnonzero(newvalues!=expected)
 if len(unplanned):assert accepted.get("is_independent_SOC_candidate")and newmodel.get("finite_line_SOC_provenance"),"Unexpected values outside the line without explicit SOC lineage"
 for old,new in zip(parentmodel["blocks"],newmodel["blocks"]):
  z=np.asarray(new["values"]);assert np.isfinite(z).all()and np.all(z>=new["lower"])and np.all(z<=new["upper"])
  if old.get("constant"):assert np.array_equal(old["values"],new["values"])
  for j in old.get("fixed_indices",[]):assert old["values"][j]==new["values"][j]
 raw={}
 for name,dt,lo,hi in [("station.f64","<f8",0.,1.),("path.u32","<u4",0.,np.inf),("initial_row.f64","<f8",0.,1079.),("initial_alpha.f64","<f8",-np.inf,np.inf)]:
  z=np.fromfile(seed/name,dt);assert z.shape==(N,)and np.isfinite(z).all()and np.all(z>=lo)and np.all(z<=hi);raw[name]=z
 pred=dict(np.load(seed/"candidate_predictions.npz"))
 sempath=verify(w["semantic"]);sem=read(sempath);active=asset(sempath.parent,sem["active_source_mask"]).astype(bool)
 pkpath=verify(w["packet"]);pk=read(pkpath)
 assert np.array_equal(np.flatnonzero(~active),[25537]) and np.array_equal(pred["active_source_mask"],active)
 assert np.array_equal(pred["source_rows"],np.arange(N))and np.array_equal(pred["native_pts"],asset(pkpath.parent,pk["arrays"]["native_pts"]))
 assert pred["RS_valid"].shape==(N,)and pred["RS_valid"][active].all()
 for key in ["uv","error_native_uv","coverage_error_px"]:
  assert pred[key].shape[0]==N and np.isfinite(pred[key][active]).all()

 for name,k in [("station.f64","stations"),("path.u32","branches"),("initial_row.f64","root_rows"),("initial_alpha.f64","profile_alpha")]:
  assert pred[k].dtype==raw[name].dtype and pred[k].shape==(N,)and pred[k].tobytes()==raw[name].tobytes(),"Prediction and raw state mismatch: "+name
 paths["accepted_native_report"]=seedreport
 return dict(spec=spec,parent=parent,primary=pr,W=w,W_final=wfinal,norm_scope=scope,paths=paths,frozen=frozen,frozen_path=fp,base_spec=base_spec,bundle=bundle,line=lr,accepted=accepted,seed=seed,state=sr,extra_SOC_ambient_indices=unplanned.tolist())
def adopt(ip,out):
 started=time.time();ctx=validate_context(ip);source=Path(__file__).resolve()
 assert not out.exists(),"Refuse overwriting an adoption run";out.mkdir(parents=True);(out/"code").mkdir()
 paths=ctx["paths"];parent=ctx["parent"];frozen=copy.deepcopy(ctx["frozen"]);w=ctx["W"];pr=ctx["primary"];lr=ctx["line"];accepted=ctx["accepted"];seed=ctx["seed"]
 shutil.copyfile(source,out/"code"/source.name)
 driver=next(verify(c["frozen"])for c in reversed(frozen["code"])if Path(c["frozen"]["path"]).name=="reduced_slp_driver.py")
 shutil.copyfile(driver,out/"code/reused_reduced_slp_driver.py");pinned_driver=out/"code/reused_reduced_slp_driver.py"
 resource=[];seen=set()
 def account(p,field,label):
  p=Path(p);digest=sha(p);workkey=(digest,field)
  if workkey in seen:return
  seen.add(workkey);doc=read(p);seconds=float(doc[field]);assert np.isfinite(seconds)and seconds>=0
  resource.append(dict(report=ref(p),seconds_field=field,seconds=seconds,role=label))
 if ctx["norm_scope"]["all_source_norm_scope"]:
  account(paths["W_report"],"elapsed_seconds","complete_all_source_norm_generation_primary_W_and_repair_outer_wall")
 else:
  account(paths["primary_report"],"total_seconds","entity_SOC_primary")
  if w.get("QP_report"):account(verify(w["QP_report"]),"elapsed_seconds","raw_entity_SOC_W_secondary")
  account(paths["W_report"],"elapsed_seconds","final_entity_W_proposal_or_repair")
  if ctx["W_final"] and "postrepair_finalization_seconds"in ctx["W_final"]:account(paths["W_final_audit_report"],"postrepair_finalization_seconds","same_delta_W_coefficients_and_final_audit")
 account(paths["actual_line_report"],"elapsed_seconds","complete_actual_full_source_line")
 for item in ctx["spec"].get("additional_resource_reports",[]):account(verify(item["report"],ip.parent),item["seconds_field"],item["role"])
 base_cumulative=float(parent["cumulative_elapsed_seconds"]);base_compute=float(parent["control"]["compute_elapsed_seconds"]);boundary=Path(parent["frozen_run"]["path"]).parent/"result.json";boundary_ref=None
 if boundary.exists():
  z=read(boundary)
  if z.get("current_complete_checkpoint",{}).get("sha256")==sha(paths["parent_checkpoint"]):
   assert z["frozen_run"]["sha256"]==parent["frozen_run"]["sha256"]
   base_cumulative=max(base_cumulative,float(z["cumulative_elapsed_seconds"]));base_compute=max(base_compute,float(z["control"]["compute_elapsed_seconds"]));boundary_ref=ref(boundary)
 extra_work=sum(x["seconds"]for x in resource);control=copy.deepcopy(parent["control"]);control["compute_elapsed_seconds"]=base_compute+extra_work
 prediction=float(w["original_q_max"])-float(w["linearized_epigraph"]);fraction=float(accepted["fraction"]);actual=float(parent["b"])-float(accepted["best"]["max_q"]);assert prediction>0 and actual>0
 ratio=actual/(prediction*fraction);iteration=int(control["iteration"])+1
 W_context=dict(declared_numerical_scope=ctx["norm_scope"],transport=w["transport"],primary_report=ref(paths["primary_report"]),final_W_proposal=ref(paths["W_report"]),proposal_step=w["proposal"],frozen_line_bundle=ref(paths["frozen_line_bundle"]),all_1692_entity_norms_reaudited=True,conditional_fixed_ordinary_S=True,PhaseII_joint_station_optimum=False)
 wp=ctx["W_final"] or w
 if ctx["W_final"]:W_context["same_delta_final_audit"]=ref(paths["W_final_audit_report"])
 wpred=None
 if all(k in wp for k in ["baseline_W_half_cost","predicted_W_linear_delta","predicted_W_quadratic_delta"]):wpred=wp["baseline_W_half_cost"]+fraction*wp["predicted_W_linear_delta"]+fraction*fraction*wp["predicted_W_quadratic_delta"]
 elif fraction==1. and "predicted_W_half_cost"in wp:wpred=wp["predicted_W_half_cost"]
 control["iteration"]=iteration;control["history"].append(dict(iteration=iteration,linearization=w["linearization"],LP=ref(paths["primary_report"]),line_search=ref(paths["actual_line_report"]),actual_reduction=actual,predicted_reduction=prediction*fraction,ratio=ratio,metrics=accepted["best"],accepted_fraction=fraction,trust_used=control["trust"],conditional_whole_W_selection=W_context,whole_W_baseline=parent["metrics"]["whole_original_W_half_cost"],whole_W_actual=accepted["best"]["whole_original_W_half_cost"],whole_W_conditional_quadratic_prediction=wpred,external_entity_SOC_primary_replaces_scalar_LP=True))
 if ratio>.75 and fraction==1.:control["trust"]*=2.
 elif ratio<.25:control["trust"]*=.5
 receipts=dict(schema="rinkfit.external_accepted_operator_receipts.v1",input=ref(ip),parent_checkpoint=ref(paths["parent_checkpoint"]),parent_frozen_run=parent["frozen_run"],parent_execution_boundary=boundary_ref,primary_report=ref(paths["primary_report"]),W_report=ref(paths["W_report"]),actual_line_report=ref(paths["actual_line_report"]),frozen_line_bundle=ref(paths["frozen_line_bundle"]),source=ref(out/"code"/source.name),reused_state_writer=ref(pinned_driver),resource_ledger=resource,external_completed_work_seconds=extra_work,parent_resource_cumulative_seconds=base_cumulative,parent_resource_compute_seconds=base_compute,all_raw_state_copied_without_recovery=True,no_Solve=True)
 if ctx["W_final"]:receipts["same_delta_W_final_audit"]=ref(paths["W_final_audit_report"])
 receipts["declared_numerical_scope"]=ctx["norm_scope"]
 atomic(out/"external_operator_receipts.json",receipts)
 frozen["parent_frozen_run"]=parent["frozen_run"];frozen["external_accepted_operator"]=ref(out/"external_operator_receipts.json")
 frozen["canonical_old_driver_resume_supported"]=False;frozen["next_execution_requires_algorithm_continue_from"]=True
 frozen["code"].append(dict(original=ref(source),frozen=ref(out/"code"/source.name),role="external_accepted_checkpoint_adopter"))
 frozen["target_id"]=identity(dict(parent_target_id=parent["target_id"],external_operator=frozen["external_accepted_operator"],immutable_target=frozen["immutable_target"]))
 atomic(out/"frozen_run.json",frozen)
 namespace=dict(Path=Path,json=json,np=np,copy=copy,time=time,shutil=shutil,os=os,sys=sys,hashlib=hashlib,N=N,D=out,sha=sha,ref=ref,identity=identity,atomic=atomic,verify=verify,spec=ctx["base_spec"],control=control,target_id=frozen["target_id"],frozen_path=out/"frozen_run.json",parent_cp=ref(paths["parent_checkpoint"]),current_cp=None,started=started,elapsed_base=base_cumulative+extra_work)
 namespace["log"]=lambda kind,**kw:print(json.dumps(dict(kind=kind,**kw)),flush=True)
 chosen={"asset_array","same_linearization_state","current_chart_state","copy_chart_state","save_state"}
 tree=ast.parse(pinned_driver.read_text(encoding="utf-8"));functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in chosen];assert {x.name for x in functions}==chosen
 sys.path.insert(0,frozen["python_frozen_dir"])
 exec(compile(ast.Module(body=functions,type_ignores=[]),str(pinned_driver),"exec"),namespace)
 doc=namespace["save_state"](seed,"external_entity_SOC_full_source_physical_step_fresh_derivatives_pending",None,None,None)
 cp=out/"checkpoints"/next((out/"checkpoints").iterdir()).name
 for name in ["report.json",*ctx["state"]["files"]]:shutil.copyfile(seed/name,cp/name)
 for name,digest in ctx["state"]["files"].items():assert sha(cp/name)==digest

 doc.update(algorithm_id="finite_RS_query_reduced_PhaseI_entity_SOC_primary_same_b_full_W_secondary_external_accepted",external_operator_receipts=ref(out/"external_operator_receipts.json"),external_accepted_native_report=ref(cp/"report.json"),external_accepted_predictions=ref(cp/"candidate_predictions.npz"),external_line_search=ref(paths["actual_line_report"]),external_primary_conic_report=ref(paths["primary_report"]),external_W_conic_or_repair_report=ref(paths["W_report"]),canonical_old_driver_resume_supported=False,next_execution_requires_algorithm_continue_from=True,external_SOC_ambient_value_deviations=ctx["extra_SOC_ambient_indices"],conic_duals_are_only_historical_linearized_program_evidence=True,source_W_E_query_and_physical_parameterization_unchanged=True,resource_ledger=resource,external_completed_work_seconds=extra_work,adoption_elapsed_seconds=time.time()-started)
 if ctx["norm_scope"]["all_source_norm_scope"]:
  doc["algorithm_id"]="finite_RS_query_reduced_PhaseI_all_source_affine_norm_working_b_same_b_full_W_secondary_external_accepted"
  if ctx["norm_scope"].get("PhaseI_budget_priority_secondary_policy"):
   doc["algorithm_id"]="finite_RS_query_reduced_PhaseI_all_source_affine_norm_inexact_working_b_budget_priority_original_W_secondary_external_accepted"
  doc["declared_numerical_scope"]=ctx["norm_scope"]
  doc["primary_optimality_or_lexicographic_optimality_claimed"]=False
  doc["PHR_multiplier_update_permitted"]=False
  doc["elapsed_accounting_complete_for_all_external_branches"]=False
  doc["elapsed_accounting"]="Known parent boundary plus explicit outer report and actual line wall; nested solver/generation times and referenced earlier external work are retained without summing them or asserting complete cumulative accounting."

 if ctx["W_final"]:doc["same_delta_W_final_audit"]=ref(paths["W_final_audit_report"])
 init=dict(schema="rinkfit.continuous_SLP_complete_seed.v1",model=doc["model"],semantic_manifest=w["semantic"],files={name:ref(cp/name)for name in ["station.f64","path.u32","initial_row.f64","initial_alpha.f64"]})
 atomic(cp/"initializer_manifest.json",init);doc["complete_seed_initializer"]=ref(cp/"initializer_manifest.json")
 doc.pop("state_token",None);doc["state_token"]=identity(doc);atomic(cp/"manifest.json",doc);atomic(out/"current.json",ref(cp/"manifest.json"))
 atomic(out/"result.json",dict(schema="rinkfit.continuous_reduced_SLP_result.v1",status="EXTERNAL_ACCEPTED_COMPLETE_STATE_PRESERVED_NO_SOLVE",current_complete_checkpoint=ref(cp/"manifest.json"),frozen_run=ref(out/"frozen_run.json"),control=control,cumulative_elapsed_seconds=doc["cumulative_elapsed_seconds"],raw_rows=N,active_rows=182016,frames=1248,no_obligations_removed=True,no_Solve=True,whole_shot_final_fit_accepted=False,unfinished_requires_parent_continuation_or_diagnosis=True))
 print(json.dumps(dict(checkpoint=ref(cp/"manifest.json"),state_token=doc["state_token"],iteration=iteration,trust=control["trust"],metrics=doc["metrics"],no_Solve=True),indent=2))
 return doc
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output");p.add_argument("--check-context-only",action="store_true");a=p.parse_args();ip=Path(a.input).resolve()
 if a.check_context_only:validate_context(ip,include_line=False);print("EXTERNAL_CONTEXT_VALID_NO_SOLVE");return
 if not a.output:p.error("--output is required for adoption")
 adopt(ip,Path(a.output).resolve())
if __name__=="__main__":main()
