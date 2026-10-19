"""Adopt one already accepted external full-source physical step; never Solve.
V8 explicitly admits exact squared-budget or radius primary coordinates and current-curvature finite original-S + original correlated-W candidates. No older proposal schema is reinterpreted; both beforeq evaluators remain separately exact.
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
 spec=read(ip);assert spec["schema"]=="rinkfit.external_curvature_finite_S_original_W_adoption_input.v2"
 keys=["parent_checkpoint","primary_report","W_report","frozen_line_bundle","before_value_report","group_W_chart"]
 paths={k:verify(spec[k],ip.parent)for k in keys}
 parent=read(paths["parent_checkpoint"]);assert parent["schema"]=="rinkfit.complete_continuous_reduced_SLP_checkpoint.v1"
 body=dict(parent);token=body.pop("state_token");assert identity(body)==token
 fp=verify(parent["frozen_run"]);frozen=read(fp);verify_frozen(frozen,fp.parent)
 base_spec=read(verify(frozen["input"]));pr=read(paths["primary_report"])
 assert pr["schema"]in ["rinkfit.same_i17_feasible_physical_direction_fresh_curvature_recovery.v1","rinkfit.finite_visible_norm_radius_primary_working_target.v1"]
 assert pr["ready_for_conditional_W"]and pr["PHR_multiplier_update_permitted"]is False
 rawW=read(paths["W_report"]);bundle=read(paths["frozen_line_bundle"])
 assert rawW["schema"]=="rinkfit.finite_visible_coupled_original_W_working_proposal.v1"and rawW["proposed"]
 assert bundle["schema"]in ["rinkfit.finite_visible_correlated_W_conditional_operator_bundle.v2","rinkfit.finite_visible_correlated_W_conditional_operator_bundle.v3"]
 verify_refs(bundle,paths["frozen_line_bundle"].parent);verify_refs(rawW,paths["W_report"].parent,("radius_array",))
 assert bundle["requires_identical_primary_curvature_TR"]and bundle["original_W_E_and_production14_unchanged"]
 assert rawW["primary_report"]["sha256"]==sha(paths["primary_report"])
 assert rawW["checkpoint"]["sha256"]==sha(paths["parent_checkpoint"])
 assert rawW["primary_working_target_declared_inexact"]and rawW["W_secondary_declared_inexact"]and rawW["numeric_solver_status_not_overridden"]
 assert rawW["not_PhaseII_station_optimality_or_true_KKT"]and rawW["PHR_multiplier_update_permitted"]is False and rawW["final_fit_accepted"]is False
 for k in ["packet","semantic"]:assert rawW[k]["sha256"]==frozen["immutable_target"][k]["sha256"]
 assert rawW["model"]["sha256"]==parent["model"]["sha256"]
 verify_refs(pr,paths["primary_report"].parent)
 C=dict(original_A=rawW["original_A"],rhs_bounds=rawW["original_rhs_bounds"],working_b=pr["canonical_b"])
 G=read(paths["group_W_chart"]);verify_refs(G,paths["group_W_chart"].parent,("radius_array",))
 assert G["authority"]==rawW["authority"]and G["raw_rows"]==N and G["active_rows"]==182016 and G["original_W_rank"]==214112 and G["groups_including_independent"]==97140
 assert G["original_W_E_Q_physics_and_source_unchanged"]and G["profile_S_V_alpha_original_implicit_operator_not_released"]and G["entity_S_fixed"]
 from scipy import sparse
 sys.path.insert(0,str(verify(bundle["norm_source"]).parent))
 from source_affine_norm_constraints import load_all_source_affine_norms
 transport=read(verify(rawW["transport"]));t_cp=read(verify(transport["checkpoint"]))
 assert t_cp["model"]["sha256"]==parent["model"]["sha256"]
 for k,d in parent["row_state_files"].items():verify(d,paths["parent_checkpoint"].parent);assert d["sha256"]==t_cp["row_state_files"][k]["sha256"]==rawW["authority"]["complete_auxiliary_sha256"][k]
 parentmodel=read(verify(parent["model"]));direction=dict(np.load(verify(rawW["proposal"])))
 parentvalues=np.concatenate([b["values"]for b in parentmodel["blocks"]]);scales=np.concatenate([b["scales"]for b in parentmodel["blocks"]]);offset=np.r_[0,np.cumsum([len(b["values"])for b in parentmodel["blocks"]])]
 assert direction["parent_values"].tobytes()==parentvalues.tobytes()and direction["scales"].tobytes()==scales.tobytes()
 free=direction["free_ambient_indices"].astype(int);d=direction["step_scaled"];eta=direction["station_eta"];selected=direction["station_source_rows"].astype(int)
 ar=lambda k:asset(paths["group_W_chart"].parent,G["arrays"][k])
 assert np.array_equal(ar("physical_free_ambient_indices"),free)and np.array_equal(ar("station_source_rows"),selected)
 expected=np.zeros(len(parentvalues));expected[free]=scales[free]*d
 assert expected.tobytes()==direction["physical_delta"].tobytes()and np.isfinite(expected).all()
 seed_delta=ar("station_seed_delta_full");total=seed_delta.copy();total[selected]+=eta
 assert total.shape==(N,)and total.tobytes()==direction["station_delta_full"].tobytes()
 sempath=verify(rawW["semantic"]);sem=read(sempath);active=asset(sempath.parent,sem["active_source_mask"]).astype(bool);kind=asset(sempath.parent,sem["arrays"]["kind"])
 assert np.flatnonzero(~active).tolist()==[25537]and np.all(total[(~active)|(kind!=0)]==0)
 tp=verify(rawW["transport"]);tan=asset(tp.parent,transport["arrays"]["station_UV_right"]).reshape(N,2)
 changed=total!=0;flo=asset(tp.parent,transport["arrays"]["station_delta_lower_right"]);fhi=asset(tp.parent,transport["arrays"]["station_delta_upper_right"])
 assert np.all(total[changed]>=flo[changed])and np.all(total[changed]<=fhi[changed])
 trust=float(parent["control"]["trust"]);x=np.r_[d,eta];lower=np.r_[np.maximum(ar("physical_lower"),-trust),ar("station_lower")];upper=np.r_[np.minimum(ar("physical_upper"),trust),ar("station_upper")]
 assert np.isfinite(x).all()and np.all(x>=lower)and np.all(x<=upper)
 V=sparse.hstack([sparse.load_npz(verify(G["arrays"]["sensor_Jphysical"])),sparse.load_npz(verify(G["arrays"]["sensor_Jstation"]))],format="csr")
 sensor=ar("sensor_row0")+V@x;assert np.all(sensor>=0)and np.all(sensor<=1079)
 full=load_all_source_affine_norms(verify(rawW["linearization"]),tp,offset,scales,free)
 pkp=verify(rawW["packet"]);pk=read(pkp);E=asset(pkp.parent,pk["arrays"]["engineering_budget_px"])
 rr=full.residual+(full.J@d).reshape(-1,2)+(tan*total[:,None])[full.rows]/E[full.rows,None]
 q=np.einsum("ij,ij->i",rr,rr);qzero=float(full.values(np.zeros(len(free)))[0].max())
 b=float(rawW["linearized_epigraph"]);assert np.all(q<=b)and b==rawW["working_target_b"]and b<min(qzero,parent["b"])
 AA=sparse.load_npz(verify(C["original_A"])).tocsr();bounds=dict(np.load(verify(C["rhs_bounds"])));coordinate=rawW["primary_epigraph_coordinate"];epigraph=float(rawW["exact_primary_epigraph_value"])
 assert coordinate in ["squared_budget_b","radius_sqrt_b"]and pr["primary_epigraph_coordinate"]==coordinate,"Primary epigraph coordinate mismatch"
 assert np.isfinite(epigraph)and epigraph>=0
 assert b.hex()==float(pr["canonical_b"]).hex()
 if coordinate=="radius_sqrt_b":
  assert epigraph.hex()==float(pr["canonical_radius"]).hex()and b.hex()==float(epigraph*epigraph).hex()
 else:assert epigraph.hex()==b.hex()
 canonical_budget=np.r_[d,b];native=np.r_[d,epigraph]
 assert direction["canonical_primal"].tobytes()==canonical_budget.tobytes()
 viol=AA@native-bounds["rhs"];allow=64*np.finfo(float).eps*(1+abs(bounds["rhs"])+abs(AA)@abs(native))
 assert np.all(viol<=allow)and np.all(native>=bounds["lower"])and np.all(native<=bounds["upper"])
 bc=np.asarray(AA[:,-1].toarray()).ravel();assert not np.any(bc>0);mask=bc<0;req=float(np.max(((AA[:,:-1]@d)[mask]-bounds["rhs"][mask])/(-bc[mask]),initial=0))
 assert b.hex()==float(C["working_b"]).hex()
 CR=verify(rawW["station_TR_application"]["curvature_manifest"]);curv=read(CR)
 assert curv["schema"]in ["rinkfit.original_station_numerical_TR_native_RS_curvature.v3","rinkfit.adaptive_original_station_numerical_RS_radius.v1"]
 for key,digest in parent["row_state_files"].items():assert curv["complete_auxiliary_sha256"][key]==digest["sha256"]
 if curv["schema"]=="rinkfit.adaptive_original_station_numerical_RS_radius.v1":
  before_curv=read(verify(curv["input_curvature"]));assert before_curv["checkpoint"]["sha256"]==curv["checkpoint"]["sha256"]and before_curv["complete_auxiliary_sha256"]==curv["complete_auxiliary_sha256"]
 assert curv["checkpoint"]["sha256"]==sha(paths["parent_checkpoint"])and curv["transport"]["sha256"]==rawW["transport"]["sha256"]
 assert bundle["current_curvature_reference"]["sha256"]==sha(CR)and G["authority"]["station_TR"]["manifest"]["sha256"]==sha(CR)
 radius=asset(CR.parent,curv["arrays"]["radius"]);assert radius.shape==(N,)and np.all(abs(total)<=radius)
 assert rawW["station_TR_application"]["radius_array"]["sha256"]==curv["arrays"]["radius"]["sha256"]
 JW=sparse.hstack([sparse.load_npz(verify(G["arrays"]["Jphysical"])),sparse.load_npz(verify(G["arrays"]["Jstation"]))],format="csr")
 r0=ar("original_rW_at_zero_seed");rseed=ar("rW");dw=(rseed-r0)+JW@x
 coeff=[float(.5*r0@r0),float(r0@dw),float(.5*dw@dw)]
 expected_coeff=[rawW["predicted_W_baseline"],rawW["predicted_W_linear_coefficient"],rawW["predicted_W_quadratic_coefficient"]]
 assert np.allclose(coeff,expected_coeff,rtol=0,atol=2e-8)
 assert sum(coeff)<float(read(verify(rawW["original_secondary_report"]))["final"]["primary_W_half_cost"])
 w=dict(rawW,original_q_max=qzero,baseline_W_half_cost=coeff[0],predicted_W_linear_delta=coeff[1],predicted_W_quadratic_delta=coeff[2])
 determinant_guard=None
 if bundle["schema"]=="rinkfit.finite_visible_correlated_W_conditional_operator_bundle.v3":
  import importlib.util
  guard_source=verify(bundle["print_guard_source"]);module_name="rinkfit_print_guard_"+sha(guard_source)
  module_spec=importlib.util.spec_from_file_location(module_name,guard_source);module=importlib.util.module_from_spec(module_spec);sys.modules[module_name]=module;module_spec.loader.exec_module(module)
  guard=module.from_full_norm_operator(full,free,scales);determinant_guard=guard.audit(d)
  assert determinant_guard["original_print_constraints"]==93 and determinant_guard["exact_original_print_g_pass"]and determinant_guard["convex_print_guard_pass"]
  determinant_guard.update(source=ref(guard_source),not_a_substitute_for_whole_nonlinear_scene=True)
 beforeq=audit_same_state_beforeq(w,parent,paths,spec,ip)
 scope=dict(schema="rinkfit.finite_S_original_W_accepted_candidate_numerical_scope.v2",primary_epigraph_coordinate=coordinate,exact_primary_epigraph_value=epigraph,canonical_budget_value=b,primary_A_uses_declared_epigraph_coordinate=True,curvature_manifest=ref(CR),optional_print_convex_guard=determinant_guard,all_source_norm_scope=True,finite_S_original_W_scope=True,selected_main=bool(spec["select_main"]),original_W_report_schema=rawW["schema"],same_state_beforeq_arithmetic_bridge=beforeq,raw_rows=N,active_rows=182016,statistics_rank=214112,full_source_affine_qmax=float(q.max()),original_A_beyond_IEEE=0,strict_box_finite_sensor=True,original_working_b=float(C["working_b"]),working_target_b=b,explicit_working_upper_not_bstar=True,primary_optimality_verified=False,W_secondary_optimality_verified=False,PHR_multiplier_update_permitted=False,original_numeric_status=rawW["raw_solver_status"],numeric_status_not_overridden=True,selected_original_S_columns=len(selected),full_rawN_seed_retained=True,unselected_S_conditionally_fixed_only_in_proposal=True,ordinary_S_physically_fixed=False,original_1692_entities_S_unchanged=True,original_961_profiles_operator_unchanged=True,new_physical_DOF=False,original_W_coefficients_reaudited=coeff,ratio_to_declared_budget_chord_not_exact_norm_prediction=True,fresh_derivatives_required=True,source_values_labels_order_W_E_unchanged=True)
 wfinal=None
 if not include_line:return dict(spec=spec,parent=parent,primary=pr,W=w,W_final=None,norm_scope=scope,paths=paths,frozen=frozen,frozen_path=fp,base_spec=base_spec,bundle=bundle)
 lp=verify(spec["actual_line_report"],ip.parent);lr=read(lp);paths["actual_line_report"]=lp
 assert lr["schema"]=="rinkfit.finite_visible_physical_and_original_S_actual_line_search.v1"
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
 # The warm trial S is the same total fraction as theta. The native inner may
 # then refine existing S/V/alpha, and every final raw array is preserved.
 warm_path=verify(accepted["original_S_trial_seed"]);warm=read(warm_path)
 assert warm["schema"]=="rinkfit.finite_visible_physical_and_original_S_trial_seed.v1"
 assert warm["parent_checkpoint"]["sha256"]==sha(paths["parent_checkpoint"])
 assert warm["proposal"]["sha256"]==sha(paths["W_report"])and warm["proposal_npz"]["sha256"]==w["proposal"]["sha256"]
 assert float(warm["fraction"]).hex()==float(accepted["fraction"]).hex()
 oldS=asset(paths["parent_checkpoint"].parent,parent["row_state_files"]["station"])
 expectedS=oldS+float(accepted["fraction"])*direction["station_delta_full"]
 assert np.fromfile(verify(warm["files"]["station.f64"]),"<f8").tobytes()==expectedS.tobytes()
 for k,name in [("branch","path.u32"),("row","initial_row.f64"),("alpha","initial_alpha.f64")]:
  assert sha(verify(warm["files"][name]))==parent["row_state_files"][k]["sha256"]
 assert warm["model"]["sha256"]==accepted["physical_model_sha256"]
 assert lr["accepted_candidate"]in lr["history"]and lr["final_physical_fit_accepted"]is False
 scope["actual_line_warm_seed"]=ref(warm_path)
 scope["actual_native_metrics_tradeoffs"]={"median_before":parent["metrics"]["quantiles_px"][0],"median_after":metrics["quantiles_px"][0],"over2_before":parent["metrics"]["over2"],"over2_after":metrics["over2"],"selection_role":"selected_main"if spec["select_main"]else"preserved_candidate_not_selected_main"}
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
  account(paths["primary_report"],"elapsed_seconds","same_state_inexact_primary_curvature_recovery")
  account(verify(w["original_secondary_report"]),"elapsed_seconds","complete_original_W_conditional_secondary_outer_wall")
  account(paths["W_report"],"elapsed_seconds","explicit_normalizer")
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
 W_context=dict(declared_numerical_scope=ctx["norm_scope"],transport=w["transport"],primary_report=ref(paths["primary_report"]),final_W_proposal=ref(paths["W_report"]),proposal_step=w["proposal"],frozen_line_bundle=ref(paths["frozen_line_bundle"]),all_1692_entity_norms_reaudited=True,conditional_fixed_ordinary_S=False,original_S_seed_and_selected_columns=True,selected_main=bool(ctx["spec"]["select_main"]),PhaseII_joint_station_optimum=False)
 wp=ctx["W_final"] or w
 if ctx["W_final"]:W_context["same_delta_final_audit"]=ref(paths["W_final_audit_report"])
 wpred=None
 if all(k in wp for k in ["baseline_W_half_cost","predicted_W_linear_delta","predicted_W_quadratic_delta"]):wpred=wp["baseline_W_half_cost"]+fraction*wp["predicted_W_linear_delta"]+fraction*fraction*wp["predicted_W_quadratic_delta"]
 elif fraction==1. and "predicted_W_half_cost"in wp:wpred=wp["predicted_W_half_cost"]
 control["iteration"]=iteration;control["history"].append(dict(iteration=iteration,linearization=w["linearization"],LP=ref(paths["primary_report"]),line_search=ref(paths["actual_line_report"]),actual_reduction=actual,predicted_reduction=prediction*fraction,ratio=ratio,metrics=accepted["best"],accepted_fraction=fraction,trust_used=control["trust"],conditional_whole_W_selection=W_context,whole_W_baseline=parent["metrics"]["whole_original_W_half_cost"],whole_W_actual=accepted["best"]["whole_original_W_half_cost"],whole_W_conditional_quadratic_prediction=wpred,external_finite_S_original_W_candidate=True,selected_main=bool(ctx["spec"]["select_main"])))
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
 doc=namespace["save_state"](seed,"external_finite_S_original_W_candidate_fresh_derivatives_pending",None,None,None)
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

 if ctx["norm_scope"].get("finite_S_original_W_scope"):
  doc["algorithm_id"]="finite_RS_query_PhaseI_finite_original_S_original_correlated_W_declared_inexact_candidate"
  doc["selected_main"]=bool(ctx["spec"]["select_main"])
  doc["candidate_selection_role"]="selected valid PhaseI main continuation with explicit tradeoff ledger"if ctx["spec"]["select_main"]else"preserved valid candidate; main baseline explicit parent"
  doc["fresh_derivatives_required"]=True
  doc["budget_chord_ratio_is_not_exact_norm_prediction"]=True
  doc["control"]["trust"]=parent["control"]["trust"]
  control["trust"]=parent["control"]["trust"]
 if ctx["W_final"]:doc["same_delta_W_final_audit"]=ref(paths["W_final_audit_report"])
 init=dict(schema="rinkfit.continuous_SLP_complete_seed.v1",model=doc["model"],semantic_manifest=w["semantic"],files={name:ref(cp/name)for name in ["station.f64","path.u32","initial_row.f64","initial_alpha.f64"]})
 atomic(cp/"initializer_manifest.json",init);doc["complete_seed_initializer"]=ref(cp/"initializer_manifest.json")
 doc.pop("state_token",None);doc["state_token"]=identity(doc);atomic(cp/"manifest.json",doc);atomic(out/"current.json",ref(cp/"manifest.json"))
 atomic(out/"result.json",dict(schema="rinkfit.continuous_reduced_SLP_result.v1",status="FINITE_ORIGINAL_S_W_ACCEPTED_COMPLETE_STATE_PRESERVED_NO_SOLVE",selected_main=bool(ctx["spec"]["select_main"]),current_complete_checkpoint=ref(cp/"manifest.json"),frozen_run=ref(out/"frozen_run.json"),control=control,cumulative_elapsed_seconds=doc["cumulative_elapsed_seconds"],raw_rows=N,active_rows=182016,frames=1248,no_obligations_removed=True,no_Solve=True,whole_shot_final_fit_accepted=False,unfinished_requires_parent_continuation_or_diagnosis=True))
 print(json.dumps(dict(checkpoint=ref(cp/"manifest.json"),state_token=doc["state_token"],iteration=iteration,trust=control["trust"],metrics=doc["metrics"],no_Solve=True),indent=2))
 return doc
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output");p.add_argument("--check-context-only",action="store_true");a=p.parse_args();ip=Path(a.input).resolve()
 if a.check_context_only:validate_context(ip,include_line=False);print("EXTERNAL_CONTEXT_VALID_NO_SOLVE");return
 if not a.output:p.error("--output is required for adoption")
 adopt(ip,Path(a.output).resolve())
if __name__=="__main__":main()
