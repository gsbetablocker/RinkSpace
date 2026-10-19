"""New-source-target reduced Phase-I preparation; continuation guards remain strict."""
from pathlib import Path
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time,traceback,numpy as np
sys.stdout.reconfigure(encoding="utf-8")
P=argparse.ArgumentParser();P.add_argument("--input",required=True);P.add_argument("--output",required=True);P.add_argument("--dry-run",action="store_true");P.add_argument("--resume");P.add_argument("--algorithm-continue-from");P.add_argument("--restore-only",action="store_true");P.add_argument("--prepare-linearization-only",action="store_true");args=P.parse_args();started=time.time();IP=Path(args.input).resolve();D=Path(args.output).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();ref=lambda p:dict(path=str(Path(p).resolve()),sha256=sha(p))
if D.exists()and any(D.iterdir()):raise ValueError("Refuse overwriting immutable run; resume into a new output directory")
D.mkdir(parents=True,exist_ok=True);spec=json.loads(IP.read_text());assert spec["schema"]=="rinkfit.continuous_reduced_SLP_input.v1";event=Path(spec["event_root"]).resolve();source=Path(r'D:\CodexData\RinkSpace_Data\runs\okada_s1_r11\model_checks');STOP=D/"STOP";env={k.upper():v for k,v in os.environ.items()};env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONIOENCODING="utf-8")
def verify(r):
 p=Path(r.get("path",r.get("file"))).resolve();assert sha(p)==r["sha256"],str(p);return p
def atomic(p,x):
 q=p.with_name(p.name+".tmp");q.write_text(json.dumps(x,indent=2),encoding="utf-8");os.replace(q,p)
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def log(kind,**kw):
 x=dict(kind=kind,unix_time=time.time(),**kw);print(json.dumps(x,ensure_ascii=False),flush=True)
 with(D/"events.jsonl").open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False)+"\n")
for key in["model","semantic","packet","query"]:verify(spec[key])
packet=json.loads(verify(spec["packet"]).read_text());semantic=json.loads(verify(spec["semantic"]).read_text());model=json.loads(verify(spec["model"]).read_text());assert packet["observations"]>0 and packet["frame_count"]>0
assert packet["semantic_packet"]["sha256"]==spec["semantic"]["sha256"]and packet["query_packet"]["sha256"]==spec["query"]["sha256"]
assert len(model["blocks"])==1376 and sum(len(b["values"])for b in model["blocks"])==4469
state=spec["initial_state"];N=int(packet["observations"])
sys.path.insert(0,str(Path(spec["python_package_root"]).parent))
from rinkfit_start.target_prepare import selected_scope
selected=selected_scope(verify(spec["packet"]),verify(spec["semantic"]))
if spec.get("selected_target_scope")!=selected:raise ValueError("Declared target scope differs from verified source authority")
ACTIVE_ROWS=selected["active_rows"];W_RANK=selected["W_rank"];PROFILES=selected["profiles"];FRAMES=selected["frames"]
if model["source_identity"]!=selected["source_identity"]:raise ValueError("Model source identity differs from selected target")

for key in["station","path","row","alpha"]:
 z=state[key];p=verify(z);assert z["shape"]==[N]and p.stat().st_size==N*np.dtype(z["dtype"]).itemsize
runtime=copy.deepcopy(spec["runtime"]);assert runtime["trust_initial"]>0 and 0<runtime["master_fraction"]<1 and 0<runtime["near_active_fraction"]<=1
if "numerical_runtime"in spec:verify(spec["numerical_runtime"])
if args.resume and args.algorithm_continue_from:raise ValueError("Choose exact code resume or explicit same-objective algorithm continuation")
if spec.get("new_target_initial_state") and (args.resume or args.algorithm_continue_from):raise ValueError("New-target initialization cannot import optimizer continuation/history")
continuation_path=args.resume or args.algorithm_continue_from
resume_doc=json.loads(Path(continuation_path).read_text())if continuation_path else None
algorithm_continuation=bool(args.algorithm_continue_from)
canonical_resume=bool(resume_doc and resume_doc.get("schema")=="rinkfit.complete_continuous_reduced_SLP_checkpoint.v1"and not algorithm_continuation)
if canonical_resume:
 frozen_path=verify(resume_doc["frozen_run"]);frozen=json.loads(frozen_path.read_text());assert frozen["immutable_target"]=={k:spec[k]for k in["model","semantic","packet","query","initial_state"]}and frozen["runtime"]==runtime
 for item in frozen["code"]:verify(item["frozen"])
 for item in frozen["python_modules"]:verify(item)
 for h in frozen["helpers"].values():
  verify(h["executable"]);verify(h["recipe"])
  for item in h.get("source_snapshot",[]):verify(item)
 for item in frozen.get("runtime_libraries",[]):verify(item)
 if not(args.dry_run or args.restore_only):assert sha(__file__)==frozen["code"][-1]["frozen"]["sha256"],"Invoke the frozen driver for exact algorithm continuation"
 helper_info=frozen["helpers"];ad_runner=verify(frozen["ad_runner"])if frozen.get("ad_runner")else None;target_id=frozen["target_id"];code_dir=Path(frozen["code_dir"]);atomic(D/"frozen_run_reference.json",dict(frozen_run=ref(frozen_path),target_id=target_id));log("FROZEN_CODE_REUSED",target_id=target_id)
else:
 helper_info={};(D/"helpers").mkdir()
 for key in["qbar","values"]:
  h=spec["helpers"].get(key)
  if h is None:continue
  exe=verify(h["executable"]);recipepath=verify(h["recipe"]);recipe=json.loads(recipepath.read_text());assert sha(exe)==recipe["executable_sha256"]
  caps=recipe.get("capabilities",{})
  if not caps and recipe.get("formal_capabilities"):caps=json.loads(verify(recipe["formal_capabilities"]).read_text())
  requires_component="source_component_recovery"in semantic
  if requires_component:assert caps.get("source_component_recovery")is True and recipe.get("source_component_evidence_policies"),"Helper must be rebuilt with the strict pinned source_component loader"
  cache=D/"helpers"/key;cache.mkdir();shutil.copyfile(exe,cache/exe.name);shutil.copyfile(recipepath,cache/"recipe.json");helper_info[key]=dict(executable=ref(cache/exe.name),recipe=ref(cache/"recipe.json"),runtime_bin_paths=recipe["runtime_bin_paths"],reader_header_sha256=recipe.get("reader_header_sha256"),source_component_evidence_policies=recipe.get("source_component_evidence_policies"),capabilities=caps)
  snapshot=[]
  for item in recipe.get("frozen_sources",{}).values():
   relative=Path(item["file"]);origin=recipepath.parent/relative;assert sha(origin)==item["sha256"];target=cache/relative;assert target.resolve().is_relative_to(cache.resolve());target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,target);snapshot.append(ref(target))
  helper_info[key].update(source_snapshot=snapshot,recipe_schema=recipe.get("schema"),tool=recipe.get("tool"))
 (D/"code").mkdir();(D/"python_frozen").mkdir();package=Path(spec["python_package_root"]);dest=D/"python_frozen/rinkfit_start";dest.mkdir();package_refs=[]
 for p in sorted(package.glob("*.py")):
  shutil.copyfile(p,dest/p.name);package_refs.append(ref(dest/p.name))
 script_names=["prepare_qbar_trial_input.py","propose_reduced_lp.py","materialize_qbar_refined_seed.py","evaluate_reduced_trial.py","reduced_physical_line_search.py","reduced_phase_I_numerics.py"]
 if spec.get("soc_script"):script_names.extend(["guarded_line_soc.py","run_reduced_phase_I_numerics.py"])
 if spec.get("whole_W_secondary"):script_names.extend(["secondary_whole_W_qp.py","repair_whole_W_qp.py","whole_W_qp_core.py"])
 if spec.get("resident"):script_names.append("resident_reduced_client.py")
 if spec.get("resident_transport"):script_names.append("resident_transport_client.py")
 if spec.get("profile_domains"):script_names.append("profile_domain_constraints.py")
 if spec.get("visibility_chart_guards"):script_names.append("temporary_visibility_chart.py")
 if spec.get("evaluation_batching"):script_names.append("batched_grid_4k.py")
 code_refs=[]
 for name in script_names:
  if name=="resident_transport_client.py":p=verify(spec["resident_transport"]["client"])
  elif name=="temporary_visibility_chart.py"and spec.get("visibility_chart_guards",{}).get("module"):p=verify(spec["visibility_chart_guards"]["module"])
  elif name=="batched_grid_4k.py"and spec.get("evaluation_batching"):p=verify(spec["evaluation_batching"]["function"])
  elif name=="evaluate_reduced_trial.py"and spec.get("evaluation_batching"):p=verify(spec["evaluation_batching"]["evaluator"])
  elif name=="prepare_qbar_trial_input.py"and spec.get("prepare_input_script"):p=verify(spec["prepare_input_script"])
  elif name=="materialize_qbar_refined_seed.py"and spec.get("materializer_script"):p=verify(spec["materializer_script"])
  elif name=="resident_reduced_client.py"and spec.get("resident"):p=verify(spec["resident"]["client"])
  elif spec.get("whole_W_secondary")and name in spec["whole_W_secondary"]["scripts"]:p=verify(spec["whole_W_secondary"]["scripts"][name])
  elif name=="reduced_physical_line_search.py"and spec.get("line_search_script"):p=verify(spec["line_search_script"])
  elif name=="guarded_line_soc.py"and spec.get("soc_script"):p=verify(spec["soc_script"])
  elif name=="propose_reduced_lp.py"and spec.get("lp_script"):p=verify(spec["lp_script"])
  elif name=="reduced_phase_I_numerics.py"and spec.get("numerics_script"):p=verify(spec["numerics_script"])
  else:p=source/name
  assert p.exists(),str(p);text=p.read_text();text=text.replace("D:/OneDrive/codex_projs/RinkSpace/calibration/rinkfit/src",str(D/"python_frozen").replace("\\","/"));text=text.replace("D:/CodexData/RinkSpace_Data/runs/okada_s1_r11",str(event).replace("\\","/"))
  if name=="reduced_physical_line_search.py":text=text.replace('evaluator=root/"model_checks/evaluate_reduced_trial.py"','evaluator=Path('+repr(str(D/"code/evaluate_reduced_trial.py"))+')')
  out=D/"code"/name;out.write_text(text,encoding="utf-8");code_refs.append(dict(original=ref(p),frozen=ref(out),transport="Import search path/event root/evaluator dispatch point only to immutable copies; model equations unchanged"))
 shutil.copyfile(__file__,D/"code/reduced_slp_driver.py");code_refs.append(dict(original=ref(__file__),frozen=ref(D/"code/reduced_slp_driver.py")))
 if spec.get("ad_runner"):
  runner=verify(spec["ad_runner"]);shutil.copyfile(runner,D/"code/ad_runner.py");ad_runner=D/"code/ad_runner.py"
 else:ad_runner=None
 whole_W_tool=None
 if spec.get("whole_W_secondary"):
  ws=spec["whole_W_secondary"];source_exe=verify(ws["executable"]);source_recipe=verify(ws["recipe"]);recipe_doc=json.loads(source_recipe.read_text());assert sha(source_exe)==recipe_doc["executable_sha256"]
  tool_dir=D/"helpers/whole_W";tool_dir.mkdir();shutil.copyfile(source_exe,tool_dir/"whole_W_transport.exe");shutil.copyfile(source_recipe,tool_dir/"recipe.json")
  sources=[]
  for item in ws["source_files"]:
   source_file=verify(item);copy_path=tool_dir/source_file.name;shutil.copyfile(source_file,copy_path);sources.append(ref(copy_path))
  qa=verify(ws["derivative_QA"]);shutil.copyfile(qa,tool_dir/"derivative_QA.json");assert json.loads(qa.read_text())["passed"]
  whole_W_tool=dict(executable=ref(tool_dir/"whole_W_transport.exe"),recipe=ref(tool_dir/"recipe.json"),source_snapshot=sources,derivative_QA=ref(tool_dir/"derivative_QA.json"),conditional_fixed_S_selection_not_PhaseII_station_optimum=True)
 profile_tool=None
 if spec.get("profile_domains"):
  ps=spec["profile_domains"];pe=verify(ps["executable"]);pr=verify(ps["recipe"]);pd=json.loads(pr.read_text());assert sha(pe)==pd["executable_sha256"]
  pc=D/"helpers/profile_domains";pc.mkdir();shutil.copyfile(pe,pc/pe.name);shutil.copyfile(pr,pc/"recipe.json");sources=[]
  for item in ps["source_files"]:
   src=verify(item);shutil.copyfile(src,pc/src.name);sources.append(ref(pc/src.name))
  qa=verify(ps["derivative_QA"]);shutil.copyfile(qa,pc/"derivative_QA.json")
  profile_tool=dict(executable=ref(pc/pe.name),recipe=ref(pc/"recipe.json"),source_snapshot=sources,derivative_QA=ref(pc/"derivative_QA.json"),original_961_finite_S_alpha_row_domains=True)
 resident_info=None
 if spec.get("resident"):
  rs=spec["resident"];rb=Path(rs["build_dir"]);rp=verify(rs["recipe"]);rd=json.loads(rp.read_text());worker=verify(rd["executable"])
  for source_name,digest in rd["sources"].items():assert sha(source_name)==digest
  resident_dir=D/"helpers/resident";resident_dir.mkdir();shutil.copyfile(worker,resident_dir/worker.name);rd["executable"]=ref(resident_dir/worker.name);rd["original_recipe"]=ref(rp);atomic(resident_dir/"recipe.json",rd)
  qas=[]
  for qa in rs["QA"]:
   qpath=verify(qa);copy_path=resident_dir/("QA_"+str(len(qas))+".json");shutil.copyfile(qpath,copy_path);qas.append(ref(copy_path))
  resident_info=dict(build_dir=str(resident_dir),executable=rd["executable"],recipe=ref(resident_dir/"recipe.json"),client=ref(D/"code/resident_reduced_client.py"),sources=[dict(path=k,sha256=v)for k,v in rd["sources"].items()],QA=qas,persistent_verified_dataset_only_not_numerical_result_cache=True)
 compiled_info=None
 if spec.get("compiled_camera"):
  cs=spec["compiled_camera"];cr=verify(cs["recipe"]);assessment_path=verify(cs["assessment"]);ca=json.loads(assessment_path.read_text());assert ca["no_original_14_math_or_OneDrive_model_fact_changes"]and ca["source_W_E_Q_and_all_original_active_rows_retained"]and ca["i13_actual_new_model_validation"]["same_X_t_all_rawN_savedrow_and_readout_ends_pass"]and ca["final_real_capture_benchmark"]["original_all_gates_passed"]
  assert ca["entrypoints"]["recipe"]["sha256"]==cs["recipe"]["sha256"];cd=D/"code/compiled_camera";cd.mkdir();copies={}
  for relative,item in cs["files"].items():
   origin=verify(item);destination=cd/relative;assert destination.resolve().is_relative_to(cd.resolve());destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,destination);copies[relative]=ref(destination)
  assert copies["camera_at.dll"]["sha256"]==ca["entrypoints"]["DLL"]["sha256"]and copies["bound_camera_sensor.py"]["sha256"]==ca["entrypoints"]["bound_factory"]["sha256"]
  compiled_info=dict(recipe=cs["recipe"],assessment=cs["assessment"],frozen_files=copies,arithmetic_is_not_claimed_bitwise_identical=True,original_NumPy_sensor_and_domain_fallback_preserved=True,model_SHA_bound_new_context_each_trial=True)
 transport_info=None
 if spec.get("resident_transport"):
  ts=spec["resident_transport"];tp=verify(ts["recipe"]);td=json.loads(tp.read_text());te=verify(td["executable"]);assert td["same_exact_projection_RS_profile_W_math"]
  if td["schema"] in ["rinkfit.resident_verified_GPU_transport.v1","rinkfit.resident_verified_GPU_transport_scope.v1"]:
   assert td["CPU_primal_RS_profile_W"]and td["raw_UV_GPU"]and not td["no_kernel_changed"]
   qa=json.loads(verify(td["backend_validation"]).read_text());assert qa["passed"]and qa["complete_state_unchanged"]and qa["end_to_end_speedup"]>1
   assert qa["build"]==td["validated_backend_build"];verify(qa["build"])
   assert qa["raw_rows"]==N and qa["active_rows"]==ACTIVE_ROWS and qa["profiles"]==PROFILES and qa["original_W_rank"]==W_RANK
  else:assert td["schema"]=="rinkfit.resident_verified_transport_prototype.v3"and td["no_kernel_changed"]
  for origin,digest in td["sources"].items():assert sha(origin)==digest
  tc=D/"helpers/transport";tc.mkdir();shutil.copyfile(te,tc/te.name);td["executable"]=ref(tc/te.name);td["original_recipe"]=ref(tp);atomic(tc/"recipe.json",td);tqas=[]
  for item in ts["QA"]:
   qp=verify(item);assert json.loads(qp.read_text())["passed"];tqas.append(ref(qp))
  transport_info=dict(build_dir=str(tc),executable=td["executable"],recipe=ref(tc/"recipe.json"),original_recipe=ref(tp),client=ref(D/"code/resident_transport_client.py"),sources=[dict(path=k,sha256=v)for k,v in td["sources"].items()],QA=tqas,roles=["whole_uv","profile_implicit"],separate_from_qbar_nuisance_refinement=True)
 runtime_libraries=[];seen_libraries=set()
 for helper in helper_info.values():
  for directory in helper["runtime_bin_paths"]:
   for dll in Path(directory).glob("*.dll"):
    if str(dll.resolve()).casefold()not in seen_libraries:seen_libraries.add(str(dll.resolve()).casefold());runtime_libraries.append(ref(dll))
 frozen=dict(runtime_libraries=runtime_libraries,schema="rinkfit.continuous_reduced_SLP_frozen_run.v1",input=ref(IP),immutable_target={k:spec[k]for k in["model","semantic","packet","query","initial_state"]},event_root=str(event),runtime=runtime,numerical_runtime=spec.get("numerical_runtime"),helpers=helper_info,code=code_refs,python_modules=package_refs,selected_target_scope=selected,source_id=packet["immutable_source_hashes"]["observations.npz"],budget_id=packet["arrays"]["engineering_budget_px"]["sha256"],raw_rows=N,active_rows=ACTIVE_ROWS,frames=FRAMES,original_W_E_unchanged=True)
 if spec.get("numerical_runtime"):shutil.copyfile(verify(spec["numerical_runtime"]),D/"helpers/numerical_runtime.json");frozen["numerical_runtime"]=ref(D/"helpers/numerical_runtime.json")
 if whole_W_tool:frozen["whole_W_tool"]=whole_W_tool
 if profile_tool:frozen["profile_domain_tool"]=profile_tool
 if spec.get("visibility_chart_guards"):
  visibility_QA=verify(spec["visibility_chart_guards"]["module_QA"]);assert json.loads(visibility_QA.read_text())["passed"];frozen["temporary_visibility_chart"]=spec["visibility_chart_guards"]
 if resident_info:frozen["resident"]=resident_info
 if transport_info:frozen["resident_transport"]=transport_info
 if compiled_info:frozen["compiled_camera"]=compiled_info
 if spec.get("evaluation_batching"):
  batchQA=verify(spec["evaluation_batching"]["QA"]);frozen["evaluation_batching"]=spec["evaluation_batching"];batch_evidence=json.loads(batchQA.read_text());assert batch_evidence["schema"]=="rinkfit.fixed_complete_state_CPU_capture_equivalence.v1"and batch_evidence["all_original_source_W_E_Q_RS_and_finite_domain_retained"]
  tested=batch_evidence["comparisons"]["batched_4k"];assert tested["scene_exact"]and all(x["byte_equal"]for x in tested["raw_arrays"].values())and all(x["byte_equal"]for x in tested["predictions"].values())and all(x["exact_equal"]for x in tested["metrics"].values())
 frozen["code_dir"]=str(D/"code");frozen["python_frozen_dir"]=str(D/"python_frozen");frozen["ad_runner"]=ref(ad_runner)if ad_runner else None
 operator_identity=dict(whole_W=None,resident=None,profile_domains=None,temporary_visibility_chart=spec.get("visibility_chart_guards"),evaluation_batching=spec.get("evaluation_batching"))
 if whole_W_tool:operator_identity["whole_W"]=dict(exe=whole_W_tool["executable"]["sha256"],recipe=spec["whole_W_secondary"]["recipe"]["sha256"],sources=sorted(x["sha256"]for x in whole_W_tool["source_snapshot"]),derivative_QA=whole_W_tool["derivative_QA"]["sha256"])
 if profile_tool:operator_identity["profile_domains"]=dict(exe=profile_tool["executable"]["sha256"],recipe=profile_tool["recipe"]["sha256"],sources=sorted(x["sha256"]for x in profile_tool["source_snapshot"]),derivative_QA=profile_tool["derivative_QA"]["sha256"])
 if compiled_info:operator_identity["compiled_camera"]=dict(recipe=compiled_info["recipe"]["sha256"],assessment=compiled_info["assessment"]["sha256"],files={k:v["sha256"]for k,v in compiled_info["frozen_files"].items()})
 if transport_info:operator_identity["resident_transport"]=dict(exe=transport_info["executable"]["sha256"],recipe=transport_info["original_recipe"]["sha256"],client=transport_info["client"]["sha256"],sources=sorted(x["sha256"]for x in transport_info["sources"]),QA=[q["sha256"]for q in transport_info["QA"]])
 if resident_info:operator_identity["resident"]=dict(exe=resident_info["executable"]["sha256"],recipe=spec["resident"]["recipe"]["sha256"],client=spec["resident"]["client"]["sha256"],source_policy_and_reader_hashes=sorted(x["sha256"]for x in resident_info["sources"]))
 frozen["operator_identity"]=operator_identity
 frozen["target_id"]=identity(dict(operator_identity=operator_identity,immutable_target=frozen["immutable_target"],runtime=runtime,numerical_runtime=frozen["numerical_runtime"],helper_identities={k:dict(exe=v["executable"]["sha256"],recipe=v["recipe"]["sha256"])for k,v in helper_info.items()},code_origins=[x["original"]["sha256"]for x in code_refs],python_modules=[x["sha256"]for x in package_refs]));atomic(D/"frozen_run.json",frozen);target_id=frozen["target_id"];log("INPUT_FROZEN",target_id=target_id,dry_run=args.dry_run)
 frozen_path=D/"frozen_run.json";code_dir=D/"code"

class StopRequested(Exception):pass
control=dict(iteration=0,trust=runtime["trust_initial"],selection_fraction=runtime["master_fraction"],compute_elapsed_seconds=0.,wall_origin_unix=started,history=[]);parent_cp=None;current_cp=None;elapsed_base=0.;initial_dir=D/"initial_state";initial_dir.mkdir();state_model=verify(spec["model"])
def model_template(z):
 z=copy.deepcopy(z)
 for b in z["blocks"]:b.pop("values",None)
 for k in["reduced_initializer_provenance","reduced_physical_proposal","ad6_finite_range_candidate","finite_line_SOC_provenance"]:z.pop(k,None)
 return identity(z)
if resume_doc:
 expected_token=resume_doc.get("state_token");body=dict(resume_doc);body.pop("state_token",None)
 if expected_token is not None:assert identity(body)==expected_token,"Checkpoint token mismatch"
 if algorithm_continuation:
  assert resume_doc["schema"]=="rinkfit.complete_continuous_reduced_SLP_checkpoint.v1"
  prior=json.loads(verify(resume_doc["frozen_run"]).read_text());oldpkt=json.loads(verify(prior["immutable_target"]["packet"]).read_text())
  for key in ["semantic","query"]:assert prior["immutable_target"][key]["sha256"]==spec[key]["sha256"],"Algorithm continuation cannot change source semantics or query"
  assert oldpkt["immutable_source_hashes"]==packet["immutable_source_hashes"]and oldpkt["arrays"]["engineering_budget_px"]["sha256"]==packet["arrays"]["engineering_budget_px"]["sha256"]
  assert model_template(json.loads(verify(resume_doc["model"]).read_text()))==model_template(model),"Algorithm continuation changed the physical model hypothesis"
  control=copy.deepcopy(resume_doc["control"]);prior_boundary=Path(resume_doc["frozen_run"]["path"]).parent/"result.json"
  if prior_boundary.exists():
   boundary=json.loads(prior_boundary.read_text())
   if boundary.get("current_complete_checkpoint",{}).get("sha256")==sha(continuation_path):control=copy.deepcopy(boundary["control"]);resume_doc["cumulative_elapsed_seconds"]=boundary["cumulative_elapsed_seconds"];atomic(D/"parent_execution_boundary.json",dict(source=ref(prior_boundary),cumulative_elapsed_seconds=boundary["cumulative_elapsed_seconds"],all_completed_failed_trial_elapsed_preserved=True))
  state_model=verify(spec["model"]);saved={key:dict(file=str(verify(value)),sha256=value["sha256"])for key,value in state.items()};parent_cp=ref(continuation_path)
  log("SAME_OBJECTIVE_ALGORITHM_CONTINUATION",parent=parent_cp,parent_target_id=resume_doc["target_id"],new_target_id=target_id,not_identical_iteration_trajectory=True,old_LP_duals_not_transplanted_as_KKT=True)
 elif canonical_resume:
  assert resume_doc["target_id"]==target_id;control=copy.deepcopy(resume_doc["control"]);state_model=verify(resume_doc["model"]);saved=resume_doc["row_state_files"];parent_cp=ref(continuation_path)
 else:
  if not(args.dry_run or args.restore_only):raise ValueError("Legacy checkpoint supports exact restore-only; initialize a fresh operator to continue without an unverifiable code context")
  state_model=verify(resume_doc["model"]);saved=resume_doc["row_state_files"];parent_cp=ref(continuation_path);old=resume_doc["control_context"];control.update(iteration=old["iteration"],trust=old.get("next_trust_scaled_infinity",runtime["trust_initial"]),wall_origin_unix=old["wall_clock_origin_unix"]);control["compute_elapsed_seconds"]=old.get("wall_elapsed_seconds",0.);control["legacy_context_preserved"]=old;control["history"]=[dict(restored_parent_checkpoint=parent_cp,original_control=copy.deepcopy(old),no_recomputed_history=True)]
 elapsed_base=resume_doc.get("cumulative_elapsed_seconds",control.get("compute_elapsed_seconds",0.));current_cp=parent_cp
 if algorithm_continuation and spec.get("discarded_execution_boundary"):
  resource_path=verify(spec["discarded_execution_boundary"]);resource=json.loads(resource_path.read_text())
  assert resource.get("whole_shot_final_fit_accepted")is False
  assert resource["frozen_run"]["sha256"]==resume_doc["frozen_run"]["sha256"],"Discarded resource branch must share the parent frozen context"
  old_elapsed=elapsed_base;elapsed_base=max(elapsed_base,float(resource["cumulative_elapsed_seconds"]))
  control["compute_elapsed_seconds"]=max(control.get("compute_elapsed_seconds",0.),resource["control"].get("compute_elapsed_seconds",0.))
  control.setdefault("discarded_execution_boundaries",[]).append(dict(source=ref(resource_path),parent_state_elapsed_seconds=old_elapsed,cumulative_resource_elapsed_seconds=elapsed_base,physical_state_history_not_adopted=True))
  atomic(D/"discarded_execution_resource_history.json",control["discarded_execution_boundaries"])
 assert model_template(json.loads(state_model.read_text()))==model_template(model),"Model hypothesis/paths/scales differ"
 for key,d in saved.items():
  path=Path(continuation_path).parent/d["file"];assert sha(path)==d["sha256"];name={"station":"station.f64","branch":"path.u32","path":"path.u32","row":"initial_row.f64","alpha":"initial_alpha.f64"}[key];shutil.copyfile(path,initial_dir/name)
else:
 for key,name in[("station","station.f64"),("path","path.u32"),("row","initial_row.f64"),("alpha","initial_alpha.f64")]:shutil.copyfile(verify(state[key]),initial_dir/name)
shutil.copyfile(state_model,initial_dir/"model.json")
for block in json.loads(state_model.read_text())["blocks"]:
 z=np.asarray(block["values"]);assert np.isfinite(z).all()and(z>=block["lower"]).all()and(z<=block["upper"]).all(),block["name"]
for name,dt,low,high in[("station.f64","<f8",0.,1.),("path.u32","<u4",0.,np.inf),("initial_row.f64","<f8",0.,1079.),("initial_alpha.f64","<f8",-np.inf,np.inf)]:
 z=np.fromfile(initial_dir/name,dt);assert len(z)==N and np.isfinite(z).all()and(z>=low).all()and(z<=high).all()
if algorithm_continuation:
 oldmodel=json.loads(verify(resume_doc["model"]).read_text());newmodel=json.loads((initial_dir/"model.json").read_text());delta=[]
 for bid,(old,new)in enumerate(zip(oldmodel["blocks"],newmodel["blocks"])):
  for col,(before,after)in enumerate(zip(old["values"],new["values"])):
   if before!=after:delta.append(dict(block=bid,name=old["name"],component=col,parent=before,proposal=after))
 changes={}
 for key,name,dt in [("station","station.f64","<f8"),("branch","path.u32","<u4"),("row","initial_row.f64","<f8"),("alpha","initial_alpha.f64","<f8")]:
  desc=resume_doc["row_state_files"][key];origin=Path(continuation_path).parent/desc["file"];assert sha(origin)==desc["sha256"];before=np.fromfile(origin,dt);after=np.fromfile(initial_dir/name,dt);changed=np.flatnonzero(before!=after).astype("<u8");file=D/("proposed_"+key+"_changed_rows.u64");changed.tofile(file);changes[key]=dict(count=len(changed),source_rows=ref(file),parent_sha256=desc["sha256"],proposal_sha256=sha(initial_dir/name))
 atomic(D/"algorithm_continuation_audit.json",dict(parent_checkpoint=ref(continuation_path),new_code_context=ref(frozen_path),physical_parameter_changes=delta,auxiliary_changes=changes,old_LP_duals_are_only_historical_context=True,identical_iteration_trajectory_claimed=False,source_W_E_query_and_physical_parameterization_unchanged=True))
initdoc=dict(schema="rinkfit.continuous_SLP_complete_seed.v1",model=ref(initial_dir/"model.json"),semantic_manifest=spec["semantic"],files={name:ref(initial_dir/name)for name in["station.f64","path.u32","initial_row.f64","initial_alpha.f64"]});atomic(initial_dir/"initializer_manifest.json",initdoc)
restore=dict(schema="rinkfit.continuous_SLP_input_restore_check.v1",no_Solve=True,input_identity=target_id,source_checkpoint=parent_cp,control_preserved=control,restored_cumulative_elapsed_seconds=elapsed_base,model=ref(initial_dir/"model.json"),files=initdoc["files"],all_raw_rows=N,active_rows=ACTIVE_ROWS,physics_checked_by_optimization=False);atomic(D/"restore_check.json",restore)
if args.dry_run or args.restore_only:
 atomic(D/"result.json",dict(status="DRY_RUN_VALIDATED"if args.dry_run else"EXACT_STATE_RESTORED_NO_SOLVE",no_Solve=True,restored=bool(resume_doc),restore=ref(D/"restore_check.json"),frozen_run=ref(frozen_path),final_fit_or_KKT_accepted=False));log("NO_SOLVE_CHECK_COMPLETE",restored=bool(resume_doc));raise SystemExit(0)
assert ad_runner and "qbar"in helper_info and "values"in helper_info and spec.get("numerical_runtime"),"Live SLP requires strict AD+values helpers and frozen numerical runtime"
for key in ["qbar","values"]:assert helper_info[key].get("recipe_schema")=="rinkfit.reduced_reader_snapshot_build.v1"and helper_info[key].get("tool")==key,"Live operator requires strict reader-aware helper snapshots"
code_dir=Path(frozen["code_dir"]);env["PYTHONPATH"]=frozen["python_frozen_dir"]+os.pathsep+env.get("PYTHONPATH","");sys.path.insert(0,frozen["python_frozen_dir"])
import importlib.util
resident=None;transport_resident=None
if frozen.get("resident_transport"):
 ti=frozen["resident_transport"];transport_spec=importlib.util.spec_from_file_location("frozen_transport_client",verify(ti["client"]));transport_module=importlib.util.module_from_spec(transport_spec);transport_spec.loader.exec_module(transport_module);transport_resident=transport_module.ResidentTransportClient(ti["build_dir"],D/"transport_owner",STOP).start();log("RESIDENT_TRANSPORT_WORKER_STARTED",pid=transport_resident.worker.pid,identity=ti)
if frozen.get("resident"):
 import importlib.util
 ri=frozen["resident"];client_spec=importlib.util.spec_from_file_location("frozen_resident_client",verify(ri["client"]));client_module=importlib.util.module_from_spec(client_spec);client_spec.loader.exec_module(client_module)
 verify(ri["executable"]);verify(ri["recipe"])
 for item in ri["sources"]:verify(item)
 resident=client_module.ResidentReducedClient(ri["build_dir"],D,STOP).start()
 log("RESIDENT_VERIFIED_DATASET_WORKER_STARTED",pid=resident.worker.pid,identity=ri)
def run_stage(name,command):
 if STOP.exists():raise StopRequested()
 out=D/"stages"/name;out.mkdir(parents=True,exist_ok=False);begin=time.time();logfile=out/"stdout.log";log("STAGE_STARTED",stage=name,command=command)
 if transport_resident is not None and(name.endswith("_whole_W_transport")or name.endswith("_profile_domains")):
  role="whole_uv"if name.endswith("_whole_W_transport")else"profile_implicit";receipt=transport_resident.transport(role,command[-2],command[-1]);logfile.write_text(json.dumps(receipt,indent=2),encoding="utf-8");elapsed=time.time()-begin;control["compute_elapsed_seconds"]+=elapsed;atomic(out/"execution.json",dict(role=role,command_interface_preserved=command,returncode=0,elapsed_seconds=elapsed,log=ref(logfile),resident_transport_receipt=receipt));log("STAGE_FINISHED",stage=name,returncode=0,elapsed_seconds=elapsed,resident_verified_transport=True)
  if STOP.exists():raise StopRequested()
  return out
 if resident is not None and(name.endswith("_AD")or name.endswith("_values")):
  role="qbar"if name.endswith("_AD")else"values";receipt=resident.request(role,command[-2],command[-1]);logfile.write_text(json.dumps(receipt,indent=2),encoding="utf-8");elapsed=time.time()-begin;control["compute_elapsed_seconds"]+=elapsed;atomic(out/"execution.json",dict(role=role,command_interface_preserved=command,returncode=0,elapsed_seconds=elapsed,log=ref(logfile),resident_receipt=receipt));log("STAGE_FINISHED",stage=name,returncode=0,elapsed_seconds=elapsed,resident_verified_dataset=True)
  if STOP.exists():raise StopRequested()
  return out
 with logfile.open("w",encoding="utf-8")as handle:
  process=subprocess.Popen(command,env=env,cwd=event,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));heartbeat=time.time();stop_reported=False
  while process.poll()is None:
   if STOP.exists()and not stop_reported:log("STOP_WAITING_FOR_COMPLETE_READ_ONLY_BOUNDARY",stage=name,pid=process.pid);stop_reported=True
   if time.time()-heartbeat>=30:log("STAGE_RUNNING",stage=name,pid=process.pid,elapsed_seconds=time.time()-begin);heartbeat=time.time()
   time.sleep(.2)
  code=process.returncode
 elapsed=time.time()-begin;control["compute_elapsed_seconds"]+=elapsed;atomic(out/"execution.json",dict(command=command,returncode=code,elapsed_seconds=elapsed,log=ref(logfile)));log("STAGE_FINISHED",stage=name,returncode=code,elapsed_seconds=elapsed)
 if STOP.exists():raise StopRequested()
 if code:raise RuntimeError("Stage failed: "+name+" exit="+str(code)+" log="+str(logfile))
 return out

def asset_array(base,d):
 p=Path(base)/d["file"];assert sha(p)==d["sha256"];return np.fromfile(p,d["dtype"]).reshape(d["shape"])
def same_linearization_state(seed,qm):
 if not qm or qm["model"]["sha256"]!=sha(seed/"model.json"):return False
 source_input=json.loads(verify(qm["input"]).read_text());root=Path(qm["input"]["path"]).parent;qr=Path(qm["_path"]).parent;chosen=asset_array(qr,qm["source_rows"]).astype(int)
 for key,name,dt in [("station","station.f64","<f8"),("path","path.u32","<u4"),("row","initial_row.f64","<f8"),("alpha","initial_alpha.f64","<f8")]:
  expected=asset_array(root,source_input["arrays"][key]).copy()
  if key in ["station","row"]:expected[chosen]=asset_array(qr,qm["linearization_"+key])
  if not np.array_equal(expected,np.fromfile(seed/name,dt)):return False
 return True
def current_chart_state(seed,out):
 from rinkfit_start.native_scene import NativeScene
 sm=verify(spec["semantic"]);sem=json.loads(sm.read_text());active=asset_array(sm.parent,sem["active_source_mask"]).astype(bool);kind=asset_array(sm.parent,sem["arrays"]["kind"]);from rinkfit_start.semantic_resolution import load_semantics;_,_,_,extra=load_semantics(sm,N,expected_sha256=sha(sm));material=extra["resolved_material_index"] if "resolved_material_index"in extra else asset_array(sm.parent,sem["resolved_material_index"]);scene=NativeScene(json.loads((seed/"model.json").read_text()));station=np.fromfile(seed/"station.f64","<f8");path=np.fromfile(seed/"path.u32","<u4");v=np.fromfile(seed/"initial_row.f64","<f8");rows=np.flatnonzero(active);mode=np.zeros(N,np.uint8)
 for i in rows[kind[rows]==0]:
  item=scene.path(int(material[i]),int(path[i]));edges=max(1,len(item.get("points",[]))-1)
  if not item.get("closed",False)and(station[i]<1e-10 or station[i]>1-1e-10):mode[i]=2
  elif item["kind"]==2 and abs(station[i]*edges-round(station[i]*edges))<1e-9:mode[i]=1
  if v[i]>1079-1e-7:mode[i]=7 if mode[i]else 3
  elif v[i]<1e-7:mode[i]=7 if mode[i]else 4
 mode[kind==1]=5;mode[kind==2]=6
 for name,z,dt in[("chart_source_rows.u64",rows,"<u8"),("chart_status.u8",np.full(len(rows),255,np.uint8),"|u1"),("chart_mode.u8",mode[rows],"|u1"),("unverified_source_rows.u64",rows,"<u8")]:np.asarray(z,dtype=dt).tofile(out/name)
 return dict(raw_active_rows=len(rows),verified_rows=0,unverified_rows=len(rows),unknown_derivatives_are_not_zero=True,current_mode_classification_only_not_KKT=True)
def copy_chart_state(qm,lpdir,out):
 if lpdir and(lpdir/"lp_duals.npz").exists():
  dual=np.load(lpdir/"lp_duals.npz");rows=dual["all_source_rows"];status=dual["all_chart_status"];mode=dual["all_chart_modes"]
 else:
  sm=Path(qm["semantic"]["path"]);doc=json.loads(sm.read_text());ad=doc["active_source_mask"];allrows=np.flatnonzero(np.fromfile(sm.parent/ad["file"],ad["dtype"]).astype(bool));rows=allrows;status=np.full(len(rows),255,np.uint8);mode=np.full(len(rows),255,np.uint8);lookup=np.full(N,-1,int);lookup[rows]=np.arange(len(rows))
  def qa(k):
   d=qm[k];return np.fromfile(Path(qm["_path"]).parent/d["file"],d["dtype"]).reshape(d["shape"])
  chosen=qa("source_rows").astype(int);status[lookup[chosen]]=qa("status");mode[lookup[chosen]]=qa("mode")
 for name,z,dt in[("chart_source_rows.u64",rows,"<u8"),("chart_status.u8",status,"|u1"),("chart_mode.u8",mode,"|u1"),("unverified_source_rows.u64",rows[status!=0],"<u8")]:np.asarray(z,dtype=dt).tofile(out/name)
 return dict(raw_active_rows=len(rows),verified_rows=int(np.count_nonzero(status==0)),unverified_rows=int(np.count_nonzero(status)),unknown_derivatives_are_not_zero=True)

def save_state(seed,reason,qm=None,lpdir=None,numerics=None):
 global current_cp,parent_cp
 state=json.loads((seed/"report.json").read_text());metrics=state["best"];generation=f"i{control['iteration']:05d}_{time.time_ns()}";out=D/"checkpoints"/generation;out.mkdir(parents=True)
 for name in["model.json","station.f64","path.u32","initial_row.f64","initial_alpha.f64","candidate_predictions.npz","report.json"]:shutil.copyfile(seed/name,out/name)
 same_derivative_state=same_linearization_state(seed,qm)
 chart=copy_chart_state(qm,lpdir,out)if same_derivative_state else current_chart_state(seed,out)
 chart["assets"]={name:ref(out/name)for name in ["chart_source_rows.u64","chart_status.u8","chart_mode.u8","unverified_source_rows.u64"]};chart["prior_linearization_is_not_current_gradient"]=not same_derivative_state
 if lpdir:
  for name in["lp_duals.npz","lp_row_mapping.json","profile_domain_row_mapping.json","visibility_chart_row_mapping.json","report.json","proposal_step.npz"]:
   if(lpdir/name).exists():shutil.copyfile(lpdir/name,out/("LP_"+name))
 files={key:dict(file=name,dtype=dt,shape=[N],sha256=sha(out/name))for key,name,dt in[("station","station.f64","<f8"),("branch","path.u32","<u4"),("row","initial_row.f64","<f8"),("alpha","initial_alpha.f64","<f8")]}
 doc=dict(schema="rinkfit.complete_continuous_reduced_SLP_checkpoint.v1",algorithm_id="finite_RS_query_reduced_PhaseI_SLP_v5_original_domains_temporary_visibility_chart_W_secondary",target_id=target_id,frozen_run=ref(frozen_path),parent_checkpoint=parent_cp,baseline_reference=spec.get("baseline_reference"),new_target_initialization=bool(spec.get("new_target_initial_state")),parent_complete_PHR_checkpoint=spec.get("parent_complete_PHR_checkpoint"),PHR_lambda_nu_rho_not_transplanted=True,reason=reason,model=ref(out/"model.json"),row_state_files=files,b=metrics["max_q"],raw_rows=N,active_rows=ACTIVE_ROWS,control=copy.deepcopy(control),cumulative_elapsed_seconds=elapsed_base+time.time()-started,elapsed_accounting="cumulative completed solver work including explicitly discarded branches plus this live segment; inactive gaps excluded",metrics=metrics,physical_scene=state["whole_static_scene"],derivative_coverage=chart,linearization=ref(qm["_path"])if qm else None,LP_duals_are_not_nonlinear_or_PHR_multipliers=True,fresh_numerics=numerics,nonlinear_stationarity=None if numerics is None else numerics.get("summary"),full_reduced_KKT_certified=False,full_PhaseII_correlated_W_station_optimum=False,final_physical_fit_accepted=False)
 doc["state_token"]=identity(doc);atomic(out/"manifest.json",doc);parent_cp=ref(out/"manifest.json");current_cp=parent_cp;atomic(D/"current.json",current_cp);log("COMPLETE_CHECKPOINT",state_token=doc["state_token"],iteration=control["iteration"],reason=reason,metrics=metrics,derivatives=chart);return doc

# Frozen child tools execute with the same reused numerical libraries.
binpaths=[]
for helper in helper_info.values():
 for value in helper["runtime_bin_paths"]:
  if value not in binpaths:binpaths.append(value)
env["PATH"]=";".join(binpaths)+";"+env.get("PATH","")
work=D/"work";work.mkdir();py=sys.executable;current_seed=initial_dir;pass_counter=0;status="UNFINISHED";diagnostic={}
def prepare_AD(seed,tag,fraction):
 inp=work/(tag+"_input");out=work/(tag+"_AD");run_stage(tag+"_prepare",[py,str(code_dir/"prepare_qbar_trial_input.py"),"--model",str(seed/"model.json"),"--initial-dir",str(seed),"--semantic-manifest",str(verify(spec["semantic"])),"--packet",str(verify(spec["packet"])),"--output",str(inp)]+(["--master-fraction",str(fraction)]if fraction else[]));run_stage(tag+"_AD",[py,str(ad_runner),"--build",str(Path(helper_info["qbar"]["executable"]["path"]).parent),"--stop-file",str(STOP),str(inp/"manifest.json"),str(out)]);qm=json.loads((out/"manifest.json").read_text());qm["_path"]=str(out/"manifest.json");return qm

def refine_complete(seed,qm,tag):
 out=work/(tag+"_canonical");run_stage(tag+"_canonical",[py,str(code_dir/"materialize_qbar_refined_seed.py"),"--linearization",qm["_path"],"--base",str(seed),"--output",str(out)]);report=json.loads((out/"report.json").read_text());assert report["raw_rows"]==N and report["active_rows"]==ACTIVE_ROWS and report["partition_exactly_once"]and report["best"]["active_RS_valid_rows"]==ACTIVE_ROWS and report["whole_static_scene"]["passed"],"Full-source strict canonicalization failed";return out

def complete_value_authority(seed,qm,values,out):
 vm=json.loads((values/"manifest.json").read_text());assert vm["raw_rows"]==N and vm["active_rows"]==ACTIVE_ROWS and vm["all_active_strict_valid"]and vm["statistical_partition_exactly_once"]
 out.mkdir();shutil.copyfile(seed/"model.json",out/"model.json");changes={}
 for key,name,dt in[("station","station.f64","<f8"),("path","path.u32","<u4"),("row","initial_row.f64","<f8"),("alpha","initial_alpha.f64","<f8")]:
  desc=vm["arrays"][key];src=values/desc["file"];assert desc["shape"]==[N]and sha(src)==desc["sha256"];shutil.copyfile(src,out/name);old=np.fromfile(seed/name,dt);new=np.fromfile(src,dt);changed=np.flatnonzero(old!=new).astype("<u8");changed.tofile(out/(key+"_witness_changed_rows.u64"));changes[key]=dict(count=len(changed),source_rows=ref(out/(key+"_witness_changed_rows.u64")),before_sha256=sha(seed/name),authoritative_sha256=desc["sha256"])
 assert same_linearization_state(out,qm)
 bd=packet["arrays"]["engineering_budget_px"];budget=asset_array(verify(spec["packet"]).parent,bd);q=asset_array(values,vm["arrays"]["q"]);active=asset_array(values,vm["arrays"]["active_mask"]).astype(bool);err=asset_array(values,vm["arrays"]["error_uv"]).reshape(N,2);coverage=np.sqrt(q)*budget;report=json.loads((seed/"report.json").read_text());report["best"].update(max_q=float(q[active].max()),maximum_error_px=float(coverage[active].max()),quantiles_px=np.quantile(coverage[active],[.5,.95,.99,1.]).tolist(),over2=int(np.count_nonzero(coverage[active]>budget[active])),whole_original_W_half_cost=vm["original_W_half_cost"],max_query_equality_px=vm["maximum_query_h"],maximum_source_row=vm["maximum_source_row"])
 pred=dict(np.load(seed/"candidate_predictions.npz"));pred.update(stations=np.fromfile(out/"station.f64","<f8"),branches=np.fromfile(out/"path.u32","<u4"),root_rows=np.fromfile(out/"initial_row.f64","<f8"),profile_alpha=np.fromfile(out/"initial_alpha.f64","<f8"),coverage_error_px=coverage,error_native_uv=err,uv=asset_array(verify(spec["packet"]).parent,packet["arrays"]["native_uv"])+err);np.savez_compressed(out/"candidate_predictions.npz",**pred)
 report["exact_value_state"]=ref(values/"manifest.json");report["prior_native_capture"]=ref(seed/"report.json");report["cpp_complete_auxiliary_state_authority"]=dict(definition="original unselected raw auxiliaries plus actual qbar linearization S/v; original path/alpha unchanged",source=qm["input"],linearization=ref(qm["_path"]),changes=changes,no_extra_true_RS_witness_substitution_after_this_state=True);report["files"]={name:sha(out/name)for name in["model.json","station.f64","path.u32","initial_row.f64","initial_alpha.f64","candidate_predictions.npz"]};atomic(out/"report.json",report);return out

def fresh_numerics(seed,qm,lpdir,tag):
 selected=qm["source_rows"]["shape"][0];recipe=json.loads(verify(helper_info["values"]["recipe"]).read_text());partial_supported=bool(recipe.get("supports_partial_linearization_values",False))
 if selected!=ACTIVE_ROWS and not partial_supported:
  return dict(summary=dict(full_reduced_PhaseI_KKT_certified=False,scaled_stationarity=None,fail_closed_reasons=["values helper has not exposed a partial-J transport capability; uncomputed derivatives remain unknown"]),actual_gradient_evaluated=False)
 values=work/(tag+"_values");num=work/(tag+"_numerics")
 if not(values/"manifest.json").exists():run_stage(tag+"_values",[str(verify(helper_info["values"]["executable"])),qm["_path"],str(values)])
 run_stage(tag+"_numerics",[py,str(code_dir/"reduced_phase_I_numerics.py"),"--values",str(values/"manifest.json"),"--runtime",str(verify(frozen["numerical_runtime"])),"--output",str(num),"--lp-report",str(lpdir/"report.json"),"--value-evaluator-recipe",str(verify(frozen["resident"]["recipe"]if resident is not None else helper_info["values"]["recipe"]))]);z=json.loads((num/"report.json").read_text());assert z["model"]["sha256"]==sha(seed/"model.json")and z["semantic"]["sha256"]==spec["semantic"]["sha256"]
 vm=json.loads((values/"manifest.json").read_text());assert vm["raw_rows"]==N and vm["active_rows"]==ACTIVE_ROWS and vm["all_active_strict_valid"]and vm["statistical_partition_exactly_once"]
 exact=work/(tag+"_exact_numerical_state");exact.mkdir();shutil.copyfile(seed/"model.json",exact/"model.json")
 for key,name in [("station","station.f64"),("path","path.u32"),("row","initial_row.f64"),("alpha","initial_alpha.f64")]:
  d=vm["arrays"][key];assert d["shape"]==[N];src=values/d["file"];assert sha(src)==d["sha256"];shutil.copyfile(src,exact/name)
 bd=packet["arrays"]["engineering_budget_px"];budget=asset_array(verify(spec["packet"]).parent,bd);q=asset_array(values,vm["arrays"]["q"]);active=asset_array(values,vm["arrays"]["active_mask"]).astype(bool);err=asset_array(values,vm["arrays"]["error_uv"]).reshape(N,2);coverage=np.sqrt(q)*budget
 state_report=json.loads((seed/"report.json").read_text());best=state_report["best"];best.update(max_q=float(q[active].max()),maximum_error_px=float(coverage[active].max()),quantiles_px=np.quantile(coverage[active],[.5,.95,.99,1.]).tolist(),over2=int(np.count_nonzero(coverage[active]>budget[active])),whole_original_W_half_cost=vm["original_W_half_cost"],max_query_equality_px=vm["maximum_query_h"],maximum_source_row=vm["maximum_source_row"]);state_report["exact_value_state"]=ref(values/"manifest.json");state_report["independent_native_capture_parent_report"]=ref(seed/"report.json");atomic(exact/"report.json",state_report)
 prediction=dict(np.load(seed/"candidate_predictions.npz"));prediction.update(stations=np.fromfile(exact/"station.f64","<f8"),branches=np.fromfile(exact/"path.u32","<u4"),root_rows=np.fromfile(exact/"initial_row.f64","<f8"),profile_alpha=np.fromfile(exact/"initial_alpha.f64","<f8"),coverage_error_px=coverage,error_native_uv=err,uv=asset_array(verify(spec["packet"]).parent,packet["arrays"]["native_uv"])+err);np.savez_compressed(exact/"candidate_predictions.npz",**prediction)
 assert same_linearization_state(exact,qm),"Fresh derivative state differs from saved full row state"
 summary={k:z.get(k)for k in["maximum_error_px","original_full_W_half_cost","normalized_primal_max","equality_max_px","domain_max_g","scene_max_g","derivative_verified_rows","derivative_unverified_rows","full_reduced_PhaseI_KKT_certified","fail_closed_reasons"]};summary["multiplier_candidates"]=[{k:v for k,v in m.items()if k not in["gradient","multipliers","arrays"]}for m in z.get("multiplier_candidates",[])];return dict(report=ref(num/"report.json"),values=ref(values/"manifest.json"),exact_state_directory=str(exact),actual_gradient_evaluated=True,summary=summary)
try:
 # Initial/resumed state is exposed exactly above, before any nuisance recovery.
 baseline=work/"initial_full_inner"
 if spec.get("new_target_initial_state"):
  authority=spec["new_target_initial_state"];rp=verify(authority["report"]);pred=verify(authority["predictions"])
  from rinkfit_start.state_integrity import validate_candidate_state
  validated=validate_candidate_state(rp.parent,verify(spec["packet"]),verify(spec["semantic"]))
  if authority.get("old_optimizer_history_or_duals_imported")is not False:raise ValueError("New target must reset history and multipliers")
  baseline.mkdir()
  for name in ["model.json","station.f64","path.u32","initial_row.f64","initial_alpha.f64"]:
   assert sha(initial_dir/name)==sha(rp.parent/name);shutil.copyfile(initial_dir/name,baseline/name)
  shutil.copyfile(rp,baseline/"report.json");shutil.copyfile(pred,baseline/"candidate_predictions.npz")
  atomic(baseline/"new_target_initialization.json",dict(source_complete_state=authority,old_control_history_or_duals_reused=False,new_target_id=target_id,baseline_reference=spec.get("baseline_reference")))
  log("NEW_SOURCE_TARGET_INITIALIZED",scope=selected,baseline_reference=spec.get("baseline_reference"),old_history_or_duals_reused=False)
 elif algorithm_continuation and spec.get("verified_complete_initial_state"):
  authority_path=verify(spec["verified_complete_initial_state"]);authority=json.loads(authority_path.read_text());assert authority["schema"]=="rinkfit.verified_complete_nuisance_warm_state.v1"and authority["parent_checkpoint"]["sha256"]==sha(continuation_path)and authority["physical_parameters_changed"]is False;verify(authority["source"])
  vp=verify(authority["values"]);vdoc=json.loads(vp.read_text());assert vdoc["all_active_strict_valid"]and not vdoc["invalid_active_rows"]and vdoc["raw_rows"]==N and vdoc["active_rows"]==ACTIVE_ROWS and vdoc["profiles"]==PROFILES and vdoc["full_W_rank"]==W_RANK and vdoc["statistical_partition_exactly_once"]and vdoc["scene_max_g"]<=0 and vdoc["maximum_query_h"]<=1e-6 and vdoc["maximum_RS_h"]<=1e-6
  assert vdoc["model"]["sha256"]==sha(initial_dir/"model.json")==authority["model"]["sha256"]
  for key in["packet","semantic"]:assert vdoc[key]["sha256"]==spec[key]["sha256"]
  parent_report=verify(authority["report"]);report=json.loads(parent_report.read_text());assert report["exact_value_state"]["sha256"]==authority["values"]["sha256"]and report["best"]["valid"]and report["whole_static_scene"]["passed"]
  baseline.mkdir();shutil.copyfile(initial_dir/"model.json",baseline/"model.json")
  for key,name in[("station","station.f64"),("path","path.u32"),("row","initial_row.f64"),("alpha","initial_alpha.f64")]:
   desc=vdoc["arrays"][key];assert sha(initial_dir/name)==desc["sha256"]==sha(parent_report.parent/name);shutil.copyfile(initial_dir/name,baseline/name)
  atomic(baseline/"report.json",report);shutil.copyfile(verify(authority["predictions"]),baseline/"candidate_predictions.npz");atomic(baseline/"verified_nuisance_warm_state.json",dict(authority=spec["verified_complete_initial_state"],parent_checkpoint=parent_cp,new_derivatives_required=True,old_KKT_not_transplanted=True))
  seconds=sum(float(json.loads(verify(item["report"]).read_text())[item["elapsed_field"]])for item in authority["actual_resource_sources"]);assert seconds==authority["actual_completed_resource_seconds"]and seconds>=0;control["compute_elapsed_seconds"]+=seconds;elapsed_base+=seconds;control.setdefault("external_completed_nuisance_work",[]).append(dict(authority=spec["verified_complete_initial_state"],elapsed_seconds=seconds));log("VERIFIED_COMPLETE_NUISANCE_WARM_STATE_RESTORED",authority=spec["verified_complete_initial_state"],auxiliary_changes=authority["auxiliary_changed_source_rows"])
 elif algorithm_continuation and spec.get("reuse_parent_full_source_initial_state"):
  assert sha(initial_dir/"model.json")==resume_doc["model"]["sha256"]
  for key,name in[("station","station.f64"),("branch","path.u32"),("row","initial_row.f64"),("alpha","initial_alpha.f64")]:assert sha(initial_dir/name)==resume_doc["row_state_files"][key]["sha256"]
  best=copy.deepcopy(resume_doc["metrics"]);assert best["valid"] and best["active_RS_valid_rows"]==ACTIVE_ROWS and best["query_numerical_roots"]==PROFILES and resume_doc["physical_scene"]["passed"]
  baseline.mkdir();shutil.copyfile(initial_dir/"model.json",baseline/"model.json")
  for name in["station.f64","path.u32","initial_row.f64","initial_alpha.f64"]:shutil.copyfile(initial_dir/name,baseline/name)
  parent_report=verify(spec["reuse_parent_full_source_initial_state"]["report"]);report=json.loads(parent_report.read_text());assert report["best"]==best and report["whole_static_scene"]==resume_doc["physical_scene"]
  for name in["model.json","station.f64","path.u32","initial_row.f64","initial_alpha.f64"]:assert sha(parent_report.parent/name)==sha(initial_dir/name)
  atomic(baseline/"report.json",report)
  prediction=verify(spec["reuse_parent_full_source_initial_state"]["predictions"]);shutil.copyfile(prediction,baseline/"candidate_predictions.npz")
  atomic(baseline/"identical_state_reuse.json",dict(parent_complete_checkpoint=parent_cp,report=ref(parent_report),predictions=ref(prediction),all_physical_and_rawN_auxiliary_file_hashes_identical=True,new_derivatives_and_full_values_are_not_reused=True))
  log("IDENTICAL_COMPLETE_PARENT_SOURCE_STATE_REUSED",parent=parent_cp)
 else:run_stage("initial_full_inner",[py,str(code_dir/"evaluate_reduced_trial.py"),"--model",str(initial_dir/"model.json"),"--initial-dir",str(initial_dir),"--semantic-manifest",str(verify(spec["semantic"])),"--output",str(baseline)])
 current_seed=baseline
 if algorithm_continuation:
  bm=json.loads((baseline/"report.json").read_text());assert bm["best"]["valid"]and bm["best"]["active_RS_valid_rows"]==ACTIVE_ROWS and bm["whole_static_scene"]["passed"]
  assert bm["best"]["max_q"]<=resume_doc["b"]+1e-10*max(1.,resume_doc["b"]),"Continuation input is not a nonincreasing same-objective state"
 save_state(current_seed,"same_objective_algorithm_continuation_full_source_state"if algorithm_continuation else"initial_complete_finite_RS_query_state")
 while True:
  if STOP.exists():raise StopRequested()
  tag=f"i{control['iteration']:05d}_p{pass_counter:05d}";pass_counter+=1;before=json.loads((current_seed/"report.json").read_text())["best"]
  prepared=spec.get("prepared_first_step")if pass_counter==1 else None
  if prepared:
   assert algorithm_continuation and prepared["parent_checkpoint"]["sha256"]==sha(continuation_path)
   qm=json.loads(verify(prepared["linearization"]).read_text());qm["_path"]=str(verify(prepared["linearization"]));assert same_linearization_state(current_seed,qm)
   lpdir=verify(prepared["primary_report"]).parent;lp=json.loads((lpdir/"report.json").read_text());assert lp["linearization"]["sha256"]==prepared["linearization"]["sha256"]
   numerics=copy.deepcopy(prepared.get("numerical_context")or resume_doc["fresh_numerics"])
   if prepared.get("fresh_numerics_report"):
    npth=verify(prepared["fresh_numerics_report"]);nz=json.loads(npth.read_text());assert nz["linearization"]["sha256"]==prepared["linearization"]["sha256"] and nz["value_state"]["sha256"]==numerics["values"]["sha256"]
    numerics["report"]=ref(npth);numerics["summary"]={k:nz.get(k)for k in["maximum_error_px","original_full_W_half_cost","normalized_primal_max","equality_max_px","domain_max_g","scene_max_g","derivative_verified_rows","derivative_unverified_rows","full_reduced_PhaseI_KKT_certified","fail_closed_reasons","multiplier_candidates"]}
   verify(numerics["report"]);vm=json.loads(verify(numerics["values"]).read_text())
   for key,name in[("station","station.f64"),("path","path.u32"),("row","initial_row.f64"),("alpha","initial_alpha.f64")]:assert vm["arrays"][key]["sha256"]==sha(current_seed/name)
   assert vm["qbar_linearization"]["sha256"]==prepared["linearization"]["sha256"] and vm["model"]["sha256"]==sha(current_seed/"model.json")
   external_work=prepared.get("completed_external_solver_work",[])
   for work_ref in external_work:
    work_path=verify(work_ref);assert work_ref["sha256"]in[prepared["QP_report"]["sha256"],prepared["repaired_proposal"]["sha256"],prepared.get("fresh_numerics_report",{}).get("sha256")]
    done=control.setdefault("external_completed_solver_work",[])
    if not any(x["report"]["sha256"]==work_ref["sha256"]for x in done):
     seconds=float(json.loads(work_path.read_text())["elapsed_seconds"]);assert seconds>=0;control["compute_elapsed_seconds"]+=seconds;elapsed_base+=seconds;done.append(dict(report=work_ref,elapsed_seconds=seconds,actual_external_same_state_solver_work=True))
   log("COMPLETED_SAME_STATE_MASTER_REUSED",context=prepared,no_new_derivative_or_solver_call_claimed=True)
  else:
   fraction=0. if before["max_q"]<=1. else control["selection_fraction"];qm=prepare_AD(current_seed,tag,fraction);current_seed=refine_complete(current_seed,qm,tag)
   if frozen.get("profile_domain_tool"):
    values=work/(tag+"_values");run_stage(tag+"_values",[str(verify(helper_info["values"]["executable"])),qm["_path"],str(values)]);current_seed=complete_value_authority(current_seed,qm,values,work/(tag+"_value_authority"))
   save_state(current_seed,"complete_current_inner_operator_before_LP",qm)
   domain_args=[]
   if frozen.get("temporary_visibility_chart"):
    preW=work/(tag+"_whole_W_transport");run_stage(tag+"_whole_W_transport",[str(verify(frozen["whole_W_tool"]["executable"])),str(verify(current_cp)),str(preW)]);domain_args=["--visibility-transport",str(preW/"manifest.json"),"--visibility-rows","auto"if frozen["temporary_visibility_chart"].get("source_selection")=="all_current_ordinary_box_touch"else",".join(str(i)for i in frozen["temporary_visibility_chart"]["source_rows"])]
   if frozen.get("profile_domain_tool"):
    dp=work/(tag+"_profile_domains");run_stage(tag+"_profile_domains",[str(verify(frozen["profile_domain_tool"]["executable"])),str(verify(current_cp)),str(dp)]);domain_args.extend(["--profile-domains",str(dp/"manifest.json")])
   if args.prepare_linearization_only:
    status="COMPLETE_SAME_STATE_OPERATORS_PREPARED_NO_MASTER_NO_PHYSICAL_STEP";diagnostic=dict(linearization=ref(qm["_path"]),whole_W_transport=ref(preW/"manifest.json"),profile_domains=ref(dp/"manifest.json"),no_old_scalar_master_or_KKT_claimed=True,no_physical_proposal_authorized=True,full_final_acceptance=False);break
   lpdir=work/(tag+"_LP");run_stage(tag+"_LP",[py,str(code_dir/"propose_reduced_lp.py"),"--linearization",qm["_path"],"--output",str(lpdir),"--trust",str(control["trust"]),"--mandatory-fraction",str(runtime["near_active_fraction"])]+domain_args);lp=json.loads((lpdir/"report.json").read_text());numerics=fresh_numerics(current_seed,qm,lpdir,tag);current_seed=Path(numerics["exact_state_directory"])if numerics.get("exact_state_directory")else current_seed;save_state(current_seed,"fresh_same_state_numerics_and_linear_master",qm,lpdir,numerics);before=json.loads((current_seed/"report.json").read_text())["best"]
  if args.prepare_linearization_only:
   status="COMPLETE_SAME_STATE_OPERATORS_AND_LINEAR_MASTER_PREPARED_NO_PHYSICAL_STEP";diagnostic=dict(linearization=ref(qm["_path"]),primary_report=ref(lpdir/"report.json"),whole_W_transport=ref(preW/"manifest.json"),profile_domains=ref(dp/"manifest.json"),numerics=numerics,no_physical_proposal_authorized=True,full_final_acceptance=False);break
  if before["max_q"]<=1.:
   status="PHASE_I_NATIVE_BUDGET_REACHED_FINAL_CERTIFICATES_AND_PHASE_II_REQUIRED";diagnostic=dict(numerics=numerics,full_final_acceptance=False);break
  prediction=lp["original_q_max"]-lp["linearized_epigraph"];floor=lp.get("unmodeled_q_constant_floor")
  if floor is not None and lp["linearized_epigraph"]<=floor+np.finfo(float).eps*max(1.,floor):
   old=control["selection_fraction"];new=min(old*.5,.5*floor/max(1.,before["max_q"]))
   if not 0<new<old:status="MASTER_SEPARATION_STAGNATED";diagnostic=lp;break
   control["selection_fraction"]=new;log("MASTER_EXPANDED",old_fraction=old,new_fraction=new);continue
  if prediction<=np.finfo(float).eps*max(1.,before["max_q"]):status="NO_RELIABLE_LINEAR_DESCENT_UNFINISHED";diagnostic=dict(LP=ref(lpdir/"report.json"),numerics=numerics);break
  assert frozen.get("whole_W_tool"),"New scientific step requires same-state whole-W secondary; no L1-only fallback"
  if prepared:
   wt=verify(prepared["whole_W_transport"]).parent;wq=verify(prepared["QP_report"]).parent;wr=verify(prepared["repaired_proposal"]).parent
   td=json.loads((wt/"manifest.json").read_text());qd=json.loads((wq/"report.json").read_text());rd=json.loads((wr/"report.json").read_text());assert qd["primary_report"]["sha256"]==prepared["primary_report"]["sha256"] and rd["QP_report"]["sha256"]==prepared["QP_report"]["sha256"]
   source_cp=json.loads(verify(td["checkpoint"]).read_text());assert source_cp["model"]["sha256"]==sha(current_seed/"model.json")
   for key,name in[("station","station.f64"),("branch","path.u32"),("row","initial_row.f64"),("alpha","initial_alpha.f64")]:assert source_cp["row_state_files"][key]["sha256"]==sha(current_seed/name)
   assert qd["transport"]["sha256"]==prepared["whole_W_transport"]["sha256"] and rd["source"]["sha256"]==spec["whole_W_secondary"]["scripts"]["repair_whole_W_qp.py"]["sha256"]
  else:
   wt=work/(tag+"_whole_W_transport");wq=work/(tag+"_W_QP");wr=work/(tag+"_W_repair")
   if not(wt/"manifest.json").exists():run_stage(tag+"_whole_W_transport",[str(verify(frozen["whole_W_tool"]["executable"])),str(verify(current_cp)),str(wt)])
   run_stage(tag+"_W_QP",[py,str(code_dir/"secondary_whole_W_qp.py"),"--primary-report",str(lpdir/"report.json"),"--transport",str(wt/"manifest.json"),"--output",str(wq),"--stop-file",str(STOP)])
   run_stage(tag+"_W_repair",[py,str(code_dir/"repair_whole_W_qp.py"),"--qp-report",str(wq/"report.json"),"--output",str(wr)])
  selection=json.loads((wr/"report.json").read_text());assert selection["proposed"]and selection["primary_epigraph_unchanged"]and selection["linearized_epigraph"]==lp["linearized_epigraph"]
  W_context=dict(transport=ref(wt/"manifest.json"),QP=ref(wq/"report.json"),raw_QP_state=ref(wq/"proposal_step.npz"),feasibility_recovery=ref(wr/"report.json"),proposal_step=ref(wr/"proposal_step.npz"),original_primary=ref(lpdir/"report.json"),conditional_fixed_ordinary_S=True,PhaseII_joint_station_optimum=False,complete_W_tertiary_minimum_movement_certificate=False)
  log("SAME_PRIMARY_W_SECONDARY_READY",context=W_context,predicted_W=selection["repaired_predicted_W_half_cost"],baseline_W=selection["baseline_W_half_cost"])
  if prepared and prepared.get("completed_line_search"):
   line=verify(prepared["completed_line_search"]).parent;lr=json.loads((line/"report.json").read_text());assert lr["proposal"]["sha256"]==sha(wr/"report.json") and lr["semantic"]["sha256"]==spec["semantic"]["sha256"] and lr["baseline"]["max_q"]==before["max_q"]
   bp=Path(lr["baseline"]["path"]).parent
   for name in["model.json","station.f64","path.u32","initial_row.f64","initial_alpha.f64"]:assert sha(bp/name)==sha(current_seed/name)
   control["compute_elapsed_seconds"]+=lr["elapsed_seconds"];elapsed_base+=lr["elapsed_seconds"];log("COMPLETED_SAME_STATE_NONLINEAR_LINE_REUSED",report=ref(line/"report.json"))
  else:
   line=work/(tag+"_line");run_stage(tag+"_line",[py,str(code_dir/"reduced_physical_line_search.py"),"--proposal",str(wr/"report.json"),"--initial-dir",str(current_seed),"--semantic-manifest",str(verify(spec["semantic"])),"--output",str(line),"--stop-file",str(STOP)]);lr=json.loads((line/"report.json").read_text())
  accepted=lr.get("accepted_candidate")
  if not accepted:status="NO_ACCEPTED_FULL_SOURCE_STEP_UNFINISHED";diagnostic=dict(line_search=ref(line/"report.json"),numerics=numerics);break
  after=accepted["best"];actual=before["max_q"]-after["max_q"];ratio=actual/(prediction*accepted["fraction"]);assert actual>0 and accepted["strict_source_scene_query_numerical_gates_passed"]
  control["iteration"]+=1;control["history"].append(dict(iteration=control["iteration"],linearization=ref(qm["_path"]),LP=ref(lpdir/"report.json"),line_search=ref(line/"report.json"),actual_reduction=actual,predicted_reduction=prediction*accepted["fraction"],ratio=ratio,metrics=after,accepted_fraction=accepted["fraction"],trust_used=control["trust"],conditional_whole_W_selection=W_context,whole_W_baseline=before["whole_original_W_half_cost"],whole_W_actual=after["whole_original_W_half_cost"],whole_W_conditional_quadratic_prediction=selection["baseline_W_half_cost"]+accepted["fraction"]*selection["predicted_W_linear_delta"]+accepted["fraction"]**2*selection["predicted_W_quadratic_delta"]));current_seed=Path(accepted["report"]["path"]).parent
  if ratio>.75 and accepted["fraction"]==1.:control["trust"]*=2.
  elif ratio<.25:control["trust"]*=.5
  save_state(current_seed,"accepted_full_source_physical_step_fresh_derivatives_pending",qm,lpdir,None)
  if actual<=np.sqrt(np.finfo(float).eps)*max(1.,before["max_q"]):status="NUMERICAL_PROGRESS_STAGNATED_UNFINISHED";diagnostic=dict(line_search=ref(line/"report.json"),actual_reduction=actual,numerical_scale=np.sqrt(np.finfo(float).eps)*max(1.,before["max_q"]));break
  if ratio<runtime.get("poor_prediction_ratio",.1):status="POOR_NONLINEAR_PREDICTION_UNFINISHED";diagnostic=dict(line_search=ref(line/"report.json"),ratio=ratio,all_obligations_preserved=True);break
except StopRequested:
 status="STOPPED_COMPLETE_ACCEPTED_STATE_PRESERVED";diagnostic=dict(unaccepted_current_trial_discarded=True)
except Exception as error:
 status="NUMERICAL_OR_INPUT_DIAGNOSTIC_UNFINISHED";diagnostic=dict(error=str(error),traceback=traceback.format_exc());log("UNFINISHED_DIAGNOSTIC",**diagnostic)
finally:
 if transport_resident is not None:
  try:transport_resident.close();log("RESIDENT_TRANSPORT_WORKER_CLOSED",returncode=getattr(transport_resident.worker,"returncode",None))
  except Exception as transport_close_error:diagnostic["transport_resident_close_error"]=str(transport_close_error)
 if resident is not None:
  try:resident.close();log("RESIDENT_WORKER_CLOSED",returncode=getattr(resident.worker,"returncode",None))
  except Exception as close_error:diagnostic["resident_close_error"]=str(close_error)
 result=dict(schema="rinkfit.continuous_reduced_SLP_result.v1",status=status,target_id=target_id,current_complete_checkpoint=current_cp,complete_input_state=ref(initial_dir/"initializer_manifest.json"),frozen_run=ref(frozen_path),control=control,cumulative_elapsed_seconds=elapsed_base+time.time()-started,diagnostic=diagnostic,raw_rows=N,active_rows=ACTIVE_ROWS,frames=FRAMES,no_obligations_removed=True,LP_success_is_not_KKT_success=True,whole_shot_final_fit_accepted=False,unfinished_requires_parent_continuation_or_diagnosis=True);atomic(D/"result.json",result);log("RUN_BOUNDARY",status=status,checkpoint=current_cp)
