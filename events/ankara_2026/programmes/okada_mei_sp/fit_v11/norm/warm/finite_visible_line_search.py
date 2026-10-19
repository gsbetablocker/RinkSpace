from pathlib import Path
import argparse,copy,hashlib,json,subprocess,sys,time,os,numpy as np
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONPATH"]=str(Path(r"D:\CodexData\RinkSpace_Data\runs\okada_s1_r11\physical_reduced_component262_rf29_r11_transport\python_frozen"))
ap=argparse.ArgumentParser();ap.add_argument("--proposal",required=True);ap.add_argument("--initial-dir",required=True);ap.add_argument("--semantic-manifest",required=True);ap.add_argument("--stop-file");ap.add_argument("--output",required=True)
a=ap.parse_args();start=time.monotonic();D=Path(a.output);P=Path(a.proposal);B=Path(a.initial_dir);SM=Path(a.semantic_manifest)
if D.exists()and any(D.iterdir()):raise ValueError("Refuse overwriting line search")
D.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads(P.read_text());assert p["proposed"] and p["semantic"]["sha256"]==sha(SM)
M=Path(p["model"]["path"]);assert sha(M)==p["model"]["sha256"];model=json.loads(M.read_text());step_path=Path(p['proposal'].get('path',p['proposal'].get('file'))) if p.get('proposal') else P.parent/'proposal_step.npz';step_path=step_path if step_path.is_absolute()else P.parent/step_path;assert not p.get('proposal')or sha(step_path)==p['proposal']['sha256'];step=np.load(step_path);delta=step["physical_delta"];old=step["parent_values"];offset=np.r_[0,np.cumsum([len(b["values"])for b in model["blocks"]])]
assert np.array_equal(old,np.concatenate([b["values"]for b in model["blocks"]]))
# The original source/UV/W/E model evaluator remains frozen. A proposed
# deltaS is an original nuisance-coordinate warm seed, never a pixel warp.
def seed_for_trial(trial,fraction,model_path):
 if 'station_delta_full' not in step:raise ValueError('Coupled proposal must contain full raw-N original-S seed')
 total=np.asarray(step['station_delta_full']);assert total.shape==(182017,)and np.isfinite(total).all()
 authority=p['authority'];cp_ref=authority['checkpoint'];CP=Path(cp_ref['path']);assert sha(CP)==cp_ref['sha256'];cp=json.loads(CP.read_text())
 oldS=np.fromfile(B/'station.f64','<f8');assert oldS.shape==(182017,)
 for key,name in [('station','station.f64'),('branch','path.u32'),('row','initial_row.f64'),('alpha','initial_alpha.f64')]:assert sha(B/name)==cp['row_state_files'][key]['sha256']
 sem=json.loads(SM.read_text());assert authority['semantic']['sha256']==sha(SM)
 def load_desc(d,base):
  x=Path(d.get('path',d.get('file')));x=x if x.is_absolute()else base/x;assert sha(x)==d['sha256'];return np.fromfile(x,d['dtype']).reshape(d['shape'])
 active=load_desc(sem['active_source_mask'],SM.parent).astype(bool);kind=load_desc(sem['arrays']['kind'],SM.parent)
 assert np.flatnonzero(~active).tolist()==[25537]and np.all(total[(~active)|(kind!=0)]==0)
 TP=Path(authority['transport']['path']);assert sha(TP)==authority['transport']['sha256'];td=json.loads(TP.read_text());lo=load_desc(td['arrays']['station_delta_lower_right'],TP.parent);hi=load_desc(td['arrays']['station_delta_upper_right'],TP.parent);changed=total!=0
 assert np.all(total[changed]>=lo[changed])and np.all(total[changed]<=hi[changed])
 S=oldS+fraction*total;assert np.isfinite(S).all()and np.all((S[active]>=0)&(S[active]<=1))
 seed=trial/'original_S_seed';seed.mkdir();S.astype('<f8').tofile(seed/'station.f64')
 for name in ['path.u32','initial_row.f64','initial_alpha.f64']:(seed/name).write_bytes((B/name).read_bytes())
 files={name:dict(path=str(seed/name),sha256=sha(seed/name))for name in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']}
 doc=dict(schema='rinkfit.finite_visible_physical_and_original_S_trial_seed.v1',semantic=dict(path=str(SM),sha256=sha(SM)),model=dict(path=str(model_path),sha256=sha(model_path)),parent_checkpoint=cp_ref,parent_complete_initial_state=str(B),proposal=dict(path=str(P),sha256=sha(P)),proposal_npz=dict(path=str(step_path),sha256=sha(step_path)),fraction=fraction,source_station_changed_rows=np.flatnonzero(S!=oldS).tolist(),source_raw_rows=182017,active_rows=182016,only_original_ordinary_S_changed=True,branch_row_alpha_bitwise_parent=True,files=files,no_nonlinear_feasibility_or_KKT_claim=True)
 (seed/'initializer_manifest.json').write_text(json.dumps(doc,indent=2));return seed,dict(path=str(seed/'initializer_manifest.json'),sha256=sha(seed/'initializer_manifest.json'))
root=Path("D:/CodexData/RinkSpace_Data/runs/okada_s1_r11");evaluator=Path(__file__).with_name('evaluate_reduced_trial.py')
base=json.loads((B/"report.json").read_text());baseline_q=float(base["best"]["max_q"]);history=[];fraction=1.;accepted=None;stop=Path(a.stop_file)if a.stop_file else None;step_floor=np.sqrt(np.finfo(float).eps)*max(1.,float(np.max(abs(old/step["scales"]))));step_size=float(np.max(abs(delta/step["scales"])))
while fraction*step_size>step_floor:
 if stop is not None and stop.exists():break
 trial=D/("trial_"+str(len(history)));trial.mkdir();C=trial/"physical_model.json";candidate=copy.deepcopy(model)
 for k,b in enumerate(candidate["blocks"]):
  v=old[offset[k]:offset[k+1]]+fraction*delta[offset[k]:offset[k+1]];assert np.isfinite(v).all()and(np.array(b["lower"])<=v).all()and(v<=np.array(b["upper"])).all();b["values"]=v.tolist()
 candidate["reduced_physical_proposal"]=dict(parent_model=str(M),parent_sha256=sha(M),proposal_manifest=str(P),proposal_sha256=sha(P),fraction=fraction,objective="PhaseI constrained source minimax with finite nuisance recovery; not the old finite-rho lifted merit",not_a_final_fit=True)
 C.write_text(json.dumps(candidate,indent=2));log=trial/"stdout.log";trial_seed,seed_ref=seed_for_trial(trial,fraction,C)
 command=[sys.executable,"-X","utf8",str(evaluator),"--model",str(C),"--initial-dir",str(trial_seed),"--semantic-manifest",str(SM),"--output",str(trial/"inner")]
 print("REAL_FULL_N_LINE_SEARCH",json.dumps(dict(fraction=fraction,trial=str(trial))),flush=True)
 with log.open("w",encoding="utf-8")as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
 reportfile=trial/"inner/report.json";rec=dict(fraction=fraction,returncode=code,log=str(log),physical_model_sha256=sha(C),original_S_trial_seed=seed_ref)
 if reportfile.exists():
  z=json.loads(reportfile.read_text());best=z["best"];physics=z["whole_static_scene"];rec.update(best=best,whole_static_scene_passed=bool(physics["passed"]),report=dict(path=str(reportfile),sha256=sha(reportfile)))
  legal=code==0 and best["valid"]and best["active_RS_valid_rows"]==182016 and best["query_numerical_roots"]==961 and best["max_query_equality_px"]<=1e-6 and physics["passed"]
  rec["strict_source_scene_query_numerical_gates_passed"]=legal
  if legal and best["max_q"]<baseline_q:
   rec["accepted_as_PhaseI_initialization_candidate"]=True;accepted=rec
 # SOC is an independently evaluated candidate from this exact rejected
 # trial. Never weaken the original scene gate or substitute an older model.
 allowed_failures={"ice.B1:segment0/ice_segment_containment","ice.B2:segment0/ice_segment_containment","ice.C:segment0/ice_segment_containment"}
 source_legal=reportfile.exists()and code==0 and best["valid"]and best["active_RS_valid_rows"]==182016 and best["query_numerical_roots"]==961 and best["max_query_equality_px"]<=1e-6
 failures=set(physics.get("violations",[]))if reportfile.exists()else set()
 if accepted is None and source_legal and failures and failures<=allowed_failures and not(stop is not None and stop.exists()):
  history.append(rec);(D/"history.json").write_text(json.dumps(history,indent=2))
  soc=trial/"guarded_SOC";soclog=trial/"guarded_SOC.log";code_root=Path(r"D:\CodexData\RinkSpace_Data\runs\okada_s1_r11\model_checks\all_source_norm_line_bundle\code");run_root=Path('D:\\CodexData\\RinkSpace_Data\\runs\\okada_s1_r11\\physical_reduced_component262_rf29_r11_transport')
  command=[sys.executable,"-X","utf8",str(code_root/"guarded_line_soc.py"),"--initial-dir",str(reportfile.parent),"--semantic-manifest",str(SM),"--packet",p["packet"]["path"],"--output",str(soc),"--prepare-script",str(code_root/"prepare_qbar_trial_input.py"),"--qbar-runner",str(code_root/"ad_runner.py"),"--qbar-build",str(run_root/"helpers/qbar"),"--value-runner",str(code_root/"run_reduced_phase_I_numerics.py"),"--value-build",str(run_root/"helpers/values"),"--numerics-driver",str(code_root/"reduced_phase_I_numerics.py"),"--numerical-runtime",str(run_root/"helpers/numerical_runtime.json")]
  print("GUARDED_SOC_ACTUAL",json.dumps(dict(primary_fraction=fraction,primary_report=str(reportfile),command=command)),flush=True)
  with soclog.open("w",encoding="utf-8")as f:soc_code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
  index=soc/"complete_index.json";sr=dict(fraction=fraction,primary_fraction=fraction,is_independent_SOC_candidate=True,original_primary_trial_feasible=False,source_trial_report=rec["report"],SOC_returncode=soc_code,SOC_log=str(soclog),report=None)
  if soc_code==0 and index.exists()and json.loads(index.read_text()).get("passed")and not(stop is not None and stop.exists()):
   command=[sys.executable,"-X","utf8",str(evaluator),"--model",str(soc/"model.json"),"--initial-dir",str(soc),"--semantic-manifest",str(SM),"--output",str(soc/"inner")]
   sl=soc/"fullN_evaluator.log"
   with sl.open("w",encoding="utf-8")as f:soc_eval=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
   sf=soc/"inner/report.json"
   sr.update(returncode=soc_eval,log=str(sl),SOC_complete_index=dict(path=str(index),sha256=sha(index)),SOC_initializer=dict(path=str(soc/"initializer_manifest.json"),sha256=sha(soc/"initializer_manifest.json")),physical_model_sha256=sha(soc/"model.json"))
   if sf.exists():
    zz=json.loads(sf.read_text());bb=zz["best"];pp=zz["whole_static_scene"]
    okay=soc_eval==0 and bb["valid"]and bb["active_RS_valid_rows"]==182016 and bb["query_numerical_roots"]==961 and bb["max_query_equality_px"]<=1e-6 and pp["passed"]
    sr.update(best=bb,whole_static_scene_passed=bool(pp["passed"]),strict_source_scene_query_numerical_gates_passed=bool(okay),report=dict(path=str(sf),sha256=sha(sf)))
    if okay and bb["max_q"]<baseline_q:sr["accepted_as_PhaseI_initialization_candidate"]=True;accepted=sr
  rec=sr
 history.append(rec);(D/"history.json").write_text(json.dumps(history,indent=2))
 if accepted is not None or(stop is not None and stop.exists()):break
 fraction*=.5
 if not np.any(old+fraction*delta!=old):break
out=dict(schema="rinkfit.finite_visible_physical_and_original_S_actual_line_search.v1",proposal=dict(path=str(P),sha256=sha(P)),numerical_step_floor_scaled=step_floor,stopped=bool(stop is not None and stop.exists()),no_final_KKT_inferred_from_line_search=True,baseline=dict(path=str(B/"report.json"),sha256=sha(B/"report.json"),max_q=baseline_q),semantic=dict(path=str(SM),sha256=sha(SM)),raw_rows=182017,active_rows=182016,source_W_E_unchanged=True,physical_model_DOF_unchanged=True,history=history,accepted_candidate=accepted,final_physical_fit_accepted=False,PhaseII_fullW_station_optimum_solved=False,global_profile_and_reduced_KKT_certified=False,elapsed_seconds=time.monotonic()-start)
(D/"report.json").write_text(json.dumps(out,indent=2));print("FULL_N_SHARED_LINE_SEARCH_DONE",json.dumps(dict(output=str(D/"report.json"),accepted=accepted is not None,elapsed_seconds=out["elapsed_seconds"])),flush=True)
