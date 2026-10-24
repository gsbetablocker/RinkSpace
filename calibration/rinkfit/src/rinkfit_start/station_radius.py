"""Adaptive DATA-only calibration of a current-state sampled station radius.
Calibrates only actually legal pure-S Taylor errors. Pure-S RS loss is recorded
as conditional visibility evidence; it never alone shrinks/fixes the station.
Not a continuous remainder enclosure or changed-theta/source feasibility proof.
"""
from pathlib import Path
import json,hashlib,sys,time,argparse,numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf8"))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(d,base=None):
 p=Path(d.get("path",d.get("file")));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=d["sha256"]:raise ValueError("Changed calibrated-radius authority "+str(p))
 return p
def arr(d,base):return np.fromfile(verify(d,base),d["dtype"]).reshape(d["shape"])
def run(input_path,output):
 start=time.monotonic();I=Path(input_path).resolve();c=read(I);T=verify(c["transport"]);tr=read(T);CP=verify(c["checkpoint"]);cp=read(CP);MP=verify(c["model"]);model=read(MP);PK=verify(c["packet"]);pk=read(PK);SP=verify(c["semantic"]);sem=read(SP)
 for k in ["checkpoint","model","packet","semantic"]:
  if tr[k]["sha256"]!=c[k]["sha256"]:raise ValueError("Different same-state radius authority "+k)
 for k in ["station","branch","row","alpha"]:
  verify(cp["row_state_files"][k],CP.parent)
  if cp["row_state_files"][k]["sha256"]!=c["complete_auxiliary_sha256"][k]:raise ValueError("Changed complete auxiliary "+k)
 package=Path(c["python_modules"][0]["path"]).parent.parent
 for d in c["python_modules"]:verify(d)
 CB=verify(c["compiled_binding"]);sys.path.insert(0,str(package));sys.path.insert(0,str(CB.parent))
 from rinkfit_start.native_scene import NativeScene
 from rinkfit_start.material_foot import _certified_candidates
 from bound_camera_sensor import bind_camera_from_model
 O=Path(output).resolve();O.mkdir(exist_ok=False);scene=NativeScene(model);camera=bind_camera_from_model(ref(MP),model=model)
 N=int(pk['observations']);S=arr(cp["row_state_files"]["station"],CP.parent);B=arr(cp["row_state_files"]["branch"],CP.parent);radius=arr(c["arrays"]["radius"],I.parent);original_radius=radius.copy();ids=np.flatnonzero(radius>0)
 from rinkfit_start.semantic_resolution import load_semantics
 _,semantic_arrays,_,verified_semantics=load_semantics(SP,N,expected_sha256=sha(SP));mi=np.asarray(verified_semantics['resolved_material_index'],int);kind=semantic_arrays['kind'];active=np.asarray(verified_semantics['source_exclusion']['active_mask'],bool);pts=arr(pk['arrays']['native_pts'],PK.parent);E=arr(pk['arrays']['engineering_budget_px'],PK.parent);times=(pts-int(pk['pts_origin']))*int(pk['native_time_base'][0])/int(pk['native_time_base'][1])
 L=arr(tr["arrays"]["station_delta_lower_right"],T.parent);U=arr(tr["arrays"]["station_delta_upper_right"],T.parent);tan=arr(tr["arrays"]["station_UV_right"],T.parent).reshape(N,2);speed=np.linalg.norm(tan,axis=1)
 if cp["raw_rows"]!=N or cp["active_rows"]!=int(active.sum())or not np.all(active[ids]&(kind[ids]==0)):raise ValueError("Radius differs from verified active ordinary source scope")
 base=_certified_candidates(scene,camera,mi[ids],B[ids],S[ids],times[ids]) if len(ids) else dict(valid=np.empty(0,bool),uv=np.empty((0,2)))
 if not base["valid"].all():raise ValueError("Current same-state base source is not valid")
 uv0=np.full((N,2),np.nan);uv0[ids]=base["uv"];target=.05*E
 probes=[(sg,f)for sg in [-1,1]for f in [.25,.5,1.]]
 valid=np.zeros((6,N),bool);error=np.full((6,N),np.nan);row=np.full((6,N),np.nan)
 def sample(jj):
  if not len(jj):return
  for p,(sg,f)in enumerate(probes):
   ds=np.maximum(L[jj],np.minimum(U[jj],sg*f*radius[jj]));z=_certified_candidates(scene,camera,mi[jj],B[jj],S[jj]+ds,times[jj])
   valid[p,jj]=z["valid"];row[p,jj]=z["row"];error[p,jj]=np.linalg.norm(z["uv"]-uv0[jj]-tan[jj]*ds[:,None],axis=1)
 sample(ids);initial_valid=valid.copy();initial_error=error.copy();initial_row=row.copy()
 bad=np.any(valid[:,ids]&(error[:,ids]>target[ids]),axis=0);pending=ids[bad];initial_bad=pending.copy();history=[]
 print("ADAPT_INITIAL",json.dumps(dict(sources=len(ids),legal_error_failed_rows=len(pending),strict_RS_conditional_rows=int(np.count_nonzero(np.any(~valid[:,ids],axis=0))),elapsed_seconds=time.monotonic()-start)),flush=True)
 while len(pending):
  old=radius[pending].copy();ratio=np.max(np.where(valid[:,pending],error[:,pending]/target[pending],0),axis=0);factor=np.minimum(.9,.9/np.sqrt(ratio));candidate=old*factor
  if np.any(~np.isfinite(candidate))or np.any(candidate<0):raise ValueError("Adaptive radius finite arithmetic")
  stalled=(candidate==old)|(S[pending]+candidate==S[pending])|(S[pending]-candidate==S[pending]);radius[pending]=np.where(stalled,0,candidate)
  active_pending=pending[~stalled]
  if len(active_pending):sample(active_pending)
  if np.any(stalled):
   jj=pending[stalled];valid[:,jj]=False;error[:,jj]=np.nan;row[:,jj]=np.nan
  bad2=np.any(valid[:,active_pending]&(error[:,active_pending]>target[active_pending]),axis=0);new_pending=active_pending[bad2]
  rec=dict(iteration=len(history),resampled_rows=len(active_pending),unresolved_float_stall_rows=pending[stalled].tolist(),remaining_bad_rows=len(new_pending),radius_shrink_min=float(np.min(np.divide(radius[pending],old))),elapsed_seconds=time.monotonic()-start);history.append(rec);print("ADAPT",json.dumps(rec),flush=True);pending=new_pending
 calibrated=ids[radius[ids]!=original_radius[ids]];conditional=ids[np.any(~valid[:,ids],axis=0)&(radius[ids]>0)];alllegal=ids[np.all(valid[:,ids],axis=0)&(radius[ids]>0)]
 status=np.full(N,2,np.uint8);status[alllegal]=0;status[conditional]=1;status[ids[radius[ids]==0]]=3
 final=[]
 for p,(sg,f)in enumerate(probes):
  g=valid[p,ids];ee=error[p,ids][g]
  final.append(dict(sign=sg,fraction=f,tested_rows=len(ids),strict_RS_valid=int(g.sum()),pureS_RS_conditional_failed_rows=ids[~g].tolist(),legal_remainder_over_0p05E=int(np.count_nonzero(ee>target[ids[g]])),maximum_legal_remainder_px=float(np.max(ee,initial=0.)),normalized_legal_remainder_quantiles=np.quantile(ee/E[ids[g]],[.5,.95,.99,1]).tolist() if len(ee) else []))
 if any(x["legal_remainder_over_0p05E"]for x in final):raise ValueError("Adaptive sampled accuracy not reached")
 camera.at.close();arrays={}
 for name,value,dtype in [("radius",radius,"<f8"),("sample_status",status,"|u1"),("sample_valid",valid,"|u1"),("sample_remainder_px",error,"<f8"),("sample_true_row",row,"<f8"),("initial_sample_valid",initial_valid,"|u1"),("initial_sample_remainder_px",initial_error,"<f8"),("initial_sample_true_row",initial_row,"<f8")]:
  p=O/(name+".bin");np.asarray(value,dtype=dtype).tofile(p);arrays[name]=dict(file=p.name,dtype=dtype,shape=list(value.shape),sha256=sha(p))
 def locate(i):
  path=model["materials"][mi[i]]["paths"][int(B[i])];p=path;out=dict(source_row=int(i),material=model["materials"][mi[i]]["label"],material_index=int(mi[i]),path=int(B[i]),path_kind=int(p["kind"]),station=float(S[i]),original_cell=[float(L[i]),float(U[i])],original_radius=float(original_radius[i]),new_radius=float(radius[i]),tangent_speed=float(speed[i]),old_first_order_motion_px=float(original_radius[i]*speed[i]),new_first_order_motion_px=float(radius[i]*speed[i]),local_sampled_C2=float(arr(c["arrays"]["sampled_curvature_envelope"],I.parent)[i]),parent_provenance=p.get("provenance"),probe_records=[])
  for n,(sg,f)in enumerate(probes):
   out["probe_records"].append(dict(sign=sg,fraction=f,initial_valid=bool(initial_valid[n,i]),initial_remainder_px=float(initial_error[n,i]),initial_true_row=float(initial_row[n,i]),final_valid=bool(valid[n,i]),final_remainder_px=float(error[n,i]),final_true_row=float(row[n,i])))
  return out
 # Closed state/model metadata and every modified source remain explicit.
 report=dict(schema="rinkfit.adaptive_original_station_numerical_RS_radius.v1",input_curvature=ref(I),checkpoint=c["checkpoint"],transport=c["transport"],model=c["model"],packet=c["packet"],semantic=c["semantic"],complete_auxiliary_sha256=c["complete_auxiliary_sha256"],source=ref(__file__),arrays=arrays,original_radius=c["arrays"]["radius"],original_radius_base=str(I.parent),full_positive_radius_sources=len(ids),calibrated_sources=len(calibrated),initial_legal_accuracy_failed_source_rows=initial_bad.tolist(),calibrated_source_rows=calibrated.tolist(),unchanged_sources=int(np.count_nonzero(radius[ids]==original_radius[ids])),conditional_pureS_RS_source_rows=conditional.tolist(),pureS_RS_failure_does_not_remove_or_zero_radius=True,original_source_reachability_and_changed_theta_must_still_be_enforced=True,main_radius_has_no_uniform_10px_cap=True,numerical_accuracy_fraction_of_E=.05,original_E_W_Q_source_or_physical_bounds_changed=False,history=history,final_six_point_actual_checks=final,sampled_checks_are_not_continuous_or_changed_theta_certificate=True,all_source_physical_line_still_required=True,sample_status_codes={"0":"all_six_pureS_probes_valid_and_legal_errors_pass","1":"pureS_RS_visibility_conditional; legal_samples_pass; newtheta_reachability_not_removed","2":"input_no_positive_radius","3":"temporary_numerical_float_stall_unresolved"},diagnostic_rows={str(i):locate(int(i))for i in c.get("diagnostic_rows",{})},all_calibrated_metadata=[locate(i)for i in calibrated],elapsed_seconds=time.monotonic()-start)
 (O/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=True),encoding="utf8");print("ADAPT_DONE",json.dumps(dict(manifest=ref(O/"manifest.json"),calibrated=len(calibrated),unchanged=report["unchanged_sources"],conditional=len(conditional),final_checks=[{k:v for k,v in x.items()if k!="pureS_RS_conditional_failed_rows"}for x in final],elapsed_seconds=report["elapsed_seconds"]),ensure_ascii=False),flush=True);return report
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--curvature",required=True);p.add_argument("--output",required=True);a=p.parse_args(argv);return run(a.curvature,a.output)
if __name__=="__main__":main()

