"""Complete official50 review from one native complete checkpoint. No Solve."""
from pathlib import Path
from fractions import Fraction
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
import av,cv2,imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/"calibration/rinkfit/src"))
from rinkfit_start.native_scene import NativeScene,camera_from_event
from rinkfit_start.state_integrity import validate_receipt_bindings
FIRST,LAST,STEP,NFRAMES=256256,575488,256,1248
WH=(1920,1080)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb")as f:
  for b in iter(lambda:f.read(4194304),b""):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def verify(d,base=None):
 p=Path(d.get("path",d.get("file")));p=p if p.is_absolute()else Path(base)/p
 assert sha(p)==d["sha256"],"Immutable file changed: "+str(p)
 return p
def write(p,x):
 with Path(p).open("x",encoding="utf-8")as f:json.dump(x,f,ensure_ascii=False,indent=2)
def source_frames(path):
 c=av.open(str(path));v=c.streams.video[0]
 assert v.time_base==Fraction(1,12800)and v.average_rate==50
 c.seek(FIRST,stream=v,backward=True,any_frame=False);expected=FIRST
 try:
  for f in c.decode(v):
   if f.pts<FIRST:continue
   if f.pts>LAST:break
   assert f.pts==expected and f.width==1920 and f.height==1080,"Source native PTS gap/size"
   yield int(f.pts),f.to_ndarray(format="bgr24");expected+=STEP
  assert expected==LAST+STEP,"Source incomplete"
 finally:c.close()
def inventory(scene,model):
 records=[];xyz=[];offset=0;mapping=[]
 def add(X,mi,pi,p,label,color):
  nonlocal offset
  records.append(dict(material_id=mi,path_index=pi,path_id=p["id"],variant_id=p["variant_id"],label=label,begin=offset,end=offset+len(X),color=color))
  xyz.append(X);offset+=len(X)
 for mi,m in enumerate(model["materials"]):
  mapping.append(dict(material_id=mi,display_id=f"m{mi:03d}",label=m["label"],paths=len(m["paths"]),variant_ids=";".join(p["variant_id"]for p in m["paths"]),path_ids=";".join(p["id"]for p in m["paths"])))
  for pi,p in enumerate(m["paths"]):
   kind=p["kind"]
   if kind==2:
    n=len(p["points"]);s0=np.linspace(0,1,n);coarse=scene.world(mi,pi,s0)
    if n==1:s=s0
    else:
     steps=np.maximum(1,np.ceil(np.linalg.norm(np.diff(coarse,axis=0),axis=1)/.0015).astype(int))
     s=np.r_[np.concatenate([np.linspace(s0[j],s0[j+1],int(k),endpoint=False)for j,k in enumerate(steps)]),1.]
   elif kind==0:
    coarse=scene.world(mi,pi,np.array([0.,1.]));n=max(2,int(np.ceil(np.linalg.norm(coarse[1]-coarse[0])/.0015))+1);s=np.linspace(0,1,n)
   elif kind==1:s=np.linspace(0,1,513)
   elif kind==3:s=np.linspace(0,1,2049 if p.get("boundary_segment",-1)<0 else 257)
   else:raise ValueError("Unrendered path kind")
   X=scene.world(mi,pi,s);assert np.isfinite(X).all()
   color=(40,245,70)if ".print."in m["label"]else(30,235,245)if kind in [0,1]else(255,190,30)
   add(X,mi,pi,p,m["label"],color)
 assert len(mapping)==len(model["materials"]) and len(records)==sum(len(m["paths"])for m in model["materials"])
 for h,name in [(0.,"continuous_board_bottom"),(scene.height,"continuous_board_top")]:
  add(scene.board(np.linspace(0,scene.perimeter,4097),h),-1,-1,dict(id=name,variant_id=name),name,(255,160,20))
 ads={}
 for m in model["materials"]:
  if not m["label"].startswith("Ad"):continue
  try:ad=int(m["label"].split(".")[0][2:])
  except ValueError:continue
  for p in m["paths"]:
   if "advertisement_block"in p:ads[ad]=p["advertisement_block"]
 for ad in sorted(ads):
  a=scene.values[ads[ad]];X=scene.board(np.array([a[0]+.5*a[1]]),a[3]);name=f"Ad{ad}"
  members=[(mi,pi)for mi,m in enumerate(model["materials"])if m["label"].startswith(name+".")for pi,pth in enumerate(m["paths"])if pth.get("advertisement_block")==ads[ad]]
  if members and all(model["materials"][mi]["paths"][pi]["kind"]==2 for mi,pi in members):
   # Print-only installation chart is a coordinate witness, not an observed panel box.
   mi,pi=members[0];X=scene.world(mi,pi,np.array([.5]))
  add(X,-2-ad,-1,dict(id="number_"+name,variant_id=name),name,(255,255,255))
 return np.concatenate(xyz),records,mapping,sorted(ads)
def project_RS(camera,X,t):
 # Same original certificate/bracket/optical predicates/root-equality gate.
 with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
  cert=camera.rs_certificate(X,t);pl=camera.at(X,t-camera.tau*.5);pr=camera.at(X,t+camera.tau*.5)
  eligible=cert["certified"]&pl["valid"]&pr["valid"]&(-pl["uv"][:,1]<=0)&(1079-pr["uv"][:,1]>=0)
 ids=np.flatnonzero(eligible);uv=np.full((len(X),2),np.nan);row=np.full(len(X),np.nan);eq=np.full(len(X),np.nan);valid=np.zeros(len(X),bool)
 if not len(ids):return dict(uv=uv,valid=valid,row=row,eq=eq,iterations=0)
 Y=X[ids];left=np.zeros(len(ids));right=np.full(len(ids),1079.);v=np.full(len(ids),539.5);pending=np.ones(len(ids),bool)
 for iteration in range(48):
  j=np.flatnonzero(pending)
  if not len(j):break
  p=camera.at(Y[j],t+camera.tau*(v[j]/1079-.5));f=v[j]-p["uv"][:,1];done=p["valid"]&np.isfinite(f)&(abs(f)<=1e-8)
  z=ids[j[done]];uv[z]=p["uv"][done];row[z]=v[j[done]];eq[z]=f[done];valid[z]=True;pending[j[done]]=False
  k=j[~done];ff=f[~done];pp=p["uv"][~done,1];right[k]=np.where(ff>=0,v[k],right[k]);left[k]=np.where(ff<0,v[k],left[k])
  v[k]=np.where(np.isfinite(pp)&(pp>left[k])&(pp<right[k]),pp,(left[k]+right[k])/2)
 if pending.any():
  k=ids[pending];fb=camera.project_rs(X[k],t);good=fb["valid"];z=k[good];uv[z]=fb["uv"][good];row[z]=fb["row"][good];eq[z]=fb["root_equality_px"][good];valid[z]=True
 assert np.all(abs(eq[valid])<=1e-8)
 return dict(uv=uv,valid=valid,row=row,eq=eq,iterations=iteration+1)
def QA(np_camera,X,t,result):
 uv=result["uv"];inside=result["valid"]&(uv[:,0]>=0)&(uv[:,0]<=1919)&(uv[:,1]>=0)&(uv[:,1]<=1079);ids=np.flatnonzero(inside);assert len(ids)>0
 orig=np_camera.project_rs(X[ids],t);assert np.all(orig["valid"]),"Visible sample fails original NumPy RS"
 diff=float(np.max(abs(orig["uv"]-uv[ids]),initial=0.));assert diff<=1e-6,"Original native implicit projection mismatch"
 return dict(visible_points=len(ids),all_original_NumPy_project_rs_valid=True,max_uv_absolute_difference_px=diff,comparison_gate_px=1e-6,original_root_equality_max_px=float(np.max(abs(orig["root_equality_px"]),initial=0.)),accelerated_root_equality_max_px=float(np.max(abs(result["eq"][ids]),initial=0.)),original_RS_tolerance_px=1e-8,no_physical_or_RS_gate_relaxed=True)
def text(canvas,label,xy,scale=.38,color=(245,245,245)):
 xy=tuple(map(int,xy));cv2.putText(canvas,label,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA);cv2.putText(canvas,label,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
REVIEW_LABEL="I20"
MODEL_MATERIALS=171
MODEL_PATHS=257
def draw(native,projection,records,pts,metrics):
 canvas=native.copy();uv=projection["uv"];valid=projection["valid"];inside=valid&(uv[:,0]>=0)&(uv[:,0]<=1919)&(uv[:,1]>=0)&(uv[:,1]<=1079);labels={};seen=[];adlabels={}
 for r in records:
  a,b=r["begin"],r["end"];ok=valid[a:b];vis=np.flatnonzero(inside[a:b])+a
  if len(vis)and r["material_id"]>=0:labels.setdefault(r["material_id"],vis[len(vis)//2]);seen.append(r["path_id"])
  if len(vis)and r["material_id"]>=0 and r["label"].startswith("Ad"):adlabels.setdefault(r["label"].split(".")[0],vis[0])
  if r["material_id"]<=-2:
   if len(vis):adlabels[r["label"]]=vis[0]
   continue
  for lo,hi in np.flatnonzero(np.diff(np.r_[False,ok,False])).reshape(-1,2):
   points=np.rint(uv[a+lo:a+hi]).astype(np.int32)
   if len(points)>1:cv2.polylines(canvas,[points],False,r["color"],1,cv2.LINE_AA)
   elif len(points)==1 and inside[a+lo]:cv2.circle(canvas,tuple(points[0]),2,r["color"],1,cv2.LINE_AA)
 for mi,j in labels.items():text(canvas,f"m{mi:03d}",uv[j]+np.array([3.,-3.]),.31,(180,255,180))
 for label,j in adlabels.items():text(canvas,label,uv[j]+np.array([8.,18.]),.55,(255,255,255))
 overlay=canvas[:67].copy();overlay[:]=(15,15,15);canvas[:67]=cv2.addWeighted(canvas[:67],.23,overlay,.77,0)
 text(canvas,"NOT ACCEPTED - complete same-model physical fit review",(16,25),.62,(50,70,255))
 text(canvas,f"{REVIEW_LABEL} | max {metrics['maximum_error_px']:.4f}px  median {metrics['quantiles_px'][0]:.4f}px | original target 2px NOT met",(16,51),.43)
 text(canvas,f"source {pts/12800:.3f}s | PTS {pts} | frame {(pts-FIRST)//STEP+1}/1248 | 50fps native | all {MODEL_PATHS} paths enabled",(16,1062),.44,(255,255,255))
 return canvas,dict(visible_model_paths=len(seen),visible_material_ids=sorted(labels),visible_model_path_ids=seen,RS_valid_sample_nodes=int(valid.sum()),inside_sensor_sample_nodes=int(inside.sum()),strict_RS_eq_max_px=float(np.max(abs(projection["eq"][valid]),initial=0.)),root_iterations=projection["iterations"])
def verify_video(path,ffprobe):
 cmd=[ffprobe,"-v","error","-count_frames","-show_entries","stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,time_base,duration:format=duration","-of","json",str(path)]
 p=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8");assert p.returncode==0,p.stderr;probe=json.loads(p.stdout);v=next(x for x in probe["streams"]if x["codec_type"]=="video")
 assert int(v["nb_read_frames"])==NFRAMES and v["width"]==WH[0] and v["height"]==WH[1] and Fraction(v["avg_frame_rate"])==50 and abs(float(v["duration"])-24.96)<1e-6
 pts=[];c=av.open(str(path))
 for f in c.decode(video=0):pts.append(Fraction(f.pts)*f.time_base)
 c.close();assert pts==[Fraction(i,50)for i in range(NFRAMES)],"Encoded clock mismatch"
 return dict(ffprobe_command=cmd,ffprobe=probe,all1248_encoded_frame_times_exact_index_div50=True,decoded_frame_count=len(pts))
def run(a):
 global REVIEW_LABEL,MODEL_MATERIALS,MODEL_PATHS
 started=time.perf_counter();CP=Path(a.checkpoint).resolve();assert sha(CP)==a.checkpoint_sha256;cp=read(CP)
 body=dict(cp)
 if "state_token" in body:
  token=body.pop("state_token");assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()==token
 else:
  assert cp["schema"]=="rinkfit.complete_recomputed_native_candidate.v1"
  assert cp["source_rows"]==182017 and cp["all_frames"]==1248 and cp["complete_state_for_fresh_relinearization"]
  complete=validate_receipt_bindings(cp,CP.parent)
  packet=complete["packet"];expected_raw=dict(packet["arrays"],string_tables=packet["string_tables"])
  assert set(cp["raw_source_arrays_verified_exact"])==set(expected_raw)
  for name,d in expected_raw.items():assert cp["raw_source_arrays_verified_exact"][name]["sha256"]==d["sha256"]
  for key,d in cp["raw_source_arrays_verified_exact"].items():verify(d)
  for key in ["packet","semantic","predictions","parent_complete_checkpoint","proposal"]:verify(cp[key])
  assert set(cp["row_state_files"])=={"station.f64","path.u32","initial_row.f64","initial_alpha.f64"}
  for name,d in cp["row_state_files"].items():
   assert verify(d).stat().st_size==182017*(4 if name=="path.u32" else 8)
 MP=verify(cp["model"]);model=read(MP);assert len(model["blocks"])==1376
 REVIEW_LABEL="I20" if "state_token" in cp else "CANDIDATE "+sha(MP)[:12]
 MODEL_MATERIALS=len(model["materials"]);MODEL_PATHS=sum(len(m["paths"])for m in model["materials"])
 for d in cp["row_state_files"].values():verify(d,CP.parent)
 selection=Path(a.source_selection).resolve();ss=read(selection)["primary_source"];source=Path(ss["path"]);assert ss["native_fps"]=="50/1"and ss["time_base"]=="1/12800"and sha(source)==ss["sha256"]
 D=Path(a.output_dir).resolve();assert "CodexData" in D.parts and "RinkSpace_Data" in D.parts;D.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(Path(a.compiled_adapter_dir).resolve()));from bound_camera_sensor import bind_camera_from_model
 scene=NativeScene(model);camera=bind_camera_from_model(ref(MP),model=model);np_camera=camera_from_event(model);X,records,mapping,adids=inventory(scene,model)
 context=dict(schema="rinkfit.same_complete_native_model_review_context.v1",checkpoint=ref(CP),model=ref(MP),complete4aux=cp["row_state_files"],source_selection=ref(selection),official_source=ss,renderer=ref(__file__),compiled_adapter=ref(Path(a.compiled_adapter_dir)/"bound_camera_sensor.py"),compiled_runtime_recipe=ref(Path(a.compiled_adapter_dir)/"runtime_recipe_sensor.json"),native_scene_source=ref(ROOT/"calibration/rinkfit/src/rinkfit_start/native_scene.py"),native_camera_source=ref(ROOT/"calibration/rinkfit/src/rinkfit_start/native_camera.py"),state_integrity_source=ref(ROOT/"calibration/rinkfit/src/rinkfit_start/state_integrity.py"),all_model_materials=MODEL_MATERIALS,all_model_paths=MODEL_PATHS,all_paths_processed_each_frame=True,world_sample_nodes=len(X),advertisement_ids=adids,continuous_board_top_and_bottom=True,unverified_or_residual_based_geometry_hiding=False,source_PTS=dict(first=FIRST,last=LAST,step=STEP,time_base="1/12800",frames=NFRAMES),model_time="(source_PTS-256256)/12800",metrics=cp["metrics"],review_state_label=REVIEW_LABEL,no_Solve=True,accuracy_accepted=False)
 if not (D/"render_context_final.json").exists():
  write(D/"render_context_final.json",context)
  if not (D/"material_ID_mapping.csv").exists():
   with(D/"material_ID_mapping.csv").open("x",newline="",encoding="utf-8-sig")as f:w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
  if not (D/"all_path_inventory.json").exists():write(D/"all_path_inventory.json",records)
  camera.at.snapshot(D/"camera_snapshot_final",dict(checkpoint=ref(CP)));(D/"code").mkdir(exist_ok=True);shutil.copyfile(__file__,D/"code/review_native_final.py")
 else:
  prev=read(D/"render_context_final.json")
  for k in ["checkpoint","model","official_source","renderer","world_sample_nodes","native_scene_source","native_camera_source","compiled_adapter","compiled_runtime_recipe","state_integrity_source","complete4aux","source_selection","metrics","review_state_label","all_model_materials","all_model_paths"]:assert prev[k]==context[k],"Changed render context "+k
 if a.first_frame_only:
  pts,native=next(source_frames(source));t=(pts-FIRST)/12800;t0=time.perf_counter();projection=project_RS(camera,X,t);projection_seconds=time.perf_counter()-t0;qa=QA(np_camera,X,t,projection);canvas,metrics=draw(native,projection,records,pts,cp["metrics"]);image=D/"first_frame_final_check.png";assert cv2.imwrite(str(image),canvas)
  report=dict(context=ref(D/"render_context_final.json"),source_pts=pts,source_frame_BGR_sha256=hashlib.sha256(native.tobytes()).hexdigest(),image=ref(image),projection_seconds=projection_seconds,whole_stage_seconds=time.perf_counter()-started,RS_QA=qa,draw=metrics,no_video_generated=True)
  write(D/"first_frame_final_check.json",report);print(json.dumps(report),flush=True);camera.at.close();return
 assert (D/"first_frame_final_check.json").exists(),"Run actual first-frame timing/RS QA first"
 stem="shot1_i20_review" if "state_token" in cp else "shot1_candidate_"+sha(MP)[:12]+"_review"
 final=D/(stem+".mp4");silent=D/(stem+".silent.partial.mp4");assert not final.exists()and not silent.exists()
 ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();ffprobe=shutil.which("ffprobe");assert ffprobe
 cmd=[ffmpeg,"-hide_banner","-loglevel","error","-f","rawvideo","-pixel_format","bgr24","-video_size",f"{WH[0]}x{WH[1]}","-framerate","50","-i","pipe:0","-an","-c:v","libx264","-preset","fast","-crf","19","-pix_fmt","yuv420p","-video_track_timescale","12800","-movflags","+faststart",str(silent)]
 ledger=D/"source_PTS_coverage.jsonl";keyframes=[];qas=[];counts=[];count=0;projection_total=0.
 with(D/"encoding.log").open("xb")as log,ledger.open("x",encoding="utf-8")as lf:
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
  try:
   for pts,native in source_frames(source):
    i=(pts-FIRST)//STEP+1;t=(pts-FIRST)/12800;t0=time.perf_counter();projection=project_RS(camera,X,t);projection_total+=time.perf_counter()-t0;canvas,metric=draw(native,projection,records,pts,cp["metrics"])
    if i in [1,625,1248]:
     qa=QA(np_camera,X,t,projection);qa.update(frame_index=i,source_PTS=pts);qas.append(qa);kf=D/f"frame_{i:06d}.png";assert cv2.imwrite(str(kf),canvas);keyframes.append(ref(kf))
    proc.stdin.write(memoryview(canvas).cast("B"));count+=1
    lf.write(json.dumps(dict(index=i,native_pts=pts,source_time_base="1/12800",source_time_seconds=pts/12800,model_time_seconds=t,source_BGR_sha256=hashlib.sha256(native.tobytes()).hexdigest(),same_model_sha256=context["model"]["sha256"],all_model_materials=MODEL_MATERIALS,all_model_paths_enabled=MODEL_PATHS,**metric))+"\n");counts.append(metric["visible_model_paths"])
    if i%50==0 or i==1:
     camera.at.assert_bound_model(ref(MP));print(json.dumps(dict(rendered_frames=count,total=1248,source_PTS=pts,elapsed_seconds=time.perf_counter()-started,projection_seconds=projection_total)),flush=True)
   proc.stdin.close();assert proc.wait()==0,"Encoder failed"
  except BaseException:
   if proc.poll()is None:proc.kill();proc.wait()
   raise
 assert count==NFRAMES
 audio=av.open(str(source));has_audio=bool(audio.streams.audio);audio.close();mux=[ffmpeg,"-hide_banner","-loglevel","error","-i",str(silent)]
 if has_audio:mux+=["-ss",str(FIRST/12800),"-t","24.96","-i",str(source),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","160k","-t","24.96"]
 else:mux+=["-map","0:v:0","-c:v","copy"]
 mux+=["-video_track_timescale","12800","-movflags","+faststart",str(final)];mr=subprocess.run(mux,capture_output=True,text=True,encoding="utf-8");assert mr.returncode==0,mr.stderr
 verification=verify_video(final,ffprobe);actualpts=[json.loads(s)["native_pts"]for s in ledger.read_text("utf-8").splitlines()];assert actualpts==list(range(FIRST,LAST+STEP,STEP));camera.at.assert_bound_model(ref(MP))
 manifest=dict(schema="rinkfit.complete_first_shot_same_native_model_review_video.v1",status="complete_NOT_ACCEPTED",context=ref(D/"render_context_final.json"),checkpoint=ref(CP),model=ref(MP),video=ref(final),video_bytes=final.stat().st_size,source=ss,frames=NFRAMES,source_PTS_first=FIRST,source_PTS_last=LAST,source_PTS_end_exclusive=LAST+STEP,source_time_base="1/12800",source_PTS_coverage=ref(ledger),source_PTS_coverage_exactly_all1248_once=True,source_PTS_array_sha256=hashlib.sha256(np.asarray(actualpts,dtype="<i8").tobytes()).hexdigest(),video_size_wh=list(WH),FPS="50/1",duration_seconds=24.96,all_model_paths_enabled=MODEL_PATHS,all_material_labels=MODEL_MATERIALS,all_path_inventory=ref(D/"all_path_inventory.json"),material_ID_mapping=ref(D/"material_ID_mapping.csv"),continuous_board_top_and_bottom_enabled=True,advertisement_numbers_enabled=adids,visible_path_count_range=[min(counts),max(counts)],strict_RS_no_tolerance_relaxation=True,accelerated_fixedpoint_with_original_certificate_bracket_equality_and_original_bisection_fallback=True,independent_original_NumPy_project_rs_all_visible_sample_nodes_on_keyframes=qas,keyframes=keyframes,ffprobe_and_decoded_clock_verification=verification,encoding_command=cmd,audio_mux_command=mux,original_source_audio_included=has_audio,projection_seconds=projection_total,elapsed_seconds=time.perf_counter()-started,no_Solve=True,source_or_model_or_auxiliary_or_W_E_modified=False,accuracy_accepted=False,full_visual_accuracy_acceptance=False,limitations=["Finite sampled rendering of the existing exact model; all geometry is shown where original optical/RS/sensor domain permits projection. No new fit or accuracy certificate.","Original2px and nonlinear KKT have not passed; NOT ACCEPTED watermark is permanent."])
 write(D/"manifest.json",manifest);print(json.dumps(dict(manifest=ref(D/"manifest.json"),video=ref(final),frames=count,elapsed_seconds=manifest["elapsed_seconds"])),flush=True);camera.at.close()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--checkpoint-sha256",required=True);p.add_argument("--source-selection",required=True);p.add_argument("--compiled-adapter-dir",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--first-frame-only",action="store_true");run(p.parse_args())
if __name__=="__main__":main()
