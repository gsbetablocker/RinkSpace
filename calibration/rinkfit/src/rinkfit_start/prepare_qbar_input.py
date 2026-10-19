from pathlib import Path
import argparse,sys,json,hashlib,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rinkfit_start.native_scene import NativeScene
from rinkfit_start.semantic_resolution import load_semantics
ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--initial-dir",required=True);ap.add_argument("--semantic-manifest",required=True);ap.add_argument("--packet",required=True);ap.add_argument("--output",required=True);ap.add_argument("--master-fraction",type=float,default=0.);a=ap.parse_args();D=Path(a.output)
if D.exists()and any(D.iterdir()):raise ValueError("Refuse prior input overwrite")
D.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();M=Path(a.model);SM=Path(a.semantic_manifest);PK=Path(a.packet);C=Path(a.initial_dir);sem=json.loads(SM.read_text());scene=NativeScene(json.loads(M.read_text()))
def load(d):
 p=SM.parent/d["file"];assert sha(p)==d["sha256"];return np.fromfile(p,d["dtype"]).reshape(d["shape"])
_,semantic_arrays,_,extra=load_semantics(SM,int(sem["observations"]),expected_sha256=sha(SM));k=semantic_arrays["kind"];active=load(sem["active_source_mask"]).astype(bool);m=extra["resolved_material_index"] if "resolved_material_index"in extra else load(sem["resolved_material_index"]);s=np.fromfile(C/"station.f64","<f8");b=np.fromfile(C/"path.u32","<u4");v=np.fromfile(C/"initial_row.f64","<f8");mode=np.zeros(len(s),np.uint8)
for i in np.flatnonzero(active&(k==0)):
 p=scene.path(int(m[i]),int(b[i]));edge=max(1,len(p.get("points",[]))-1)
 # Physical open endpoints have a one-sided normal cone, not two segments.
 if not p.get("closed",False)and(s[i]<1e-10 or s[i]>1-1e-10):mode[i]=2
 elif p["kind"]==2 and abs(s[i]*edge-round(s[i]*edge))<1e-9:mode[i]=1
 if v[i]>1079-1e-7:mode[i]=7 if mode[i]else 3
 elif v[i]<1e-7:mode[i]=7 if mode[i]else 4
mode[k==1]=5;mode[k==2]=6;mode[~active]=255;mode.tofile(D/"mode.u8")
def desc(p,dtype):return dict(file=str(p),dtype=dtype,shape=[len(s)],sha256=sha(p))
arrays={key:desc(C/name,dtype)for key,name,dtype in [("station","station.f64","<f8"),("path","path.u32","<u4"),("row","initial_row.f64","<f8"),("alpha","initial_alpha.f64","<f8")]};arrays["mode"]=desc(D/"mode.u8","|u1")
x=dict(schema="rinkfit.qbar_sparse_linearization_input.v1",packet=dict(path=str(PK),sha256=sha(PK)),model=dict(path=str(M),sha256=sha(M)),semantic=dict(path=str(SM),sha256=sha(SM)),arrays=arrays,raw_rows=len(s),active_rows=int(active.sum()),no_Solve=True,purpose="Full reviewed v2 source; exact station chart/refinement status is mandatory for any envelope Jacobian.")
if a.master_fraction:
 assert 0<a.master_fraction<1
 pred=np.load(C/"candidate_predictions.npz");pk=json.loads(PK.read_text());bd=pk["arrays"]["engineering_budget_px"];budget=np.fromfile(PK.parent/bd["file"],bd["dtype"]);q=(pred["coverage_error_px"]/budget)**2;assert np.isfinite(q[active]).all();q.astype("<f8").tofile(D/"full_source_q.f64");selected=np.flatnonzero(active&(q>=a.master_fraction*q[active].max())).astype("<u8");selected.tofile(D/"selected_source_rows.u64")
 x["source_rows"]=dict(file=str(D/"selected_source_rows.u64"),dtype="<u8",shape=[len(selected)],sha256=sha(D/"selected_source_rows.u64"));x["full_source_q"]=desc(D/"full_source_q.f64","<f8");x["active_source_rows_full"]=np.flatnonzero(active).tolist();x["constraint_generation"]=dict(selection="q >= fraction * full-active current maxq",fraction=a.master_fraction,unmodeled_rows_are_not_zero_gradient=True,unmodeled_current_q_is_epigraph_constant_floor=True,all_active_actual_trial_required=True)
(D/"manifest.json").write_text(json.dumps(x,indent=2));print(json.dumps(dict(manifest=str(D/"manifest.json"),sha256=sha(D/"manifest.json"),mode_counts={str(z):int((mode==z).sum())for z in np.unique(mode)})),flush=True)
