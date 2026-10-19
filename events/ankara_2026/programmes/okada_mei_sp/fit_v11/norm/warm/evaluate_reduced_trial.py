from pathlib import Path
import sys,json,hashlib,time,copy,argparse,numpy as np
sys.stdout.reconfigure(encoding='utf-8');sys.path.insert(0,'D:/CodexData/RinkSpace_Data/runs/okada_s1_r11/physical_reduced_component262_rf29_r11_transport/python_frozen')
from rinkfit_start.native_scene import NativeScene,camera_from_event
from rinkfit_start.material_capture import capture_finite_arrays
from rinkfit_start.material_foot import _certified_candidates,_implicit_capture,_true_rs_grid_recovery,select_certified_upper_bound
from rinkfit_start.native_query import recover_query_roots,rational_query
_CODE_SUPPORT=Path(__file__).parent
if not (_CODE_SUPPORT/'batched_grid_4k.py').exists():_CODE_SUPPORT=Path(r'D:\CodexData\RinkSpace_Data\runs\okada_s1_r11\model_checks\finite_visible_line_bundle_v4_1\code')
sys.path.insert(0,str(_CODE_SUPPORT));sys.path.insert(0,str(Path(__file__).parent));from batched_grid_4k import _true_rs_grid_recovery,BATCH_STATS
_COMPILED_SUPPORT=_CODE_SUPPORT/'compiled_camera'
sys.path.insert(0,str(_COMPILED_SUPPORT));from compiled_camera_guarded_sensor import _STATS as CAMERA_STATS;from bound_camera_sensor import bind_camera_from_model
from warm_all_ordinary_RS import refine_all_ordinary
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--initial-dir',required=True);ap.add_argument('--semantic-manifest',required=True);ap.add_argument('--output',required=True);ap.add_argument('--review-proof');args=ap.parse_args()
R=Path('D:/CodexData/RinkSpace_Data/runs/okada_s1_r11');M=Path(args.model);BASE=Path(args.initial_dir);SP=Path(args.semantic_manifest);OUT=Path(args.output)
if OUT.exists()and any(OUT.iterdir()):raise ValueError('Refuse overwriting previous reduced initializer')
OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();model=json.loads(M.read_text());sem=json.loads(SP.read_text());scene=NativeScene(model);camera=bind_camera_from_model(dict(path=str(M),sha256=sha(M)),model=model)
# The GPU policy changes only camera.at dispatch; final native CPU/RS/Q/scene gates stay in this evaluator.
import importlib.util as _gpu_import
_gpu_path=Path(__file__).parent/'gpu/camera.py'
if not _gpu_path.exists():_gpu_path=Path(__file__).resolve().parents[2]/'gpu/camera.py'
_gpu_spec=_gpu_import.spec_from_file_location('event_GPU_camera',_gpu_path);_gpu_module=_gpu_import.module_from_spec(_gpu_spec);_gpu_spec.loader.exec_module(_gpu_module)
camera=_gpu_module.install(camera,dict(path=str(M),sha256=sha(M)),OUT)
camera.at.snapshot(OUT/'camera_snapshot',dict(model=dict(path=str(M),sha256=sha(M)),semantic=dict(path=str(SP),sha256=sha(SP)),initial_state={name:dict(path=str(BASE/name),sha256=sha(BASE/name))for name in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']},physical_blocks_values_bounds_scales_and_gauge_sha256=hashlib.sha256(json.dumps(model['blocks'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),compiled_recipe=dict(path=str(_COMPILED_SUPPORT/'runtime_recipe_sensor.json'),sha256=sha(_COMPILED_SUPPORT/'runtime_recipe_sensor.json'))))
def array(d,base=SP.parent):
 p=base/d['file'];assert sha(p)==d['sha256'];return np.fromfile(p,d['dtype']).reshape(d['shape'])
N=182017;uv=np.fromfile(R/'backend_input/uv.f64','<f8').reshape(N,2);pts=np.fromfile(R/'backend_input/pts.i64','<i8');frames=np.fromfile(R/'backend_input/frame.i32','<i4');times=(pts-256256)/12800.;budget=np.fromfile(R/'backend_input/budget.f64','<f8');kind=array(sem['arrays']['kind']);normals=array(sem['arrays']['profile_normal']);active=array(sem['active_source_mask']).astype(bool);material=array(sem['resolved_material_index']).copy();variants=array(sem['resolved_variant_index']).copy();vids=sem['resolved_variant_ids'];expected=np.array(vids)[variants];oldmaterial=np.fromfile(R/'semantic_true_order'/json.loads((R/'semantic_true_order/manifest.json').read_text())['resolved_material_index']['file'],'<u4')
s0=np.fromfile(BASE/'station.f64','<f8');b0=np.fromfile(BASE/'path.u32','<u4');v0=np.fromfile(BASE/'initial_row.f64','<f8');a0=np.fromfile(BASE/'initial_alpha.f64','<f8');ordinary=active&(kind==0);profile=np.flatnonzero(active&(kind==2));corners=active&(kind==1)
assert N==len(s0)==len(b0)==len(v0)==len(a0) and active.sum()==182016 and len(profile)==961
preview=None
if args.review_proof:
 proofpath=Path(args.review_proof);proof=json.loads(proofpath.read_text());preview=dict(path=str(proofpath),sha256=sha(proofpath),authority='explicit Data preview of frozen reviewed generator facts; not a new semantic acceptance')
 for rec in proof['bindings']:
  i=rec['source_row'];material[i]=rec['resolved_model_material_index'];expected[i]=rec['resolved_variant_id']
for i in range(N):
 if scene.model['materials'][int(material[i])]['paths'][int(b0[i])]['variant_id']!=expected[i]:
  legal=[b for b,p in enumerate(scene.model['materials'][int(material[i])]['paths'])if p['variant_id']==expected[i]]
  if not legal:raise ValueError('Resolved variant absent from shared model')
  b0[i]=legal[0]
changed=np.flatnonzero(material!=oldmaterial)
seed_identity=BASE/'initializer_manifest.json'
if seed_identity.exists():
 seed_doc=json.loads(seed_identity.read_text());seed_sem=seed_doc.get('semantic_manifest',seed_doc.get('semantic',{}));assert seed_sem.get('sha256')==sha(SP),'Complete seed semantics differs from selected source authority'
 for name in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']:
  if name in seed_doc.get('files',{}):assert sha(BASE/name)==seed_doc['files'][name]['sha256']
else:
 seed_doc=json.loads((BASE/'report.json').read_text());assert seed_doc['semantic']['sha256']==sha(SP),'Complete candidate seed semantics differs'
active_runs=SP.parent/sem['assets']['source_runs.jsonl']['file']
assert not active_runs.read_text().strip(),'This independent per-row inner mode requires verified hard-order count zero'
if False: # Caller must supply a complete seed bound to the exact same semantic manifest.
 cap=capture_finite_arrays(scene,camera,uv[changed],frames[changed],material[changed],kind[changed],np.array([vids.index(v)for v in expected[changed]]),vids,s0[changed],b0[changed],progress=None);s0[changed]=cap['stations'];b0[changed]=cap['branches'];s0[changed]=_implicit_capture(scene,camera,material[changed],b0[changed],s0[changed],times[changed],uv[changed],iterations=20);rr=_certified_candidates(scene,camera,material[changed],b0[changed],s0[changed],times[changed]);_true_rs_grid_recovery(scene,camera,material[changed],frames[changed],times[changed],uv[changed],np.ones(len(changed),bool),expected[changed],s0[changed],b0[changed],rr,same_branch_only=False,threshold_px=4.,grid_count=1025,progress=None);v0[changed]=rr['row']
 print('RECOVERED_REVIEWED_BINDINGS',len(changed),flush=True)
qm=json.loads((R/'query_packet/manifest.json').read_text());query={k:array(d,R/'query_packet')for k,d in qm['arrays'].items()};assert np.array_equal(query['source_rows'],profile)
weights=array(sem['arrays']['weights']);members=array(sem['arrays']['group_rows']);sigma=array(sem['arrays']['source_sigma']);independent=array(sem['arrays']['ordinary_rows']).astype(int);independent=independent[active[independent]];statref=SP.parent/sem['active_statistics']['file'];stats=json.loads(statref.read_text());replacement={x['group_id']:(np.array(x['source_rows']),array(x['W'],statref.parent))for x in stats['replacements']};units=[];partition=np.zeros(N,int);rank=0
for line in (SP.parent/sem['assets']['groups.jsonl']['file']).read_text().splitlines():
 g=json.loads(line);rr=members[g['member_start']:g['member_start']+g['member_count']].astype(int);W=weights[g['weight_start']:g['weight_start']+g['rank']*2*len(rr)].reshape(g['rank'],2*len(rr))
 if g['group_id']in replacement:rr,W=replacement[g['group_id']]
 units.append((rr,W));partition[rr]+=1;rank+=len(W)
partition[independent]+=1;rank+=2*len(independent);assert np.array_equal(partition,active.astype(int))and rank==214112
pp0=np.array(camera.optics['principal_point']);visibility_seed=ordinary&((v0<1e-5)|(v0>1079-1e-5))
def coverage(e):
 c=np.linalg.norm(e,axis=1);c[profile]=abs(np.einsum('ij,ij->i',normals[profile],e[profile]));return c
# This resolves actual image visibility endpoints; it does not clamp S/v.
def visibility_capture(ids,s,b):
 for i in ids:
  m=int(material[i]);bi=int(b[i]);center=s[i];path=scene.path(m,bi);layout_time=times[i];X=scene.world(m,bi,np.array([center]));bottom=camera.at(X,[layout_time+camera.tau*.5])['uv'][0,1];vb=1079. if v0[i]>539.5 or bottom>1079 else 0.;time_bound=layout_time+camera.tau*(vb/1079-.5)
  def G(z):return vb-float(camera.at(scene.world(m,bi,np.array([z])),[time_bound])['uv'][0,1])
  edges=max(1,len(path.get('points',[]))-1);lo=max(0.,center-.05);hi=min(1.,center+.05);knots=np.arange(edges+1)/edges if path['kind']==2 else np.array([]);nodes=np.unique(np.r_[np.linspace(lo,hi,65),center,knots[(knots>=lo)&(knots<=hi)]]);xx=scene.world(m,bi,nodes);vals=vb-camera.at(xx,np.full(len(nodes),time_bound))['uv'][:,1];brackets=np.flatnonzero(vals[:-1]*vals[1:]<=0);candidates=[]
  for edge in brackets:
   a,bound=nodes[edge:edge+2];fa=vals[edge]
   for _ in range(56):
    middle=(a+bound)*.5;fm=G(middle)
    if fa*fm<=0:bound=middle
    else:a=middle;fa=fm
   candidates.extend([a,bound,np.nextafter(a,-np.inf),np.nextafter(bound,np.inf)])
  if not candidates:continue
  candidates=np.array(candidates);test=_certified_candidates(scene,camera,np.full(len(candidates),m),np.full(len(candidates),bi),candidates,np.full(len(candidates),layout_time));legal=np.flatnonzero(test['valid'])
  if len(legal):s[i]=candidates[legal[np.argmin(np.sum((test['uv'][legal]-uv[i])**2,axis=1))]]

def evaluate(delta,label):
 begin=time.monotonic();camera.optics['principal_point']=pp0+np.array(delta);s=s0.copy();b=b0.copy();alpha=a0.copy();r=_certified_candidates(scene,camera,material,b,s,times);initial_s=s.copy();initial_b=b.copy();initial_r={k:v.copy()for k,v in r.items()if isinstance(v,np.ndarray)and v.shape[:1]==(N,)};invalid=ordinary&~r['valid'];visibility_capture(np.flatnonzero(visibility_seed|invalid),s,b);r=_certified_candidates(scene,camera,material,b,s,times)
 qr=recover_query_roots(scene,camera,material[profile],b[profile],s[profile],times[profile],query,alpha_seed=alpha[profile]);s[profile]=qr['stations'];alpha[profile]=qr['alpha'];r=_certified_candidates(scene,camera,material,b,s,times)
 if not qr['valid'].all():return None,dict(label=label,delta=list(delta),valid=False,reason='genuine finite query root unresolved',rows=profile[~qr['valid']].tolist())
 # Every ordinary source receives current-theta local true-RS refinement.
 # The original 8px value is grid scheduling only; all source gates remain.
 warm=refine_all_ordinary(scene,camera,material,b,s,r['row'],times,uv,active,kind,model_reference=dict(path=str(M),sha256=sha(M)))
 s=warm['station'];b=warm['branch'];wi=warm['source_rows'];wr=warm['ordinary_result']
 for k,v in r.items():
  if isinstance(v,np.ndarray)and v.shape[:1]==(N,)and k in wr:v[wi]=wr[k]
 grid_report=_true_rs_grid_recovery(scene,camera,material,frames,times,uv,ordinary,expected,s,b,r,same_branch_only=False,threshold_px=8.,grid_count=1025,progress=None)
 # Offer the original same-theta legal ordinary witness again, including
 # before legacy visibility recovery. Never discard a better lawful witness.
 oi=np.flatnonzero(ordinary);saved={k:v[oi]for k,v in initial_r.items()}
 current={k:v[oi]for k,v in r.items()if isinstance(v,np.ndarray)and v.shape[:1]==(N,)}
 ss,bb,chosen,take=select_certified_upper_bound(initial_s[oi],initial_b[oi],saved,s[oi],b[oi],current,uv[oi])
 initial_witness_restored=int(np.count_nonzero(initial_r['valid'][oi]&(~take)))
 s[oi]=ss;b[oi]=bb
 for k,v in r.items():
  if isinstance(v,np.ndarray)and v.shape[:1]==(N,)and k in chosen:v[oi]=chosen[k]
 if not r['valid'][active].all():return None,dict(label=label,delta=list(delta),valid=False,reason='strict full-source RS unresolved after all-ordinary warm and mandatory invalid grid',rows=np.flatnonzero(active&~r['valid']).tolist(),all_ordinary_warm=warm['report'],grid=grid_report)
 e=r['uv']-uv;c=coverage(e);q=(c/budget)**2;optimized=ordinary.copy()
 iterations=[dict(iteration=0,optimized_rows=int(optimized.sum()),newly_active=0,max_q=float(q[active].max()),all_ordinary_warm=warm['report'],grid=grid_report,initial_same_theta_witness_retained=True,initial_witness_retained_count=initial_witness_restored)]
 W=sum(.5*float((W@e[rr].reshape(-1))@(W@e[rr].reshape(-1)))for rr,W in units)+.5*float(np.sum((e[independent]/sigma[independent,None])**2));q_uv,_,_=rational_query(query,alpha[profile]);qe=np.linalg.norm(r['uv'][profile]-q_uv,axis=1)
 result=dict(label=label,delta=list(delta),valid=bool(r['valid'][active].all()),active_RS_valid_rows=int(r['valid'][active].sum()),max_q=float(q[active].max()),maximum_error_px=float(c[active].max()),quantiles_px=np.quantile(c[active],[.5,.95,.99,1]).tolist(),over2=int(np.count_nonzero(active&(c>2))),whole_original_W_half_cost=W,max_query_equality_px=float(qe.max()),query_numerical_roots=int(qr['valid'].sum()),profile_global_interval_certified=False,locally_captured_ordinary_candidates=int(optimized.sum()),inactive_under_current_worst_bound_rows=int(np.count_nonzero(ordinary&~optimized)),inner_active_set_history=iterations,maximum_source_row=int(np.flatnonzero(active)[np.argmax(q[active])]),elapsed_seconds=time.monotonic()-begin)
 print('FULL_N_TRIAL',json.dumps(result),flush=True);return dict(station=s,branch=b,row=r['row'],alpha=alpha,uv=r['uv'],valid=r['valid'],q=q,coverage=c,error=e),result
history=[];best=None;bestkey=None
for delta in [(0.,0.)]:
 data,rec=evaluate(delta,str(len(history)));history.append(rec);(OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
 if data is not None:
  key=(rec['max_q'],rec['whole_original_W_half_cost'])
  if bestkey is None or key<bestkey:bestkey=key;best=(data,rec)
if best is None:raise ValueError('No all-active legally projected candidate')
data,rec=best;data['row'][~active]=v0[~active];candidate=copy.deepcopy(model);candidate['blocks'][0]['values'][20:22]=(pp0+rec['delta']).tolist();candidate['reduced_initializer_provenance']=dict(parent_model=str(M),parent_sha256=sha(M),semantic=str(SP),semantic_sha256=sha(SP),review_preview=preview,physical_parameter_columns=[],not_formal_final_acceptance=True)
(OUT/'model.json').write_text(json.dumps(candidate,indent=2),encoding='utf-8')
for name,key,dtype in [('station.f64','station','<f8'),('path.u32','branch','<u4'),('initial_row.f64','row','<f8'),('initial_alpha.f64','alpha','<f8')]:np.asarray(data[key],dtype=dtype).tofile(OUT/name)
np.savez_compressed(OUT/'candidate_predictions.npz',source_rows=np.arange(N),native_pts=pts,active_source_mask=active,uv=data['uv'],stations=data['station'],branches=data['branch'],root_rows=data['row'],profile_alpha=data['alpha'],error_native_uv=data['error'],coverage_error_px=data['coverage'],RS_valid=data['valid'])
report=dict(schema='rinkfit.full_active_finite_RS_inner_trial.v1',initialization_only=True,final_physical_fit_accepted=False,raw_rows=N,active_rows=int(active.sum()),frames=1248,raw_material_labels=171,source_W_E_unchanged=True,statistics_rank=rank,partition_exactly_once=True,model_parent=dict(path=str(M),sha256=sha(M)),semantic=dict(path=str(SP),sha256=sha(SP)),source_generator_preview=preview,only_physical_columns_changed=[],reviewed_rebound_rows=changed.tolist(),best=rec,history=history,PhaseI_reduced_constrained_target_definition_equivalent=True,exact_inner_KKT_certified=False,old_finite_rho_lifted_merit_identical_claimed=False,PhaseII_full_W_station_elimination_solved=False,profile_implicit_root_numerical_only=True,final_profile_global_KKT_pending=True,global_station_optimum_certified=False,finite_RS_nearest_seed_and_local_KKT_charts=True,whole_static_scene=scene.check_physics(),files={p.name:sha(p)for p in OUT.iterdir()if p.is_file()},elapsed_seconds=time.monotonic()-start,CPU_grid_batch=BATCH_STATS,CPU_compiled_camera=CAMERA_STATS,GPU_camera=getattr(camera,'gpu_acceleration',dict(enabled=False)),GPU_camera_calls=getattr(camera.at,'stats',{}),compiled_camera_snapshot=dict(path=str(OUT/'camera_snapshot/manifest.json'),sha256=sha(OUT/'camera_snapshot/manifest.json')))
(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('FULL_N_REDUCED_DONE',str(OUT/'report.json'),flush=True)
