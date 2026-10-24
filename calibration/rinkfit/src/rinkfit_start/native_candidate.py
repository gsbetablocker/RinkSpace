"""Complete fixed-physical-model native RS/station/query recapture and original W audit.

Initial implementation frozen in Data, SHA256 90e232681a58b021d3f89dc3939d6c5d9cdd2a8da931aa4bc22313cd50e1f9bd.
This recomputes the selected source authority; it does not certify global station/KKT
optimality or accept a physical fit. Compiled/GPU camera adapters are explicit inputs.
"""
from pathlib import Path
import sys,json,hashlib,time,copy,argparse,numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from rinkfit_start.native_scene import NativeScene,camera_from_event
from rinkfit_start.material_capture import capture_finite_arrays
from rinkfit_start.material_foot import _certified_candidates,_implicit_capture,_true_rs_grid_recovery,select_certified_upper_bound
from rinkfit_start.native_query import recover_query_roots,rational_query
from rinkfit_start.batched_grid import _true_rs_grid_recovery,BATCH_STATS
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--initial-dir',required=True);ap.add_argument('--semantic-manifest',required=True);ap.add_argument('--output',required=True);ap.add_argument('--run',required=True);ap.add_argument('--packet',required=True);ap.add_argument('--compiled-dir',required=True);ap.add_argument('--gpu-camera');ap.add_argument('--recapture-material',action='append',default=[],help='Force all ordinary rows of each exact material label through all legal finite branches, including new source-contour revisions');args=ap.parse_args(argv)
 COMPILED=Path(args.compiled_dir);sys.path.insert(0,str(COMPILED));from compiled_camera_guarded_sensor import _STATS as CAMERA_STATS;from bound_camera_sensor import bind_camera_from_model
 R=Path(args.run);PK=Path(args.packet);packet=json.loads(PK.read_text(encoding='utf-8'));M=Path(args.model);BASE=Path(args.initial_dir);SP=Path(args.semantic_manifest);OUT=Path(args.output)
 if OUT.exists()and any(OUT.iterdir()):raise ValueError('Refuse overwriting an existing complete evaluation')
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();model=json.loads(M.read_text(encoding='utf-8'));sem=json.loads(SP.read_text(encoding='utf-8'));scene=NativeScene(model);camera=bind_camera_from_model(dict(path=str(M),sha256=sha(M)),model=model)
 if args.gpu_camera:
  import importlib.util
  gs=importlib.util.spec_from_file_location('candidate_gpu_camera',args.gpu_camera);gm=importlib.util.module_from_spec(gs);gs.loader.exec_module(gm);camera=gm.install(camera,dict(path=str(M),sha256=sha(M)),OUT)
 print('GPU_CAMERA',getattr(camera,'gpu_acceleration',{}),flush=True)
 camera.at.snapshot(OUT/'camera_snapshot',dict(model=dict(path=str(M),sha256=sha(M)),semantic=dict(path=str(SP),sha256=sha(SP)),initial_state={name:dict(path=str(BASE/name),sha256=sha(BASE/name))for name in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']},physical_blocks_values_bounds_scales_and_gauge_sha256=hashlib.sha256(json.dumps(model['blocks'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),compiled_recipe=dict(path=str((COMPILED/'runtime_recipe_sensor.json')),sha256=sha((COMPILED/'runtime_recipe_sensor.json')))))
 def array(d,base=SP.parent):
  p=base/d['file'];assert sha(p)==d['sha256'];return np.fromfile(p,d['dtype']).reshape(d['shape'])
 N=int(packet['observations']);raw={name:array(d,PK.parent)for name,d in packet['arrays'].items()};assert tuple(packet['native_image_wh'])==(1920,1080),'Current camera/RS adapter supports the native 1920x1080 sensor';assert np.array_equal(raw['source_rows'],np.arange(N));uv=raw['native_uv'];pts=raw['native_pts'];frames=raw['frame_index'];time_base=float(packet['native_time_base'][0])/float(packet['native_time_base'][1]);times=(pts-int(packet['pts_origin']))*time_base;budget=raw['engineering_budget_px']
 from rinkfit_start.semantic_resolution import load_semantics
 _,semantic_arrays,_,validated=load_semantics(SP,N,expected_sha256=sha(SP))
 kind=semantic_arrays['kind'];normals=semantic_arrays['profile_normal'];active=np.asarray(validated['source_exclusion']['active_mask'],bool);material=np.asarray(validated['resolved_material_index'],int).copy();variants=np.asarray(validated['resolved_variant_index'],int).copy();vids=validated['resolved_variant_ids'];expected=np.array(vids)[variants]
 assert validated['source_identity']==model['source_identity']==packet['immutable_source_hashes']['observations.npz'],'Selected semantic/model/native packet source identity differs'
 table_path=PK.parent/packet['string_tables']['file'];assert sha(table_path)==packet['string_tables']['sha256'];tables=json.loads(table_path.read_text(encoding='utf-8'));signatures=sem['immutable_source_arrays'];source_values={'source_uv':uv,'native_pts':pts,'frame_indices':frames}
 for key,table_key,index_key in [('source_point_ids','source_point_ids','point_id_index'),('source_target_labels','material_labels','material_index')]:source_values[key]=np.asarray(tables[table_key],dtype=signatures[key]['dtype_str'])[raw[index_key]]
 for key,value in source_values.items():
  identity=signatures[key]
  assert list(value.shape)==identity['shape']and value.dtype.str==identity['dtype_str']and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==identity['sha256_c_contiguous_array_bytes'],'Native packet differs from immutable semantic source array: '+key
 s0=np.fromfile(BASE/'station.f64','<f8');b0=np.fromfile(BASE/'path.u32','<u4');v0=np.fromfile(BASE/'initial_row.f64','<f8');a0=np.fromfile(BASE/'initial_alpha.f64','<f8');ordinary=active&(kind==0);profile=np.flatnonzero(active&(kind==2));corners=active&(kind==1)
 assert N==len(s0)==len(b0)==len(v0)==len(a0)==len(active)==len(material);assert np.isfinite(budget).all()and(budget>0).all()
 preview=None
 labels=np.array([scene.model['materials'][int(m)]['label']for m in material]);forced=ordinary&np.isin(labels,args.recapture_material)
 missing=set(args.recapture_material)-set(labels.tolist())
 if missing:raise ValueError('Requested finite recapture material is absent: '+str(sorted(missing)))
 branch_legal=np.array([0<=int(b0[i])<len(scene.model['materials'][int(material[i])]['paths'])for i in range(N)])
 unseeded=(~np.isfinite(s0))|(s0<0)|(s0>1)|~branch_legal
 if np.any(unseeded&~forced):raise ValueError('Illegal station/branch seed outside explicitly recaptured ordinary material rows: '+str(np.flatnonzero(unseeded&~forced).tolist()))
 for i in np.flatnonzero(unseeded):
  legal=[b for b,p in enumerate(scene.model['materials'][int(material[i])]['paths'])if p['variant_id']==expected[i]]
  if not legal:raise ValueError('No legal source variant for unseeded row')
  b0[i]=legal[0];s0[i]=.5;v0[i]=uv[i,1]
 # The value .5 is only a logged legal search seed for an explicitly unseeded row;
 # it does not transport/clamp the removed source segment or create an observation.
 for i in range(N):
  if scene.model['materials'][int(material[i])]['paths'][int(b0[i])]['variant_id']!=expected[i]:
   legal=[b for b,p in enumerate(scene.model['materials'][int(material[i])]['paths'])if p['variant_id']==expected[i]]
   if not legal:raise ValueError('Resolved variant absent from shared model')
   b0[i]=legal[0]
 changed=np.array([],dtype=int)
 seed_identity=BASE/'initializer_manifest.json'
 if seed_identity.exists():
  seed_doc=json.loads(seed_identity.read_text());seed_sem=seed_doc.get('semantic_manifest',seed_doc.get('semantic',{}));assert seed_sem.get('sha256')==sha(SP),'Complete seed semantics differs from selected source authority'
  for name in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']:
   if name in seed_doc.get('files',{}):assert sha(BASE/name)==seed_doc['files'][name]['sha256']
 else:
  seed_doc=json.loads((BASE/'report.json').read_text());assert seed_doc['semantic']['sha256']==sha(SP),'Complete candidate seed semantics differs'
 active_runs=SP.parent/sem['assets']['source_runs.jsonl']['file'];assert sha(active_runs)==sem['assets']['source_runs.jsonl']['sha256']
 assert not active_runs.read_text().strip(),'This independent per-row inner mode requires verified hard-order count zero'
 QP=Path(packet['query_packet']['manifest']);assert sha(QP)==packet['query_packet']['sha256'];qm=json.loads(QP.read_text(encoding='utf-8'));query={k:array(d,QP.parent)for k,d in qm['arrays'].items()};assert np.array_equal(query['source_rows'],profile)
 weights=semantic_arrays['weights'];members=semantic_arrays['group_rows'];sigma=semantic_arrays['source_sigma'];independent=semantic_arrays['ordinary_rows'].astype(int);independent=independent[active[independent]];statref=SP.parent/sem['active_statistics']['file'];assert sha(statref)==sem['active_statistics']['sha256'];stats=json.loads(statref.read_text(encoding='utf-8'));assert int(stats['active_observations'])==int(active.sum());replacement={x['group_id']:(np.array(x['source_rows']),array(x['W'],statref.parent))for x in stats['replacements']}
 if 'source_audit_revision'in sem:
  replacement={gid:(np.asarray(x['source_rows'],int),x['W'])for gid,x in validated['source_exclusion']['replacements'].items()}
 units=[];partition=np.zeros(N,int);rank=0
 for line in (SP.parent/sem['assets']['groups.jsonl']['file']).read_text().splitlines():
  g=json.loads(line);rr=members[g['member_start']:g['member_start']+g['member_count']].astype(int);W=weights[g['weight_start']:g['weight_start']+g['rank']*2*len(rr)].reshape(g['rank'],2*len(rr))
  if g['group_id']in replacement:rr,W=replacement[g['group_id']]
  units.append((rr,W));partition[rr]+=1;rank+=len(W)
 partition[independent]+=1;rank+=2*len(independent);assert np.array_equal(partition,active.astype(int))and rank==int(stats['formal_residual_rows'])
 recapture_report=None
 if np.any(forced):
  ii=np.flatnonzero(forced);ss=s0[ii].copy();bb=b0[ii].copy();rr=_certified_candidates(scene,camera,material[ii],bb,ss,times[ii])
  recapture_report=_true_rs_grid_recovery(scene,camera,material[ii],frames[ii],times[ii],uv[ii],np.ones(len(ii),bool),expected[ii],ss,bb,rr,same_branch_only=False,threshold_px=-1.,grid_count=1025,progress=None)
  if not rr['valid'].all():raise ValueError('No legal true-RS branch found for explicitly recaptured source rows: '+str(ii[~rr['valid']].tolist()))
  s0[ii]=ss;b0[ii]=bb;v0[ii]=rr['row']
  recapture_report.update(source_rows=ii.tolist(),unseeded_source_rows=np.flatnonzero(unseeded).tolist(),algorithm_seed_only_not_observation=True,all_same_variant_finite_branches_visited=True,source_UV_unchanged=True)
  print('FORCED_FINITE_BRANCH_RECAPTURE',json.dumps(recapture_report),flush=True)
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
  begin=time.monotonic();camera.optics['principal_point']=pp0+np.array(delta);s=s0.copy();b=b0.copy();alpha=a0.copy();r=_certified_candidates(scene,camera,material,b,s,times);invalid=ordinary&~r['valid'];visibility_capture(np.flatnonzero(visibility_seed|invalid),s,b);r=_certified_candidates(scene,camera,material,b,s,times)
  qr=recover_query_roots(scene,camera,material[profile],b[profile],s[profile],times[profile],query,alpha_seed=alpha[profile]);s[profile]=qr['stations'];alpha[profile]=qr['alpha'];r=_certified_candidates(scene,camera,material,b,s,times)
  if not qr['valid'].all():return None,dict(label=label,delta=list(delta),valid=False,reason='genuine finite query root unresolved',rows=profile[~qr['valid']].tolist())
  if not r['valid'][active].all():return None,dict(label=label,delta=list(delta),valid=False,reason='strict full-source RS unresolved',rows=np.flatnonzero(active&~r['valid']).tolist())
  e=r['uv']-uv;c=coverage(e);q=(c/budget)**2;threshold=.25*float(q[active].max());target=np.flatnonzero(ordinary);iterations=[]
  # Recompute every ordinary station; local capture is not a global-optimum certificate.
  optimized=np.zeros(N,bool)
  it=0
  while len(target):
   if not len(target):break
   old={k:v[target]for k,v in r.items()if isinstance(v,np.ndarray)and v.shape[:1]==(N,)};cs=_implicit_capture(scene,camera,material[target],b[target],s[target],times[target],uv[target],iterations=16,initial_row=r['row'][target]);cr=_certified_candidates(scene,camera,material[target],b[target],cs,times[target]);cs,cb,cr,take=select_certified_upper_bound(s[target],b[target],old,cs,b[target],cr,uv[target]);_true_rs_grid_recovery(scene,camera,material[target],frames[target],times[target],uv[target],np.ones(len(target),bool),expected[target],cs,cb,cr,same_branch_only=False,threshold_px=8.,grid_count=1025,progress=None);s[target]=cs;b[target]=cb
   for k,v in r.items():
    if isinstance(v,np.ndarray)and v.shape[:1]==(N,)and k in cr:v[target]=cr[k]
   optimized[target]=True;e=r['uv']-uv;c=coverage(e);q=(c/budget)**2;dominant=float(q[active&(optimized|~ordinary)].max());target=np.flatnonzero(ordinary&~optimized&(q>=dominant));iterations.append(dict(iteration=it,optimized_rows=int(optimized.sum()),newly_active=len(target),max_q=float(q[active].max())));it+=1
  if len(target):return None,dict(label=label,delta=list(delta),valid=False,reason='inner active-set expansion unfinished')
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
 report=dict(schema='rinkfit.full_active_finite_RS_inner_trial.v1',initialization_only=True,final_physical_fit_accepted=False,implementation_source=dict(path=str(Path(__file__)),sha256=sha(Path(__file__))),run=str(R),raw_rows=N,active_rows=int(active.sum()),frames=int(packet['frame_count']),raw_material_labels=int(sem['material_labels']),packet=dict(path=str(PK),sha256=sha(PK)),query_packet=dict(path=str(QP),sha256=sha(QP)),source_W_E_unchanged=True,statistics_rank=rank,partition_exactly_once=True,model_parent=dict(path=str(M),sha256=sha(M)),semantic=dict(path=str(SP),sha256=sha(SP)),source_generator_preview=preview,forced_finite_branch_recapture=recapture_report,only_physical_columns_changed=[],reviewed_rebound_rows=changed.tolist(),best=rec,history=history,fixed_theta_source_residual_operator_unchanged=True,physical_model_family_provenance=model.get('provenance',{}).get('time_subspace'),full_fine_and_restricted_time_families_claimed_equivalent=False,exact_inner_KKT_certified=False,old_finite_rho_lifted_merit_identical_claimed=False,PhaseII_full_W_station_elimination_solved=False,profile_implicit_root_numerical_only=True,final_profile_global_KKT_pending=True,global_station_optimum_certified=False,finite_RS_nearest_seed_and_local_KKT_charts=True,whole_static_scene=scene.check_physics(),files={p.name:sha(p)for p in OUT.iterdir()if p.is_file()},elapsed_seconds=time.monotonic()-start,CPU_grid_batch=BATCH_STATS,CPU_compiled_camera=CAMERA_STATS,compiled_camera_snapshot=dict(path=str(OUT/'camera_snapshot/manifest.json'),sha256=sha(OUT/'camera_snapshot/manifest.json')))
 (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('FULL_N_REDUCED_DONE',str(OUT/'report.json'),flush=True)
 

if __name__=='__main__':main()
