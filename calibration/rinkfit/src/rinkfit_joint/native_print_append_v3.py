"""Append reviewed native print readouts without a per-pixel JSON object.

The original NPZ store remains immutable. Typed evidence columns preserve every
selected UV, normal, profile and original store ordinal exactly. Missing semantic
bindings stay in the source store with an explicit withheld denominator.
"""
from pathlib import Path
from collections import defaultdict
import json,time,hashlib,numpy as np,cv2,av
from .io_utils import read_json,write_json,file_sha256
from .joint_metrics import PacketArrays
from .dense_joint_packet import _ref,_array,_fast_seed_frame,REGISTRY
from .joint_model import as_event_view
from .screen_native_motion_v2 import _camera
from rinkfit_start.native_scene import NativeScene

def actual_frames(cfg,records):
 wanted={f['native_pts']for f in records}
 with av.open(cfg['source_video'])as con:
  stream=con.streams.video[0]
  if str(stream.time_base)!='1/12800':raise ValueError('Source native timebase changed')
  con.seek(min(wanted),stream=stream,backward=True)
  for frame in con.decode(stream):
   pt=int(frame.pts)
   if pt>max(wanted):break
   if pt in wanted:yield pt,frame.to_ndarray(format='bgr24')

def append(*,parent_packet,parent_qualification,parent_weights,model,candidate_manifest,source_review,output):
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();mf=read_json(candidate_manifest);cfg=read_json(mf['config_file']);review=read_json(source_review);m=read_json(model)
 for path,sha in[(mf['config_file'],mf['config_sha256']),(mf['reader_file'],mf['reader_sha256'])]:
  if file_sha256(path)!=sha:raise ValueError('Changed current native readout dependency: '+str(path))
 if review['source_model_residual_used'] or review['source_pixels_changed'] or mf['source_model_projection_used']:raise ValueError('Source readout must be independent of model projections')
 if review['source_config']!={'file':mf['config_file'].replace('\\','/'),'sha256':mf['config_sha256']}:
  if Path(review['source_config']['file']).resolve()!=Path(mf['config_file']).resolve() or review['source_config']['sha256']!=mf['config_sha256']:raise ValueError('Source review/config mismatch')
 frames=read_json(Path(candidate_manifest).parent/'frame_index.json');seeds=cfg['seeds'];eligible={x['label']:x for x in review['eligible_materials']}
 for label,q in eligible.items():
  ma=m['materials'][q['material_index']]
  if ma['label']!=label or not label.startswith('Ad') or q['physical_surface']!='board':raise ValueError('Unconfirmed physical print identity')
  if not q['allowed_paths'] or any(p<0 or p>=len(ma['paths']) or ma['paths'][p].get('physical_point',False) for p in q['allowed_paths']):raise ValueError('Invalid source finite print path')
 seed_material=np.array([eligible[s['label']]['material_index']if s['label']in eligible else -1 for s in seeds],'<i4')
 def save(name,a):
  a=np.asarray(a);a.tofile(out/name);return dict(_ref(out/name),dtype=a.dtype.str,shape=list(a.shape),layout='C',byte_order='little')
 seed_ref=save('seed_material.i32',seed_material);columns={};chunks=defaultdict(list);records=[];missing=defaultdict(int);raw_ordinal=0;selected=0
 from .native_print_track_readout import batch_interfaces,clear_profile
 from .source_worker import _channels
 from calibration.program_tracking.extract_first_shot_ice import current_broadcast_exclusion
 import rinkfit_joint.source_worker as source_channels
 import calibration.program_tracking.extract_first_shot_ice as source_graphics
 current_frames=iter(actual_frames(cfg,frames));reread_failed=0;cv2.setNumThreads(2)
 for fi,f in enumerate(frames):
  pt,im=next(current_frames)
  if pt!=f['native_pts']or hashlib.sha256(im.tobytes()).hexdigest()!=f['source_BGR_sha256']:raise ValueError('Actual native exposure changed')
  mask=cv2.imread(f['person_mask']['file'],0)
  if f['unknown_mask']:mask=cv2.bitwise_or(mask,cv2.imread(f['unknown_mask']['file'],0))
  graphics,_=current_broadcast_exclusion(im);mask[graphics]=255
  if file_sha256(f['file'])!=f['sha256']:raise ValueError('Changed actual source readout')
  with np.load(f['file'])as z:
   si=z['seed_index'];uv=z['uv'];normal=z['normal'];profile=z['profile'];fb=z['FB_error_px']
   if len(si)!=f['rows'] or len(set(map(int,si)))!=len(si):raise ValueError('Raw native source count or seed uniqueness')
   if not(np.isfinite(uv).all()and np.isfinite(normal).all()and np.isfinite(profile).all()and np.isfinite(fb).all()):raise ValueError('Nonfinite native profile')
   if np.any(profile[:,0]<.75)or np.any(profile[:,1]<4.)or np.any(abs(profile[:,2])>=3.125)or np.any(fb>.8+1e-6):raise ValueError('Changed native profile qualification')
   if np.any(uv[:,0]<0)or np.any(uv[:,0]>1919)or np.any(uv[:,1]<0)or np.any(uv[:,1]>1079):raise ValueError('Source point outside native exposure')
   fresh=batch_interfaces(uv-profile[:,2,None]*normal,normal,_channels(im[:,:,::-1]));same=np.linalg.norm(fresh['uv']-uv,axis=1)<=1e-6;current=fresh['valid']&same&clear_profile(uv,normal,mask)&(fresh['profile'][:,3]==profile[:,3])&(fresh['profile'][:,4]==profile[:,4]);reread_failed+=int(np.count_nonzero(~current));ok=(seed_material[si]>=0)&current;ix=np.flatnonzero(ok);mat=seed_material[si[ix]]
   for k in si[~ok]:missing[seeds[int(k)]['label']]+=1
   for key,a in dict(seed_index=si[ix],frame_index=np.full(len(ix),fi,'<u4'),source_ordinal=(raw_ordinal+ix).astype('<u8'),uv=uv[ix],normal=normal[ix],profile=profile[ix],flow_fb=fb[ix],material_index=mat.astype('<u4')).items():chunks[key].append(a.copy())
   fr=dict(native_pts=int(f['native_pts']),shot_id=next(s['id']for s in cfg['shots']if s['first_pts']<=f['native_pts']<=s['last_pts']),frame_role='native_material_repair_training',source_BGR_sha256=f['source_BGR_sha256'],source_time_base='1/12800',raw_readout=_ref(f['file']),raw_row_start=raw_ordinal,raw_rows=len(si),selected_row_start=selected,selected_rows=len(ix),person_mask=_ref(f['person_mask']['file']),unknown_mask=_ref(f['unknown_mask']['file'])if f['unknown_mask']else None)
   records.append(fr);raw_ordinal+=len(si);selected+=len(ix)
  if(fi+1)%250==0:print(json.dumps(dict(stage='source_columns',frames=fi+1,selected=selected,seconds=time.monotonic()-started)),flush=True)
 if raw_ordinal!=mf['rows']:raise ValueError('Incomplete candidate denominator')
 data={k:np.concatenate(v)for k,v in chunks.items()};chunks.clear();count=selected
 if not count:raise ValueError('No source-qualified print observations')
 for key,a in data.items():columns[key]=save('source_'+key+'.bin',a)
 write_json(out/'frames.json',records);source_dataset='native_named_print_1022';ev=dict(schema='rinkfit.native_named_print_source_readout_evidence.v3',source_dataset_id=source_dataset,source_video={'file':cfg['source_video'],'sha256':cfg['source_video_sha256']},source_time_base='1/12800',source_config=_ref(mf['config_file']),candidate_manifest=_ref(candidate_manifest),source_entity_review=_ref(source_review),source_reader=_ref(mf['reader_file']),source_profile_reader=_ref(mf['reader_file']),frames=_ref(out/'frames.json'),columns=columns,seed_material=seed_ref,source_model_residual_used=False,source_pixels_changed=False,physical_surface='board',all_native_exposures_remeasured=True,all_selected_original_UV_reproduced_from_actual_RGB=True,source_reread_unconfirmed_rows=reread_failed,source_channel_reader=_ref(source_channels.__file__),broadcast_mask_reader=_ref(source_graphics.__file__),source_tracking_forward_backward_limit_px=.8,raw_candidate_rows=raw_ordinal,selected_rows=count,withheld_rows=raw_ordinal-count,withheld_by_entity=dict(missing),source_identity_review_scope=review['review_scope'],statistical_precision_claimed=False,fit_accuracy_accepted=False)
 write_json(out/'source_evidence.json',ev);evidence=out/'source_evidence.json'
 rd=PacketArrays(parent_packet);p=rd.packet;n0=int(p['row_count']);N=n0+count;scene=NativeScene(as_event_view(m,'s01'));cameras={sid:_camera(m,sid)for sid in m['shots']};paths=np.zeros(count,'<u4');stations=np.zeros(count,'<f8');allowed={str(x['material_index']):x['allowed_paths']for x in eligible.values()};groups=defaultdict(list);ids=[];family_names=[];family_index={};families=np.zeros(count,'<u4');shot=np.zeros(count,'<u2');pts=np.zeros(count,'<i8')
 for fi,f in enumerate(records):
  lo=f['selected_row_start'];hi=lo+f['selected_rows'];mat=data['material_index'][lo:hi];uv=data['uv'][lo:hi];block=[dict(material_index=int(mi),material_label=m['materials'][int(mi)]['label'],source_uv=q.tolist(),allowed_paths=allowed[str(int(mi))],path_index=0)for mi,q in zip(mat,uv)]
  if block:
   t=(f['native_pts']-m['shots'][f['shot_id']]['time_origin_pts'])/12800;pa,st,_,_=_fast_seed_frame(scene,cameras[f['shot_id']],t,block);paths[lo:hi]=pa;stations[lo:hi]=st
  pts[lo:hi]=f['native_pts'];shot[lo:hi]=p['shot_ids'].index(f['shot_id'])
  for j in range(lo,hi):
   mi=int(data['material_index'][j]);panel=m['materials'][mi]['label'].split('.')[0];family=f'native_print/{f["shot_id"]}/{f["native_pts"]}/{panel}'
   if family not in family_index:family_index[family]=len(family_names);family_names.append(family)
   families[j]=family_index[family];groups[family].append(n0+j);ids.append(f'{source_dataset}/{f["shot_id"]}/{f["native_pts"]}/seed{int(data["seed_index"][j])}'.encode('utf8'))
  if(fi+1)%100==0:print(json.dumps(dict(stage='finite_curve_initialization',frames=fi+1,total_frames=len(records),rows=hi,seconds=time.monotonic()-started)),flush=True)
 suffix=dict(uv=data['uv'],native_pts=pts,shot_index=shot,kind=np.zeros(count),material_index=data['material_index'],path=paths,admission=np.ones(count),fit_role=np.ones(count),E=np.full(count,2.),station=stations,row=data['uv'][:,1],alpha=np.zeros(count),source_sigma=np.ones(count),profile_normal=data['normal'],query_uv=data['uv'],alpha_bounds=np.zeros((count,2)),stroke_pair_index=np.full(count,-1),stroke_source_side=np.full(count,-1))
 if set(suffix)!=set(p['arrays']):raise ValueError('Unsupported original source ABI')
 arrays={}
 for key in p['arrays']:
  old=rd.load(key);tail=np.asarray(suffix[key],dtype=old.dtype);joined=np.concatenate([old,tail]);assert np.array_equal(joined[:n0],old,equal_nan=True);arrays[key]=save(key+'.bin',joined)
 blob=b''.join(ids);offsets=np.r_[0,np.cumsum([len(s)for s in ids])].astype('<u8');registry=dict(schema=REGISTRY,row_start=n0,count=count,source_id_utf8=save('source_id_utf8.u8',np.frombuffer(blob,'u1')),source_id_offsets=save('source_id_offsets.u64',offsets),source_family_ids=family_names,source_family_index=save('source_family_index.u32',families),source_store_ordinal=columns['source_ordinal'],source_dataset_id=source_dataset,allowed_paths_are_explicit=True,allowed_paths_by_material=allowed,original_source_selection_IDs_retained=True,source_evidence=_ref(evidence));ng=[dict(id='native_print_unknown_W:'+name,source_group_id=name,rows=ii,rank=0,W=[],statistic_kind='constraint_only_unknown_noise',unknown_noise_source_refs=[_ref(evidence)],source_noise_covariance_claimed=False)for name,ii in groups.items()]
 val=dict(schema=p['schema'],dense_parent_packet=_ref(parent_packet),row_count=N,arrays=arrays,native_ad_source_binding=registry,native_ad_source_bindings=p.get('native_ad_source_bindings',[p['native_ad_source_binding']]if'native_ad_source_binding'in p else[])+[registry],active_rows=int(p['active_rows'])+count,admitted_rows=int(p.get('admitted_rows',p['active_rows']))+count,native_ad_source_append=dict(schema='rinkfit.native_ad_source_append.v1',parent_packet=_ref(parent_packet),row_start=n0,count=count,evidence=_ref(evidence)),dense_parent_append={'groups':ng,'source_manifests':[_ref(evidence),_ref(__file__),_ref(model)],'engineering_pixel_dependency_groups.groups':[dict(id=name,rows=ii)for name,ii in groups.items()]},material_binding=dict(p['material_binding'],model_material_prefix_count=len(m['materials']),materials=[dict(index=i,label=ma['label'],paths=[dict(index=k,id=pa['id'],physical_point=pa.get('physical_point',False))for k,pa in enumerate(ma['paths'])])for i,ma in enumerate(m['materials'])],model_topology_ref=_ref(model)))
 write_json(out/'packet.json',val);packetref=_ref(out/'packet.json');q=dict(schema='rinkfit.native_named_print_append_qualification.v3',source_packet=packetref,parent_packet=_ref(parent_packet),parent_qualification=_ref(parent_qualification),parent_rows=n0,raw_rows=N,new_rows=count,active_new_rows=count,source_evidence=_ref(evidence),issuer=_ref(__file__),membership_array_sha256={k:v['sha256']for k,v in arrays.items()},compact_registry=registry,original_prefix_values_unchanged=True,native_profile_normals_preserved=True,source_UV_from_model=False,quality_accepted=False);write_json(out/'source_qualification.json',q);qr=_ref(out/'source_qualification.json');w=read_json(parent_weights);wf={};values={};budgets=[]
 for name,dt in[('weights.f64','<f8'),('active.u8','u1'),('information_class.u8','u1')]:
  old=_array(w['files'][name],Path(parent_weights).parent);tail=np.ones(count,dtype=dt)
  if name=='information_class.u8':tail*=5
  if name=='weights.f64':
   for name2,ii in groups.items():
    ix=np.array(ii)-n0;tail[ix]/=len(ix);budgets.append(dict(id=name2,rows=len(ix),per_row_weight=1./len(ix),engineering_budget=1.,identity='one actual printed panel in one native exposure; aliases do not each obtain a new budget'))
  values[name]=np.concatenate([old,tail]);wf[name]=save(name,values[name])
 retained=float(values['weights.f64'][:n0].sum());added=float(values['weights.f64'][n0:].sum());weights=dict(schema='rinkfit.native_curve_append_engineering_weights.v1',source_packet=packetref,parent_packet=_ref(parent_packet),parent_weights=_ref(parent_weights),issuer=_ref(__file__),dense_source_qualification=qr,parent_rows=n0,raw_rows=N,active_rows=int(np.count_nonzero(values['active.u8'])),new_active_rows=count,component_budgets=budgets,files=wf,retained_parent_budget=retained,new_dense_budget=added,actual_total_engineering_budget=retained+added,source_noise_W_or_sigma_issued=False,statistical_precision_or_noise_independence_claimed=False);write_json(out/'engineering_weights.json',weights);receipt=dict(packet=packetref,qualification=qr,weights=_ref(out/'engineering_weights.json'),source_evidence=_ref(evidence),parent_rows=n0,new_raw_rows=count,new_active_rows=count,all_original_raw_arrays_bitwise_preserved=True,raw_candidates=raw_ordinal,withheld_candidates=raw_ordinal-count,elapsed_seconds=time.monotonic()-started,physical_solver_started=False);write_json(out/'receipt.json',receipt);print(json.dumps(receipt),flush=True);return receipt
