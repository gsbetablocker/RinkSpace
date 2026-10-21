"""Actual source fixed-birth affine remeasurement. Never updates an immutable packet."""
from pathlib import Path
from collections import Counter
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import json,gzip,hashlib,struct,copy
import av,cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .source_texture_tracklets import _source_mask
from .native_entity_tracklets import NAVIGATION
from .source_drift_audit_v1 import direct,patch,rect_ok,ref
HELPER_SHA256='d55c7262842f4ca5297c41b3f2699773f2aa56996ea3d357ff46c2181e86b277'
PARAMETERS=dict(template_pixels=33,current_search_pixels=65,sampling_halo_px=1,affine_model='source-local only; no planar whole-Ad or world H',template_policy='fixed birth actual RGB; never updated from previous frame',navigation='original same-row observed UV centres the search only',NCC_min=.75,centre_FB_max_px=1.,all_four_template_corner_cycle_max_px=1.,NCC_peak_is_initializer_not_mean=True,ECC_motion='MOTION_AFFINE',ECC_criteria=dict(max_iterations=40,epsilon=.001,gaussian_filter_size=3),invalid_policy='retain old row UV as uncertain candidate; no interpolation/zero/subset success',source_covariance_issued=False)
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
def meanhash(uv):return hashlib.sha256(struct.pack('<dd',*uv)).hexdigest()
def generate(config,output):
 from . import source_drift_audit_v1 as helper
 if file_sha256(helper.__file__)!=HELPER_SHA256:raise ValueError('Changed fixed-birth helper')
 conf=read_json(config);gd=Path(conf['source_generation_directory']);cfg=read_json(gd/'source_config.json');tid=conf['track_id'];out=Path(output)
 if out.exists():raise ValueError('New immutable observation generation required')
 cfgsrc=cfg['source'];man=read_json(gd/'manifest.json');qual=read_json(gd/'qualification.json');qualified=next(q for q in qual['per_track'] if q['track_id']==tid)
 if not qualified['physical_static_surface_qualified']:raise ValueError('Unqualified original physical source')
 srcrows={r['source_row_id']:r for r in read_jsonl(conf['typed_source_rows']['file']) if r['track_id']==tid}
 if file_sha256(conf['typed_source_rows']['file'])!=conf['typed_source_rows']['sha256']:raise ValueError('Changed exact old typed source')
 rows=[];rawline={}
 with gzip.open(gd/'candidate_rows.jsonl.gz','rt',encoding='utf8') as f:
  for i,l in enumerate(f,1):
   q=json.loads(l)
   if q['track_id']==tid:q['candidate_source_line_1based']=i;rows.append(q);rawline[q['native_pts']]=hashlib.sha256(l.rstrip('\n').encode('utf8')).hexdigest()
 rows.sort(key=lambda q:q['native_pts']);sid=rows[0]['shot_id'];oldrefs=[ref(gd/n) for n in ('manifest.json','qualification.json','candidate_rows.jsonl.gz','source_config.json')]
 if len(rows)!=qualified['rows'] or any(b['native_pts']!=a['native_pts']+256 for a,b in zip(rows,rows[1:])):raise ValueError('Incomplete finite original track')
 if set(srcrows)!={'static_board_texture/'+tid+'/'+str(q['native_pts']) for q in rows}:raise ValueError('Missing typed original rowIDs')
 if file_sha256(cfgsrc['path'])!=cfgsrc['sha256']:raise ValueError('Wrong official native source')
 roles=read_json(cfg['roles'])['frame_roles'];ledgers={q['native_pts']:q for p in cfg['actor_ledgers'] for q in read_jsonl(p) if rows[0]['native_pts']<=q['native_pts']<=rows[-1]['native_pts']}
 contract=dict(schema='rinkfit.fixed_birth_readout_input.v1',reader=ref(__file__),helper=ref(helper.__file__),parameters=PARAMETERS,source_generation=ref(gd/'manifest.json'),source_qualification=ref(gd/'qualification.json'),source_candidate_rows=ref(gd/'candidate_rows.jsonl.gz'),original_typed_source_rows=conf['typed_source_rows'],source_roles=ref(cfg['roles']),source_configuration=ref(gd/'source_config.json'),actor_ledgers=[ref(p) for p in cfg['actor_ledgers']],source_video_sha256=cfgsrc['sha256'],track_id=tid,shot_id=sid,all_original_rows=len(rows),first_pts=rows[0]['native_pts'],last_pts=rows[-1]['native_pts'],time_base=[1,12800],no_model_used=True)
 generation_id='birth_affine_'+hashlib.sha256(canon(contract)).hexdigest();out.mkdir(parents=True);write_json(out/'input_contract.json',contract)
 bypts={q['native_pts']:q for q in rows};records=[];patchrows=[];updated=[];seed=None;cv2.setNumThreads(1)
 with av.open(cfgsrc['path']) as con:
  st=con.streams.video[0];con.seek(rows[0]['native_pts'],stream=st,backward=True)
  for fr in con.decode(st):
   p=int(fr.pts)
   if p<rows[0]['native_pts']:continue
   if p>rows[-1]['native_pts']:break
   if p not in bypts:raise ValueError('Decoded gap in source')
   q=bypts[p];rid='static_board_texture/'+tid+'/'+str(p);old=srcrows[rid];uv=np.asarray(q['source_uv'],float)
   if struct.pack('<dd',*uv)!=struct.pack('<dd',*old['source_uv']):raise ValueError('Old typed UV differs from original candidate')
   if roles.get(str(p),{}).get('role')!='development_candidate':raise ValueError('Protected frame requested')
   rgb=fr.to_ndarray(format='rgb24');mask,digest=_source_mask(rgb,ledgers[p],cfgsrc,p,sid)
   if digest!=q['current_rgb_sha256']:raise ValueError('Wrong current native RGB')
   selected=[z for z in cfg['regions'] if z['native_pts']==p]
   for z in selected:
    if z.get('native_person_mask'):
     if file_sha256(z['native_person_mask']['file'])!=z['native_person_mask']['sha256']:raise ValueError('Changed original person mask')
     mm=cv2.imread(z['native_person_mask']['file'],cv2.IMREAD_GRAYSCALE);mask=np.maximum(mask,cv2.dilate((mm>0).astype(np.uint8)*255,np.ones((33,33),np.uint8)))
   guards=cfg.get('guard_support_bbox_by_pts',{}).get(str(p),[]);g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);bad=rect_ok(uv,33,g.shape,guards);x,y=np.rint(uv).astype(int)
   if bad is None and mask[max(0,y-17):y+18,max(0,x-17):x+18].any():bad='actor_graphics_or_native_person_mask_actual_search_support'
   rec=dict(schema='rinkfit.fixed_birth_current_source_readout.v1',observation_generation_id=generation_id,source_row_id=rid,track_id=tid,shot_id=sid,surface_class='board',surface_instance_id=q['surface_instance_id'],native_pts=p,time_base=[1,12800],current_RGB_sha256=digest,original_source_uv=q['source_uv'],original_source_uv_f64_sha256=meanhash(q['source_uv']),original_candidate_line_1based=q['candidate_source_line_1based'],original_candidate_line_sha256=rawline[p],current_search_ROI_bbox_native=[*(uv-33).tolist(),*(uv+33).tolist()],current_search_patch_origin_native=(uv-32).tolist(),current_search_patch_pixels=[65,65],guard_support_rectangles=guards,source_geometry_model_used=False,ice_or_global_H_used=False,source_material_identity_inherited=True,world_position_remains_one_shared_unknown=True,source_precision_or_covariance_issued=False,original_source_qualification_unchanged=True)
   newuv=uv.copy();strong=False
   if seed is None:
    seed=dict(pts=p,uv=uv.copy(),RGB_sha256=digest,template=patch(g,uv,33) if bad is None else None,buffer=patch(g,uv,65) if bad is None else None,bad=bad)
    rec.update(status='birth_original_UV_preserved' if bad is None else 'uncertain_birth_support',unavailable_reason=bad)
   elif bad or seed['bad']:rec.update(status='uncertain_support',unavailable_reason=bad or 'birth_'+seed['bad'])
   else:
    d=direct(seed['template'],patch(g,uv,65));rec['direct']=d
    if d['status']!='affine_registration':rec.update(status='uncertain_forward_registration',unavailable_reason=d['status'])
    else:
     dUV=uv+np.asarray(d['centre_in_search'])-32;re=direct(patch(g,dUV,33),seed['buffer']);rec['reverse']=re;rec['direct_candidate_source_uv']=dUV.tolist();rec['candidate_minus_original_uv']=(dUV-uv).tolist();rec['candidate_minus_original_px']=float(np.linalg.norm(dUV-uv))
     if re['status']!='affine_registration':rec.update(status='uncertain_reverse_registration',unavailable_reason=re['status'])
     else:
      A=np.asarray(d['warp'])[:,:2];B=np.asarray(re['warp'])[:,:2];bias=np.asarray(re['centre_in_search'])-32;fb=float(np.linalg.norm(bias));e=np.array([[-16,-16],[-16,16],[16,-16],[16,16]],float)@(B@A-np.eye(2)).T+bias;cycle=float(np.linalg.norm(e,axis=1).max());rec.update(long_baseline_forward_backward_px=fb,affine_patch_cycle_corner_errors_px=np.linalg.norm(e,axis=1).tolist(),affine_patch_cycle_max_px=cycle,affine_jacobian_reciprocal_error_fro=float(np.linalg.norm(B@A-np.eye(2))))
      dbad=rect_ok(dUV,17,g.shape,guards);di=np.rint(dUV).astype(int)
      if dbad is None and mask[di[1],di[0]]:dbad='registered_actual_support_actor_graphics_or_native_mask'
      if dbad:rec.update(status='uncertain_registered_support',unavailable_reason=dbad)
      elif min(d['registered_NCC'],re['registered_NCC'])<.75:rec.update(status='uncertain_appearance',unavailable_reason='existing_NCC075_not_met')
      elif fb>1.:rec.update(status='uncertain_centre_reciprocity',unavailable_reason='existing_1px_reference_not_met')
      elif cycle>1.:rec.update(status='uncertain_full_affine_patch_reciprocity',unavailable_reason='one_or_more_actual_template_corner_cycles_over_existing_1px_reference')
      else:strong=True;newuv=dUV;rec.update(status='new_current_source_UV_candidate',unavailable_reason=None)
   rec.update(birth_pts=seed['pts'],birth_RGB_sha256=seed['RGB_sha256'],birth_source_uv=seed['uv'].tolist(),fixed_template_origin_native=(seed['uv']-16).tolist(),fixed_template_pixels=[33,33],age_native_steps=(p-seed['pts'])//256,remeasured_source_uv=newuv.tolist(),remeasured_source_uv_f64_sha256=meanhash(newuv),source_uv_updated=strong,readout_status='direct_full_affine_cycle_supported_candidate' if strong else 'original_birth_reference' if p==seed['pts'] and bad is None else 'uncertain_original_UV_retained')
   nr=copy.deepcopy(old);nr.update(source_uv=newuv.tolist(),current_uv=newuv.tolist(),original_source_uv=q['source_uv'],original_source_uv_f64_sha256=meanhash(q['source_uv']),observation_generation_id=generation_id,current_readout_strategy=contract['reader'],current_readout_parameters=PARAMETERS,readout_update_status=rec['readout_status'],source_measurement='current_native_RGB_fixed_birth_local_affine_readout' if strong else old['source_measurement'],source_mean_remeasured_from_current_pixels=strong,source_mean_is_truth=False,source_precision_or_covariance_issued=False,original_source_qualification=contract['source_qualification'],original_generation=contract['source_generation'],original_reader_refs=man['reader_modules'])
   patchrows.append(dict(source_row_id=rid,track_id=tid,native_pts=p,shot_id=sid,current_RGB_sha256=digest,original_source_uv=q['source_uv'],original_source_uv_f64_sha256=meanhash(q['source_uv']),next_source_uv=newuv.tolist(),next_source_uv_f64_sha256=meanhash(newuv),source_uv_changed=strong,action='replace_source_mean_at_next_immutable_batch' if strong else 'retain_original_source_mean_with_explicit_uncertain_readout' if p!=seed['pts'] or bad else 'retain_original_birth_reference',readout_status=rec['readout_status'],observation_generation_id=generation_id,readout_record_index_1based=len(records)+1))
   records.append(rec);updated.append(nr)
 if len(records)!=len(rows):raise ValueError('Incomplete readout denominator')
 for name,rr in [('current_readout_rows.jsonl',records),('source_rows.jsonl',updated)]:
  with (out/name).open('w',encoding='utf8') as h:
   for r in rr:h.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
 write_json(out/'correction_patch.json',dict(schema='rinkfit.source_rowID_current_readout_correction_patch.v1',observation_generation_id=generation_id,input_contract=ref(out/'input_contract.json'),old_source_generation=contract['source_generation'],old_source_qualification=contract['source_qualification'],original_typed_source_rows=contract['original_typed_source_rows'],reader=contract['reader'],parameters=PARAMETERS,rows=patchrows,all_original_rows_preserved=True,apply_only_at_next_batch=True,requires_source_identity_guard_and_fresh_RMS_RS_auxiliary_binding=True,unknown_UV_not_replaced=True,confidence_is_not_truth=True))
 # Input files are immutable: retain and check their exact final bytes.
 for rr in oldrefs:
  if file_sha256(rr['file'])!=rr['sha256']:raise ValueError('Input generation mutated during readout')
 manifest=dict(schema='rinkfit.source_current_fixed_birth_affine_observation_generation.v1',observation_generation_id=generation_id,input_contract=ref(out/'input_contract.json'),source_video=cfgsrc,original_generation=contract['source_generation'],original_source_qualification=contract['source_qualification'],reader=contract['reader'],helper=contract['helper'],source_rows=ref(out/'source_rows.jsonl'),readout_records=ref(out/'current_readout_rows.jsonl'),correction_patch=ref(out/'correction_patch.json'),all_source_rows=len(records),source_uv_updated_rows=sum(r['source_uv_updated'] for r in records),original_birth_rows=sum(r['status']=='birth_original_UV_preserved' for r in records),uncertain_rows=sum(r['readout_status']=='uncertain_original_UV_retained' for r in records),counts=dict(Counter(r['status'] for r in records)),no_original_source_or_packet_mutation=True,no_model_or_interpolated_UV=True,no_new_human_intervention=True,fit_packet_consumption_status='awaiting_next_immutable_batch_binding',statistical_covariance_issued=False,physical_static_ownership_qualification_unchanged=True,full_track_not_success_subset=True,source_qualification_new_policy_effect='Current source mean changes only when full fixed-birth bidirectional local-affine source readout supports it; all failures stay visible with original UV.',limitations=['Small-source local affine patch model is a conditional image approximation; no whole-board flat rectangle or global H was fitted.','NCC/reciprocity source evidence does not issue covariance or establish metric/world positions.','Original physical ownership is inherited from current exact source qualification; the parent unknown Ad point stays shared and must be reinitialized at the next batch.','No old individual template/glyph coordinates, camera or world point were used to create current UV.'])
 write_json(out/'manifest.json',manifest);return manifest
