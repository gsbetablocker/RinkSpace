import json,gzip,time,hashlib,base64,collections
from pathlib import Path
import numpy as np,cv2
from scipy.spatial import cKDTree
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');R=O.parent;F=O/'source_v5_frames';S=O/'static_v5_frames';S.mkdir(exist_ok=True)
PTS=list(range(948992,1425408+1,256))
CODES={0:'unknown_no_same_shot_adjacent_static_material_support',1:'actual_adjacent_static_board_motion_normal_or_2D_supported',2:'original_current_native_known_advertisement_material_readout',3:'original_user_manual_source_click_static_board_entity',4:'actual_adjacent_motion_conflict_outside_static_board_motion_support',5:'unknown_local_board_texture_motion_support_unavailable',6:'excluded_outside_current_board_body_or_actor_or_source_unknown_or_actual_RGB_graphics_complete_profile',7:'current_recognized_static_print_material_semantic_proof_normal_only_unbound_no_2D_world_correspondence'}
def write(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def waitload(p):
 while True:
  try:return json.loads(p.read_text(encoding='utf8'))
  except(FileNotFoundError,json.JSONDecodeError):time.sleep(.3)
def read(pts):
 ref=dict(waitload(F/f'{pts}.json'));rows=[]
 for part in ref['source_parts']:
  with open(part['canonical_file'],'rb')as f:f.seek(part['gzip_member_offset']);member=f.read(part['gzip_member_bytes'])
  assert hashlib.sha256(member).hexdigest()==part['gzip_member_sha256'];rr=[json.loads(x)for x in gzip.decompress(member).decode('utf8').splitlines()];assert len(rr)==part['rows'];rows.extend(rr)
 surface=np.frombuffer(base64.b64decode(ref['surface_eligible_uint8_base64']),dtype='u1')>0;assert len(surface)==len(rows);n=len(rows);code=np.zeros(n,np.uint8)
 sf=ref['semantic_normal_ref'];assert hashlib.sha256(Path(sf['file']).read_bytes()).hexdigest()==sf['sha256'];sj=json.loads(Path(sf['file']).read_text(encoding='utf8'));semantic=np.zeros(n,bool)
 if sj['independent_static_print_semantic_support']:
  sm=sj['mask_ref'];assert hashlib.sha256(Path(sm['file']).read_bytes()).hexdigest()==sm['sha256'];mask=cv2.imread(sm['file'],0)>0;guard=cv2.dilate(mask.astype('u1'),np.ones((7,7),'u1'))>0;xy=np.rint(np.asarray([r['uv']for r in rows],float)).astype(int);candidate=guard[np.clip(xy[:,1],0,1079),np.clip(xy[:,0],0,1919)]
  for i,r in enumerate(rows):
   p=r.get('pixel_support',{});normal=p.get('normal');semantic[i]=bool(candidate[i]and surface[i]and normal is not None and np.isfinite(normal).all()and np.linalg.norm(normal)>.5 and p.get('contrast_native_DN',0)>=12)
 for i,r in enumerate(rows):
  if not surface[i]:continue
  if r['source_kind']=='preserved_user_manual_native_image_click':code[i]=3
  elif r['source_kind']=='preserved_original_native_advertisement_readout':code[i]=2
 c=dict(gzip_member_sha256=ref['source_parts_binding_sha256'],board_file=ref['source_motion_file'],board_file_sha256=ref['source_motion_sha256'])
 return dict(cov=c,source_ref=ref,surface=surface,semantic_normal=semantic,motion_refs=[],rows=rows,code=code,sid=ref['shot_id'],pts=ref['native_pts'],xy=np.asarray([r['uv']for r in rows]),ids={r['point_id']:i for i,r in enumerate(rows)},reasons=collections.Counter(),backward_status=np.zeros(n,np.uint8),forward_status=np.zeros(n,np.uint8),backward_kind=np.zeros(n,np.uint8),forward_kind=np.zeros(n,np.uint8),motion_error=np.zeros(n,np.float32)+np.nan,local_motion_unavailable=0)
def qualify_pair(prev,cur):
 if prev['sid']!=cur['sid'] or cur['pts']-prev['pts']!=256:return
 motionref=dict(file=cur['cov']['board_file'],sha256=cur['cov']['board_file_sha256']);cur['motion_refs'].append(motionref);prev['motion_refs'].append(motionref)
 bp=Path(cur['cov']['board_file']);board=json.loads(bp.read_text(encoding='utf8'));assert hashlib.sha256(bp.read_bytes()).hexdigest()==cur['cov']['board_file_sha256'];t=board;src=np.asarray(t.get('source_uv',[]),float).reshape(-1,2);dst=np.asarray(t.get('current_uv',[]),float).reshape(-1,2);fb=np.asarray(t.get('forward_backward_error_px',[]),float)
 if len(src)<8 or len(src)!=len(dst) or len(fb)!=len(src):
  cur['local_motion_unavailable']=len(cur['rows']);prev['local_motion_unavailable']+=len(prev['rows']);return
 good=np.isfinite(src).all(1)&np.isfinite(dst).all(1)&(fb<.75);src=src[good];dst=dst[good]
 if len(src)<8:return
 delta=dst-src;tree=cKDTree(src);common=[(i,prev['ids'][r['point_id']])for i,r in enumerate(cur['rows'])if r['point_id']in prev['ids'] and cur['surface'][i] and prev['surface'][prev['ids'][r['point_id']]]]
 if not common:return
 ci=np.array([x[0]for x in common]);pi=np.array([x[1]for x in common]);q=prev['xy'][pi];dist,ix=tree.query(q,k=min(12,len(src)));localdelta=delta[ix];med=np.median(localdelta,axis=1);res=np.linalg.norm(localdelta-med[:,None,:],axis=2);keep=res<=np.maximum(3.,np.median(res,axis=1)*3)[:,None];expected=np.array([np.median(d[k],axis=0)if k.sum()>=4 else np.array([np.nan,np.nan])for d,k in zip(localdelta,keep)]);support=(dist[:,min(3,dist.shape[1]-1)]<=320)&np.isfinite(expected).all(1)
 actual=cur['xy'][ci]-prev['xy'][pi];err=actual-expected
 for j,(ii,jj)in enumerate(zip(ci,pi)):
  r=cur['rows'][ii];p=r.get('pixel_support',{});normal=p.get('normal')
  if not support[j]:cur['backward_status'][ii]=5;prev['forward_status'][jj]=5;continue
  if p.get('two_dimensional_texture_support'):
   e=float(np.linalg.norm(err[j]));kind='2D'
  elif normal is not None:
   a=np.asarray(normal,float);a/=max(np.linalg.norm(a),1e-12);e=abs(float(err[j]@a));kind='normal'
  else:continue
  # Engineering domain qualification only: four native pixels allows subpixel reader variation and local plane/curve motion.
  passed=e<=4.;status=1 if passed else 4;cur['backward_status'][ii]=status;prev['forward_status'][jj]=status;cur['motion_error'][ii]=e;cur['backward_kind'][ii]=2 if kind=='2D'else 1;prev['forward_kind'][jj]=2 if kind=='2D'else 1
  cur['reasons']['backward_'+kind+('_supported'if passed else'_conflict')]+=1;prev['reasons']['forward_'+kind+('_supported'if passed else'_conflict')]+=1

def finish(frame):
 code=frame['code'];back=frame['backward_status'];forward=frame['forward_status'];auto=code==0;passed=(back==1)|(forward==1);conflict=((back==4)|(forward==4))&~passed;unavailable=((back==5)|(forward==5))&~passed&~conflict
 code[auto&passed]=1;code[auto&conflict]=4;code[auto&unavailable]=5
 semantic=frame['semantic_normal']&((code==0)|(code==5));code[semantic]=7
 code[~frame['surface']]=6
 eligible=(code==1)|(code==2)|(code==3)|(code==7);texture=(code==1)&((frame['backward_kind']==2)&(back==1)|(frame['forward_kind']==2)&(forward==1));normal=(code==7)|((code==1)&~texture);types=np.zeros(len(code),'u1');types[normal]=1;types[texture]=2;types[code==2]=3;types[code==3]=4;conf=[]
 for i in np.flatnonzero(code==4):
  r=frame['rows'][int(i)];e=frame['motion_error'][int(i)];conf.append(dict(point_id=r['point_id'],uv=r['uv'],physical_surface='board_candidate_not_static_qualified',current_native_image_motion_conflict_px=float(e)if np.isfinite(e)else None))
 out=dict(schema='rinkspace.actual_same_shot_board_material_static_qualification.v1',shot_id=frame['sid'],native_pts=frame['pts'],source_gzip_member_sha256=frame['cov']['gzip_member_sha256'],source_row_count=len(code),eligible_rows=int(eligible.sum()),eligible_uint8_base64=base64.b64encode(eligible.astype('u1').tobytes()).decode('ascii'),status_uint8_base64=base64.b64encode(code.tobytes()).decode('ascii'),status_code_definition=CODES,normal_only_uint8_base64=base64.b64encode(normal.astype('u1').tobytes()).decode(),texture_2D_tracking_uint8_base64=base64.b64encode(texture.astype('u1').tobytes()).decode(),observation_type_uint8_base64=base64.b64encode(types.tobytes()).decode(),observation_type_definition={0:'unqualified',1:'normal_only_static_material_boundary_unbound',2:'actual_image_2D_texture_tracking_supported_no_world_identity',3:'preserved_original_native_print_readout',4:'preserved_user_native_click'},status_counts=dict(collections.Counter(map(int,code))),actual_adjacent_static_motion_refs=frame['motion_refs'],semantic_normal_ref=frame['source_ref']['semantic_normal_ref'],adjacent_motion_source='actual board body 2D LK-FB from same source exposures; local robust median supported in source image, no camera/rink model',local_motion_support_nearest4_radius_native_px=320,engineering_static_motion_capture_native_px=4,engineering_threshold_is_not_sigma=True,source_coordinate_or_material_station_moved=False,pure_edge_tangent_world_correspondence_not_claimed=True,source_sigma_unknown=True,cut_not_crossed=True,source_original_unknown_and_conflict_rows_preserved=True,reason_counts=dict(frame['reasons']),motion_conflict_rows=conf)
 write(S/f'{frame["pts"]}.json',out)
 ref=frame['source_ref'];parts=[dict(file=p['canonical_file'],offset_bytes=p['gzip_member_offset'],length_bytes=p['gzip_member_bytes'],sha256=p['gzip_member_sha256'],row_count=p['rows'])for p in ref['source_parts']]
 final=dict(schema='rinkspace.final_dense_advertisement_source_consumer_frame.v1',shot_id=frame['sid'],native_pts=frame['pts'],time_base='1/12800',decoded_native_BGR_sha256=ref['decoded_native_BGR_sha256'],decoded_native_RGB_sha256=ref['decoded_native_RGB_sha256'],source_members=parts,source_row_count=len(code),eligible_rows=int(eligible.sum()),eligible_uint8_base64=out['eligible_uint8_base64'],status_uint8_base64=out['status_uint8_base64'],status_code_definition=CODES,normal_only_uint8_base64=out['normal_only_uint8_base64'],texture_2D_tracking_uint8_base64=out['texture_2D_tracking_uint8_base64'],observation_type_uint8_base64=out['observation_type_uint8_base64'],observation_type_definition=out['observation_type_definition'],status_counts=out['status_counts'],final_board_ref=dict(file=ref['final_board_file'],sha256=ref['final_board_sha256']),mask_ref=dict(file=ref['mask_file'],sha256=ref['mask_sha256']),unknown_exclusion_ref=ref.get('unknown_exclusion_ref'),person_mask_semantics='qualified_current_person_SAM_pixels',unknown_exclusion_semantics='actual_current_source_mixed_or_reflective_pixels_unknown_not_person_foreground_not_formal_static_material',static_motion_ref=dict(file=ref['source_motion_file'],sha256=ref['source_motion_sha256']),adjacent_static_motion_refs=frame['motion_refs'],static_qualification_ref=dict(file=str(S/f'{frame["pts"]}.json'),sha256=hashlib.sha256((S/f'{frame["pts"]}.json').read_bytes()).hexdigest()),body_material_ref=ref['body_material_ref'],graphics_ref=ref['graphics_ref'],semantic_normal_ref=ref['semantic_normal_ref'],composite_mask_index_ref=ref['composite_mask_index_ref'],eligible_physical_surface='board',legacy_candidate_visibility_fields_superseded_by_final_status=True,legacy_candidate_geometry_fields_superseded_by_final_observation_type=True,world_correspondence_or_independent_weights_not_issued=True,adjacent_static_support_requires_both_exposures_surface_profile_qualified=True,source_parts_binding_sha256=ref['source_parts_binding_sha256'],complete_current_RGB_profile_definition=dict(auto_points_normal_offsets_native_px=np.arange(-3,3.01,.25).tolist(),preserved_original_points_full_square_radius_native_px=3,all_samples_in_current_material_and_person_and_unknown_and_UI_qualified_domain=True),surface_and_static_both_consumed=True,source_UV_or_ID_moved=False,world_material_crosswalk_not_issued=True,source_sigma_unknown=True,source_detail_file=str(F/f'{frame["pts"]}.json'),static_detail_file=str(S/f'{frame["pts"]}.json'))
 finaldir=O/'final_frames_v5';finaldir.mkdir(exist_ok=True);write(finaldir/f'{frame["pts"]}.json',final)
 return out

def run():
 previous=None;out=[];started=time.perf_counter()
 for pts in PTS:
  current=read(pts)
  if previous is not None:qualify_pair(previous,current);out.append(finish(previous))
  previous=current
  if len(out)%25==0 and out:write(O/'static_v5_progress.json',dict(frames=len(out),native_pts=out[-1]['native_pts'],eligible_rows=sum(q['eligible_rows']for q in out),elapsed_seconds=time.perf_counter()-started));print(len(out),flush=True)
 if previous is not None:out.append(finish(previous))
 assert len(out)==1862
 small=[{k:v for k,v in r.items()if k not in ('eligible_uint8_base64','status_uint8_base64','normal_only_uint8_base64','texture_2D_tracking_uint8_base64','observation_type_uint8_base64','motion_conflict_rows')}for r in out]
 with gzip.open(O/'static_v5_index.jsonl.gz','wt',encoding='utf8')as f:
  for r in small:f.write(json.dumps(r,separators=(',',':'))+'\n')
 summary=dict(schema='rinkspace.actual_all_native_board_material_static_qualification.v1',complete=True,frames=1862,per_shot={sid:dict(frames=sum(r['shot_id']==sid for r in out),eligible_rows=sum(r['eligible_rows']for r in out if r['shot_id']==sid),semantic_normal_only_rows=sum(r['status_counts'].get(7,0)for r in out if r['shot_id']==sid),minimum_eligible_per_frame=min(r['eligible_rows']for r in out if r['shot_id']==sid),surface_excluded=sum(r['status_counts'].get(6,0)for r in out if r['shot_id']==sid),unknown=sum(r['status_counts'].get(0,0)+r['status_counts'].get(5,0)for r in out if r['shot_id']==sid),motion_conflicts=sum(r['status_counts'].get(4,0)for r in out if r['shot_id']==sid))for sid in ('s05','s06')},row_order_is_source_canonical_member_order=True,status_directory=str(S),source_rows_not_rewritten=True,no_camera_model_or_solver=True,unknown_and_motion_conflicts_retained_not_eligible=True,source_sigma_unknown=True,elapsed_seconds=time.perf_counter()-started)
 summary['final_consumer_frame_directory']=str(O/'final_frames_v5');write(O/'static_v5_manifest.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':run()
