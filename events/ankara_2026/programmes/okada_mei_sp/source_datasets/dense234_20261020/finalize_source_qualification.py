"""Immutable next source generation with an observed actor-boundary uncertainty guard."""
from pathlib import Path
from collections import Counter
import gzip,json,hashlib,time,cv2,numpy as np
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense234_v1');R=O.parent;A=O/'generation_v1';B=O/'generation_v2'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def run():
 parent=read(A/'manifest.json')
 if not parent['complete']:raise ValueError('Parent source generation not sealed')
 if B.exists():raise FileExistsError(B)
 source_frames=A/'selected_frames.jsonl.gz';source_rows=A/'selected_rows.jsonl.gz'
 for p in(source_frames,source_rows):
  if sha(p)!=parent['files'][p.name]['sha256']:raise ValueError('Parent source rows changed')
 binding=A/'consumer_bindings.jsonl.gz'
 if sha(binding)!=parent['files'][binding.name]['sha256']:raise ValueError('Parent mask bindings changed')
 with gzip.open(binding,'rt',encoding='utf8')as bf:mask_bindings={r['native_pts']:r['mask']for r in map(json.loads,bf)}
 p_sha=sha(A/'manifest.json');recipe_sha=sha(__file__);B.mkdir();removed={};counts=[];byshot={};cl=Counter();t=time.perf_counter();pcount=0;delta=gzip.open(B/'eligibility_delta.jsonl.gz','wt',encoding='utf8',compresslevel=4);fr=gzip.open(B/'selected_frames.jsonl.gz','wt',encoding='utf8',compresslevel=4);cov=gzip.open(B/'coverage.jsonl.gz','wt',encoding='utf8',compresslevel=4)
 with gzip.open(source_frames,'rt',encoding='utf8')as f:
  for l in f:
   row=json.loads(l);p=row['native_pts'];mask=cv2.imread(str(R/'dense_masks_v1/masks'/f'{p}.png'),0)
   if mask is None:raise ValueError('Missing actual native person mask')
   if sha(mask_bindings[p]['file'])!=mask_bindings[p]['sha256']:raise ValueError('Person pixel mask changed')
   distance=cv2.distanceTransform((mask==0).astype('uint8'),cv2.DIST_L2,5);kept=[];before=len(row['points'])
   for point in row['points']:
    x,y=np.rint(point['uv']).astype(int);d=float(distance[y,x])
    if d<=8.:
     r=dict(shot_id=row['shot_id'],native_pts=p,selected_row_id=point['selected_row_id'],uv=point['uv'],reason='current_actor_silhouette_blur_boundary_not_certified_static_material',distance_to_actual_person_mask_native_px=d,guard_native_px=8.,guard_is_engineering_source_visibility_domain_not_sigma=True,source_UV_unchanged=True,parent_source_frame=str(source_frames),actual_counterexample_source_pts=745728)
     delta.write(json.dumps(r,separators=(',',':'))+'\n');removed[point['selected_row_id']]=r
    else:kept.append(point)
   row['points']=kept;fr.write(json.dumps(row,separators=(',',':'))+'\n');cov.write(json.dumps(dict(shot_id=row['shot_id'],native_pts=p,mask_ref=mask_bindings[p],before=before,selected=len(kept),excluded_actual_actor_blur_boundary=before-len(kept),source_UV_not_moved=True),separators=(',',':'))+'\n');pcount+=1;counts.append(len(kept));byshot[row['shot_id']]=byshot.get(row['shot_id'],0)+len(kept);cl.update(x['display_class']for x in kept)
 delta.close();fr.close();cov.close();outrows=gzip.open(B/'selected_rows.jsonl.gz','wt',encoding='utf8',compresslevel=4);rowcount=0;found=set();bykind=Counter();bypart=Counter();manual=0
 with gzip.open(source_rows,'rt',encoding='utf8')as f:
  for l in f:
   row=json.loads(l);rid=row['selected_row_id']
   if rid in removed:found.add(rid);continue
   outrows.write(l);rowcount+=1;bykind[row['source_kind']]+=1;bypart[row['physical_surface']]+=1;manual+=row['source_kind']=='preserved_user_manual_native_image_click'
 outrows.close()
 if found!=set(removed)or pcount!=1458 or rowcount!=sum(counts):raise ValueError('Final source point membership did not bind exact input rows')
 if sha(A/'manifest.json')!=p_sha or sha(__file__)!=recipe_sha:raise ValueError('Immutable qualification input changed')
 files={p.name:dict(file=str(p),sha256=sha(p))for p in B.glob('*.gz')};cfg=read(A/'video_config.json');cfg['selected_frame_file']=files['selected_frames.jsonl.gz'];write(B/'video_config.json',cfg)
 manifest=dict(schema='rinkspace.source_only_dense234_generation.v2',complete=True,frames=pcount,point_rows=rowcount,per_frame=dict(min=min(counts),max=max(counts)),by_shot=byshot,by_physical_surface=dict(bypart),source_kind_counts=dict(bykind),manual_source_rows=manual,display_classes=dict(cl),parent_generation=dict(file=str(A/'manifest.json'),sha256=p_sha),source_files=files,source_UV_moved=False,new_points_issued=False,physical_solver_or_packet_changed=False,source_qualifier=dict(file=str(Path(__file__).resolve()),sha256=recipe_sha),actual_actor_boundary_support_case=dict(native_pts=745728,source_review_image=str(A/'source_QA/raised_boot_check.jpg'),toe_and_boot_boundary_return_distance_native_px=[7.3938,4.9969],actual_person_mask_file=str(R/'dense_masks_v1/masks/745728.png'),guard_native_px=8,domain_is_not_calibrated_noise_or_bbox=True),source_row_exclusions=len(removed),all_visible_features_perfect_claimed=False,elapsed_seconds=time.perf_counter()-t)
 write(B/'manifest.json',manifest);print(json.dumps(dict(frames=pcount,rows=rowcount,per_frame=manifest['per_frame'],by_shot=byshot,excluded=len(removed),files=files,elapsed_seconds=manifest['elapsed_seconds'])))
if __name__=='__main__':run()
