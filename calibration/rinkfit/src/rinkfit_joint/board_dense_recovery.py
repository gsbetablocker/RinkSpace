"""Independent current-RGB recovery of continuous paired board color bands.

This branch reads no optical flow, camera projection, or pixel template from
another exposure. Both narrow color strips and the neutral board body between
them must form the same long contiguous current source interval.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,hashlib
import cv2,numpy as np
from scipy.signal import find_peaks
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .source_worker import _channels,_sample,_remeasure
from .role_guard import FrameRoleGuard

CONFIG={'version':'current_paired_continuous_board_bands_v1','source_column_step_px':8,'minimum_contiguous_columns':33,'maximum_adjacent_band_change_px':12,'board_body_height_px':[24,600],'material_band_thickness_px':[2,48],'both_current_material_bands_required':True,'darker_external_background_above_blue_and_neutral_ice_below_yellow_required':True,'source_actor_or_graphics_gaps_bridged':False,'cross_exposure_pixel_template_used':False,'model_or_camera_projection_used':False}

def source_band_paths(rgb,mask):
 channels=_channels(rgb);blue=channels[2];luma=channels[0];height,width=mask.shape;grad=np.gradient(blue,axis=0);runs=[];active=[]
 for x in range(8,width-8,8):
  extrema=find_peaks(-grad[:,x],height=.75,prominence=.5,distance=3)[0];top=[];bottom=[]
  for y in extrema:
   if not 180<y<980 or mask[y-8:y+9,x-3:x+4].any():continue
   above=float(np.median(blue[y-7:y-2,x-1:x+2]));below=float(np.median(blue[y+2:y+7,x-1:x+2]));score=above-below
   if score<6:continue
   if above>8 and below>-12 and np.median(luma[y+2:y+7,x-1:x+2])>70:
    thick=0
    for yy in range(y-2,max(0,y-55),-1):
     if blue[yy,x]>6:thick+=1
     else:break
    if 2<=thick<=48:top.append((y,score,thick))
   if below<-8 and above<20 and np.median(luma[y-7:y-2,x-1:x+2])>70:
    thick=0
    for yy in range(y+2,min(height,y+55)):
     if blue[yy,x]<-6:thick+=1
     else:break
    if 2<=thick<=48:bottom.append((y,score,thick))
  pairs=[]
  for a,sa,ta in top:
   for b,sb,tb in bottom:
    if not 24<=b-a<=600:continue
    body=blue[a+8:b-8,x-2:x+3];light=luma[a+8:b-8,x-2:x+3]
    if body.size and np.median(abs(body))<20 and np.median(light)>80:
     upper=a-ta-5;lower=b+tb+5
     if not(4<upper<1075 and 4<lower<1075) or mask[upper-2:upper+3,x-2:x+3].any() or mask[lower-2:lower+3,x-2:x+3].any():continue
     background=float(np.median(luma[upper-2:upper+3,x-2:x+3]));ice=float(np.median(luma[lower-2:lower+3,x-2:x+3]));icecolor=float(np.median(abs(blue[lower-2:lower+3,x-2:x+3])));board=float(np.median(light))
     if background<=board-20 and ice>=board-30 and icecolor<20:pairs.append((a,b,min(sa,sb),ta,tb))
  # The uppermost narrow blue strip above the same neutral board/yellow body
  # has priority over internal lettering. Long uninterrupted support is required.
  pairs.sort(key=lambda x:x[0]);next_active=[];used=set()
  for path in active:
   last=path[-1];eligible=[(abs(a-last[1])+abs(b-last[2]),j,pair) for j,pair in enumerate(pairs) for a,b,*_ in [pair] if j not in used and abs(a-last[1])<=12 and abs(b-last[2])<=12]
   if eligible:
    _,j,pair=min(eligible,key=lambda x:x[0]);used.add(j);next_active.append(path+[(x,*pair)])
   else:
    if len(path)>=33:runs.append(path)
  for j,pair in enumerate(pairs):
   if j not in used:next_active.append([(x,*pair)])
  active=next_active
 for path in active:
  if len(path)>=33:runs.append(path)
 # Overlapping lower blue letter paths are not separate cap interfaces.
 kept=[]
 for path in sorted(runs,key=lambda r:(r[0][0],np.median([v[1] for v in r]))):
  xs={p[0] for p in path};higher=any(len(xs&{v[0] for v in old})>=.8*len(xs) and np.median([v[1] for v in old])<np.median([v[1] for v in path])-3 for old in kept)
  if not higher:kept.append(path)
 return kept,channels

def recover_board_gaps(run,material_source,gap_receipts,outdir):
 run=Path(run);mat=Path(material_source);out=Path(outdir)
 if out.exists():raise ValueError('New immutable board-native source generation required')
 targets=[r for r in read_jsonl(gap_receipts) if not r['source_rows']];guard=FrameRoleGuard(run/'roles/frame_roles.json');mapping=read_json(mat/'material_map.json')['semantic_to_material'];registry={m['id']:m for m in read_json(mat/'joint_material_registry.json')['materials']};ledgers={r['native_pts']:r for n in ['dense_s03_v1','dense_remaining_v1'] for r in read_jsonl(run/n/'frame_ledger.jsonl.gz')};sources=[run/n for n in ['dense_image_match_v2','dense_image_match_v1','dense_remaining_reacquire_v2']];out.mkdir(parents=True);rows=[];receipts=[];profiles=[];index=[];puv=[];pn=[];pch=[];off=np.arange(-3,3.0001,.25);toff=np.arange(-2,3,dtype=float);counts=Counter()
 for frame in targets:
  pts=int(frame['native_pts']);sid=frame['shot_id'];guard.assert_training_pixels([pts],shot_id=sid);ip=next((d/'native_cache'/f'{pts}.png' for d in sources if (d/'native_cache'/f'{pts}.png').is_file()),None);mp=next((d/'mask_cache'/f'{pts}.png' for d in sources if (d/'mask_cache'/f'{pts}.png').is_file()),None)
  if ip is None or mp is None:raise ValueError('Expected cached actual source/mask not present')
  bgr=cv2.imread(str(ip));bgrsha=hashlib.sha256(bgr.tobytes()).hexdigest()
  if bgrsha!=ledgers[pts]['decoded_native_BGR_sha256']:raise ValueError('Native source RGB identity mismatch')
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)>0;paths,channels=source_band_paths(rgb,mask);current=[];cert=[]
  for pathidx,path in enumerate(paths):
   uv=np.asarray([[x,yb,yy] for x,yb,yy,*_ in path],float);cert.append({'source_path_index':pathidx,'current_paired_material_band_points_uv':uv.tolist(),'actual_band_column_data_x_blueY_yellowY_contrast_blueThickness_yellowThickness':[[float(x) for x in v] for v in path],'native_png':str(ip),'native_png_sha256':file_sha256(ip),'current_mask_png':str(mp),'current_mask_png_sha256':file_sha256(mp),'source_BGR_sha256':bgrsha,'configuration':CONFIG,'roles_share_same_current_board_body':True,'other_exposure_or_world_model_used':False})
   for ri,role in enumerate(['blue_cap_body_interface','yellow_strip_body_interface']):
    target='board.'+role;key='|'.join([target,'board','material_interface',role])
    if not registry[key]['identity_accepted']:raise ValueError('Physical material role not source-certified')
    for j in range(1,len(path)-1,2):
     q=np.asarray([uv[j,0],uv[j,ri+1]]);d=np.asarray([uv[j+1,0]-uv[j-1,0],uv[j+1,ri+1]-uv[j-1,ri+1]]);normal=np.array([-d[1],d[0]]);normal/=np.linalg.norm(normal);r=_remeasure(q,normal,channels,'material_interface')
     if r.get('current_uv') is None:continue
     point=np.asarray(r['current_uv']);u,v=np.rint(point).astype(int)
     if mask[v,u]:continue
     # Verify the same actual role again at the newly measured material edge.
     side=_sample(channels,point+np.asarray([-5,-3,3,5])[:,None]*normal)[:,2];high=np.mean(side[:2]);low=np.mean(side[2:]);blueok=high>8 and low>-12 and high-low>6;yellowok=low<-8 and high<20 and high-low>6
     if not(blueok if ri==0 else yellowok):continue
     rid=f'dense_board/{sid}/{pts}/{role}/{pathidx}/{j}';family=f'dense_board/{sid}/{pts}/{role}';row={'shot_id':sid,'native_pts':pts,'time_base':'1/12800','source_row_id':rid,'source_family_id':family,'joint_material_id':key,'material_index':mapping[key]['material_index'],'target_id':target,'measurement_operator':'material_interface','point_correspondence':False,'source_uv':point.tolist(),'current_uv':point.tolist(),'navigation_seed_uv':q.tolist(),'source_measurement_audit':r,'anchor_source_row_id':None,'anchor_pts':pts,'source_acquisition_mode':'current_native_physical_material_topology','source_identity_basis':'current_native_paired_board_material_topology','native_material_identity_certificate_index':pathidx,'native_material_identity_current_RGB_verified':True,'decoded_native_BGR_sha256':bgrsha,'current_pixels_measured':True,'pixels_from_model':False,'numeric_statistics_ready':False,'solver_eligible':False,'role':'development','pending_reason':'current source-family covariance and same shared board height physical replay required'};current.append(row);rows.append(row);n=np.asarray(r['normal']);t=np.array([-n[1],n[0]]);coords=point+toff[:,None,None]*t+off[None,:,None]*n;allc=_sample(channels,coords.reshape(-1,2)).reshape(5,25,3);ch=int(abs(np.gradient(allc[2],.25,axis=0)).max(0).argmax());index.append({'profile_index':len(profiles),'source_row_id':rid,'source_family_id':family,'shot_id':sid,'native_pts':pts,'side':0,'decoded_native_BGR_sha256':bgrsha});profiles.append(allc[:,:,ch]);puv.append(point);pn.append(n);pch.append(ch)
  receipts.append({'shot_id':sid,'native_pts':pts,'source_rows':len(current),'status':'current_paired_board_material_source_recovered' if current else 'current_paired_board_material_topology_not_supported','source_BGR_sha256':bgrsha,'source_material_certificates':cert,'native_png':str(ip),'native_png_sha256':file_sha256(ip),'current_mask_png':str(mp),'current_mask_png_sha256':file_sha256(mp),'protected_pixels_consumed':False});counts['source_recovered_frames' if current else 'source_still_pending_frames']+=1;counts['actual_current_board_interface_rows']+=len(current);print(json.dumps({'native_pts':pts,'completed':len(receipts),'counts':dict(counts)}),flush=True)
 write_jsonl(out/'rows.jsonl.gz',rows);write_jsonl(out/'frame_receipts.jsonl.gz',receipts);write_json(out/'profile_index.json',index);np.savez_compressed(out/'profiles.npz',intensity=np.asarray(profiles,np.float32).reshape(-1,5,25),uv=np.asarray(puv,float).reshape(-1,2),normal=np.asarray(pn,float).reshape(-1,2),channel=np.asarray(pch,np.uint8),normal_offsets=off,tangent_offsets=toff);summary={'schema':'rinkfit.joint9.current_native_paired_board_recovery.v1','input_gap_frames':len(targets),'counts':dict(counts),'all_gap_frames_accounted':len(receipts)==len(targets),'no_other_exposure_template_or_flow_used':True,'no_world_or_camera_source_predictions_used':True,'protected_pixels_consumed':False,'source_uv_is_first_actual_native_measurement':True,'numeric_statistics_ready':False,'physical_model_requires_shared_height_fraction':True};write_json(out/'manifest.json',summary);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--material-source',required=True);p.add_argument('--gap-receipts',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(recover_board_gaps(a.run,a.material_source,a.gap_receipts,a.out)))
