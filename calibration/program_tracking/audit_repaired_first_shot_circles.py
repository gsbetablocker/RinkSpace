"""Independently audit frozen repaired circle evidence and render ordered source pages."""
from calibration.program_tracking.extract_first_shot_ice import *
from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts
from collections import defaultdict,Counter

def conic_signed(points,geometry):
 angle=np.deg2rad(geometry['angle_degrees']);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 Q=R@np.diag(1/(np.asarray(geometry['axes_px'])*.5)**2)@R.T;d=np.asarray(points)-geometry['center_uv']
 return np.einsum('ij,ij->i',d@Q,d)-1

def circle_overlay(index,cache_root):
 packet=read(cache_root/f'{index:06d}.json');image=cv2.imread(str(DATA/f'frames/{index:06d}.png'))
 for cid,f in qualify_extra_circle_material(packet['features_by_curve']).items():
  if not f['point_count']:continue
  role=cid.rsplit(':',1)[1];colour={'outer':(0,50,255),'middle':(30,240,30),'inner':(255,200,0)}.get(role,(200,60,230))
  if f['target_id']=='ice.C0:outer':colour=(30,210,255)
  if f['target_id'].startswith('ice.A'):colour=(255,60,200)
  for segment in f['segments']:
   pts=np.rint(segment['uv']).astype(np.int32)
   if len(pts)>1:cv2.polylines(image,[pts],False,colour,1,cv2.LINE_AA)
   for point in pts:cv2.circle(image,tuple(point),3,colour,-1,cv2.LINE_AA)
  pos=np.rint(f['segments'][0]['uv'][0]).astype(int);cv2.putText(image,cid,tuple(pos),0,.35,colour,1,cv2.LINE_AA)
 cv2.putText(image,str(index),(20,1045),0,.7,(0,255,255),2,cv2.LINE_AA)
 return image

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-root',required=True);ap.add_argument('--contacts',action='store_true');args=ap.parse_args();root=Path(args.output_root);cache=root/'circle_candidates';out=root/'circle_diagnostics';out.mkdir(exist_ok=True)
 issues=[];counts=[];coverage=defaultdict(list);midpoint=0.;maxdev=0.;direct_count=0;nesting=0;material_reject=0;frames=0;changes=[]
 for index in range(201,1249):
  file=cache/f'{index:06d}.json'
  if not file.exists():continue
  packet=read(file);source=packet['source'];bgr=cv2.imread(str(DATA/f'frames/{index:06d}.png'));person=cv2.imread(source['mask_path'],0)>0;bc,_=current_broadcast_exclusion(bgr);dom,domain=current_ice_visible_domain(index);mask=person|bc|dom
  if packet.get('algorithm_sha256')!=sha(ROOT/'calibration/program_tracking/native_ice_material_repair.py'):issues.append(dict(frame=index,kind='stale_circle_measurement_algorithm'))
  if sha(source['mask_path'])!=source['mask_sha256']:issues.append(dict(frame=index,kind='stale_mask_hash'))
  if source['native_pts']!=256256+(index-1)*256:issues.append(dict(frame=index,kind='wrong_source_clock'))
  identity_rejections=conflicts([f for f in packet['features_by_curve'].values() if f['point_count']]);material_reject+=len(identity_rejections)
  qualified=qualify_extra_circle_material(packet['features_by_curve'])
  features=[f for f in qualified.values() if f['point_count']];byid={f['target_id']:f for f in features};count=0
  for f in features:
   tid=f['target_id'];coverage[tid].append(index);count+=f['point_count'];rows=f['source_circle_fit']['direct_readouts'];direct_count+=len(rows)
   for row in rows:
    actual=np.asarray(row['uv']);e=row['source_evidence'];sides=np.asarray(e['actual_source_sides_uv']);dev=float(np.linalg.norm(actual-sides.mean(0)));midpoint=max(midpoint,dev)
    if dev>8e-5:issues.append(dict(frame=index,target_id=tid,kind='native_readout_not_interface_midpoint',error=dev))
    context=np.vstack([actual[None],sides,np.asarray(e['actual_source_side_quantiles_uv']).reshape(-1,2)])
    if (context<[0,0]).any() or (context>[1919,1079]).any():issues.append(dict(frame=index,target_id=tid,kind='material_out_of_native'));continue
    ip=np.rint(context).astype(int)
    if mask[ip[:,1],ip[:,0]].any():issues.append(dict(frame=index,target_id=tid,kind='raw_material_hits_mask_or_above_contact'))
    dev=row['representative_to_actual_material_px'];maxdev=max(maxdev,dev)
    if dev>3.0002:issues.append(dict(frame=index,target_id=tid,kind='source_fit_exceeds_material_tolerance',error=dev))
   for segment in f['segments']:
    uv=np.asarray(segment['uv']);ip=np.rint(uv).astype(int)
    if (uv<[0,0]).any() or (uv>[1919,1079]).any():issues.append(dict(frame=index,target_id=tid,kind='representative_out_of_native'));continue
    if mask[ip[:,1],ip[:,0]].any():issues.append(dict(frame=index,target_id=tid,kind='representative_hits_mask_or_above_contact'))
    if segment['individual_native_pixel_readouts'] or segment['pixels_measured_in_this_frame']:issues.append(dict(frame=index,target_id=tid,kind='fitted_representative_misrepresented_as_direct_pixel'))
   if tid.startswith(('ice.F','ice.R')):
    if f['measurement_operator']!='material_interface':issues.append(dict(frame=index,target_id=tid,kind='filled_ring_not_material_contour'))
    outer=byid.get(tid.split(':')[0]+':outer')
    if not outer or tid.endswith(':outer'):continue
    pts=np.concatenate([np.asarray(seg['uv']) for seg in f['segments']]);sign=conic_signed(pts,outer['geometric_interpretation']);nesting+=1
    if (sign>1e-5).any():issues.append(dict(frame=index,target_id=tid,kind='nested_ring_outside_same_family_outer',outside_count=int((sign>1e-5).sum())))
    if f['joint_group_navigation']['shared_source_affine_parameters']!=outer['joint_group_navigation']['shared_source_affine_parameters']:issues.append(dict(frame=index,target_id=tid,kind='independent_ring_projection_not_shared'))
  counts.append(count);frames+=1
  old=read(DATA/f'ice_candidates/{index:06d}.json');oldc={f['target_id']:f['point_count'] for f in old['features'] if f.get('geometric_interpretation',{}).get('kind')=='projected_physical_circle_ellipse'};newc={f['target_id']:f['point_count'] for f in features}
  changes.append(dict(frame=index,old_counts=oldc,new_counts=newc,reason='current whole narrow-circle paint midpoint or declared filled-house interface under one shared current source circle family; no independent layer fitting or prior-frame shape state'))
 result=dict(schema='rinkspace.repaired.circle.current_native_audit.v1',checked_frame_count=frames,issue_count=len(issues),issues=issues,point_count_range=[min(counts),max(counts)] if counts else None,total_circle_representatives=sum(counts),direct_material_readouts=direct_count,
  maximum_paint_center_midpoint_rounding_error_px=midpoint,maximum_representative_to_actual_native_readout_px=maxdev,nested_family_pair_count=nesting,
  filled_annulus_conflicts_for_final_merge_filter=material_reject,coverage={k:dict(frame_count=len(v),first=min(v),last=max(v)) for k,v in coverage.items()},
  visual_review=dict(contact_pages_rendered=False,actually_inspected_pages=[],actually_inspected_native_frames=[]),
  limitation='This machine audit checks evidence and shared geometry. Actual source identity and completeness require ordered native visual and independent source-motion review; source measurements remain awaiting user review.')
 write(out/'candidate_changes.json',dict(changes=changes));write(out/'circle_material_audit.json',result)
 if args.contacts:
  folder=out/'contact_pages';folder.mkdir(exist_ok=True)
  for start in range(201,1249,16):
   if not (cache/f'{min(start+15,1248):06d}.json').exists():continue
   canvas=np.zeros((4*288,4*480,3),np.uint8)
   for offset,index in enumerate(range(start,min(start+16,1249))):
    image=cv2.resize(circle_overlay(index,cache),(480,270),interpolation=cv2.INTER_AREA);y,x=divmod(offset,4);canvas[y*288:y*288+270,x*480:x*480+480]=image;cv2.putText(canvas,str(index),(x*480+10,y*288+286),0,.45,(230,230,230),1)
   cv2.imwrite(str(folder/f'{start:06d}_{min(start+15,1248):06d}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
  result['visual_review']['contact_pages_rendered']=True;write(out/'circle_material_audit.json',result)
 print(json.dumps({k:result[k] for k in ('checked_frame_count','issue_count','point_count_range','maximum_representative_to_actual_native_readout_px','nested_family_pair_count')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
