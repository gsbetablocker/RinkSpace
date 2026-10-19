"""Audit native ice evidence and render every frame into ordered contact pages."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from calibration.program_tracking.extract_first_shot_ice import DATA,MODEL,read,write,image_line,LONG_LINE_ORDER,current_broadcast_exclusion,current_ice_visible_domain
from calibration.program_tracking.extract_continuation_ice_source import broadcast_exclusion
from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts as circle_material_conflicts

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
COLORS={
 'ice.lane4:edge0':(220,100,230),'ice.lane4:edge2':(30,30,240),
 'ice.house_connection_F4_R4:segment0':(180,40,210),
 'ice.lane3:edge0':(20,225,225),'ice.house_connection_F3_R3:segment0':(0,160,230),
 'ice.lane3:edge2':(40,215,255),'ice.lane2:edge0':(20,100,245),
 'ice.house_connection_F2_R2:segment0':(180,0,255),'ice.lane2:edge2':(60,150,250),
 'ice.lane1:edge0':(60,75,255),'ice.house_connection_F1_R1:segment0':(145,60,200),
 'ice.lane1:edge2':(40,50,180),'ice.C:segment0':(255,210,100),
 'ice.B1:segment0':(255,220,40),'ice.B2:segment0':(250,170,0)}
def overlay(index,labels=True):
 packet=read(DATA/f'ice_candidates/{index:06d}.json');bgr=cv2.imread(str(DATA/f'frames/{index:06d}.png'))
 for f in packet['features']:
  color=COLORS.get(f['target_id'],(70,200,40) if f['target_id']=='ice.C0:outer' else (215,40,170))
  for s in f['segments']:
   uv=np.asarray(s['uv']);points=np.rint(uv).astype(np.int32)
   if len(points)>1:cv2.polylines(bgr,[points],False,color,1,cv2.LINE_AA)
   for point in points:cv2.circle(bgr,tuple(point),3,color,-1,cv2.LINE_AA)
  if labels and f['segments']:
   uv=np.asarray(f['segments'][0]['uv']);pos=np.rint(uv[len(uv)//2]).astype(int)
   label=f['target_id'].removeprefix('ice.').replace('house_connection_','').replace(':segment0','')
   cv2.putText(bgr,label,tuple(pos),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
 cv2.putText(bgr,f'FRAME {index}  ICE {packet["point_count"]}',(16,1052),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2,cv2.LINE_AA)
 return bgr

def render_contact_pages(starts):
 folder=DATA/'ice_diagnostics/contact_pages';folder.mkdir(parents=True,exist_ok=True)
 for start in starts:
  canvas=np.full((4*288,4*480,3),35,np.uint8)
  for offset,index in enumerate(range(start,min(start+16,1249))):
   image=cv2.resize(overlay(index,labels=False),(480,270),interpolation=cv2.INTER_AREA)
   y,x=divmod(offset,4);canvas[y*288:y*288+270,x*480:(x+1)*480]=image
   cv2.putText(canvas,f'{index}',(x*480+10,y*288+286),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
  cv2.imwrite(str(folder/f'{start:06d}_{min(start+15,1248):06d}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])

def main():
 p=argparse.ArgumentParser();p.add_argument('--contacts',action='store_true');args=p.parse_args()
 registry=read(MODEL/'feature_registry.json');known=set(registry['targets'])
 issues=[];counts=[];geometry=Counter();rejections=Counter();coverage=defaultdict(list);removed=[];total=0
 midpoint=defaultdict(lambda:dict(native_readout_count=0,maximum_midpoint_error_px=0.,frames=set()))
 operators=defaultdict(set);timer_present=0;legacy_timer_misses=0;domain_pass=0
 for index in range(201,1249):
  packet=read(DATA/f'ice_candidates/{index:06d}.json');source=packet['source']
  maskfile=DATA/f'masks_current/{index:06d}.png';person=cv2.imread(str(maskfile),0)>0
  bgr=cv2.imread(str(DATA/f'frames/{index:06d}.png'));broadcast,timer=current_broadcast_exclusion(bgr);outside_ice,domain=current_ice_visible_domain(index)
  # Independent fixed-ROI assertion: this first-shot timer is visually
  # present throughout 201-1248. Do not let a failed component detector
  # certify its own exclusions, which was the original grey-timer failure.
  independent_timer=np.zeros((1080,1920),bool);independent_timer[940:1007,1688:1869]=True
  mask=person|broadcast|outside_ice|independent_timer
  timer_present+=bool(timer['current_source_timer_present'])
  legacy_timer_misses+=not timer['legacy_component_detector']['current_source_timer_present']
  if not source.get('ice_visible_domain',{}).get('straight_and_arc_parts_preserved'):
   issues.append(dict(frame=index,kind='ice_domain_does_not_preserve_measured_contact_parts'))
  else:domain_pass+=1
  if source.get('ice_visible_domain',{}).get('current_source_contact_material_uv_sha256')!=domain['current_source_contact_material_uv_sha256']:
   issues.append(dict(frame=index,kind='stale_source_board_contact_visible_domain'))
  if not source.get('broadcast_timer_exclusion',{}).get('current_source_timer_present'):
   issues.append(dict(frame=index,kind='visible_broadcast_timer_detector_miss'))
  if source['mask_sha256']!=digest(maskfile):issues.append(dict(frame=index,kind='stale_mask_hash'))
  if source['native_pts']!=256256+(index-1)*256:issues.append(dict(frame=index,kind='source_pts_mismatch'))
  if packet['programme_frame_index_1based']!=index:issues.append(dict(frame=index,kind='frame_index_mismatch'))
  points=0
  for collision in circle_material_conflicts(packet['features']):
   issues.append(dict(frame=index,kind='extra_circle_reads_existing_house_fill_band',target_id=collision['target_id'],uv=collision['uv']))
  for f in packet['features']:
   tid=f['target_id'];coverage[tid].append(index)
   operators[tid].add(f['measurement_operator'])
   if tid.startswith(('ice.R','ice.F')) and f['measurement_operator']!='material_interface':issues.append(dict(frame=index,kind='house_ring_not_material_contour_interface',target_id=tid))
   if tid not in known:issues.append(dict(frame=index,kind='unregistered_target',target_id=tid))
   n=sum(len(s['uv']) for s in f['segments']);points+=n
   if n!=f['point_count']:issues.append(dict(frame=index,kind='feature_count_mismatch',target_id=tid))
   evidence=[]
   for s in f['segments']:
    uv=np.asarray(s['uv']);inside=(uv>=[0,0]).all(1)&(uv<=[1919,1079]).all(1)
    if not inside.all():issues.append(dict(frame=index,kind='source_uv_outside_native',target_id=tid))
    ip=np.rint(uv[inside]).astype(int)
    if mask[ip[:,1],ip[:,0]].any():issues.append(dict(frame=index,kind='representative_hits_person_or_broadcast',target_id=tid))
    if s['vertex_count']!=len(uv):issues.append(dict(frame=index,kind='segment_count_mismatch',target_id=tid))
    if 'source_line_fit' not in f:evidence.extend(s.get('source_evidence',[]))
   if 'source_line_fit' in f:
    evidence.extend(r['source_evidence'] for r in f['source_line_fit']['direct_readouts'])
    if f['source_line_fit']['actual_native_readout_count']!=len(f['source_line_fit']['direct_readouts']):issues.append(dict(frame=index,kind='line_readout_count_mismatch',target_id=tid))
    for row in f['source_line_fit']['direct_readouts']:
     direct_uv=np.asarray(row['uv']);direct_inside=(direct_uv>=[0,0]).all() and (direct_uv<=[1919,1079]).all()
     if not direct_inside:issues.append(dict(frame=index,kind='direct_native_line_readout_outside_native',target_id=tid))
     elif mask[tuple(np.rint(direct_uv).astype(int)[::-1])]:issues.append(dict(frame=index,kind='direct_native_line_readout_hits_mask',target_id=tid))
     error=float(np.linalg.norm(np.asarray(row['uv'])-np.asarray(row['source_evidence']['actual_source_sides_uv']).mean(0)))
     st=midpoint[tid];st['native_readout_count']+=1;st['maximum_midpoint_error_px']=max(st['maximum_midpoint_error_px'],error);st['frames'].add(index)
     if error>8e-5:issues.append(dict(frame=index,kind='paint_center_direct_readout_not_two_signed_interface_midpoint',target_id=tid,error_px=error))
   for row in evidence:
    context=np.asarray(row['actual_source_side_quantiles_uv']).reshape(-1,2)
    inside=(context>=[0,0]).all(1)&(context<=[1919,1079]).all(1)
    if not inside.all():issues.append(dict(frame=index,kind='source_evidence_outside_native',target_id=tid));continue
    ip=np.rint(context).astype(int)
    if mask[ip[:,1],ip[:,0]].any():issues.append(dict(frame=index,kind='source_evidence_hits_mask',target_id=tid))
   if f.get('parallel_family_geometry_check',{}).get('orientation_departure_over_readout_span_px',0)>8.00001:
    issues.append(dict(frame=index,kind='parallel_family_check_failed',target_id=tid))
  report={r['target_id']:r for r in packet['per_feature_coverage']}
  for tid,row in report.items():
   actual=sum(f['point_count'] for f in packet['features'] if f['target_id']==tid)
   if row['points']!=actual:issues.append(dict(frame=index,kind='coverage_count_mismatch',target_id=tid))
   if not actual and not row.get('missing_reason'):issues.append(dict(frame=index,kind='unexplained_visible_feature_missing',target_id=tid))
  if points!=packet['point_count']:issues.append(dict(frame=index,kind='packet_count_mismatch'))
  if points>170:issues.append(dict(frame=index,kind='ice_representative_cap_exceeded',points=points))
  geometry[packet.get('parallel_family_geometry_audit',{}).get('status','missing')]+=1
  removed.extend(dict(frame=index,**r) for r in packet.get('representative_cap_removals',[]))
  rejected_path=packet.get('rejected_material_candidate_report_path')
  if rejected_path:
   for r in read(rejected_path)['rejected_candidates']:rejections[r['reason']]+=1
  counts.append(points);total+=points
 report=dict(schema='rinkspace.first_shot_ice_source_audit.v1',frame_range=[201,1248],native_frame_count=1048,
   issue_count=len(issues),issues=issues,point_count_range=[min(counts),max(counts)],mean_point_count=float(np.mean(counts)),total_representatives=total,
   parallel_family_status=dict(geometry),rejected_material_candidate_counts=dict(rejections),representative_cap_removals=removed,
   target_frame_coverage={k:dict(frame_count=len(v),first_frame=min(v),last_frame=max(v)) for k,v in sorted(coverage.items())},
   paint_center_line_midpoint_checks={k:dict(native_readout_count=v['native_readout_count'],maximum_midpoint_error_px=v['maximum_midpoint_error_px'],passed_frame_count=len(v['frames'])) for k,v in sorted(midpoint.items())},
   maximum_native_line_midpoint_rounding_error_px=max(v['maximum_midpoint_error_px'] for v in midpoint.values()),
   material_operator_by_target={k:sorted(v) for k,v in sorted(operators.items())},
   current_timer_present_frame_count=timer_present,legacy_timer_component_missed_frame_count=legacy_timer_misses,
   piecewise_source_board_contact_domain_checked_frame_count=domain_pass,
   checked_every_frame=['native PTS','target registry','float UV bounds','latest person mask hash','person and broadcast exclusion for representatives and signed-material quantiles',
     'independent visually confirmed same-shot fixed timer ROI','piecewise actual measured board-contact ice-visible domain including arc parts',
     'all packet/feature/segment/coverage counts','parallel-family geometry acceptance','explanation for every searched but absent feature','two signed-interface midpoint of every paint-center native line readout','house R/F ring contour operators',
     'extra A outline circles do not reuse outer-middle or middle-inner filled house band interfaces as separate outline centers'],
   visual_review=dict(all_frames_contact_pages_rendered=False,manually_inspected_page_ranges=[],full_resolution_manually_inspected_frames=[]),
   limitations=['Source native material and shape consistency establish candidate observations; user approval remains pending.',
     'A geometrically ambiguous source reflection can be rejected even though a model guide exists.',
     '500 F2-R2 lower-left short visible navigation segment has zero eligible two-sided signed RGB material readouts at the minimum accepted source contrast; it is explicitly absent rather than copied from the manual guide.'])
 if args.contacts:
  render_contact_pages(range(201,1249,16))
  for i in (300,400,500,800,1000,1248):cv2.imwrite(str(DATA/f'ice_diagnostics/{i:06d}_ice_overlay.png'),overlay(i))
  report['visual_review']['all_frames_contact_pages_rendered']=True
 write(DATA/'ice_diagnostics/final_ice_audit_summary.json',report)
 print(json.dumps({k:report[k] for k in ('native_frame_count','issue_count','point_count_range','parallel_family_status','rejected_material_candidate_counts')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
