"""Recover the source-visible lane 1 long edge using physical-width profiles."""
from __future__ import annotations
import argparse,hashlib,json
from copy import deepcopy
from pathlib import Path
import cv2,numpy as np
from .extract_first_shot_ice import (DATA,ROOT,Extractor,NativeProfiles,read,write,sha,
 current_broadcast_exclusion,current_ice_visible_domain,_source_queries,image_line)
from .native_ice_material_repair import narrow_dark_paint
from .physical_ice_width import (projected_width_profile,sample_physical_width_field,
 restore_physical_profile_readout,source_at_profile_offsets,valid_profile_contributors)

RD=DATA/'repair_20261014';OUT=RD/'temporal_diagnostics/lane1_physical_width'
MODEL=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014'
TARGET='ice.lane1:edge0'

class PhysicalLaneProfiles(NativeProfiles):
 def __init__(self,rgb,source,exclusion,camera):
  super().__init__(rgb,source,exclusion);self.camera=camera
  f=np.asarray(self.frame.rgb,np.float32);self.luminance=.2126*f[...,0]+.7152*f[...,1]+.0722*f[...,2]
 def sample(self,path,guide,radius=90,spacing=55):
  h,w=self.frame.rgb.shape[:2]
  queries,_,_=_source_queries(path,guide,np.ones(len(guide)),(h,w),spacing,90)
  rows=[]
  for k,query in enumerate(queries):
   section=projected_width_profile(self.camera,path,query,image_hw=(h,w),radius_px=2.)
   if section is None:continue
   g0,g1=np.asarray(guide)[[0,-1]];v=g1-g0;v/=np.linalg.norm(v);normal=np.array([-v[1],v[0]])
   alignment=float(abs(section['query_width_direction_uv']@normal))
   if alignment<.04:continue
   # Preserve the physical transverse direction. Its search length must
   # increase at an oblique view; 90 image pixels along that axis otherwise
   # cover just a few pixels across the visible band. Local reader offsets
   # are expressed in normal-coverage units so a 75 px wide-source gate is
   # not misapplied to a long, foreshortened physical section.
   section=projected_width_profile(self.camera,path,query,image_hw=(h,w),radius_px=60./alignment,step_px=1./alignment)
   if section is None or section['valid'].sum()<20:continue
   section['offsets']=section['offsets']*alignment
   values=sample_physical_width_field(self.frame.rgb,section,'dark',field=self.luminance)
   # Every crossing is first localized in this current native luminance
   # profile; its scalar coordinates are then mapped to the exact section.
   valid=valid_profile_contributors(section,self.mask);edges=np.flatnonzero(np.r_[True,valid[1:]!=valid[:-1],True]);measured=[]
   for first,last in zip(edges[:-1],edges[1:]):
    if valid[first]and last-first>=15:
     measured.extend(narrow_dark_paint(np.zeros(2),np.array([1.,0.]),section['offsets'][first:last],values[first:last],np.ones(last-first,bool)))
   for scalar in measured:
    ev=scalar['source_evidence'];lo,hi=ev['complete_band_native_interval_px']
    row=restore_physical_profile_readout(self.camera,section,scalar)
    if row is None:continue
    e=row['source_evidence'];sides=np.asarray(e['actual_source_sides_uv']);width=float(np.linalg.norm(sides[1]-sides[0]))
    uv=np.asarray(row['uv'])
    # This query is a known physical long edge. Adjacent lane/circle bands
    # cannot acquire its identity merely by being a strong dark valley.
    distance=float(abs((uv-query)@normal))
    if distance>30 or width<2 or width>105:continue
    pixel=np.rint(uv).astype(int);r,g,b=np.asarray(self.frame.rgb[pixel[1],pixel[0]],float)
    blue=b-(r+g)*.5;red=r-(b+g)*.5
    if blue < -10. and red > 12.:
     # This named long edge is grey-blue paint. A dark luminance valley
     # entirely inside the independently identified broad red C strip is
     # not evidence for the missing blue edge (1235--1236 source case).
     continue
    support=source_at_profile_offsets(section,np.arange(np.ceil(lo),np.floor(hi)+1))
    context=np.vstack([support,np.asarray(e['actual_source_side_quantiles_uv']).reshape(-1,2)])
    if (context<[0,0]).any()or(context>[w-1,h-1]).any():continue
    ip=np.rint(context).astype(int)
    if self.mask[ip[:,1],ip[:,0]].any():continue
    e.pop('filled_house_band_is_not_an_outline',None)
    e.update(query_index=k,material_signal='dark',joint_band_width_px=width,
      measurement_origin='current_native_source_identified_lane_long_edge_two_luminance_flanks_physical_transverse_section',
      physical_measurement='physical_width_midpoint_of_current_paint_ice_interfaces',
      source_identity_normal_distance_px=distance,
      transverse_profile_image_normal_coverage_scale=alignment,
      scalar_profile_units='local_image_normal_coverage_px_with_exact_physical_transverse_Brown_UV',
      all_bilinear_source_contributors_inside_frame_and_outside_exclusion_mask=True,
      physical_profile_valid_runs_split_before_finding_own_flanks=True)
    rows.append(row)
  return rows,dict(query_count=len(queries),measured_count=len(rows),
   operator='physical_ice_plane_width_section_current_two_luminance_flanks_ground_midpoint')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def produce(indices):
 cv2.setNumThreads(1);e=Extractor(RD,MODEL);records=[]
 for index in indices:
  packet=read(RD/f'ice_candidates/{index:06}.json');source=packet['source'];path=dict(next(p for p in e.paths if p['curve_id']=='lane1:edge0'))
  guide,origin=e.guide(path,index)
  bgr=cv2.imread(source['image_path']);h,w=bgr.shape[:2]
  inside=(guide>=[0,0]).all(1)&(guide<=[w-1,h-1]).all(1)
  near=(guide>=[-30,-30]).all(1)&(guide<=[w+29,h+29]).all(1)
  if inside.sum()<2:
   if near.sum()<2:continue
   # The old camera is a search seed with a measured 8--14 px offset at
   # 1248. A line just beyond the image edge must still query actual pixels.
   # Move only the navigation line into the image; its RGB reading remains
   # mandatory and the shifted coordinate is never emitted as evidence.
   close=guide[near];shift=np.zeros(2)
   if close[:,1].min()>h-15:shift[1]=h-15-close[:,1].min()
   elif close[:,1].max()<14:shift[1]=14-close[:,1].max()
   guide=guide+shift;origin+=':offscreen_seed_search_halo_only'
  broadcast,_=current_broadcast_exclusion(bgr);outside,_=current_ice_visible_domain(index)
  native=PhysicalLaneProfiles(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source,(cv2.imread(source['mask_path'],0)>0)|broadcast|outside,e.cameras[index])
  path['_line_identity_policy']=dict(current_identity_constrained=True,
   identity_source='8a28_named_lane1_long_edge_ground_width_section_current_source_material',reference_frame=index,
   normal_identity_tolerance_px=30.,search_radius_px=90.,physical_width_not_compared_to_unrelated_narrow_lane_paint=True)
  slope=float((guide[-1,1]-guide[0,1])/(guide[-1,0]-guide[0,0]))
  feature,_,_=e.line(path,guide,native,family_slope=slope)
  feature['navigation_origin']=origin
  feature['physical_width_recovery']=dict(reason='Physical lane long-edge paint is wider than unrelated narrow lane rows; use its own current measured full width',
   source_camera_direction_only=True,old_correct_other_targets_preserved=True,physical_width_operator_module='calibration.program_tracking.physical_ice_width')
  if feature['point_count']:
   feature['note_zh']='本帧原片实体lane1长边两侧材料经地面物理宽方向测量，中点反投地面平均再投回；未复制导航点。'
  write(OUT/f'proposals/{index:06}.json',dict(index=index,source=source,feature=feature,
    before_packet_sha256=sha(RD/f'ice_candidates/{index:06}.json'),frozen_other_features_sha256=digest([f for f in packet['features']if f['target_id']!=TARGET])))
  marked=bgr.copy()
  for row in feature.get('source_line_fit',{}).get('direct_readouts',[]):
   sides=np.rint(row['source_evidence']['actual_source_sides_uv']).astype(int);center=np.rint(row['uv']).astype(int)
   cv2.line(marked,tuple(sides[0]),tuple(sides[1]),(0,220,220),1)
   for p in sides:cv2.circle(marked,tuple(p),2,(0,240,240),-1)
   cv2.circle(marked,tuple(center),2,(0,0,255),-1)
  for seg in feature['segments']:cv2.polylines(marked,[np.rint(seg['uv']).astype(int)],False,(0,0,255),1,cv2.LINE_AA)
  cv2.imwrite(str(OUT/f'native/{index:06}.jpg'),np.vstack([bgr[550:],marked[550:]]),[cv2.IMWRITE_JPEG_QUALITY,96]) if (OUT/'native').exists() else None
  records.append(dict(index=index,old_points=sum(f['point_count']for f in packet['features']if f['target_id']==TARGET),new_points=feature['point_count'],
   actual_readouts=len(feature.get('source_line_fit',{}).get('direct_readouts',[])),line=image_line(feature)if feature['point_count']else None,source_sha256=sha(source['image_path']),proposal_sha256=sha(OUT/f'proposals/{index:06}.json')))
  print(json.dumps(records[-1]),flush=True)
 write(OUT/'proposal_receipt.json',dict(records=records,candidates_changed=False,current_source_physical_width_operator=True))

def apply():
 receipt=read(OUT/'proposal_receipt.json');review=read(OUT/'native_visual_review_receipt.json');audit=read(OUT/'source_and_continuity_audit.json')
 if audit['issues']or audit['new_missing_existing_frames']:raise ValueError('Unresolved lane physical-width source issues')
 if set(review['reviewed_frame_indices'])!={r['index']for r in receipt['records']}:raise ValueError('Source review scope mismatch')
 if review['source_and_continuity_audit_sha256']!=sha(OUT/'source_and_continuity_audit.json'):raise ValueError('Stale source audit review')
 reviewed={r['index']:r['sha256']for r in review['proposal_packets']};applied=[]
 previous_receipt=read(OUT/'applied_receipt.json')if(OUT/'applied_receipt.json').exists()else None
 for record in receipt['records']:
  index=record['index'];proposal_path=OUT/f'proposals/{index:06}.json'
  if sha(proposal_path)!=reviewed[index]:raise ValueError('Proposal changed after visual review')
  proposal=read(proposal_path);feature=proposal['feature'];path=RD/f'ice_candidates/{index:06}.json';packet=read(path);before=sha(path)
  if packet['source']!=proposal['source']:raise ValueError('Source binding changed since lane measurement')
  others=[f for f in packet['features']if f['target_id']!=TARGET];other_digest=digest(others)
  packet['features']=others+([feature]if feature['point_count']else [])
  packet['point_count']=sum(f['point_count']for f in packet['features'])
  if packet['point_count']>500:raise ValueError('Hard point cap exceeded')
  packet['per_feature_coverage']=[r for r in packet['per_feature_coverage']if r['target_id']!=TARGET]+[
   dict(target_id=TARGET,points=feature['point_count'],navigation_origin=feature['navigation_origin'],native_material_diagnostic=feature['extraction_diagnostic'],
    missing_reason=None if feature['point_count']else 'No complete current two-side grey-blue lane paint readout in the in-frame physical-width query; partial frame-edge or red-C interior valleys excluded')]
  packet['source_correlated_groups']=[g for g in packet['source_correlated_groups']if g['target_id']!=TARGET]
  if feature['point_count']:
   packet['source_correlated_groups'].append(dict(group_id=feature['source_evidence_group_id'],target_id=TARGET,native_source_pts=packet['source']['native_pts'],representative_point_count=feature['point_count'],
    native_individual_readout_count=feature['source_line_fit']['actual_native_readout_count'],correlation_reason='same current-frame measured physical-width lane line estimate; representatives are not independent measurements'))
  oldaudit=packet.get('parallel_family_geometry_audit',{})
  packet['parallel_family_geometry_audit']=dict(oldaudit,local_lane1_physical_width_reread=True,
   lane1_direction_source='same named finite 8a28 source guide with current source fit; no previous points emitted',
   lane1_source_visual_review=str(OUT/'native_visual_review_receipt.json'))
  packet['lane1_physical_width_revision']=dict(receipt_path=str(OUT/'applied_receipt.json'),source_visual_review_sha256=sha(OUT/'native_visual_review_receipt.json'),
   only_target_id=TARGET,measurement='actual current paint sides along physical ground transverse section; inverse/projective world midpoint')
  if digest([f for f in packet['features']if f['target_id']!=TARGET])!=other_digest:raise ValueError('Non-lane target changed')
  write(path,packet)
  applied.append(dict(index=index,before_sha256=before,after_sha256=sha(path),proposal_sha256=sha(proposal_path),frozen_all_non_lane1_features_sha256=other_digest,
    old_points=record['old_points'],new_points=feature['point_count']))
 write(OUT/'applied_receipt.json',dict(records=applied,only_target_geometry_changed=TARGET,source_mask_clock_model_circle_board_wide_other_thin_preserved=True,
  previous_application=previous_receipt,
  current_physical_width_module_sha256=sha(ROOT/'calibration/program_tracking/physical_ice_width.py'),current_generator_sha256=sha(Path(__file__)),native_visual_review_sha256=sha(OUT/'native_visual_review_receipt.json')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--indices',nargs='*',type=int,default=[1248]);p.add_argument('--all',action='store_true');p.add_argument('--apply',action='store_true');a=p.parse_args();(OUT/'native').mkdir(parents=True,exist_ok=True)
 apply()if a.apply else produce(range(201,1249)if a.all else a.indices)
