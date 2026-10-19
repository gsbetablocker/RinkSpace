"""Current physical-width reread of source-verified B1 entry and exit only."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np
from .extract_first_shot_ice import (DATA,ROOT,Extractor,NativeProfiles,read,write,sha,
 _source_queries,current_broadcast_exclusion,current_ice_visible_domain,image_line)
from .native_ice_material_repair import narrow_dark_paint
from .physical_ice_width import (projected_width_profile,sample_physical_width_field,
 restore_physical_profile_readout,source_at_profile_offsets,valid_profile_contributors)

RD=DATA/'repair_20261014';OUT=RD/'temporal_diagnostics/B1_physical_visibility'
MODEL=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014';TARGET='ice.B1:segment0'

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

class LocalPhysicalBlue(NativeProfiles):
 def __init__(self,rgb,source,mask,camera):
  super().__init__(rgb,source,mask);self.camera=camera
  rgb=np.asarray(rgb,np.float32);self.luminance=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2]
 def sample_complete_colour_band(self,path,guide,spacing):
  q,n,_=_source_queries(path,guide,np.ones(len(guide)),(1080,1920),35,40);rows=[];rejected=[]
  for k,query in enumerate(q):
   profile=projected_width_profile(self.camera,path,query,radius_px=1.)
   if profile is None:continue
   cross=float(abs(profile['query_width_direction_uv']@n[k]))
   if cross<.04:continue
   profile=projected_width_profile(self.camera,path,query,radius_px=42./cross,step_px=1./cross)
   profile['offsets']=profile['offsets']*cross
   values=sample_physical_width_field(self.frame.rgb,profile,'dark',field=self.luminance)
   valid=valid_profile_contributors(profile,self.mask);edges=np.flatnonzero(np.r_[True,valid[1:]!=valid[:-1],True]);candidates=[]
   for first,last in zip(edges[:-1],edges[1:]):
    if valid[first]and last-first>=15:
     candidates.extend(narrow_dark_paint(np.zeros(2),np.array([1.,0.]),profile['offsets'][first:last],values[first:last],np.ones(last-first,bool)))
   for scalar in candidates:
    row=restore_physical_profile_readout(self.camera,profile,scalar)
    if row is None:continue
    ev=row['source_evidence'];uv=np.asarray(row['uv']);distance=float(abs((uv-query)@n[k]));sides=np.asarray(ev['actual_source_sides_uv'])
    width=float(abs((sides[1]-sides[0])@n[k]))
    if distance>18 or width<4 or width>65:
     rejected.append(dict(query=k,reason='outside_source_verified_B1_identity_or_not_complete_blue_strip_width',normal_distance_px=distance,normal_width_px=width));continue
    lo,hi=ev['complete_band_native_interval_px'];support=source_at_profile_offsets(profile,np.arange(np.ceil(lo),np.floor(hi)+1))
    context=np.vstack([support,np.asarray(ev['actual_source_side_quantiles_uv']).reshape(-1,2)])
    if (context<[0,0]).any()or(context>[1919,1079]).any():continue
    ip=np.rint(context).astype(int)
    if self.mask[ip[:,1],ip[:,0]].any():continue
    ev.pop('filled_house_band_is_not_an_outline',None);ev.pop('narrow_paint_width_px',None)
    ev.update(query_index=k,material_signal='blue',joint_band_width_px=width,
     measurement_origin='current_native_source_verified_B1_physical_transverse_luminance_two_ice_side_flanks',
     boundary_signal='current_luminance_interfaces_in_independently_verified_blue_material_domain',
     blue_tint_threshold_not_applied_only_in_authorized_source_entry_exit_domains=True,
     local_profile_reason=path['_wide_local_identity_reason'],transverse_profile_image_normal_coverage_scale=cross,
     scalar_profile_units='local_image_normal_coverage_px_exact_physical_transverse_Brown_UV',
     all_bilinear_source_contributors_inside_frame_and_outside_exclusion_mask=True,
     physical_profile_valid_runs_split_before_finding_own_flanks=True)
    rows.append(row)
  return rows,dict(query_count=len(q),measured_count=len(rows),operator='current_source_verified_B1_physical_width_two_luminance_interfaces_ground_midpoint',rejections=rejected)

def produce(indices):
 cv2.setNumThreads(1);e=Extractor(RD,MODEL);p=next(p for p in e.paths if p['feature_id']=='B1');records=[]
 (OUT/'native').mkdir(parents=True,exist_ok=True)
 for i in indices:
  output=RD/f'physical_wide_proposals/{i:06}.json';packet=read(output);source=packet['source'];raw=cv2.imread(source['image_path']);broadcast,_=current_broadcast_exclusion(raw);outside,_=current_ice_visible_domain(i)
  native=LocalPhysicalBlue(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),source,(cv2.imread(source['mask_path'],0)>0)|broadcast|outside,e.cameras[i])
  guide,origin=e.guide(p,i)
  policy=dict(current_identity_constrained=True,identity_source='previous_native_source_reviewed_B1_entry_exit_material_domain_no_old_uv_used',reference_frame=i,
   normal_identity_tolerance_px=18.,search_radius_px=42.,registration_is_navigation_only=True,current_source_pixels_reread=True)
  local=dict(p,_line_identity_policy=policy,_wide_local_identity_reason='source_entry_yellow_reflection_suppresses_blue_tint'if i<500 else 'source_exit_frame_edge_truncates_global_colour_background')
  feature,_,_=e.line(local,guide,native);feature['navigation_origin']=origin
  if i in (427,625)and feature['point_count']:
   # Actual native source review found a false grey valley at the first
   # entry and a clipped/unequal pair at the final exit. Keep the evidence
   # for audit, never turn the existence of two numerical flanks into proof
   # that both belong to this physical paint band.
   write(OUT/f'rejected_native_candidates/{i:06}.json',dict(source=source,feature=feature,
    reason='427 source valley lies below the still only partial blue entry; 625 first readout has a clipped lower blue flank and the remaining single readout cannot determine a current line'))
   from .extract_first_shot_ice import blank_feature
   rejected_count=feature['point_count'];feature=blank_feature(p);feature['navigation_origin']=origin
   feature['extraction_diagnostic']=dict(query_count=0,measured_count=0,eligible_native_material_readouts=0,
    acceptance='current source visual ownership review rejects a partial/ambiguous paint flank',
    source_rejected_candidate_path=str(OUT/f'rejected_native_candidates/{i:06}.json'),rejected_candidate_representatives=rejected_count)
  feature['physical_B1_local_visibility_review']=dict(current_native_luminance_flanks=True,physical_transverse_ground_midpoint=True,source_domain='entry426_449'if i<500 else'exit615_628',
   no_previous_image_normal_uv_imported=True,native_review_receipt=str(OUT/'native_visual_review_receipt.json'))
  old=[f for f in packet['features']if f['target_id']==TARGET];before=sha(output);others=[f for f in packet['features']if f['target_id']!=TARGET];frozen=digest(others)
  packet['features']=others+([feature]if feature['point_count']else[]);packet['point_count']=sum(f['point_count']for f in packet['features'])
  packet['per_feature_coverage']=[r for r in packet['per_feature_coverage']if r['target_id']!=TARGET]+[
   dict(target_id=TARGET,points=feature['point_count'],navigation_origin=origin,native_material_diagnostic=feature['extraction_diagnostic'],
    missing_reason=None if feature['point_count']else'current source verified entry/exit query has fewer than two complete in-frame physical-width paint/ice interface readouts')]
  if digest([f for f in packet['features']if f['target_id']!=TARGET])!=frozen:raise ValueError('Other physical wide feature changed')
  write(output,packet)
  marked=raw.copy()
  for row in feature.get('source_line_fit',{}).get('direct_readouts',[]):
   sides=np.rint(row['source_evidence']['actual_source_sides_uv']).astype(int);center=np.rint(row['uv']).astype(int)
   cv2.line(marked,tuple(sides[0]),tuple(sides[1]),(0,230,230),1)
   for point in sides:cv2.circle(marked,tuple(point),2,(0,230,230),-1)
   cv2.circle(marked,tuple(center),2,(0,0,255),-1)
  for seg in feature['segments']:cv2.polylines(marked,[np.rint(seg['uv']).astype(int)],False,(0,0,255),1)
  roi=[0,500,900,1000]if i<500 else[1000,500,1920,1080];x0,y0,x1,y1=roi
  cv2.imwrite(str(OUT/f'native/{i:06}.jpg'),np.hstack([raw[y0:y1,x0:x1],marked[y0:y1,x0:x1]]),[cv2.IMWRITE_JPEG_QUALITY,97])
  records.append(dict(index=i,before_sha256=before,after_sha256=sha(output),old_B1_points=sum(f['point_count']for f in old),new_B1_points=feature['point_count'],
   native_readouts=len(feature.get('source_line_fit',{}).get('direct_readouts',[])),source_image_sha256=sha(source['image_path']),all_non_B1_features_sha256=frozen,
   physical_width_proposal_path=str(output),line=image_line(feature)if feature['point_count']else None))
  print(json.dumps(records[-1]),flush=True)
 write(OUT/'repair_receipt.json',dict(records=records,only_authorized_B1_entry_exit_proposals_changed=True,ice_candidates_sources_models_circles_board_other_wide_unchanged=True,
  module_sha256=sha(ROOT/'calibration/program_tracking/physical_ice_width.py'),generator_sha256=sha(Path(__file__))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--indices',nargs='*',type=int);a=p.parse_args();produce(a.indices or list(range(426,450))+list(range(615,629)))
