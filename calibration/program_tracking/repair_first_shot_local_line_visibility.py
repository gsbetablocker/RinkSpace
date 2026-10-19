"""Reread specifically reviewed short visibility gaps from current source pixels.

Neighbouring measured lines and the same 8a28 Brown cameras select a finite
material search only. No projected or transported pixel becomes an observation.
The dark two-flank reader handles known paint whose weak chroma lost visibility.
"""
from __future__ import annotations
import argparse,hashlib,json
from copy import deepcopy
from pathlib import Path
import cv2,numpy as np
from .extract_first_shot_ice import (DATA,ROOT,Extractor,NativeProfiles,read,write,sha,
 current_broadcast_exclusion,current_ice_visible_domain,_source_queries,image_line)
from .native_ice_line_continuity import straight_guide
from .native_ice_material_repair import sample_chroma,narrow_dark_paint
from .projection import project_fast,ground_xy

RD=DATA/'repair_20261014';OUT=RD/'temporal_diagnostics/local_line_visibility'
MODEL=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014'
CASES={1075:(1076,['lane4:edge2','lane3:edge0']),
       1078:(1077,['lane4:edge2','lane3:edge0']),
       1082:(1081,['lane4:edge2','lane3:edge0']),
       1133:(1134,['lane4:edge2','lane3:edge0','house_connection_F3_R3:segment0'])}

def frozen_digest(value):
 return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()

class CurrentNarrowPaint(NativeProfiles):
 def sample(self,path,guide,radius=18,spacing=48):
  q,n,interval=_source_queries(path,guide,np.ones(len(guide)),(1080,1920),spacing,radius)
  if not len(q):return [],dict(query_count=0,measured_count=0)
  offsets,profiles,valid=sample_chroma(self.frame.rgb,q,n,interval,'dark');rows=[]
  for k in range(len(q)):
   candidates=narrow_dark_paint(q[k],n[k],offsets[k],profiles[k],valid[k])
   for row in candidates:
    evidence=row['source_evidence'];width=evidence['narrow_paint_width_px']
    if width>path.get('_maximum_width',16) or abs(evidence['native_profile_shift_px'])>radius-4:continue
    lo,hi=evidence['complete_band_native_interval_px']
    support=q[k]+np.arange(np.ceil(lo),np.floor(hi)+1)[:,None]*n[k]
    context=np.vstack([support,np.asarray(evidence['actual_source_side_quantiles_uv']).reshape(-1,2)])
    if (context<[0,0]).any() or (context>[1919,1079]).any():continue
    ip=np.rint(context).astype(int)
    if self.mask[ip[:,1],ip[:,0]].any():continue
    evidence.pop('filled_house_band_is_not_an_outline',None)
    evidence.update(query_index=k,material_signal='dark',
      measurement_origin='current_native_source_identified_thin_paint_two_local_luminance_flanks',
      joint_band_width_px=evidence.pop('narrow_paint_width_px'),
      physical_measurement='midpoint_of_two_current_narrow_paint_ice_interfaces')
    rows.append(row)
  return rows,dict(query_count=len(q),measured_count=len(rows),
    operator='current_source_identified_narrow_paint_two_local_luminance_flanks')

def propose():
 cv2.setNumThreads(1);extractor=Extractor(RD,MODEL);records=[]
 for index,(reference,cids) in CASES.items():
  packet=read(RD/f'ice_candidates/{index:06}.json');before=deepcopy(packet)
  reference_path=RD/f'ice_candidates/{reference:06}.json';reference_packet=read(reference_path)
  source=packet['source'];bgr=cv2.imread(source['image_path']);broadcast,_=current_broadcast_exclusion(bgr);outside,_=current_ice_visible_domain(index)
  native=CurrentNarrowPaint(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source,(cv2.imread(source['mask_path'],0)>0)|broadcast|outside)
  revised={};case_records=[]
  for cid in cids:
   feature0=next(f for f in reference_packet['features']if f['target_id']=='ice.'+cid)
   fit=feature0['source_line_fit'];a=np.asarray(fit['point_origin']);v=np.asarray(fit['unit_tangent']);n=np.asarray(fit['unit_normal']);coef=fit['normal_offset_coefficients']
   stations=np.linspace(0,2200,257);sourceguide=a+stations[:,None]*v+np.polyval(coef,stations)[:,None]*n
   world,valid=ground_xy(extractor.cameras[reference],sourceguide);uv,depth=project_fast(extractor.cameras[index],np.c_[world[valid],np.zeros(valid.sum())]);uv=uv[depth>.05]
   path=dict(next(p for p in extractor.paths if p['curve_id']==cid));extent,origin=extractor.guide(path,index);guide=straight_guide(uv,extent)
   neighbor_widths=[np.linalg.norm(np.diff(r['source_evidence']['actual_source_sides_uv'],axis=0)) for r in fit['direct_readouts']]
   width_limit=max(12.,1.75*float(np.median(neighbor_widths)))
   policy=dict(current_identity_constrained=True,identity_source='adjacent_native_material_identity_with_same_Brown_camera_search_transport',reference_frame=reference,
    normal_identity_tolerance_px=9.,search_radius_px=18.,registration_is_navigation_only=True,native_image_coordinates_not_generated=True,
    reference_source_packet=str(reference_path),reference_source_packet_sha256=sha(reference_path),unsupported_global_source_H_not_used_as_measurement=True)
   path.update(material_signal='dark',_maximum_width=width_limit,_line_identity_policy=policy)
   feature,_,_=extractor.line(path,guide,native,family_slope=(guide[-1,1]-guide[0,1])/(guide[-1,0]-guide[0,0]))
   feature['navigation_origin']=policy['identity_source'];feature['source_schematic_material_signal']=feature0['material_signal']
   feature['local_visibility_repair']=dict(reason='native source paint persists but weak chroma caused original RGB material sampler to miss it',
    reference_frame=reference,current_native_two_flank_luminance_readouts=True,predicted_pixels_emitted=False,
    native_visual_identity_review=str(OUT/'native_review_receipt.json'),reference_narrow_material_width_median_px=float(np.median(neighbor_widths)),maximum_width_px=width_limit)
   revised['ice.'+cid]=feature
   case_records.append(dict(target_id=feature['target_id'],old_points=next((f['point_count']for f in packet['features']if f['target_id']==feature['target_id']),0),
     new_points=feature['point_count'],current_source_line=image_line(feature)if feature['point_count']else None,actual_readouts=len(feature.get('source_line_fit',{}).get('direct_readouts',[]))))
  packet['features']=[f for f in packet['features']if f['target_id']not in revised]+[f for f in revised.values()if f['point_count']]
  packet['point_count']=sum(f['point_count']for f in packet['features'])
  packet['per_feature_coverage']=[r for r in packet['per_feature_coverage']if r['target_id']not in revised]+[
   dict(target_id=f['target_id'],points=f['point_count'],navigation_origin=f['navigation_origin'],native_material_diagnostic=f['extraction_diagnostic'],
     missing_reason=None if f['point_count']else 'reviewed_source_identity_but_insufficient_current_two_side_material_readout')for f in revised.values()]
  packet['source_correlated_groups']=[g for g in packet['source_correlated_groups']if g['target_id']not in revised]
  for f in revised.values():
   if not f['point_count']:continue
   packet['source_correlated_groups'].append(dict(group_id=f['source_evidence_group_id'],target_id=f['target_id'],native_source_pts=source['native_pts'],representative_point_count=f['point_count'],
    native_individual_readout_count=f['source_line_fit']['actual_native_readout_count'],correlation_reason='same current-frame measured physical line estimate; representatives are not independent measurements'))
  write(OUT/f'proposals/{index:06}.json',packet)
  marked=bgr.copy()
  for f in revised.values():
   for row in f.get('source_line_fit',{}).get('direct_readouts',[]):
    ev=row['source_evidence'];sides=np.rint(ev['actual_source_sides_uv']).astype(int);center=np.rint(row['uv']).astype(int)
    cv2.line(marked,tuple(sides[0]),tuple(sides[1]),(0,255,255),1)
    for point in sides:cv2.circle(marked,tuple(point),2,(0,255,255),-1)
    cv2.circle(marked,tuple(center),2,(0,0,255),-1)
  cv2.imwrite(str(OUT/f'current_material_{index:06}.jpg'),np.vstack([bgr[820:1080],marked[820:1080]]),[cv2.IMWRITE_JPEG_QUALITY,97])
  records.append(dict(index=index,source_sha256=sha(source['image_path']),before_sha256=sha(RD/f'ice_candidates/{index:06}.json'),proposal_sha256=sha(OUT/f'proposals/{index:06}.json'),
    frozen_other_features_sha256=frozen_digest([f for f in before['features']if f['target_id']not in revised]),targets=case_records))
  print(index,case_records,flush=True)
 write(OUT/'proposal_receipt.json',dict(records=records,current_candidates_unchanged=True,source_or_model_unchanged=True))

def apply():
 receipt=read(OUT/'proposal_receipt.json');review=read(OUT/'native_review_receipt.json');approved=set(review['approved_frame_indices'])
 for record in receipt['records']:
  index=record['index']
  if index not in approved:continue
  current=RD/f'ice_candidates/{index:06}.json';proposal=OUT/f'proposals/{index:06}.json'
  if sha(current)!=record['before_sha256']or sha(proposal)!=record['proposal_sha256']:raise ValueError('Candidate changed since proposal')
  packet=read(proposal);targets={r['target_id']for r in record['targets']}
  if frozen_digest([f for f in packet['features']if f['target_id']not in targets])!=record['frozen_other_features_sha256']:raise ValueError('Nonreviewed feature changed')
  packet['local_line_visibility_revision']=dict(receipt_path=str(OUT/'applied_receipt.json'),native_review_receipt_sha256=sha(OUT/'native_review_receipt.json'))
  write(current,packet);record['after_sha256']=sha(current)
 write(OUT/'applied_receipt.json',dict(records=[r for r in receipt['records']if r['index']in approved],only_reviewed_current_source_thin_lines_changed=True,source_masks_clocks_circles_wide_materials_preserved=True))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args();apply()if args.apply else propose()
