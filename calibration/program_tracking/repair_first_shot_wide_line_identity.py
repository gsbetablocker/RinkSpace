"""Reread wide paint independently, preserving the already repaired circles/lines."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from .extract_first_shot_ice import (Extractor,NativeProfiles,DATA,ROOT,read,write,sha,
 current_broadcast_exclusion,current_ice_visible_domain,_source_queries,blank_feature)

class EntryProfiles(NativeProfiles):
 """Read both luminance flanks of a source-identified blue strip.

 The yellow board reflection suppresses blue chroma here. The bounded current
 source profile retains the two paint/ice transitions; navigation only selects
 the material domain, and never provides a measured side or midpoint.
 """
 def sample_complete_colour_band(self,path,guide,spacing):
  from .native_ice_material_repair import sample_chroma,narrow_dark_paint
  q,n,interval=_source_queries(path,guide,np.ones(len(guide)),(1080,1920),35,36)
  if not len(q):return [],dict(query_count=0,measured_count=0)
  offsets,profiles,valid=sample_chroma(self.frame.rgb,q,n,interval,'dark');rows=[]
  for k in range(len(q)):
   choices=narrow_dark_paint(q[k],n[k],offsets[k],profiles[k],valid[k])
   choices.sort(key=lambda r:abs(r['source_evidence']['native_profile_shift_px']))
   for row in choices:
    ev=row['source_evidence'];lo,hi=ev['complete_band_native_interval_px']
    support=q[k]+np.arange(np.ceil(lo),np.floor(hi)+1)[:,None]*n[k]
    context=np.vstack([support,np.asarray(ev['actual_source_side_quantiles_uv']).reshape(-1,2)])
    if (context<[0,0]).any() or (context>[1919,1079]).any():continue
    ip=np.rint(context).astype(int)
    if self.mask[ip[:,1],ip[:,0]].any() or abs(ev['native_profile_shift_px'])>18:continue
    ev.update(query_index=k,material_signal='dark',
      measurement_origin='current_native_source_identified_blue_paint_two_local_luminance_flanks',
      joint_band_width_px=ev.pop('narrow_paint_width_px'),
      physical_measurement='midpoint_of_two_current_blue_paint_ice_interfaces',
      local_profile_reason=path.get('_wide_local_identity_reason','source_entry_blue_chroma_suppressed_by_yellow_board_reflection'))
    ev.pop('filled_house_band_is_not_an_outline',None)
    rows.append(row);break
  return rows,dict(query_count=len(q),measured_count=len(rows),
    operator='source_identified_blue_band_two_ice_side_luminance_interfaces')

def repair_entry():
 rd=DATA/'repair_20261014';model=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014'
 extractor=Extractor(rd,model);path=next(p for p in extractor.paths if p['feature_id']=='B1');records=[]
 # The current user-labelled frame 500 is an identity reset. Reread through
 # that reset, so changed entry measurements cannot leave stale later guides.
 for index in range(420,501):
  output=rd/f'wide_line_candidates/{index:06d}.json';packet=read(output)
  old=[f for f in packet['features'] if f['target_id']=='ice.B1:segment0']
  source=extractor.rows[index]['source'];bgr=cv2.imread(source['image_path'])
  broadcast,_=current_broadcast_exclusion(bgr);outside,_=current_ice_visible_domain(index)
  cls=EntryProfiles if index<=449 else NativeProfiles
  native=cls(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source,(cv2.imread(source['mask_path'],0)>0)|broadcast|outside)
  guide,origin=extractor.guide(path,index)
  if index<=449:
   policy=dict(current_identity_constrained=False,identity_source='source_visually_verified_B1_entry_domain',
     native_image_coordinates_not_generated=True)
  else:guide,origin,policy=extractor.line_identity.select_guide(path,index,guide,origin)
  if index<426:
   feature=blank_feature(path)
   feature['extraction_diagnostic']=dict(source_identity='source_review_no_B1_material_in_current_entry_query')
  else:feature,_,_=extractor.line(dict(path,_line_identity_policy=policy),guide,native)
  feature['navigation_origin']=origin
  extractor.line_identity.record(index,[feature])
  packet['features']=[f for f in packet['features'] if f['target_id']!='ice.B1:segment0']+([feature] if feature['point_count'] else [])
  packet['per_feature_coverage']=[r for r in packet['per_feature_coverage'] if r['target_id']!='ice.B1:segment0']+[
    dict(target_id='ice.B1:segment0',points=feature['point_count'],navigation_origin=origin,
      native_material_diagnostic=feature.get('extraction_diagnostic'),
      missing_reason=None if feature['point_count'] else 'source_review_absent_or_short_entry_with_insufficient_two_sided_material_support')]
  packet['point_count']=sum(f['point_count'] for f in packet['features'])
  records.append(dict(index=index,before_sha256=sha(output),source_image_sha256=sha(source['image_path']),
    previous_B1_features=old,current_B1_feature=feature))
  write(output,packet);records[-1]['after_sha256']=sha(output)
 write(rd/'wide_line_entry_material_repair.json',dict(range=[420,500],records=records,
   source_identity_review='native 420–449 images reviewed; 430–449 current two-side measurements visually rechecked',
   projected_or_transported_coordinates_emitted=False,other_wide_features_preserved=True))

def repair_exit():
 rd=DATA/'repair_20261014';model=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014'
 extractor=Extractor(rd,model);path=next(p for p in extractor.paths if p['feature_id']=='B1');records=[]
 for index in range(615,629):
  output=rd/f'wide_line_candidates/{index:06d}.json';packet=read(output)
  source=extractor.rows[index]['source'];bgr=cv2.imread(source['image_path'])
  broadcast,_=current_broadcast_exclusion(bgr);outside,_=current_ice_visible_domain(index)
  native=EntryProfiles(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source,(cv2.imread(source['mask_path'],0)>0)|broadcast|outside)
  guide,origin=extractor.guide(path,index)
  feature,_,_=extractor.line(dict(path,_wide_local_identity_reason='source_verified_B1_exit_domain_frame_edge_clips_whole_chroma_background'),guide,native)
  feature['navigation_origin']=origin
  records.append(dict(index=index,before_sha256=sha(output),source_image_sha256=sha(source['image_path']),
    previous_B1_features=[f for f in packet['features'] if f['target_id']=='ice.B1:segment0'],current_B1_feature=feature))
  packet['features']=[f for f in packet['features'] if f['target_id']!='ice.B1:segment0']+([feature] if feature['point_count'] else [])
  packet['per_feature_coverage']=[r for r in packet['per_feature_coverage'] if r['target_id']!='ice.B1:segment0']+[
    dict(target_id='ice.B1:segment0',points=feature['point_count'],navigation_origin=origin,
      native_material_diagnostic=feature.get('extraction_diagnostic'),
      missing_reason=None if feature['point_count'] else 'visible_exit_segment_lacks_two_current_ice_side_transitions_inside_image')]
  packet['point_count']=sum(f['point_count'] for f in packet['features']);write(output,packet);records[-1]['after_sha256']=sha(output)
 write(rd/'wide_line_exit_material_repair.json',dict(range=[615,628],records=records,
   source_identity='current source B1 blue strip at lower right; whole-band search had mistaken unrelated upper gray textures at 619',
   projected_or_transported_coordinates_emitted=False,other_wide_features_preserved=True))

def produce():
 rd=DATA/'repair_20261014';model=ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014'
 extractor=Extractor(rd,model)
 for index in range(201,1249):
  source=extractor.rows[index]['source'];bgr=cv2.imread(source['image_path'])
  broadcast,_=current_broadcast_exclusion(bgr);outside,_=current_ice_visible_domain(index)
  native=NativeProfiles(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source,(cv2.imread(source['mask_path'],0)>0)|broadcast|outside)
  features=[];coverage=[]
  for path in extractor.paths:
   if path['feature_id'] not in ('B1','B2','C'):continue
   guide,origin=extractor.guide(path,index)
   guide,origin,policy=extractor.line_identity.select_guide(path,index,guide,origin)
   if len(guide)<2:continue
   feature,_,_=extractor.line(dict(path,_line_identity_policy=policy),guide,native)
   feature['navigation_origin']=origin
   if feature['point_count']:features.append(feature)
   coverage.append(dict(target_id=feature['target_id'],points=feature['point_count'],navigation_origin=origin,
      native_material_diagnostic=feature.get('extraction_diagnostic'),
      missing_reason=None if feature['point_count'] else 'no_coherent_current_whole_band_under_source_motion_identity'))
  extractor.line_identity.record(index,features)
  write(rd/f'wide_line_candidates/{index:06d}.json',dict(programme_frame_index_1based=index,source=source,
    features=features,per_feature_coverage=coverage,point_count=sum(f['point_count'] for f in features),
    only_wide_materials=True,model_scene_sha256=sha(model/'scene.json')))
  if index%100==0:print('wide material source reread',index,flush=True)

def apply():
 rd=DATA/'repair_20261014';changed=[];wide={'ice.B1:segment0','ice.B2:segment0','ice.C:segment0'}
 for index in range(201,1249):
  path=rd/f'ice_candidates/{index:06d}.json';packet=read(path);replacement=read(rd/f'wide_line_candidates/{index:06d}.json')
  original=[f for f in packet['features'] if f['target_id'] in wide]
  frozen=[f for f in packet['features'] if f['target_id'] not in wide]
  packet['features']=frozen+replacement['features']
  packet['point_count']=sum(f['point_count'] for f in packet['features'])
  packet['per_feature_coverage']=[r for r in packet['per_feature_coverage'] if r['target_id'] not in wide]+replacement['per_feature_coverage']
  packet['source_correlated_groups']=[g for g in packet['source_correlated_groups'] if g['target_id'] not in wide]
  for f in replacement['features']:
   packet['source_correlated_groups'].append(dict(group_id=f['source_evidence_group_id'],target_id=f['target_id'],
    native_source_pts=packet['source']['native_pts'],representative_point_count=f['point_count'],
    native_individual_readout_count=f['source_line_fit']['actual_native_readout_count'],
    correlation_reason='same current-frame measured physical curve estimate; representatives are not independent measurements'))
  record=dict(index=index,before_sha256=sha(path),old_wide_features=original,new_source_packet=str(rd/f'wide_line_candidates/{index:06d}.json'),
    source_packet_sha256=sha(rd/f'wide_line_candidates/{index:06d}.json'),all_nonwide_features_preserved_exactly=True)
  packet['wide_source_identity_revision']=dict(path=str(rd/'wide_line_identity_repair_receipt.json'),programme_frame_index_1based=index)
  write(path,packet);record['after_sha256']=sha(path);changed.append(record)
 write(rd/'wide_line_identity_repair_receipt.json',dict(frames=1048,rows=changed,model_and_source_clocks_unchanged=True))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');parser.add_argument('--entry-repair',action='store_true');parser.add_argument('--exit-repair',action='store_true');args=parser.parse_args()
 cv2.setNumThreads(1)
 apply() if args.apply else repair_entry() if args.entry_repair else repair_exit() if args.exit_repair else produce()
