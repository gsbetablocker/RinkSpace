"""Split every missing source station, without refitting or changing native reads."""
from calibration.program_tracking.extract_first_shot_ice import *
import datetime

def source_segments(feature,native_mask):
 if not feature.get('point_count') or 'source_circle_fit' not in feature:return feature
 rows=feature['source_circle_fit']['direct_readouts'];parts=[];part=[];last=None
 for row in rows:
  q=np.asarray(row['representative_uv']);ip=np.rint(q).astype(int)
  if (q<[0,0]).any() or (q>[1919,1079]).any() or native_mask[ip[1],ip[0]]:continue
  qi=row['query_index']
  if part and (qi-last>1 or np.linalg.norm(np.asarray(row['uv'])-part[-1]['uv'])>120):parts.append(part);part=[]
  part.append(row);last=qi
 if part:parts.append(part)
 segments=[];group_id=feature['source_evidence_group_id']
 for part in parts:
  uv=np.asarray([r['representative_uv'] for r in part]);take=simplify(uv,tolerance=2.,maximum_gap=160,closed=False);selected=[part[k] for k in take]
  segments.append(dict(uv=[r['representative_uv'] for r in selected],vertex_count=len(selected),closed=False,
   sigma_px=[r['sigma_px'] for r in selected],source_evidence=[dict(r['source_evidence'],kind='shared_current_source_circle_representative',actual_native_readout_uv=r['uv'],representative_to_actual_material_px=r['representative_to_actual_material_px'],source_evidence_group_id=group_id,individual_native_pixel_readout=False) for r in selected],
   measurement_origin='same_frame_joint_physical_circle_family_fit',pixels_measured_in_this_frame=False,representative_points_share_circle_family_estimate=True,source_evidence_group_id=group_id,source_evidence_measured_in_this_frame=True,individual_native_pixel_readouts=False,hidden_span_filled=False,
   current_native_query_index_interval=[part[0]['query_index'],part[-1]['query_index']],missing_material_station_spanned=False))
 feature['segments']=segments;feature['point_count']=sum(s['vertex_count'] for s in segments);feature['native_visibility_segmentation']=dict(any_missing_source_query_splits_arc=True,no_raw_material_readout_changed=True,no_curve_geometry_changed=True,single_station_segments_are_correlated_source_supported_points=True)
 return feature

def main():
 p=DATA/'repair_20261014';manifest={r['index']:r for r in read(p/'frame_manifest.json')['frames']};changes=[];hashes=[]
 for i in range(201,1249):
  source=manifest[i]['source'];b=cv2.imread(str(DATA/f'frames/{i:06d}.png'));person=cv2.imread(source['mask_path'],0)>0;bc,_=current_broadcast_exclusion(b);domain,_=current_ice_visible_domain(i);mask=person|bc|domain
  cachefile=p/f'circle_candidates/{i:06d}.json';cache=read(cachefile);icefile=p/f'ice_candidates/{i:06d}.json';packet=read(icefile);before=sha(icefile);thin=lambda:hashlib.sha256(json.dumps([f for f in packet['features'] if 'source_circle_fit' not in f],sort_keys=True).encode()).hexdigest();thinbefore=thin();items=[]
  for feature in packet['features']:
   if 'source_circle_fit' not in feature:continue
   old=deepcopy(feature['segments']);oldcount=feature['point_count'];fitbefore=hashlib.sha256(json.dumps([feature['source_circle_fit'],feature['geometric_interpretation'],feature['joint_group_navigation']],sort_keys=True).encode()).hexdigest();source_segments(feature,mask)
   assert fitbefore==hashlib.sha256(json.dumps([feature['source_circle_fit'],feature['geometric_interpretation'],feature['joint_group_navigation']],sort_keys=True).encode()).hexdigest()
   if old!=feature['segments']:items.append(dict(target_id=feature['target_id'],old_points=oldcount,new_points=feature['point_count'],old_segment_count=len(old),new_segment_count=len(feature['segments'])))
  for feature in cache['features_by_curve'].values():source_segments(feature,mask)
  packet['point_count']=sum(f['point_count'] for f in packet['features']);counts={f['target_id']:f['point_count'] for f in packet['features']}
  for cov in packet['per_feature_coverage']:
   if cov['target_id'] in counts:cov['points']=counts[cov['target_id']]
  for group in packet['source_correlated_groups']:
   if group['target_id'] in counts:group['representative_point_count']=counts[group['target_id']]
  assert thinbefore==thin()
  for file,value in ((cachefile,cache),(icefile,packet)):
   tmp=file.with_suffix('.tmp');write(tmp,value);tmp.replace(file)
  if items:changes.append(dict(frame=i,changes=items,candidate_before_sha256=before,candidate_after_sha256=sha(icefile),thin_feature_sha256_before_and_after=thinbefore))
  hashes.append(dict(frame=i,ice_candidate_sha256=sha(icefile),circle_cache_sha256=sha(cachefile)))
 write(p/'circle_diagnostics/native_visibility_segmentation_receipt.json',dict(records=changes,changed_frames=[r['frame'] for r in changes],candidate_integrity=hashes,all_native_readouts_and_fitted_geometry_preserved_exactly=True,all_non_circle_features_preserved_exactly=True,reason='A missing station on a very flat circle may separate upper and lower visible arcs; no straight chord is a material observation across it.',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 print('changed',len(changes),flush=True)

if __name__=='__main__':main()
