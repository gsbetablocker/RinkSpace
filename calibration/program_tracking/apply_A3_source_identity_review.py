from calibration.program_tracking.extract_first_shot_ice import *
p=DATA/'repair_20261014';folder=p/'circle_diagnostics/A3_identity_review';inventory=read(folder/'manifest.json')['items'];receipt=[];rejected=[]
qualification_file=p/'temporal_diagnostics/extra_ice_unqualified_query_domains.json';qualification=read(qualification_file)
reason='actual current source shows advertisement reflection bars, filled/grey textures or isolated straight streaks at these positions; no independently identifiable A3 thin painted circle material is established'
for item in inventory:
 i=item['frame'];file=p/f'ice_candidates/{i:06d}.json';packet=read(file);f=next((f for f in packet['features'] if f['target_id']=='ice.A3:outer'),None)
 if f is None:continue
 before=sha(file);rejected.append(dict(frame=i,source=packet['source'],candidate=f,reason=reason,source_native_review_image=item['path']))
 thin_before=hashlib.sha256(json.dumps([g for g in packet['features'] if 'source_circle_fit' not in g],sort_keys=True).encode()).hexdigest()
 packet['features']=[g for g in packet['features'] if g['target_id']!='ice.A3:outer'];packet['point_count']=sum(g['point_count'] for g in packet['features']);packet['source_correlated_groups']=[g for g in packet['source_correlated_groups'] if g['target_id']!='ice.A3:outer']
 for cov in packet['per_feature_coverage']:
  if cov['target_id']=='ice.A3:outer':cov.update(points=0,missing_reason=reason,source_native_identity_review_path=item['path'])
 after_thin=hashlib.sha256(json.dumps([g for g in packet['features'] if 'source_circle_fit' not in g],sort_keys=True).encode()).hexdigest();assert thin_before==after_thin
 packet['source_identity_review_A3_rejection']=dict(reason=reason,source_review_image=item['path'],reviewers=['recover_manual_annotations' if i<900 else 'first_shot_board_ads'])
 tmp=file.with_suffix('.tmp');write(tmp,packet);tmp.replace(file)
 cachefile=p/f'circle_candidates/{i:06d}.json';cache=read(cachefile);old=cache['features_by_curve']['A3:outer'];old['historical_rejected_native_candidate']=dict(segments=old['segments'],source_circle_fit=old.get('source_circle_fit'),geometric_interpretation=old.get('geometric_interpretation'))
 old['segments']=[];old['point_count']=0
 for name in ('source_circle_fit','joint_group_navigation','geometric_interpretation','source_evidence_group_id'):old.pop(name,None)
 old['extraction_diagnostic'].update(physical_identity_rejected=True,missing_reason=reason,source_native_review_image=item['path']);tmp=cachefile.with_suffix('.tmp');write(tmp,cache);tmp.replace(cachefile)
 qualification['records'].append(dict(target_id='ice.A3:outer',programme_frame_range=[i,i],allowed_as_fitting_observation=False,rejection_reason=reason,source_native_identity_review_path=item['path'],physical_model_target_removed=False))
 receipt.append(dict(frame=i,old_points=f['point_count'],new_points=0,candidate_before_sha256=before,candidate_after_sha256=sha(file),thin_features_sha256_before_and_after=thin_before))
write(folder/'rejected_native_candidates.json',dict(records=rejected,not_fitting_observations=True));write(folder/'rejection_receipt.json',dict(records=receipt,total_rejected_points=sum(r['old_points'] for r in receipt),entity_identity_uncertain_not_geometric_failure=True,scene_geometry_unchanged=True))
write(qualification_file,qualification);print(len(receipt),sum(r['old_points'] for r in receipt))
