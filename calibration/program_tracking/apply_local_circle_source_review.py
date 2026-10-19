"""Remeasure only source-reviewed circle families, preserving all other targets."""
from calibration.program_tracking.extract_first_shot_ice import *
import argparse,datetime
ap=argparse.ArgumentParser();ap.add_argument('--indices');args=ap.parse_args()
p=DATA/'repair_20261014';policies=read(p/'circle_diagnostics/native_material_identity_domains.json');frames={r['frame'] for r in policies['records']}
if args.indices:frames&=set(map(int,args.indices.split(',')))
records=[];archives=[]
for i in sorted(frames):
 ex=Extractor(p,ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014');ex.use_circle_cache=False
 families={r['feature_id'] for r in policies['records'] if r['frame']==i};ex.house_reader.groups={fid:g for fid,g in ex.house_reader.groups.items() if fid in families}
 source=deepcopy(ex.rows[i]['source']);b=cv2.imread(str(DATA/f'frames/{i:06d}.png'));m=cv2.imread(source['mask_path'],0);bc,_=current_broadcast_exclusion(b);dom,_=current_ice_visible_domain(i);native=NativeProfiles(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),source,(m>0)|bc|dom)
 fresh,diagnostics=ex.nested_house_features(i,native);cachefile=p/f'circle_candidates/{i:06d}.json';cache=read(cachefile);icefile=p/f'ice_candidates/{i:06d}.json';packet=read(icefile)
 thin=lambda:hashlib.sha256(json.dumps([f for f in packet['features'] if 'source_circle_fit' not in f],sort_keys=True).encode()).hexdigest()
 before=thin();changes=[]
 for cid,f in fresh.items():
  old=cache['features_by_curve'][cid];archives.append(dict(frame=i,target_id=f['target_id'],previous_candidate=old));cache['features_by_curve'][cid]=f
  tid=f['target_id'];packet['features']=[g for g in packet['features'] if g['target_id']!=tid]
  if f['point_count']:packet['features'].append(f)
  reason=None if f['point_count'] else 'No currently confirmed material readouts remain sufficient after localized native-source identity review; source evidence was not replaced by model or temporal pixels'
  for cov in packet['per_feature_coverage']:
   if cov['target_id']==tid:cov.update(points=f['point_count'],native_material_diagnostic=f.get('extraction_diagnostic'),missing_reason=reason,native_source_identity_review=True)
  packet['source_correlated_groups']=[g for g in packet['source_correlated_groups'] if g['target_id']!=tid]
  if f['point_count']:packet['source_correlated_groups'].append(dict(group_id=f['source_evidence_group_id'],target_id=tid,native_source_pts=source['native_pts'],representative_point_count=f['point_count'],native_individual_readout_count=f['source_circle_fit']['actual_native_readout_count'],correlation_reason='same current frame joint physical circle family after source material identity exclusions'))
  changes.append(dict(target_id=tid,old_points=old['point_count'],new_points=f['point_count'],native_readouts=f.get('source_circle_fit',{}).get('actual_native_readout_count',0)))
 cache['diagnostics'].update(diagnostics);cache['localized_current_source_identity_review']=dict(policy_path=str(p/'circle_diagnostics/native_material_identity_domains.json'),policy_sha256=sha(p/'circle_diagnostics/native_material_identity_domains.json'))
 packet['point_count']=sum(f['point_count'] for f in packet['features']);packet['localized_circle_source_identity_review']=cache['localized_current_source_identity_review'];assert thin()==before
 for file,value in ((cachefile,cache),(icefile,packet)):
  temporary=file.with_suffix('.tmp');write(temporary,value);temporary.replace(file)
 records.append(dict(frame=i,changes=changes,noncircle_features_sha256_before_and_after=before,candidate_sha256=sha(icefile)));print(i,changes,flush=True)
receipt=p/'circle_diagnostics/localized_identity_review_receipt.json';previous=read(receipt) if receipt.exists() else dict(records=[])
previous['records']=[r for r in previous['records'] if r['frame'] not in frames]+records;previous.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),thin_features_preserved_exactly=True,policy_sha256=sha(p/'circle_diagnostics/native_material_identity_domains.json'))
write(receipt,previous);write(p/'circle_diagnostics/localized_identity_review_previous_source_candidates.json',dict(records=archives))
