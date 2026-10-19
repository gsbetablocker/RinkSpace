"""Apply only measured clipped-house outer arcs; preserve every other candidate byte-value."""
from calibration.program_tracking.extract_first_shot_ice import *
from collections import defaultdict
import datetime
p=DATA/'repair_20261014';plan=read(p/'circle_diagnostics/short_arc_repair_plan.json');byframe=defaultdict(list)
for item in plan['records']:byframe[item['frame']].append(item['house'])
ex=Extractor(p,ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014');ex.use_circle_cache=False
records=[];unchanged_thin_hashes=[]
for i,houses in byframe.items():
 source=deepcopy(ex.rows[i]['source']);b=cv2.imread(str(DATA/f'frames/{i:06d}.png'));m=cv2.imread(source['mask_path'],0);bc,_=current_broadcast_exclusion(b);dom,_=current_ice_visible_domain(i);n=NativeProfiles(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),source,(m>0)|bc|dom);features,diag=ex.nested_house_features(i,n)
 file=p/f'circle_candidates/{i:06d}.json';cache=read(file);before_sha=sha(file);changed=[]
 for fid in houses:
  cid=fid+':outer';old=cache['features_by_curve'][cid];new=features[cid]
  if not new['point_count'] and not old['point_count']:continue
  cache['features_by_curve'][cid]=new;cache['diagnostics'][fid]=diag[fid];changed.append(dict(target_id=new['target_id'],old_points=old['point_count'],new_points=new['point_count'],native_readouts=new.get('source_circle_fit',{}).get('actual_native_readout_count',0),geometry_policy=new.get('joint_group_navigation'),missing_reason=None if new['point_count'] else 'No two current source material stations survive the confirmed contour identity and fixed physical circle geometry; prior trial withdrawn'))
 if not changed:records.append(dict(frame=i,changed=False,targets=houses,reason='fewer than two current material stations remain consistent with fixed physical circle shape; no invented tangent or shape'));continue
 cache['short_arc_repair']=dict(algorithm_sha256=sha(ROOT/'calibration/program_tracking/native_ice_short_arc_repair.py'),source_before_cache_sha256=before_sha,changed_targets=changed)
 tmp=file.with_suffix('.tmp');write(tmp,cache);tmp.replace(file)
 # Only replace these outer-circle entries in final full ice candidate.
 icefile=p/f'ice_candidates/{i:06d}.json';packet=read(icefile)
 thinbefore=hashlib.sha256(json.dumps([f for f in packet['features'] if 'source_circle_fit' not in f],sort_keys=True).encode()).hexdigest()
 for change in changed:
  tid=change['target_id'];packet['features']=[f for f in packet['features'] if f['target_id']!=tid];f=features[tid.removeprefix('ice.')]
  if f['point_count']:packet['features'].append(f)
  for cov in packet['per_feature_coverage']:
   if cov['target_id']==tid:cov.update(points=f['point_count'],native_material_diagnostic=f.get('extraction_diagnostic'),missing_reason=change['missing_reason'],short_arc_repair=True)
  packet['source_correlated_groups']=[g for g in packet.get('source_correlated_groups',[]) if g['target_id']!=tid]
  if f['point_count']:packet['source_correlated_groups'].append(dict(group_id=f['source_evidence_group_id'],target_id=tid,native_source_pts=packet['source']['native_pts'],representative_point_count=f['point_count'],native_individual_readout_count=f['source_circle_fit']['actual_native_readout_count'],correlation_reason='same current-frame fixed physical circle shape with only source-identifiable translation; correlated representatives'))
 packet['point_count']=sum(f['point_count'] for f in packet['features']);packet['clipped_circle_arc_repair']=cache['short_arc_repair']
 thinafter=hashlib.sha256(json.dumps([f for f in packet['features'] if 'source_circle_fit' not in f],sort_keys=True).encode()).hexdigest();assert thinbefore==thinafter
 tmp=icefile.with_suffix('.tmp');write(tmp,packet);tmp.replace(icefile);unchanged_thin_hashes.append(dict(frame=i,sha256=thinbefore))
 records.append(dict(frame=i,changed=True,changes=changed,ice_candidate_sha256=sha(icefile)))
write(p/'circle_diagnostics/short_arc_repair_receipt.json',dict(schema='rinkspace.current_native_clipped_circle_arc_repair.v1',records=records,changed_frames=[r['frame'] for r in records if r['changed']],thin_features_preserved_exactly=unchanged_thin_hashes,algorithm_sha256=sha(ROOT/'calibration/program_tracking/native_ice_short_arc_repair.py'),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_approved_frame_1_200_modified=True))
print('changed',sum(r['changed'] for r in records),'of',len(records));print('still_absent',[(r['frame'],r.get('targets')) for r in records if not r['changed']])

