"""Full joint source accounting, exact full-W costs and material/time regressions."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,gzip
import numpy as np
from .io_utils import read_json,write_json,file_sha256

class PacketArrays:
 def __init__(self,path):self.path=Path(path);self.root=self.path.parent;self.packet=read_json(path);self.checked={}
 def load(self,d):
  if isinstance(d,str):d=self.packet['arrays'][d]
  p=Path(d['file']);p=p if p.is_absolute() else self.root/p
  if str(p) not in self.checked:self.checked[str(p)]=file_sha256(p)
  if self.checked[str(p)]!=d['sha256']:raise ValueError('Source array hash changed: '+str(p))
  return np.fromfile(p,dtype=d.get('dtype','<f8'),offset=d.get('offset_bytes',0),count=int(np.prod(d['shape']))).reshape(d['shape'])

def genuine_query_operators(packet,n):
 """Use each actual Query and its original normal; kind is not Query presence."""
 query_mask=np.zeros(n,bool);normals=np.zeros((n,2),float)
 for query in packet.get('queries',[]):
  i=int(query['row_index'])
  if not 0<=i<n or query_mask[i]:raise ValueError('Genuine Query row is out of range or duplicated')
  normal=np.asarray(query['normal'],float)
  if normal.shape!=(2,) or not np.isfinite(normal).all() or not np.any(normal):raise ValueError('Invalid original genuine Query normal')
  query_mask[i]=True;normals[i]=normal
 return query_mask,normals

def summarize(distance,ids,valid,E):
 ids=np.asarray(ids,dtype=int);vv=valid[ids]&np.isfinite(distance[ids]);q=distance[ids[vv]];norm=q/E[ids[vv]]
 return dict(active_rows=len(ids),valid_rows=int(vv.sum()),invalid_rows=int((~vv).sum()),max_px=float(np.max(q)) if len(q) else None,p95_px=float(np.quantile(q,.95)) if len(q) else None,median_px=float(np.median(q)) if len(q) else None,rms_px=float(np.sqrt(np.mean(q*q))) if len(q) else None,max_budget_ratio=float(np.max(norm)) if len(q) else None,rows_above_original_E=int(np.sum(norm>1.)),all_original_E_passed=bool(len(ids)>0 and vv.all() and np.all(norm<=1.)))

def metrics(model_file,packet_file,state_dir,output):
 out=Path(output)
 if out.exists():raise ValueError('New immutable full-source quality generation required')
 arrays=PacketArrays(packet_file);packet=arrays.packet;model=read_json(model_file);state=Path(state_dir);value=read_json(state/'value_report.json');n=packet['row_count']
 if file_sha256(model_file)!=value['model_sha256']:raise ValueError('Metrics model differs from this exact prediction generation')
 for f,desc in value['files'].items():
  if file_sha256(state/f)!=desc['sha256']:raise ValueError('State array changed '+f)
 uv=arrays.load('uv');pred=np.fromfile(state/'predicted_uv.f64','<f8').reshape(n,2);error=np.fromfile(state/'error_native_uv.f64','<f8').reshape(n,2);valid=np.fromfile(state/'valid.u8','u1').astype(bool);admit=arrays.load('admission').astype(bool);role=arrays.load('fit_role');active=admit&(role==1);ids=np.flatnonzero(active);kinds=arrays.load('kind');mat=arrays.load('material_index');pts=arrays.load('native_pts');shot_index=arrays.load('shot_index');E=arrays.load('E').reshape(n);queries,normal=genuine_query_operators(packet,n);sid=np.asarray(packet['shot_ids'])[shot_index];labels=np.asarray([m['label'] for m in model['materials']]);target=np.full(n,'UNASSIGNED',object);legal=mat<len(labels);target[legal]=labels[mat[legal]];distance=np.linalg.norm(error,axis=1);distance[queries]=np.abs(np.sum(error[queries]*normal[queries],axis=1))
 if np.max(np.abs((pred-uv-error)[ids][valid[ids]]),initial=0)>1e-8:raise ValueError('Raw native residual changed definition')
 out.mkdir(parents=True);partition=np.zeros(n,np.uint8);bykind=defaultdict(lambda:dict(groups=0,rank=0,half_cost=0.,invalid_groups=0));rank=0
 with gzip.open(out/'source_group_costs.jsonl.gz','wt',encoding='utf8') as f:
  for g in packet['groups']:
   rr=np.asarray(g['rows'],int);kind=g['statistic_kind'];unknown=kind=='constraint_only_unknown_noise'
   if unknown and (g['rank']!=0 or g['W']!=[]):raise ValueError('Unknown-noise constraint group must have rank 0 and no W')
   if not unknown and g['rank']<=0:raise ValueError('Statistical source group has no statistical rank')
   W=None if unknown else arrays.load(g['W']).reshape(g['rank'],2*len(rr));partition[rr]+=1;rank+=g['rank'];ok=bool(active[rr].all() and valid[rr].all() and np.isfinite(error[rr]).all());v=W@error[rr].ravel() if ok and not unknown else None;cost=.5*float(v@v) if v is not None else None;bykind[kind]['groups']+=1;bykind[kind]['rank']+=g['rank'];bykind[kind]['invalid_groups']+=not ok
   bykind[kind].setdefault('rows',0);bykind[kind]['rows']+=len(rr)
   if ok and not unknown:bykind[kind]['half_cost']+=cost
   record=dict(id=g['id'],statistic_kind=kind,rows=len(rr),rank=g['rank'],full_W_half_cost=cost,valid=ok,shot_ids=sorted(set(sid[rr].tolist())),material_labels=sorted(set(target[rr].tolist())),native_pts_min=int(pts[rr].min()),native_pts_max=int(pts[rr].max()),W_sha256=None if unknown else g['W']['sha256'],engineering_constraints_retained=True,noise_model_known=not unknown);f.write(json.dumps(record,separators=(',',':'))+'\n')
  independent=np.asarray(packet.get('independent_rows',[]),int)
  if len(independent):
   sigma=arrays.load('source_sigma');sigma=sigma[:,None] if sigma.ndim==1 else sigma;partition[independent]+=1;rank+=2*len(independent);ok=valid[independent]&np.isfinite(error[independent]).all(1);whitened=error[independent]/sigma[independent];cost=.5*np.sum(whitened*whitened,1);bykind['original']['groups']+=len(independent);bykind['original']['rank']+=2*len(independent);bykind['original']['invalid_groups']+=int((~ok).sum());bykind['original']['half_cost']+=float(np.sum(cost[ok]))
   for i,c,v in zip(independent,cost,ok):f.write(json.dumps(dict(id='independent/'+str(i),statistic_kind='original',rows=1,rank=2,full_W_half_cost=float(c) if v else None,valid=bool(v),shot_ids=[sid[i]],material_labels=[target[i]],native_pts_min=int(pts[i]),native_pts_max=int(pts[i])),separators=(',',':'))+'\n')
 if not np.array_equal(partition,active.astype(np.uint8)):raise ValueError('Original/new source units not counted exactly once')
 def stratify(keys):
  keys=np.asarray(keys,object);return {str(k):summarize(distance,ids[keys[ids]==k],valid,E) for k in sorted(set(keys[ids].tolist()))}
 rig=np.asarray([model['shots'][s]['rig_id'] for s in sid]);category=np.asarray(['ice' if t.startswith('ice.') else 'board' if t.startswith('board.') else 'board_print' if t.startswith('print_component.') else 'ad_print' if '.print.' in t else 'ad_entity' if t.startswith('Ad') else 'unknown' for t in target]);operator=np.asarray(['profile_query' if is_query else 'fixed_entity_point' if k==1 else 'width_midline' if t.split(':')[0] in {'ice.B1','ice.B2','ice.C','ice.C0','ice.A1','ice.A2','ice.A3','ice.A4'} else 'material_interface' for k,t,is_query in zip(kinds,target,queries)]);pair_index=arrays.load('stroke_pair_index').reshape(n) if 'stroke_pair_index' in packet['arrays'] else np.full(n,-1);operator[(pair_index>=0)&active]='paint_interface_profile_query';timekeys=np.asarray([s+'/'+str(int(p)//12800) for s,p in zip(sid,pts)]);s01=active&(sid=='s01');frame=(pts-256256)//256+1
 selections={'s01_all':np.flatnonzero(s01),'s01_Ad2_all':np.flatnonzero(s01&np.asarray([t.startswith('Ad2.') for t in target])),'s01_Ad2_entity':np.flatnonzero(s01&(kinds==1)&np.asarray([t.startswith('Ad2.') for t in target])),'s01_pulse_f135_140':np.flatnonzero(s01&(frame>=135)&(frame<=140)),'s01_first_native_frame':np.flatnonzero(s01&(frame==1)),'s01_Ad4_all':np.flatnonzero(s01&np.asarray([t.startswith('Ad4.') for t in target]))}
 for t in ('ice.R3:inner','ice.R3:middle','ice.R3:outer','ice.C0:outer'):selections['s01_'+t]=np.flatnonzero(s01&(target==t))
 for label,lo,hi in (('opening',1,200),('pulse_context',100,180),('later',201,1248)):
  selections['s01_Ad2_'+label]=np.flatnonzero(s01&(frame>=lo)&(frame<=hi)&np.asarray([t.startswith('Ad2.') for t in target]))
 entity_mask=s01&(kinds==1)&np.asarray([t.startswith('Ad2.') for t in target])
 entity_parts={}
 for label,lo,hi in (('opening',1,134),('anomaly_135_140',135,140),('later',141,1248)):
  rr=np.flatnonzero(entity_mask&(frame>=lo)&(frame<=hi));selections['s01_Ad2_entity_'+label]=rr;entity_parts[label]=len(rr)
 if sum(entity_parts.values())!=int(entity_mask.sum()):raise ValueError('Ad2 entity time partition lost or duplicated a corner')
 focus_groups=defaultdict(list)
 for i in ids:
  if target[i].startswith('ice.R3:') or target[i].split(':')[0]=='ice.C0':focus_groups[sid[i]+'/'+target[i]+'/'+str(int(pts[i])//12800)].append(i)
 focused={key:summarize(distance,rr,valid,E) for key,rr in sorted(focus_groups.items())}
 regressions={key:summarize(distance,rr,valid,E) for key,rr in selections.items()};boundaries=[]
 for rid,r in model['rigs'].items():
  b=model['blocks'][r['optics_block']]
  for k,name in ((0,'cx'),(1,'cy'),(11,'tau')):boundaries.append(dict(rig_id=rid,parameter=name,value=b['values'][k],lower=b['lower'][k],upper=b['upper'][k],lower_margin=b['values'][k]-b['lower'][k],upper_margin=b['upper'][k]-b['values'][k]))
 sourceids=packet['source_row_ids'];excluded_s01=[dict(row_index=int(i),source_row_id=sourceids[i],active=False) for i in (25537,34524)]
 report=dict(schema='rinkfit.joint9.full_source_quality.v1',model=dict(path=str(Path(model_file).resolve()),sha256=file_sha256(model_file)),packet=dict(path=str(Path(packet_file).resolve()),sha256=file_sha256(packet_file)),state_directory=str(state.resolve()),raw_rows=n,admitted_rows=int(admit.sum()),temporarily_held_admitted_rows=int(np.sum(admit&(role!=1))),active_rows=len(ids),inactive_rows=n-len(ids),all_active_sources_counted_once=True,statistical_rank=rank,unknown_noise_constraint_rows=bykind.get('constraint_only_unknown_noise',{}).get('rows',0),unknown_noise_constraint_groups=bykind.get('constraint_only_unknown_noise',{}).get('groups',0),known_statistical_active_rows=len(ids)-bykind.get('constraint_only_unknown_noise',{}).get('rows',0),full_W_by_kind=dict(bykind),full_W_half_cost=sum(v['half_cost'] for v in bykind.values()) if all(v['invalid_groups']==0 for v in bykind.values()) else None,all_new_statistics_formal=all(k in {'original','propagated_source_covariance','native_joint_readout'} for k in bykind),overall=summarize(distance,ids,valid,E),by_shot=stratify(sid),by_rig=stratify(rig),by_material=stratify(target),by_category=stratify(category),by_operator=stratify(operator),by_source_second=stratify(timekeys),s01_regressions=regressions,Ad2_entity_disjoint_time_partition=dict(counts=entity_parts,total=int(entity_mask.sum()),all_623_expected_corners_present=int(entity_mask.sum())==623),R3_C0_by_shot_material_source_second=focused,source_second_definition='floor(native_pts/12800), official original source clock; native_frame retained in source packet',s01_excluded_original_sources=excluded_s01,optics_boundary_margins=boundaries,measurement_width_scopes=model.get('measurement_widths',[]),stroke_pair_bindings=model.get('stroke_pair_model_bindings',[]),actual_paint_interface_rows=int(np.sum(active&(pair_index>=0))),paint_interface_residual_definition='Each actual source flank uses its own world-side width offset, finite station, true RS row and fixed profile query; no midpoint residual substitution',genuine_profile_query_rows=int(np.sum(active&queries)),legacy_kind2_rows=int(np.sum(active&(kinds==2))),query_presence_definition='Exact packet queries row identities, independent of kind; normal copied from each original Query without renormalization',engineering_distance_definition='Euclidean native finite/entity residual; original fixed-normal distance for actual genuine profile queries',statistical_distance_definition='original native 2D residual with complete W; no replacement with engineering distance',independent_accuracy_claimed=False,metric_scale_verified=False,formal_Phase_I_II_complete=False,joint_fit_complete=False)
 def tail_records(errors,good):
  dist=np.linalg.norm(errors,axis=1);dist[queries]=np.abs(np.sum(errors[queries]*normal[queries],axis=1));available=ids[good[ids]&np.isfinite(dist[ids])];chosen=list(available[np.argsort(dist[available])[-20:][::-1]])
  for shot_id in packet['shot_ids']:
   rr=available[sid[available]==shot_id];chosen.extend(rr[np.argsort(dist[rr])[-3:][::-1]])
  result=[];seen=set()
  for i in chosen:
   i=int(i)
   if i in seen:continue
   seen.add(i);result.append(dict(row_index=i,source_row_id=sourceids[i],shot_id=str(sid[i]),material=str(target[i]),kind=int(kinds[i]),native_pts=int(pts[i]),source_seconds=float(pts[i]/12800),source_uv=uv[i].tolist(),predicted_uv=(uv[i]+errors[i]).tolist(),error_native_uv=errors[i].tolist(),engineering_distance_px=float(dist[i]),original_E_px=float(E[i]),budget_ratio=float(dist[i]/E[i]),strict_certificate_valid=bool(valid[i])))
  return result
 report['native_replay_nuisance_semantics']=value.get('native_replay_nuisance_semantics','core.replay locally re-solves station, row and alpha per source at the fixed physical model; its query/domain-valid subset does not certify the explicit AL checkpoint auxiliaries.')
 report['model_hash_identity_limit']='Physical scene/rig/trajectory SHA alone does not bind locally profiled replay auxiliaries to the complete explicit AL checkpoint.'
 report['largest_strict_valid_native_errors']=tail_records(error,valid)
 report['projection_classification_counts']=value.get('projection_classification_counts')
 raw_names=('raw_station_RS_predicted_uv.f64','raw_station_RS_error_native_uv.f64','raw_station_RS_valid.u8')
 if all(name in value['files'] for name in raw_names):
  raw_pred=np.fromfile(state/raw_names[0],'<f8').reshape(n,2);raw_error=np.fromfile(state/raw_names[1],'<f8').reshape(n,2);raw_valid=np.fromfile(state/raw_names[2],'u1').astype(bool)
  if np.max(np.abs((raw_pred-uv-raw_error)[ids][raw_valid[ids]]),initial=0)>1e-8:raise ValueError('Independent local-RS residual changed source pixel definition')
  raw_distance=np.linalg.norm(raw_error,axis=1);raw_distance[queries]=np.abs(np.sum(raw_error[queries]*normal[queries],axis=1))
  report['raw_fixed_current_station_local_RS']=dict(semantics=value.get('raw_projection_semantics'),overall=summarize(raw_distance,ids,raw_valid,E),by_shot={s:summarize(raw_distance,ids[sid[ids]==s],raw_valid,E) for s in packet['shot_ids']},by_material={t:summarize(raw_distance,ids[target[ids]==t],raw_valid,E) for t in sorted(set(target[ids]))},s01_regressions={key:summarize(raw_distance,rr,raw_valid,E) for key,rr in selections.items()},largest_errors=tail_records(raw_error,raw_valid),strict_certificate_checks_relaxed=False,whole_interval_RS_uniqueness_or_physical_support_certified=False,full_W_cost_uses_existing_strict_certificate_scope=True)
 expected=value['full_W_half_cost'];computed=report['full_W_half_cost'];report['CXX_full_W_recompute_difference']=None if expected is None or computed is None else computed-expected
 if computed is not None and expected is not None and abs(computed-expected)>1e-6*max(1.,abs(expected)):raise ValueError('Independent W cost differs from backend')
 write_json(out/'quality.json',report)
 print(json.dumps(dict(active=len(ids),invalid=report['overall']['invalid_rows'],max_px=report['overall']['max_px'],full_W_half_cost=computed,statistics_formal=report['all_new_statistics_formal'],report=str(out/'quality.json')),ensure_ascii=False));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--packet',required=True);p.add_argument('--state',required=True);p.add_argument('--out',required=True);a=p.parse_args();metrics(a.model,a.packet,a.state,a.out)
