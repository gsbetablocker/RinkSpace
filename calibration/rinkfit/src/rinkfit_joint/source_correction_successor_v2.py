"""Actual source-mean/visibility correction at an immutable batch boundary.

Only literal typed-source rows are changed here. Old physical/auxiliary state is
not silently restored or reinterpreted: root selects a valid complete capture
parent separately and affected q/RS values must then be rebound explicitly.
"""
from pathlib import Path
import copy,collections,importlib
import numpy as np
from .io_utils import read_json,write_json,read_jsonl,file_sha256
from .packet_storage import write_packet
from .r_inner_prefix_v1 import array
from .board_middle_append_v1 import checked,require,ref
SCHEMA='rinkfit.immutable_actual_source_correction_successor.v1'
def generation_rows(generation,consumer,seen):
 checked(consumer,seen);name=Path(consumer['file']).stem;require(name in('source_readout_contract_v3','source_readout_contract_v4'),'Unknown current native-source consumer');module=importlib.import_module('rinkfit_joint.'+name);return module.source_generic_readout_rows(generation,lambda r:checked(r,seen))
def source_chain_dependencies(generations,packet,seen):
 edges=[];applied=[];ids={rid:i for i,rid in enumerate(packet['source_row_ids'])}
 for spec in generations:
  g=read_json(checked(spec['generation'],seen));c=read_json(checked(g['input_contract'],seen))
  if 'adjacent_all_readout_records'not in c:continue
  full={q['source_row_id']:q for q in read_jsonl(checked(c['adjacent_all_readout_records'],seen))};targets=[q for q in read_jsonl(checked(g['readout_records'],seen))if q.get('reference_kind')=='previous_direct_source_verified_exposure_same_physical_track'];done=set()
  for q in targets:
   applied.append(q['source_row_id']);cur=q['source_row_id']
   while full[cur].get('reference_kind')=='previous_direct_source_verified_exposure_same_physical_track':
    r=full[cur];prev=r['reference_source_row_id'];require(r['status']=='wide_context_direct_source_supported'and full[prev]['status']in('wide_context_direct_source_supported','original_birth_reference'),'Applied source dependency crosses a failed readout')
    if cur not in done:edges.append(dict(source_row_id=cur,reference_source_row_id=prev,source_row=ids[cur],reference_row=ids[prev],root_birth_row_id=r['root_birth_row_id'],direct_reader=c['adjacent_reader'],complete_source_graph=c['adjacent_all_readout_records'],source_error_is_not_independent=True));done.add(cur)
    cur=prev
 return dict(schema='rinkfit.actual_adjacent_source_chain_dependency_ledger.v1',applied_source_rows=applied,consumed_reference_edges=edges,calibrated_noise_covariance_issued=False,native_v18_cross_frame_weighting_consumed=False,native_v18_splits_dependency_units_by_exposure=True)

def merge_temporal_groups(base,ledger):
 result=copy.deepcopy(base);groups=result['groups'];parent={}
 def root(i):
  parent.setdefault(i,i)
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def join(a,b):
  a=root(a);b=root(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for g in groups:
  for i in g['rows'][1:]:join(g['rows'][0],i)
 for e in ledger['consumed_reference_edges']:join(e['source_row'],e['reference_row'])
 affected={root(e['source_row'])for e in ledger['consumed_reference_edges']};merged=collections.defaultdict(list);oldids=collections.defaultdict(list);unchanged=[]
 for g in groups:
  if not g['rows']:continue
  r=root(g['rows'][0])
  if r in affected:merged[r]+=g['rows'];oldids[r].append(g['id'])
  else:unchanged.append(g)
 for r,rows in sorted(merged.items()):unchanged.append(dict(id='source_chain:'+__import__('hashlib').sha256(np.asarray(sorted(rows),dtype='<u8').tobytes()).hexdigest()[:24],rows=sorted(rows),inherited_dependency_group_ids=oldids[r],actual_source_reference_chain_correlated=True,independent_source_noise_claimed=False))
 result['groups']=unchanged;return result

def build(parent_packet,generations,output):
 src=Path(parent_packet).resolve();out=Path(output).resolve();require(not out.exists(),'Immutable source correction exists');seen=set();old=read_json(src);A={k:np.array(array(old,k,src.parent,seen))for k in old['arrays']};rows={};byid=collections.defaultdict(list)
 for i,rid in enumerate(old['source_row_ids']):byid[rid].append(i)
 for spec in generations:
  current=generation_rows(spec['generation'],spec['consumer'],seen);require(not set(rows).intersection(current),'Overlapping source corrections require explicit precedence');rows.update(current)
 changed=[];pending=[];affected=set();bindings=[]
 for rid,r in rows.items():
  require(len(byid[rid])==1,'Corrected native typed source identity is not unique');i=byid[rid][0];require(int(A['native_pts'][i])==r['native_pts']and old['shot_ids'][int(A['shot_index'][i])]==r['shot_id']and np.array_equal(A['uv'][i],r['original_source_uv'])and int(A['kind'][i])==1,'Current source correction redirects original UV/clock/material kind');require(A['admission'][i]==1 and A['fit_role'][i]==1,'Corrected cohort was not active source-qualified data')
  if r['source_mean_changed']:A['uv'][i]=r['current_uv'];A['query_uv'][i]=r['current_uv'];A['row'][i]=r['current_uv'][1];changed.append(i);affected.add(r['physical_track_id'])
  if r.get('source_only_pending'):A['admission'][i]=0;pending.append(i);affected.add(r['physical_track_id'])
  bindings.append(dict(row_index=i,source_row_id=rid,physical_track_id=r['physical_track_id'],native_pts=r['native_pts'],old_source_uv=r['original_source_uv'],current_uv=r['current_uv'],source_mean_changed=r['source_mean_changed'],source_geometry_eligible=r['source_geometry_eligible'],source_only_pending=bool(r.get('source_only_pending')),source_readout_generation_id=r['source_readout_generation_id'],source_readout_status=r['source_readout_status'],source_readout_class=r['source_readout_class']))
 out.mkdir();doc=copy.deepcopy(old)
 for k in ('uv','query_uv','row','admission'):
  if np.array_equal(A[k],array(old,k,src.parent,seen)):continue
  f=out/(k+'.bin');A[k].tofile(f);doc['arrays'][k]={**old['arrays'][k],**ref(f)}
 bad=set(pending)
 for g in doc['groups']:
  if bad.intersection(g['rows']):require(g['statistic_kind']=='constraint_only_unknown_noise'and g['rank']==0 and g['W']==[],'Source correction would slice actual noise W; fresh statistical generation required');g['rows']=[i for i in g['rows']if i not in bad]
 doc['groups']=[g for g in doc['groups']if g['rows']or g.get('statistic_kind')!='constraint_only_unknown_noise']
 for g in doc['engineering_pixel_dependency_groups']['groups']:g['rows']=[i for i in g['rows']if i not in bad]
 doc['engineering_pixel_dependency_groups']['groups']=[g for g in doc['engineering_pixel_dependency_groups']['groups']if g['rows']]
 chain=source_chain_dependencies(generations,doc,seen);write_json(out/'source_chain_dependencies.json',chain);doc['engineering_pixel_dependency_groups']=merge_temporal_groups(doc['engineering_pixel_dependency_groups'],chain)
 active=(A['admission']==1)&(A['fit_role']==1);doc['active_rows']=int(active.sum());doc['recapture_rows']=np.flatnonzero(active&(np.arange(len(active))>=doc['new_source_start'])).tolist();doc['constraint_only_unknown_noise_rows']=sum(len(g['rows'])for g in doc['groups']if g.get('statistic_kind')=='constraint_only_unknown_noise');write_json(out/'source_bindings.json',bindings)
 doc['source_current_correction_successor']=dict(schema=SCHEMA,parent_packet=ref(src),generations=generations,source_bindings=ref(out/'source_bindings.json'),source_chain_dependencies=ref(out/'source_chain_dependencies.json'),native_temporal_weighting_not_consumed=True,consumer=ref(__file__),raw_rows_unchanged=True,source_mean_changes=len(changed),source_wrong_identity_pending=len(pending),pending_row_indices=pending,affected_tracks=sorted(affected),source_noise_covariance_issued=False,model_or_projection_used_for_source_UV=False,source_only_prototype=True,actual_accepted_capture_parent_bound=False,fresh_affected_q_RS_auxiliary_rebind_required=True)
 doc['source_manifests']+=[ref(__file__),ref(out/'source_bindings.json'),ref(out/'source_chain_dependencies.json')]+[x[k]for x in generations for k in('generation','consumer')];write_packet(out/'packet.json',doc);report=assert_current(out/'packet.json');write_json(out/'source_guard_positive.json',report);return report

def assert_current(packet_path):
 path=Path(packet_path).resolve();doc=read_json(path);b=doc['source_current_correction_successor'];seen=set();require(b['schema']==SCHEMA and b['consumer']==ref(__file__)and b['source_only_prototype']and not b['actual_accepted_capture_parent_bound'],'Unknown source correction semantics');src=checked(b['parent_packet'],seen);old=read_json(src);require(doc['row_count']==old['row_count']and doc['source_row_ids']==old['source_row_ids'],'Source correction lost raw denominator');A={k:array(doc,k,path.parent,seen)for k in doc['arrays']};B={k:array(old,k,src.parent,seen)for k in old['arrays']};expected={}
 for spec in b['generations']:
  rr=generation_rows(spec['generation'],spec['consumer'],seen);require(not set(expected).intersection(rr),'Overlapping current source generations');expected.update(rr)
 bindings=read_json(checked(b['source_bindings'],seen));require(len(bindings)==len(expected)and{r['source_row_id']for r in bindings}==set(expected),'Correction binding loses full source generation')
 changed=set();pending=set()
 for d in bindings:
  r=expected[d['source_row_id']];i=d['row_index'];require(doc['source_row_ids'][i]==r['source_row_id']and np.array_equal(B['uv'][i],r['original_source_uv'])and np.array_equal(A['uv'][i],r['current_uv'])and np.array_equal(A['query_uv'][i],r['current_uv']),'New packet is not the exact source reread effect');require(int(A['admission'][i])==(0 if r.get('source_only_pending')else int(B['admission'][i])),'Source wrong-identity pending not installed');require(int(A['native_pts'][i])==r['native_pts']and doc['shot_ids'][int(A['shot_index'][i])]==r['shot_id'],'Source corrected clock/shot changed')
  if r['source_mean_changed']:changed.add(i)
  if r.get('source_only_pending'):pending.add(i)
 for k in B:
  differences=np.flatnonzero(np.any(A[k]!=B[k],axis=tuple(range(1,A[k].ndim)))if A[k].ndim>1 else A[k]!=B[k]);expected_indices=changed if k in('uv','query_uv','row')else pending if k=='admission'else set();require(set(map(int,differences))==expected_indices,'Unexpected corrected source/auxiliary mutation '+k)
 for k in ('source_dataset_ids','source_family_ids','queries','stroke_pairs','source_station_domains','material_binding','independent_rows','independent_statistic_kind','joint_statistics_rank'):require(doc[k]==old[k],'Source correction changes unaffected mathematics '+k)
 expectedgroups=copy.deepcopy(old['groups'])
 for g in expectedgroups:
  if pending.intersection(g['rows']):require(g['rank']==0 and g['W']==[]and g['statistic_kind']=='constraint_only_unknown_noise','Source correction changed nonzero W');g['rows']=[i for i in g['rows']if i not in pending]
 expectedgroups=[g for g in expectedgroups if g['rows']or g.get('statistic_kind')!='constraint_only_unknown_noise'];require(doc['groups']==expectedgroups,'Source correction modifies other W/groups');ed=copy.deepcopy(old['engineering_pixel_dependency_groups'])
 for g in ed['groups']:g['rows']=[i for i in g['rows']if i not in pending]
 ed['groups']=[g for g in ed['groups']if g['rows']];chain=read_json(checked(b['source_chain_dependencies'],seen));require(chain==source_chain_dependencies(b['generations'],doc,seen),'Actual reference source-chain ledger differs');ed=merge_temporal_groups(ed,chain);require(doc['engineering_pixel_dependency_groups']==ed,'Source correction changes unrelated E dependency components');active=(A['admission']==1)&(A['fit_role']==1);require(doc['active_rows']==int(active.sum())and sorted(pending)==b['pending_row_indices'],'New source membership/count differs')
 return dict(schema=SCHEMA,passed=True,packet=ref(path),parent_packet=b['parent_packet'],rows=doc['row_count'],active_rows=doc['active_rows'],complete_reread_source_rows=len(expected),actual_source_mean_changes=len(changed),source_wrong_identity_pending=len(pending),raw_UV_ID_PTS_history_preserved=True,all_other_source_arrays_and_W_Query_exact=True,new_noise_W_or_rank_issued=False,accepted_physical_parent_bound=False,numeric_rebind_required=True,source_chain_edges=len(chain['consumed_reference_edges']),source_chain_applied_rows=len(chain['applied_source_rows']),native_cross_frame_information_weight_consumed=False,solver_started=False)
