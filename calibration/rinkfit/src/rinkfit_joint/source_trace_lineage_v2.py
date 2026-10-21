"""Exact direct-readout provenance for sparse/current trace source ordering.

The original representative seed is linked by binary64 equality to its stored
actual pixel measurement, then to that measurement's current support vertex.
No world coordinate, angle, nearest point, or tolerance supplies the relation.
"""
from pathlib import Path
from collections import Counter,defaultdict
import copy,struct
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from . import source_trace_lineage_v1 as previous
SCHEMA='rinkfit.selected_source_direct_readout_trace_lineage.v2'
def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def checked(r,seen):
 p=Path(r.get('file',r.get('path'))).resolve();key=(str(p),r['sha256'])
 if key not in seen:require(file_sha256(p)==r['sha256'],'Changed exact direct-readout source dependency '+str(p));seen.add(key)
 return p
def binary_uv(q):return np.asarray(q,dtype='<f8').tobytes()
def _construct(previous_manifest):
 old=read_json(previous_manifest);v=previous.assert_current(previous_manifest,return_records=True);seen=set();binding=read_json(checked(old['actual_source_reader_input'],seen));support=defaultdict(list)
 for part in binding['source_support_parts']:
  for s in read_jsonl(checked(part['ref'],seen)):
   if s['source_packet']not in part.get('exclude_source_packets',[]):support[(s['statistical_input_id'],s['source_packet_sha256'])].append((s,part['ref']))
 traces={c['component_id']:c for c in read_jsonl(checked(old['components'][0]['source_trace_ref'],seen))};components=copy.deepcopy(v['components']);proofs=[];counts=Counter();mapped={};fixed=[]
 for c in components:
  if not c['unknown_rows']:
   mapped.update({q['raw_index']:dict(component_id=c['component_id'],**q)for q in c['Query_rows']});counts['inherited_exact_current_vertex_components']+=1;continue
  trace=traces[c['component_id']];options=support[(trace['statistical_input_id'],trace['source_packet_sha256'])];unique={object_sha256(s):(s,r)for s,r in options};require(len(unique)==1,'Ambiguous actual source support generation');s,sref=next(iter(unique.values()));require(s['shot_id']==trace['shot_id']and s['native_pts']==trace['native_pts'],'Actual readout source exposure changed');segments=s['current_support_segments'];require(sum(seg==trace['ordered_raw_indices']for seg in segments)==1,'Source component not exactly one current visible support segment');require(s['source_visibility_link_rejections']==trace['explicit_visibility_link_rejections'],'Actual source visibility relation changed');meta=s['source_curve_family_metadata'];direct=meta.get('source_circle_fit',{}).get('direct_readouts',[]);require(direct,'Missing actual source direct-readout lineage');audits={a['raw_index']:a for a in s['actual_source_audits']};newrows=[]
  for q in c['Query_rows']:
   seed=q['finite_Query_selected_mean_proof']['source_seed_uv'];candidates=[(j,d)for j,d in enumerate(direct)if binary_uv(d['representative_uv'])==binary_uv(seed)];require(len(candidates)==1,'Representative source seed is not one unique original pixel readout');j,d=candidates[0];raw_matches=[k for k,uv in enumerate(trace['raw_uv'])if binary_uv(uv)==binary_uv(d['uv'])];require(len(raw_matches)==1,'Actual original pixel readout is not one trace raw vertex');vi=raw_matches[0];ri=trace['ordered_raw_indices'][vi];a=audits[ri]
   require(binary_uv(a['raw_uv'])==binary_uv(d['uv'])and binary_uv(a['current_uv'])==binary_uv(trace['current_uv'][vi]),'Current visible trace vertex differs from actual reread');require(a['original_profile_evidence']==d['source_evidence'],'Original material profile evidence changed');e=d['source_evidence'];require(d['query_index']==e['query_index']and e['actual_source_support_outside_person_and_broadcast_masks']is True and e['physical_layer']==d['role'],'Actual native material query/layer evidence changed');require(a['status']=='current_native_material_interface'and a['profile']['valid']and a['support_guard']['status']=='reader_support_clear'and a['original_source_profile_guard']['status']=='reader_support_clear','Actual current material profile/whole support is not qualified')
   for fp,hk in(('native_png','native_png_sha256'),('mask_path','mask_sha256')):checked(dict(file=s[fp],sha256=s[hk]),seen)
   chain=dict(actual_retained_support=sref,actual_retained_support_sha256=object_sha256(s),original_direct_readout_index=j,original_direct_readout_sha256=object_sha256(d),original_direct_readout=d,original_seed_binary64_le=binary_uv(seed).hex(),direct_actual_original_UV_binary64_le=binary_uv(d['uv']).hex(),producer_raw_source_index=ri,ordered_trace_vertex=vi,actual_current_source_audit_sha256=object_sha256(a),actual_current_source_audit=a,trace_current_UV=trace['current_uv'][vi],representative_seed_exact_unique=True,original_readout_UV_exact_unique=True,trace_vertex_is_a_different_current_readout_from_selected_mean=True,shared_source_query_identity_not_equal_world_angle=True)
   q.update(mapping_status='exact_original_direct_readout_lineage',ordered_trace_coordinate=vi,producer_raw_source_index=ri,actual_direct_readout_chain=chain);q.pop('unknown_reason',None);proofs.append(dict(component_id=c['component_id'],raw_index=q['raw_index'],source_row_id=q['source_row_id'],**chain));newrows.append(q['raw_index']);counts['new_exact_direct_readout_rows']+=1
  ordered=sorted(c['Query_rows'],key=lambda q:q['ordered_trace_coordinate']);require(len({q['ordered_trace_coordinate']for q in ordered})==len(ordered),'Distinct selected Queries collapse onto one original source pixel readout');c.update(status='source_lineage_order_available',source_trace_mapping_rule='original representative seed -> actual original direct material readout -> current visible source trace vertex',unknown_rows=[],ordered_selected_rows=[q['raw_index']for q in ordered],source_adjacency_edges=[[a['raw_index'],b['raw_index']]for a,b in zip(ordered,ordered[1:])]);mapped.update({q['raw_index']:dict(component_id=c['component_id'],**q)for q in c['Query_rows']});fixed.append(c['component_id']);counts['new_exact_direct_readout_components']+=1
 counts['order_available_components']=len(components);counts['source_Query_rows']=sum(len(c['Query_rows'])for c in components);counts['source_order_edges']=sum(len(c['source_adjacency_edges'])for c in components);require(counts['source_Query_rows']==old['counts']['source_Query_rows']and len(mapped)==counts['source_Query_rows'],'Full source-order denominator changed')
 report=dict(schema=SCHEMA,issuer=ref(__file__),previous_issuer=ref(previous.__file__),previous_source_lineage=ref(previous_manifest),source_packet=old['source_packet'],actual_source_reader_input=old['actual_source_reader_input'],counts=dict(counts),components=components,new_direct_readout_proofs=proofs,actual_dependencies=[dict(file=k[0],sha256=k[1])for k in sorted(seen)],all_original_Query_UV_W_source_branches_unchanged=True,world_angle_station_or_model_coordinates_used=False,nearest_or_tolerance_or_array_ordinal_assignment_used=False,source_connectivity_comes_from_actual_current_visible_support_segments=True,source_noise_or_independence_issued=False,quality_accepted=False,solver_started=False)
 return report,mapped

def build(previous_manifest,output):
 out=Path(output);require(not out.exists(),'New immutable source lineage generation required');report,mapping=_construct(previous_manifest);out.mkdir(parents=True);write_json(out/'manifest.json',report);return ref(out/'manifest.json')
def assert_current(manifest,*,return_records=False):
 m=read_json(manifest);require(m['schema']==SCHEMA and m['issuer']==ref(__file__),'Wrong direct-readout source lineage issuer');actual,mapping=_construct(m['previous_source_lineage']['file']);require(m==actual,'Source direct-readout certificate differs from exact current inputs');r=dict(schema=SCHEMA,passed=True,manifest=ref(manifest),source_packet=m['source_packet'],counts=m['counts'],world_or_model_coordinates_used=False,Query_UV_or_W_changed=False)
 return dict(report=r,components=m['components'],mapping_by_raw_row=mapping)if return_records else r
