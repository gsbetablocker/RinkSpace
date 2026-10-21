"""Exact selected native source-mean lineage for Query/trace source order.

This grants source order only. It does not select world station, alter Query,
or infer source depth, covariance, independence or missing connectivity.
"""
from pathlib import Path
import struct,collections
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .r_inner_prefix_v1 import array
SCHEMA='rinkfit.selected_source_mean_trace_lineage.v1'
def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def require(ok,text):
 if not ok:raise ValueError(text)
def checked(r,seen):
 p=Path(r.get('file',r.get('path'))).resolve();key=(str(p),r['sha256'])
 if key not in seen:require(file_sha256(p)==r['sha256'],'Changed source lineage input '+str(p));seen.add(key)
 return p
def _construct(feedback):
 seen=set();f=read_json(feedback);pp=checked(f['source_packet'],seen);p=read_json(pp);components=list(read_jsonl(checked(f['files']['components'],seen)));require(len(components)==f['unresolved_multirow_components']==55,'Full55 unresolved-source denominator');inputpath=checked(f['source_reader_input_binding'],seen);binding=read_json(inputpath);actual_trace_manifest=read_json(checked(f['actual_source_component_manifest'],seen));tracepath=checked(f['actual_support_components'],seen);traceby={c['component_id']:c for c in read_jsonl(tracepath)};wanted={q['source_row_id']for c in components for q in c['actual_current_raw_Queries']};sourcepath=checked(binding['rows'],seen);rby={r['source_row_id']:r for r in read_jsonl(sourcepath)if r['source_row_id']in wanted};require(len(rby)==len(wanted)==166,'Whole selected source-row lineage');qs={q['row_index']:q for q in p['queries']};A={k:array(p,k,pp.parent,seen)for k in('uv','profile_normal','admission','fit_role','native_pts','material_index')};records=[];counts=collections.Counter();mapped={}
 for x in f['source_reader_qualifications'].values():checked(x,seen)
 for part in f['source_support_parts']:checked(part.get('ref',part),seen)
 for c in components:
  cid=c['component_id'];trace=c['actual_source_component_trace'];require(cid in traceby and traceby[cid]==trace,'Feedback trace differs from actual producer source trace');require(trace['edges_come_only_from_ordered_current_support_segments'],'Trace lacks real source adjacency');uv=np.asarray(trace['current_uv'],float);raw_indices=trace['ordered_raw_indices'];require(len(uv)==len(raw_indices)==len(trace['raw_uv'])and len(uv)>0,'Actual trace full raw-index denominator');queries=[];fail=[]
  for q in c['actual_current_raw_Queries']:
   i=int(q['raw_index']);rid=q['source_row_id'];actual=qs.get(i);r=rby[rid];mu=np.asarray(q['original_source_UV'],dtype='<f8');normal=np.asarray(q['original_Query_exact']['normal'],dtype='<f8');require(actual==q['original_Query_exact']and actual['source_row_id']==rid==p['source_row_ids'][i],'Entire original source Query/ID changed');require(A['admission'][i]==1 and A['fit_role'][i]==1 and r['source_admitted']is True,'Inactive or unqualified source cannot issue order');require(int(A['native_pts'][i])==q['native_pts']==trace['native_pts']and int(A['material_index'][i])==q['material_index'],'Source clock/material changed');require(np.array_equal(A['uv'][i],mu)and np.array_equal(mu,r['current_uv'])and np.array_equal(A['profile_normal'][i],normal),'Actual selected source mean/normal changed');a=r['source_measurement_audit'];require(np.array_equal(a['normal'],normal),'Recorded operator normal changed');alpha=np.asarray(actual['alpha'],dtype='<f8');line=np.asarray(actual['native_uv'],dtype='<f8');zero=np.flatnonzero(alpha==0);require(len(zero)==1 and line.shape==(len(alpha),2),'Original finite Query lacks unique source seed');seed=line[int(zero[0])];profile=a.get('profile',{});t=a.get('normal_offset_px',profile.get('offset_px'));require(t is not None and alpha[0]<=t<=alpha[-1],'Selected actual optical mean outside original finite Query');require(np.array_equal(seed+float(t)*normal,mu),'Original source operator arithmetic does not reproduce selected mean');require(np.array_equal(seed[None,:]+alpha[:,None]*normal[None,:],line),'Original Query sample construction changed');require(a.get('numeric_valid',profile.get('valid',False))is True,'Source optical readout not valid')
   if len(alpha)==49:require(np.array_equal(a['actual_profile_normal_offsets'],alpha),'Original cubic49 domain changed')
   else:require(len(alpha)==25 and np.array_equal(alpha,np.arange(-3,3.0001,.25)),'Unknown original finite source25 domain')
   for rr in q['actual_frame_and_mask_refs']:checked(rr,seen)
   exact=np.flatnonzero(np.all(uv==mu,axis=1));jr=dict(raw_index=i,source_row_id=rid,source_dataset_id=q['source_dataset_id'],native_pts=q['native_pts'],material_index=q['material_index'],original_Query=actual,original_Query_sha256=object_sha256(actual),selected_source_UV=mu.tolist(),selected_source_UV_binary64_le=mu.tobytes().hex(),actual_source_record_sha256=object_sha256(r),source_reader_input_binding=f['source_reader_input_binding'],actual_source_rows=binding['rows'],source_component_ref=c['source_trace_ref'],exact_current_trace_vertex_indices=exact.tolist(),finite_Query_selected_mean_proof=dict(source_operator_status=a['status'],source_operator_audit_sha256=object_sha256(a),normal=normal.tolist(),source_seed_uv=seed.tolist(),selected_original_optical_alpha=float(t),original_alpha_bounds=[float(alpha[0]),float(alpha[-1])],actual_sample_count=len(alpha),selected_mean_binary64_forward_replay_exact=True,all_literal_Query_samples_binary64_forward_replay_exact=True,Query_or_mean_or_source_material_not_reselected=True),actual_source_frame_mask_refs=q['actual_frame_and_mask_refs'])
   if len(exact)==1:jr.update(mapping_status='exact_unique_selected_source_mean_vertex',ordered_trace_coordinate=int(exact[0]),producer_raw_source_index=int(raw_indices[int(exact[0])]))
   else:jr.update(mapping_status='unknown_current_trace_correspondence',unknown_reason='selected optical mean is not exactly one current source trace vertex; no nearest, tolerance, inferred station or ordinal substituted');fail.append(i)
   queries.append(jr);counts['source_Query_rows']+=1;counts['exact_unique_source_mean_rows']+=len(exact)==1;counts['unknown_source_relation_rows']+=len(exact)!=1
  good=not fail;rec=dict(component_id=cid,status='source_lineage_order_available'if good else'unknown_source_trace_mapping_retained',source_trace_ref=c['source_trace_ref'],actual_source_component_sha256=object_sha256(trace),shot_id=trace['shot_id'],native_pts=trace['native_pts'],source_packet_sha256=trace['source_packet_sha256'],source_statistical_input_id=trace['statistical_input_id'],producer_ordered_raw_indices=raw_indices,actual_source_visibility_link_rejections=trace['explicit_visibility_link_rejections'],source_edges_from_actual_current_segments_only=True,Query_rows=queries,unknown_rows=fail,world_direction_is_not_measured=True,allowed_world_direction_candidates=[1,-1],world_or_model_coordinates_used=False,nearest_or_array_ordinal_assignment_used=False)
  if good:
   ordered=sorted(queries,key=lambda x:x['ordered_trace_coordinate']);require(len({q['ordered_trace_coordinate']for q in ordered})==len(ordered),'Two Queries have same source point; cannot manufacture strict order');rec['ordered_selected_rows']=[q['raw_index']for q in ordered];rec['source_adjacency_edges']=[[a['raw_index'],b['raw_index']]for a,b in zip(ordered,ordered[1:])];mapped.update({q['raw_index']:dict(component_id=cid,**q)for q in queries});counts['order_available_components']+=1;counts['source_order_edges']+=len(ordered)-1
  else:counts['unknown_components_retained']+=1
  records.append(rec)
 require(counts['source_Query_rows']==166 and len(records)==55,'Source relation full denominator changed');report=dict(schema=SCHEMA,source_feedback=ref(feedback),source_packet=ref(pp),issuer=ref(__file__),actual_source_reader_input=f['source_reader_input_binding'],actual_source_component_manifest=f['actual_source_component_manifest'],counts=dict(counts),components=records,actual_dependencies=[dict(file=k[0],sha256=k[1])for k in sorted(seen)],source_queries_UV_W_or_station_arrays_unchanged=True,source_or_world_numeric_validity_not_promoted=True,physical_world_directions_not_selected=True,normal_navigations_not_signed_as_order=True,source_uncertainty_or_noise_independence_not_issued=True,quality_accepted=False,solver_started=False)
 return report,mapped

def build(feedback_manifest,output):
 out=Path(output);require(not out.exists(),'Immutable source trace lineage output exists');report,mapping=_construct(feedback_manifest);out.mkdir(parents=True);write_json(out/'manifest.json',report);return ref(out/'manifest.json')

def assert_current(manifest,*,return_records=False):
 path=Path(manifest);m=read_json(path);require(m['schema']==SCHEMA and m['issuer']==ref(__file__),'Unknown source trace lineage issuer');actual,mapped=_construct(m['source_feedback']['file']);require(m==actual,'Source trace lineage certificate differs from real current source inputs');report=dict(schema=SCHEMA,passed=True,manifest=ref(path),source_packet=m['source_packet'],counts=m['counts'],world_or_model_coordinates_used=False,Query_UV_or_W_changed=False)
 return dict(report=report,components=m['components'],mapping_by_raw_row=mapped)if return_records else report
