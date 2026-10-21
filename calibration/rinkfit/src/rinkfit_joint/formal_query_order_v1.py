"""Actual source-component order audit for current Query roots.

Uses only producer-declared current connected support, never source-family or
representative array adjacency. Both world orientations remain alternatives.
This is a read-only trial validity consumer; no source admission is changed.
"""
from pathlib import Path
import json,gzip,re,hashlib
from collections import defaultdict,Counter
import numpy as np
from .io_utils import file_sha256
from .joint_model import as_event_view
from rinkfit_start.native_scene import NativeScene
from rinkfit_start.semantic_order import _direction_candidate

def audit(packet_path,model_path,root_audit,component_manifest,output):
 pp,mp,ra,cm,out=map(Path,(packet_path,model_path,root_audit,component_manifest,output))
 if out.exists():raise FileExistsError('immutable order audit exists')
 packet=json.loads(pp.read_text(encoding='utf8'));model=json.loads(mp.read_text(encoding='utf8'));manifest=json.loads(cm.read_text(encoding='utf8'));report=json.loads((ra/'report.json').read_text(encoding='utf8'))
 if report['packet']['sha256']!=file_sha256(pp)or report['model']['sha256']!=file_sha256(mp):raise ValueError('order root/model/source mismatch')
 ref=manifest['support_components'];cf=Path(ref['path']);
 if file_sha256(cf)!=ref['sha256']:raise ValueError('current support component changed')
 roots=np.fromfile(ra/'root_station_row_alpha.f64','<f8').reshape(packet['row_count'],3);valid=np.fromfile(ra/'root_valid.u8','u1').astype(bool)
 wanted=defaultdict(list)
 for q in packet['queries']:
  cid=q.get('source_component_id')
  if cid:wanted[cid].append(q)
 scene=NativeScene(as_event_view(model,'s01'));lengths=np.asarray([scene.long,scene.arc,scene.short,scene.arc]*2);prefix=np.r_[0,np.cumsum(lengths)];records=[];seen=set();failed=0;unresolved=0;edges=0;eligible=[]
 with gzip.open(cf,'rt',encoding='utf8')as h:
  for line in h:
   c=json.loads(line);cid=c['component_id']
   if cid not in wanted:continue
   if cid in seen:raise ValueError('duplicate support component id')
   seen.add(cid)
   if not c['edges_come_only_from_ordered_current_support_segments']:raise ValueError('component lacks actual ordered current support')
   selected=[];mapping=[];mapping_failed=[]
   trace=np.asarray(c['current_uv'],float)
   def cross(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]
   for q in wanted[cid]:
    # Source-only intersection: representative ordinal is not raw trace index.
    # No world/template point or model interpolation establishes this order.
    line=np.asarray(q['native_uv'],float);hits=[]
    for j in range(len(line)-1):
     d=trace[1:]-trace[:-1];v=line[j+1]-line[j];rhs=line[j]-trace[:-1];den=cross(d,v);good=abs(den)>1e-12
     u=np.divide(cross(rhs,v),den,out=np.full(len(den),np.nan),where=good);vpar=np.divide(cross(rhs,d),den,out=np.full(len(den),np.nan),where=good)
     ix=np.flatnonzero(good&(u>=0)&(u<=1)&(vpar>=0)&(vpar<=1))
     hits.extend((float(k+u[k]),float(j+vpar[k]))for k in ix)
    unique=[]
    for hit in sorted(hits):
     if not unique or abs(hit[0]-unique[-1][0])>1e-8:unique.append(hit)
    if len(unique)!=1:mapping_failed.append(dict(row=int(q['row_index']),intersection_count=len(unique)));continue
    selected.append((unique[0][0],int(q['row_index'])));mapping.append(dict(row=int(q['row_index']),ordered_trace_coordinate=unique[0][0],source_Query_polyline_coordinate=unique[0][1],source_only_finite_Query_trace_intersection=True))
   if mapping_failed:
    rows=[int(q['row_index'])for q in wanted[cid]];records.append(dict(component_id=cid,rows=rows,status='source_Query_trace_intersection_not_unique_or_absent',mapping_failures=mapping_failed,no_world_or_model_association_used=True));unresolved+=1;continue
   selected.sort();rows=np.array([i for _,i in selected],int)
   rec=dict(component_id=cid,rows=rows.tolist(),source_packet_sha256=c['source_packet_sha256'],source_statistical_input_id=c['statistical_input_id'],source_family_id=c['source_family_id'],shot_id=c['shot_id'],native_pts=c['native_pts'],producer_declared_ordered_raw_indices=c['ordered_raw_indices'],source_visibility_rejected_links=c['explicit_visibility_link_rejections'],no_new_edge_across_components=True,world_direction_is_not_measured=True,allowed_direction_candidates=[1,-1],source_raw_components_not_world_template_order=True,source_Query_trace_mapping=mapping)
   if len(rows)<2:rec.update(status='single_selected_query_no_cross_row_order');records.append(rec);continue
   edges+=len(rows)-1
   if not valid[rows].all():rec.update(status='root_unresolved_whole_component_retained',failed_root_rows=rows[~valid[rows]].tolist());unresolved+=1;records.append(rec);continue
   ma=packet['arrays']['material_index'];mi=np.memmap(ma['file'],dtype=ma['dtype'],mode='r',shape=tuple(ma['shape']))[rows].astype(int)
   saved=Path(report['model']['file']).parent/'explicit_nuisance/path.u32'
   # The immutable source CP RLE/current path stays untouched by root audit.
   path=np.fromfile(saved,'<u4')[rows].astype(int)
   paths=[model['materials'][m]['paths'][p]for m,p in zip(mi,path)];s=roots[rows,0];kinds={p['kind']for p in paths};closed=False;reason=None
   if kinds=={3} and len(set(mi))==1:
    segments=[int(p.get('boundary_segment',-1))for p in paths]
    if all(k>=0 for k in segments):coordinate=np.array([(prefix[k]+lengths[k]*x)/lengths.sum()for k,x in zip(segments,s)]);closed=True;rec['coordinate']='one continuous full board perimeter arclength'
    elif all(k==-1 for k in segments):coordinate=s;closed=True;rec['coordinate']='one continuous full board perimeter normalized station'
    else:reason='mixed_full_and_segment_board_charts'
   elif len(set(zip(mi,path)))==1:
    coordinate=s;closed=bool(paths[0].get('closed',False));rec['coordinate']='same finite physical material path'
   else:reason='different_material_or_disconnected_finite_paths_not_joined'
   if reason:rec.update(status='unsupported_exact_world_connection',reason=reason);unresolved+=1;records.append(rec);continue
   candidates={str(d):_direction_candidate(coordinate,d,closed,1e-10)for d in (1,-1)};legal=[d for d in (1,-1)if candidates[str(d)]['valid']];rec.update(status='source_order_valid'if legal else'no_legal_monotone_direction',world_path_closed=closed,station=s.tolist(),coordinate_values=coordinate.tolist(),direction_candidates=candidates,legal_directions=legal)
   failed+=not bool(legal);eligible.append(rec);records.append(rec)
 if seen!=set(wanted):raise ValueError('Query component absent from actual producer graph')
 out.mkdir();(out/'components.json').write_text(json.dumps(records,indent=2),encoding='utf8');result=dict(schema='rinkfit.actual_Query_current_source_component_order_audit.v1',packet=dict(file=str(pp),sha256=file_sha256(pp)),model=dict(file=str(mp),sha256=file_sha256(mp)),root_report=dict(file=str(ra/'report.json'),sha256=file_sha256(ra/'report.json')),source_component_manifest=dict(file=str(cm),sha256=file_sha256(cm)),support_components=dict(file=str(cf),sha256=file_sha256(cf)),queries_with_source_component=sum(map(len,wanted.values())),component_count=len(records),components_with_2_or_more_selected=sum(len(r['rows'])>=2 for r in records),source_adjacency_inequality_edges=edges,order_evaluable_components=len(eligible),failed_order_components=failed,unresolved_components_retained=unresolved,all_relevant_current_components_accounted=True,source_raw_or_Query_or_W_or_domain_changed=False,production_Ceres_order_factors_installed=False,trial_accepted=False,quality_accepted=False,status_counts=dict(Counter(r['status']for r in records)))
 (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');return result
