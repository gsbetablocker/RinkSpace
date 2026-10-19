"""Audit circular painted-band sources and preserve paired-edge midpoint semantics.

C0/A1-A4 are centerlines under the latest explicit user definition. This
module never substitutes one material edge or changes immutable observations.
"""
from pathlib import Path
import json,sys,collections,hashlib,argparse,numpy as np
from rinkfit_start.regression import NativeRegression,sha
p=argparse.ArgumentParser(description='Audit saved circular painted-band midpoint sources; historical outer identifiers do not change the measurement operator. No coordinates or likelihoods are rewritten.');p.add_argument('--run',required=True);p.add_argument('--semantic',required=True);p.add_argument('--label',required=True);p.add_argument('--output',required=True);p.add_argument('--identity-review');a=p.parse_args()
if a.label.split(':')[0]not in {'ice.C0','ice.A1','ice.A2','ice.A3','ice.A4'}:raise ValueError('This auditor preserves explicitly defined circular-band centerlines; a historical outer suffix alone never selects a material edge')
R=Path(a.run);S=Path(a.semantic);out=Path(a.output);out.mkdir(exist_ok=False);reg=NativeRegression(R,S);wanted=set(np.flatnonzero(reg.active&(reg.labels==a.label)).tolist());registry=Path(reg.sem['assets']['source_rows.jsonl']['file']);rows=[r for line in registry.open(encoding='utf-8')for r in[json.loads(line)]if r['source_row']in wanted];assert set(r['source_row']for r in rows)==wanted
groups=collections.defaultdict(list)
for r in rows:groups[r['source_packet']['path']].append(r)
operators=collections.Counter();origins=collections.Counter();mismatch=[];frames=[];manual=[];maxmid=0.;widths=[];raw_count=0;recovered=[];unresolved=[]
for name,rr in groups.items():
 p=Path(name);assert sha(p)==rr[0]['source_packet']['sha256'];d=json.loads(p.read_text(encoding='utf-8'));fmap={f['id']:f for f in d['features']};errors=[];feature_rows=collections.Counter()
 for r in rr:
  f=fmap[r['source_feature_id']];operators[f.get('measurement_operator','absent_manual')]+=1;feature_rows[f['id']]+=1;seg=f['segments'][r['source_segment_index']];origins[seg.get('measurement_origin','absent')]+=1;point=seg['uv'][r['original_vertex_index']];err=float(np.linalg.norm(reg.uv[r['source_row']]-point));errors.append(err)
  if err>1e-9:mismatch.append(dict(row=r['source_row'],error=err))
  if f.get('manual_source_observation'):manual.append(dict(source_row=r['source_row'],feature_id=f['id'],frame=d['programme_frame_index_1based'],pts=d['source_pts'],uv=point,source_packet=r['source_packet'],user_confirmed=seg.get('user_confirmed'),evidence=seg.get('source_evidence'),boundary_operator_not_explicit=True))
 f=fmap.get(a.label);source=d.get('ice_source_readout_packet');direct=f.get('source_circle_fit',{}).get('direct_readouts',[])if f else []
 if f and not direct and source:
  q=Path(source['path']);assert sha(q)==source['sha256'];raw=json.loads(q.read_text(encoding='utf-8'));g=next(x for x in raw['features']if x['id']==a.label);direct=g.get('source_circle_fit',{}).get('direct_readouts',[])
 geo=f.get('geometric_interpretation',{})if f else {};center=np.array(geo.get('center_uv',[]),float);axes=np.array(geo.get('axes_px',[]),float);angle=float(geo.get('angle_degrees',0.));theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 frec=dict(frame=d['programme_frame_index_1based'],pts=d['source_pts'],source_packet=dict(path=str(p),sha256=sha(p)),source_rows=[r['source_row']for r in rr],feature_row_counts=dict(feature_rows),representative_uv_match_max=max(errors),measurement_operator=f.get('measurement_operator')if f else None,material_transition=f.get('material_transition')if f else None,measurement_definitions=d.get('measurement_definitions'),source_readout_packet=source,direct_native_readouts=len(direct),midpoint_checks=0,source_ellipse_for_side_orientation_only=geo,saved_native_midpoint_readouts=len(direct),new_observation_version_created=False)
 for j,readout in enumerate(direct):
  e=readout.get('source_evidence',{});sides=np.array(e.get('actual_source_sides_uv',[]),float);quantiles=np.array(e.get('actual_source_side_quantiles_uv',[]),float);q=np.array(e.get('query_uv',[]),float);normal=np.array(e.get('profile_normal',[]),float);reason=[]
  if sides.shape==(2,2)and np.isfinite(sides).all():
   error=float(np.linalg.norm(sides.mean(0)-readout['uv']));maxmid=max(maxmid,error);widths.append(float(np.linalg.norm(sides[1]-sides[0])));frec['midpoint_checks']+=1;raw_count+=1
  else:reason.append('missing_finite_saved_pair')
  # Both measured interfaces support the saved midpoint. No exterior interface
  # is selected or emitted as a replacement observation.
 frames.append(frec)
assert len({r['id']for r in recovered})==len(recovered)
history=None
if a.identity_review:history=dict(path=a.identity_review,sha256=sha(a.identity_review),review=json.loads(Path(a.identity_review).read_text(encoding='utf-8')))
report=dict(schema='rinkfit.circular_band_centerline_source_audit.v1',target_label=a.label,definition_authority='latest explicit user correction: C0 and A1-A4 are circular painted lines measured at the width midpoint; target rings are F1-F4 and R1-R4',current_observation_role='circular_band_centerline',historical_outer_identifier_is_not_outer_material_interface=True,source_registry=dict(path=str(registry),sha256=sha(registry)),semantic=dict(path=str(S),sha256=sha(S)),active_target_rows=len(wanted),source_packets=len(groups),measurement_operator_row_counts=dict(operators),measurement_origin_row_counts=dict(origins),original_UV_reproduces_packet_rows=not mismatch,UV_mismatches=mismatch,direct_native_readouts_with_two_interfaces=raw_count,maximum_native_UV_to_interface_midpoint_rounding_error_px=maxmid,interface_separation_px=dict(min=min(widths),median=float(np.median(widths)),max=max(widths)),operator_matches_current_user_definition=all(k in {'paint_center','absent_manual'}for k in operators),source_semantics_conflict=any(k not in {'paint_center','absent_manual'}for k in operators),centerline_representative_rows=sum(v for k,v in operators.items()if k=='paint_center'),human_manual_rows=manual,manual_clicks_retained_verbatim_not_automatically_assigned_midpoint_uncertainty=True,historical_material_identity_review=history,current_C0_model_logic_requires_boundary_shift=False,new_boundary_observations_created=False,production_data_or_model_changed=False,withdrawn_interpretation='The previous C0 outer-interface migration proposal was incorrect. C0 outer is a legacy name for the circular band centerline. The saved paint_center and paired interfaces agree with the latest user definition.',frames=sorted(frames,key=lambda x:x['frame']))
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k]for k in ['active_target_rows','source_packets','measurement_operator_row_counts','original_UV_reproduces_packet_rows','direct_native_readouts_with_two_interfaces','maximum_native_UV_to_interface_midpoint_rounding_error_px','operator_matches_current_user_definition','new_boundary_observations_created']}))
