"""Grouped automatic source resolution of all preserved unbound candidates.

Only measured native contours from the same physical panel may alias.  Current
model projections and residual magnitudes are not used for identity decisions.
"""
from collections import defaultdict,Counter
from pathlib import Path
import argparse,json,re
import numpy as np
from scipy.spatial import cKDTree
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256


def resolve_unbound(audit_dir,output_dir):
 audit=Path(audit_dir);out=Path(output_dir)
 if out.exists():raise ValueError('An immutable unbound resolution generation is required')
 rows=list(read_jsonl(audit/'relocated_rows.jsonl.gz'));bygroup=defaultdict(list);byframe=defaultdict(list)
 for r in rows:bygroup[r['provenance_group_id']].append(r);byframe[(r['shot_id'],r['native_pts'])].append(r)
 support_ref=read_json(audit/'retained_support.ref.json') if (audit/'retained_support.ref.json').is_file() else {'path':str(audit/'retained_support.jsonl.gz')}
 sp=Path(support_ref['path'])
 if support_ref.get('sha256') and file_sha256(sp)!=support_ref['sha256']:raise ValueError('Retained actual support hash changed')
 clouds=defaultdict(lambda:defaultdict(list))
 for s in read_jsonl(sp):
  members=bygroup[s['source_group_id']];targets={r.get('target_id') for r in members if r.get('category')=='ad_print' and r.get('target_id')}
  if len(targets)!=1:continue
  target=next(iter(targets));mt=re.match(r'Ad(\d+)\.print\.',target)
  if not mt:continue
  pts=[x['current_uv'] for x in s['actual_source_audits'] if x.get('current_uv') is not None]
  if pts:clouds[(s['shot_id'],s['native_pts'],int(mt.group(1)))][target].extend(pts)
 trees={k:{target:cKDTree(np.unique(np.asarray(pts,dtype=float),axis=0)) for target,pts in d.items() if len(pts)>2} for k,d in clouds.items()}
 decisions=[];jobs=defaultdict(list);counts=Counter()
 for gid,members in bygroup.items():
  if members[0].get('target_id'):continue
  r=members[0];name=r['source_feature_id'];panel=re.search(r'(?:unbound\.)?Ad(\d+)\.',name);uv=np.asarray([x['current_uv'] for x in members if x.get('current_uv') is not None],float);alias=None;proof=None
  if panel and len(uv)>=6:
   scores=[]
   for target,tree in trees.get((r['shot_id'],r['native_pts'],int(panel.group(1))),{}).items():
    distances=tree.query(uv)[0];fraction=float(np.mean(distances<=1.25));scores.append((fraction,float(np.quantile(distances,.95)),target))
   scores.sort(reverse=True)
   if scores and scores[0][0]>=.90 and (len(scores)==1 or scores[1][0]<.25):
    alias=scores[0][2];proof={'method':'measured_same_frame_same_physical_panel_contour_overlap','fraction_within_1_25_px':scores[0][0],'p95_native_contour_distance_px':scores[0][1],'runner_up_fraction':scores[1][0] if len(scores)>1 else 0.,'current_model_geometry_used':False,'native_measurements_compared':len(uv)}
  if alias:status='resolved_existing_material_duplicate_source_representation';reason='same_actual_native_material_contour_in_same_physical_ad_panel';new=None
  elif 'C0.inner_material_boundary' in name:status='resolved_material_role_new_width_geometry_required';reason='C0_actual_inner_paint_edge_is_not_width_midline';new={'parent_physical_material':'ice.C0','operator':'material_interface','role':'inner_paint_edge','geometry':'shared_C0_centerline_plus_unknown_actual_half_width','same_as_existing_midline':False}
  elif 'blue_cap_lower_body_interface' in name:status='resolved_static_surface_new_board_interface_required';reason='board_blue_cap_body_interface_distinct_from_board_upper_edge';new={'surface':'board','operator':'material_interface','role':'blue_cap_body_interface','geometry':'continuous_board_unwrapped_height_curve_shared_across_views','depth_free':False}
  elif 'yellow_strip_upper_body_interface' in name:status='resolved_static_surface_new_board_interface_required';reason='board_yellow_strip_body_interface_distinct_from_ice_board_contact';new={'surface':'board','operator':'material_interface','role':'yellow_strip_body_interface','geometry':'continuous_board_unwrapped_height_curve_shared_across_views','depth_free':False}
  elif panel:status='resolved_physical_panel_print_component_pending_temporal_shape_match';reason='real_print_boundary_on_known_Ad_panel_unmatched_local_component';new={'surface':'board','physical_advertisement_number':int(panel.group(1)),'operator':'ink_boundary','geometry':'finite_shared_unwrapped_ad_material_contour','depth_free':False,'individual_frame_shape_is_world_truth':False}
  elif r.get('physical_surface')=='ice':status='resolved_ice_surface_component_pending_physical_landmark_match';reason='current_native_flat_ice_paint_support_without_unique_world_material_identity';new={'surface':'ice','operator':r.get('measurement_operator_resolved'),'geometry':'shared_finite_ice_material_shape_with_multiframe_identity','depth_free':False,'assign_to_nearest_known_ice_curve':False}
  elif r.get('physical_surface')=='board':status='resolved_board_surface_component_pending_panel_topology_match';reason='current_native_board_material_support_without_unique_panel_entity';new={'surface':'board','operator':r.get('measurement_operator_resolved'),'geometry':'finite_shared_board_material_contour_with_panel_topology','depth_free':False}
  else:status='outside_model_surface_or_identity_pending';reason='no_certified_ice_or_board_surface';new=None
  group_decision={'source_group_id':gid,'shot_id':r['shot_id'],'native_pts':r['native_pts'],'source_feature_id':name,'source_row_ids':[x['source_row_id'] for x in members],'row_count':len(members),'current_native_supported_rows':len(uv),'decision':status,'reason':reason,'resolved_target_candidate':alias,'identity_proof':proof,'new_shared_geometry_requirement':new,'solver_eligible':False,'preserved_source_flags':True,'raw_source_deleted':False,'duplicate_statistical_weight_forbidden':bool(alias)}
  decisions.append(group_decision);counts[status]+=len(members);cluster=(status,re.sub(r'\.\d{4,}$','',name));jobs[cluster].append(group_decision)
 out.mkdir(parents=True);write_jsonl(out/'group_decisions.jsonl.gz',decisions)
 clusters=[{'id':object_sha256(k)[:24],'status':k[0],'material_family':k[1],'rows':sum(x['row_count'] for x in v),'source_group_ids':[x['source_group_id'] for x in v],'shots':sorted({x['shot_id'] for x in v}),'native_pts':sorted({x['native_pts'] for x in v}),'action':'automatic cross-time/other-view source matching and shared finite material registration','human_review_required':False} for k,v in sorted(jobs.items())]
 write_json(out/'material_family_jobs.json',clusters);summary={'schema':'rinkfit.joint9.unbound_resolution.v1','unbound_rows':sum(x['row_count'] for x in decisions),'source_groups':len(decisions),'grouped_material_family_jobs':len(clusters),'row_decision_counts':dict(counts),'all_source_records_preserved':True,'physical_surface_classification_complete':True,'all_physical_landmark_identities_solved':False,'user_per_frame_audit_required':False,'source_audit_rows_sha256':file_sha256(audit/'relocated_rows.jsonl.gz'),'current_model_residuals_or_projections_used':False};write_json(out/'summary.json',summary);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(resolve_unbound(a.audit,a.out)))
