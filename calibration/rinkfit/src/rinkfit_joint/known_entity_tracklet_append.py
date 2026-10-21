"""Source-supported versions of already identified physical board corners.

A raw corner mean is preserved whenever both actual RGB material transitions
support it. Native two-edge peaks are separate measurements in the audit, not
replacement truth. This module contains no world, camera, H, or fitting calls.
"""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import json
import av
import cv2
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .source_worker import _channels,_sample,_remeasure
from .source_texture_tracklets import _source_mask
from .entity_corners import remeasure_corner,CORNER_SOURCE_OPERATOR

BUNDLE_SCHEMA='rinkfit.native_known_entity_tracklet_qualified_bundle.v1'
GATE_SCHEMA='rinkfit.known_corner_source_support_qualification.v1'
RECEIPT_SCHEMA='rinkfit.source_supported_known_corner_qualification.v1'
ESTIMATOR='source_supported_original_UV_or_native_corner_peak_v1'


def _ref(path):
 p=Path(path).resolve();return dict(file=str(p),sha256=file_sha256(p))


def _verified(ref):
 p=Path(ref.get('file',ref.get('path'))).resolve()
 if file_sha256(p)!=ref['sha256']:raise ValueError('Changed actual known-corner dependency: '+str(p))
 return p


def _curve_residual(uv,edge):
 q=np.asarray(edge['curve_origin_uv']);t=np.asarray(edge['curve_tangent']);n=np.asarray(edge['curve_normal']);a=np.asarray(edge['curve_coefficients'])
 x=float((np.asarray(uv)-q)@t)
 return float((np.asarray(uv)-q)@n-(a[0]+a[1]*x+a[2]*x*x))


def _contains(uv,edges):
 for e in edges:
  lo,hi=e['normal_residual_support_px'];v=float((np.asarray(uv)-e['origin_uv'])@e['normal'])
  clo,chi=e['curve_residual_support_px'];c=_curve_residual(uv,e)
  if not(lo<=v<=hi and clo<=c<=chi):return False
 return len(edges)==2


def material_edge_support(channels,operator):
 """Read signed transition lobes in the original channels and finite profiles.

 The support is the contiguous same-polarity derivative lobe containing the
 original peak. It is a photometric support set, without noise or confidence
 interpretation. Six inward profiles constrain each current material edge.
 Curved-edge residuals and the corner tangent-plane bands are both retained.
 """
 op=operator['source_operator'];q=np.asarray(op['intersection_seed_uv']);peak=np.asarray(op['current_intersection_uv']);out=[]
 off=np.arange(-3.,3.0001,.25)
 for f in op['edge_fits']:
  t=np.asarray(f['tangent']);n=np.asarray(f['normal']);a=np.asarray(f['coefficients']);profiles=[];intervals=[]
  for audit,res in zip(f['profile_audits'],f['fit_residuals_px']):
   d=audit['inward_station_px'];seed=q+d*t;v=_sample(channels,seed[None,:]+off[:,None]*n)
   g=np.gradient(v,.25,axis=0);c=int(audit['profile']['channel']);pol=int(audit['profile']['polarity']);signed=pol*g[:,c];k=int(np.abs(g[:,c]).argmax())
   if signed[k]<=0:raise ValueError('Actual material transition polarity changed')
   left=right=k
   while left>0 and signed[left-1]>0:left-=1
   while right<len(off)-1 and signed[right+1]>0:right+=1
   # Boundary samples are finite observed support, never extrapolated tails.
   low=float(off[left]);high=float(off[right]);shift=float(audit['profile']['offset_px'])
   band=[float(res+low-shift),float(res+high-shift)];intervals.append(band)
   profiles.append(dict(inward_station_px=d,seed_uv=seed.tolist(),channel=c,polarity=pol,
    profile_offsets_px=off.tolist(),actual_channel_intensities=v[:,c].tolist(),signed_gradient=signed.tolist(),
    peak_sample_index=k,original_peak_offset_px=shift,same_polarity_lobe_offset_px=[low,high],
    fit_residual_px=float(res),curve_residual_support_px=band,original_source_profile=audit['profile']))
  lo=max(z[0] for z in intervals);hi=min(z[1] for z in intervals)
  x=float((peak-q)@t);jac=n-(a[1]+2*a[2]*x)*t;scale=float(np.linalg.norm(jac));normal=jac/scale
  out.append(dict(origin_uv=peak.tolist(),normal=normal.tolist(),normal_residual_support_px=[lo/scale,hi/scale],
   profile_normal_residual_support_px=[[l/scale,h/scale]for l,h in intervals],
   curve_origin_uv=q.tolist(),curve_tangent=t.tolist(),curve_normal=n.tolist(),curve_coefficients=a.tolist(),
   curve_residual_support_px=[lo,hi],profile_curve_residual_support_px=intervals,
   normal_band_is_corner_linearization=True,chosen_exact_curve_residual_also_required=True,
   profiles=profiles,support_is_photometric_transition_not_covariance=True))
 return out


def load_source_support_qualification(manifest_file,manifest,reference,verify):
 for name in ('input_cumulative_journal','qualification_receipts','current_model','source_mean_and_floating_estimator_replay_QA'):
  verify(manifest[name])
 if not manifest['old_raw_source_flags_false_are_not_overwritten']or not manifest['new_measurement_version_geometry_qualification_only']:
  raise ValueError('Known corner source support cannot change old authority')
 qa=read_json(verify(manifest['source_mean_and_floating_estimator_replay_QA']))
 if qa.get('passed')is not True:raise ValueError('Known corner actual source replay incomplete')
 checks={r['source_row_id']:r for r in read_jsonl(verify(qa['rows_ref']))}
 rows={r['source_row_id']:r for r in read_jsonl(verify(manifest['input_cumulative_journal']))}
 receipts={r['source_row_id']:r for r in read_jsonl(verify(manifest['qualification_receipts']))}
 if set(rows)!=set(receipts)or len(rows)!=manifest['qualified_new_measurement_rows']:raise ValueError('Known corner qualified source denominator differs')
 for rid,receipt in receipts.items():
  r=rows[rid];q=checks.get(rid)
  if (q is None or not q['valid']or q['chosen_uv']!=r['current_uv']or not _contains(r['current_uv'],q['material_edge_support'])
      or q['current_rgb_sha256']!=r['current_rgb_sha256']or not receipt['new_actual_source_geometry_qualified']
      or receipt['solver_eligible']or receipt['statistical_objective_eligible']):raise ValueError('Known corner receipt lacks actual two-edge RGB support')
 return receipts,[dict(file=str(manifest_file),sha256=reference['sha256']),manifest['qualification_receipts'],dict(manifest['input_cumulative_journal'],role='qualified_current_entity_source_journal')]


def validate_source_support_receipt(row,receipt,material,material_index):
 if not row.get('point_correspondence')or row.get('measurement_operator_resolved')!='physical_corner':raise ValueError('Known corner requires fixed physical-corner operator')
 expected=object_sha256(dict(source_row_id=row['source_row_id'],native_pts=row['native_pts'],current_2D_UV=row['current_uv'],estimator=ESTIMATOR))
 if (expected!=receipt['current_2D_source_mean_sha256']or receipt['native_pts']!=row['native_pts']or receipt['original_UV_preserved']!=row['source_uv']or receipt['existing_material_index']!=material_index):raise ValueError('Known corner source-support receipt redirects UV/entity')
 paths=[i for i,p in enumerate(material['paths'])if p['id']==receipt['existing_point_path']]
 if len(paths)!=1:raise ValueError('Known corner has no unique existing point path')
 p=material['paths'][paths[0]]
 if not p.get('physical_point')or p['kind']!=receipt['point_path_kind']or (p['kind']==2 and len(p.get('points',[]))!=1 and len(p.get('geometry_indices',[]))!=2):raise ValueError('Known corner is not an existing native fixed-point path')
 return paths[0]


def prepare_known_entity_tracklets(parent_packet,model_path,candidate_manifest,ownership_reviews,output):
 """Re-read exact source RGB and seal only the reviewed supported subset."""
 out=Path(output).resolve()
 if out.exists():raise ValueError('New immutable known-corner source generation required')
 parent=read_json(parent_packet);model=read_json(model_path);gen=read_json(candidate_manifest)
 scope=read_json(_verified(gen['scope']));source=scope['source'];roles_path=_verified(gen['roles']);roles=read_json(roles_path)['frame_roles']
 if source['time_base']!='1/12800'or source['native_fps']!='50/1'or file_sha256(source['path'])!=source['sha256']:raise ValueError('Actual official native programme clock required')
 inventory_path=_verified(gen['anchor_inventory']);inventory=read_json(inventory_path)
 if inventory['source_packet']['sha256']!=file_sha256(parent_packet):raise ValueError('Known corner alias ledger must start at exact anchor packet')
 anchors=list(read_jsonl(_verified(inventory['source_rows_ref'])));anchorbyid={a['source_row_id']:a for a in anchors}
 episodes={};reviewrefs=[]
 for file in ownership_reviews:
  review=read_json(file);reviewrefs.append(_ref(file))
  if review.get('source_body_material_ownership_supported')is not True or review['source_candidate_manifest']['sha256']!=file_sha256(candidate_manifest):raise ValueError('Exact source candidate ownership review required')
  for e in review['reviewed_episodes']:
   if e['track_id']in episodes:raise ValueError('Repeated physical continuity ownership review')
   episodes[e['track_id']]=e
 candidates=[]
 for c in read_jsonl(_verified(gen['source_rows_ref'])):
  if c['track_id']not in episodes or c['original_packet_measurement_priority']:continue
  e=episodes[c['track_id']]
  if not(e['native_pts_interval'][0]<=c['native_pts']<=e['native_pts_interval'][1])or c['existing_material_index']!=e['existing_material_index']or c['existing_point_path']!=e['existing_point_path']:raise ValueError('Candidate exceeds reviewed physical corner episode')
  candidates.append(c)
 expected=sum(e['new_exposure_candidates']for e in episodes.values())
 if len(candidates)!=expected:raise ValueError('Reviewed source candidate denominator changed')
 def array(key):
  d=parent['arrays'][key];return np.memmap(_verified(d),mode='r',dtype=d['dtype'],shape=tuple(d['shape']))
 arrays={k:array(k)for k in ('kind','material_index','path','native_pts','admission','fit_role','uv')}
 old=defaultdict(list)
 for i in np.flatnonzero(arrays['kind']==1):old[(int(arrays['material_index'][i]),int(arrays['path'][i]),int(arrays['native_pts'][i]))].append(int(i))
 ledgers={}
 for ref in gen['independent_source_bound_actor_ledgers']:
  for l in read_jsonl(_verified(ref)):ledgers[int(l['native_pts'])]=l
 bypts=defaultdict(list)
 for c in candidates:bypts[int(c['native_pts'])].append(c)
 out.mkdir(parents=True);rows=[];receipts=[];checks=[];pending=[];counts=Counter();max_replay=0.;decoded=0;seen=set();mapping={}
 cv2.setNumThreads(2)
 with av.open(source['path'])as con:
  stream=con.streams.video[0]
  if str(stream.time_base)!='1/12800':raise ValueError('Wrong decoder time base')
  con.seek(min(bypts),stream=stream,backward=True)
  for frame in con.decode(stream):
   pts=int(frame.pts)
   if pts>max(bypts):break
   if pts not in bypts:continue
   rgb=frame.to_ndarray(format='rgb24');ledger=ledgers[pts];mask,digest=_source_mask(rgb,ledger,source,pts,ledger['shot_id']);channels=_channels(rgb);decoded+=1
   for c in bypts[pts]:
    if c['current_rgb_sha256']!=digest or roles[str(pts)]['role']!='development_candidate':raise ValueError('Actual current RGB/role differs from source candidate')
    op=c['current_source_operator'];actual=remeasure_corner(channels,op['source_operator']['intersection_seed_uv'],op['edge_tangents'],mask)
    err=float(np.max(np.abs(np.asarray(actual.get('current_intersection_uv',np.inf))-c['current_uv'])))
    if not actual['valid']or err>1e-8:raise ValueError('Actual native two-edge replay differs from recorded source peak')
    max_replay=max(max_replay,err);mi=c['existing_material_index'];mat=model['materials'][mi];path=[i for i,p in enumerate(mat['paths'])if p['id']==c['existing_point_path']]
    if len(path)!=1 or not mat['paths'][path[0]].get('physical_point'):raise ValueError('Reviewed source corner does not have existing fixed material path')
    prior=old.get((mi,path[0],pts),[])
    active_prior=[i for i in prior if arrays['admission'][i]==1 and arrays['fit_role'][i]==1]
    conflicting_old_means=len({tuple(arrays['uv'][i])for i in prior})>1
    preferred=active_prior[0]if active_prior else (prior[0]if prior else None)
    chosen=np.asarray(arrays['uv'][preferred]).tolist()if prior else c['current_uv'];edges=material_edge_support(channels,op)
    supported=_contains(chosen,edges);diag=_remeasure(chosen,[0,0],channels,'texture',corner=True)
    valid=bool(supported and diag['status']=='entity_corner_two_direction_support'and not active_prior and not conflicting_old_means)
    rid='native_corner_version_v1/'+c['shot_id']+'/'+str(mi)+'/'+str(pts)
    if rid in seen:raise ValueError('Reviewed episodes duplicate one existing physical corner exposure')
    seen.add(rid)
    anchor=anchorbyid[c['qualified_anchor_source_row_id']]
    qa=dict(source_row_id=rid,track_id=c['track_id'],native_pts=pts,shot_id=c['shot_id'],valid=valid,
     chosen_uv=chosen,new_peak_uv=c['current_uv'],source_uv=chosen,current_rgb_sha256=digest,native_peak_replay_max_abs_px=err,
     material_edge_support=edges,chosen_within_both_material_edge_support=supported,
     source_support_interval_uv=None,chosen_two_direction_diagnostic=diag,prior_inactive_row_indices=prior,
     original_UV_retained=bool(prior),source_position_correction_confirmed=False,
     actual_source_replay_passed=True,source_mean_pixels_from_model=False,model_or_H_used=False)
    # The displayed UV interval is merely the parallelogram envelope. Both
    # normal bands AND exact quadratic residuals remain the acceptance test.
    A=np.asarray([e['normal']for e in edges]);center=np.asarray(edges[0]['origin_uv']);corners=[]
    for u in edges[0]['normal_residual_support_px']:
     for v in edges[1]['normal_residual_support_px']:corners.append(center+np.linalg.solve(A,[u,v]))
    qa['source_support_interval_uv']=[np.min(corners,axis=0).tolist(),np.max(corners,axis=0).tolist()]
    checks.append(qa)
    if not valid:
     reason=('existing_active_original_measurement_priority'if active_prior else 'conflicting_old_inactive_original_UV_versions'if conflicting_old_means else 'original_UV_outside_both_actual_edge_support'if not supported else 'chosen_original_UV_has_no_local_two_direction_support')
     pending.append(dict(source_row_id=rid,track_id=c['track_id'],native_pts=pts,reason=reason,prior_inactive_row_indices=prior,old_UV=chosen,new_peak_uv=c['current_uv'],old_raw_unchanged=True))
     counts['existing_active_original_priority'if active_prior else 'pending']+=1;continue
    counts['recovered_inactive_existing_exposures'if prior else 'genuinely_new_exposures']+=1
    r=dict(c,source_row_id=rid,source_family_id=c['shot_id']+'/'+str(mi)+'/'+str(pts),
     joint_material_index=mi,joint_material_id=mat['label'],source_uv=chosen,current_uv=chosen,normal=[0.,0.],
     source_geometry_eligible=True,measurement_operator_resolved='physical_corner',frame_role='development_candidate',
     anchor_raw_index=int(anchor['raw_index']),anchor_packet_ref=_ref(parent_packet),
     prior_inactive_row_indices=prior,explicit_new_measurement_version=bool(prior),original_UV_preserved=chosen,
     source_support_qualified=True,source_position_correction_confirmed=False,
     physical_identity_status='same_actual_physical_'+c['target_id'].split('.')[0]+'_source_review',
     current_corner_operator=dict(original_corner_seed_native_uv=op['source_operator']['intersection_seed_uv'],edge_tangents=op['edge_tangents']),
     same_source_alias_lineage=[dict(raw_index=i,source_row_id=parent['source_row_ids'][i],original_uv=np.asarray(arrays['uv'][i]).tolist(),admission=int(arrays['admission'][i]),fit_role=int(arrays['fit_role'][i]))for i in prior])
    receipt=dict(schema=RECEIPT_SCHEMA,source_row_id=rid,track_id=c['track_id'],target_id=c['target_id'],native_pts=pts,shot_id=c['shot_id'],
     existing_material_index=mi,existing_point_path=c['existing_point_path'],point_path_kind=mat['paths'][path[0]]['kind'],
     physical_point=True,entity_kind=1,entity_station_fixed=True,new_actual_source_geometry_qualified=True,
     original_UV_preserved=chosen,current_2D_source_mean_sha256=object_sha256(dict(source_row_id=rid,native_pts=pts,current_2D_UV=chosen,estimator=ESTIMATOR)),
     current_native_RGB_sha256=digest,current_rgb_sha256=digest,source_uv=chosen,current_uv=chosen,
     solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,
     original_flags_preserved=dict(manual_reference_preserved=False),prior_inactive_row_indices=prior,
     source_mean_estimator=ESTIMATOR,native_peak_uv=c['current_uv'],original_UV_preserved_when_source_supported=True)
    validate_source_support_receipt(r,receipt,mat,mi);rows.append(r);receipts.append(receipt);mapping[mat['label']]=dict(material_index=mi,geometry_ready=True)
   print(json.dumps(dict(native_pts=pts,source_frames=decoded,source_candidates_complete=len(checks),qualified=len(rows),pending=len(pending))),flush=True)
 if len(checks)!=len(candidates):raise ValueError('Native decoder did not cover every reviewed source row')
 write_jsonl(out/'source_rows.jsonl',rows);write_jsonl(out/'qualification_receipts.jsonl',receipts);write_jsonl(out/'source_replay_rows.jsonl',checks);write_json(out/'pending.json',pending)
 ownership=dict(schema='rinkfit.combined_native_corner_ownership_review.v1',source_body_material_ownership_supported=True,reviewed_episodes=list(episodes.values()),original_reviews=reviewrefs)
 write_json(out/'ownership_review.json',ownership);write_json(out/'material_map.json',dict(semantic_to_material=mapping))
 qa=dict(schema='rinkfit.actual_native_corner_source_support_replay.v1',passed=True,source_candidates=len(candidates),qualified_rows=len(rows),pending_rows=len(pending),counts=dict(counts),
  rows_ref=_ref(out/'source_replay_rows.jsonl'),pending_ref=_ref(out/'pending.json'),native_source_frames=decoded,
  peak_replay_max_abs_px=max_replay,all_source_candidates_accounted=True,all_original_raw_UV_and_flags_unchanged=True,
  no_covariance_or_displacement_factor_invented=True,actual_profile_support_definition='intersection_of_same_polarity_gradient_lobes_of_original_six_finite_material_profiles',
  old_UV_policy='preserve old source UV when supported by both actual material edges; ambiguous rows remain pending; no automatic source position corrections',
  source_generation=_ref(candidate_manifest),reader_module=_ref(__file__))
 write_json(out/'source_mean_replay_QA.json',qa)
 gate=dict(schema=GATE_SCHEMA,input_cumulative_journal=_ref(out/'source_rows.jsonl'),qualification_receipts=_ref(out/'qualification_receipts.jsonl'),current_model=_ref(model_path),
  source_mean_and_floating_estimator_replay_QA=_ref(out/'source_mean_replay_QA.json'),qualified_new_measurement_rows=len(rows),
  old_raw_source_flags_false_are_not_overwritten=True,new_measurement_version_geometry_qualification_only=True,source_mean_estimator=ESTIMATOR)
 write_json(out/'qualification_gate.json',gate)
 chunk=dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',all_requested_frames_accounted=True,source_dataset_tag='known_board_corner_native_versions_v1',source_rows_ref=_ref(out/'source_rows.jsonl'),geometry_rows_ref=_ref(out/'source_rows.jsonl'),source_gate_manifest=_ref(out/'qualification_gate.json'),actual_source_rows=len(rows),by_shot=dict(Counter(r['shot_id']for r in rows)))
 write_json(out/'chunk_manifest.json',chunk)
 bundle=dict(schema=BUNDLE_SCHEMA,model=_ref(model_path),model_used_only_for_existing_material_topology=True,anchor_packet=_ref(parent_packet),source_rows_ref=_ref(out/'source_rows.jsonl'),qualification_receipts=_ref(out/'qualification_receipts.jsonl'),
  qualification_gate=_ref(out/'qualification_gate.json'),source_mean_replay_QA=_ref(out/'source_mean_replay_QA.json'),ownership_review=_ref(out/'ownership_review.json'),anchor_inventory=_ref(inventory_path),source_generation=_ref(candidate_manifest),
  reader_module_refs=[_ref(__file__),_ref(Path(__file__).with_name('entity_corners.py')),_ref(Path(__file__).with_name('native_entity_tracklets.py')),_ref(Path(__file__).with_name('source_worker.py'))],
  source_ledger_preserved=True,source_rows=len(rows),candidate_denominator=len(candidates),counts=dict(counts),by_shot=chunk['by_shot'],
  closed_chunks=[_ref(out/'chunk_manifest.json')],typed_entity_qualification=[_ref(out/'qualification_gate.json')],material_map=_ref(out/'material_map.json'),roles=_ref(roles_path),
  native_station_zero_fixed_existing_materials=True,new_geometry_parameters_added=False,independent_motion_factors_added=False,unknown_noise_has_statistical_precision=False)
 write_json(out/'manifest.json',bundle)
 load_source_support_qualification(out/'qualification_gate.json',gate,_ref(out/'qualification_gate.json'),_verified)
 return bundle


def append_prepared_known_entity_tracklets(parent_packet,bundle_path,model_path,output,finalize):
 from .reference_source_append import append_constraint_source_chunks
 b=read_json(bundle_path)
 if b['schema']!=BUNDLE_SCHEMA:raise ValueError('Wrong known-corner source bundle')
 # Numeric model values are supplied by the actual current parent. Source
 # qualification in the bundle depends only on existing material identity.
 return append_constraint_source_chunks(parent_packet,b['closed_chunks'],model_path,_verified(b['material_map']),_verified(b['roles']),output,
  typed_entity_qualification=b['typed_entity_qualification'],finalize_packet_with_parent=lambda result,out,parent:finalize(parent_packet,result,bundle_path,out/'qualified_certificate.json',parent_document=parent))
