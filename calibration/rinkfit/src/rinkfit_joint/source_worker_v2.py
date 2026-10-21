"""Versioned native-source audit with exact reader footprint exclusions.

Historical inputs and v1 remain unchanged. Person masks are read from pinned
native source manifests; static-ice rectangles are optional, source-bound
foreground evidence, never observations or model-residual filters.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256, object_sha256
from .materials import measurement_definition, semantic_key
from .source_worker import (_points, _channels, _remeasure, _raw_support,
    _visible_indices, _normal, _descriptor, _family, POLICY as V1_POLICY)
from calibration.program_tracking.static_ice_source_exclusions import source_records, profile_roi_hits

POLICY = {**V1_POLICY, 'schema':'rinkfit.native_source_policy.v2',
    'reader_support':'actual Gaussian radius-3 +/-3 remeasurement stencil; separately recorded original full transition interval x tangent[-3,3], cv2 1/32 bilinear pixels; two flanks and recorded contexts',
    'person_mask_policy':'use existing pinned bitmap; no detector, dilation or mask rewrite',
    'static_ice_exclusions':'explicit pinned source authority; full profile interval and source visibility links',
    'manual_click_policy':'preserve exact UV and original point-level visibility authority; never invent an automatic profile for a click',
    'v1_source_worker_sha256':'93efe3a5cbca580d63a75837a2405b7f9bccc980edb7057185dbd473dee1fd9d',
    'self_review_is_distinct_from_user_acceptance':True, 'implementation_sha256':file_sha256(__file__), 'source_exclusions_helper_sha256':file_sha256(source_records.__code__.co_filename)}

def _raw_support(group,feature,target):
    raw=_points(group.get('source_native_readout_uv'));ev=[];kind='segment_native_readout'
    if len(raw):
        evidence=group.get('point_source_evidence')
        if isinstance(evidence,list) and len(evidence)==len(raw):ev=evidence
        return raw,ev,kind
    fit=feature.get('source_circle_fit') or feature.get('source_line_fit') or {}
    reads=fit.get('direct_readouts') or []
    role=target.rsplit(':',1)[-1] if target and ':' in target else None
    if role and any(x.get('role')==role for x in reads):reads=[x for x in reads if x.get('role')==role]
    reads=[x for x in reads if len(_points([x.get('uv')]))]
    if reads:
        raw=_points([x['uv'] for x in reads]);ev=[x.get('source_evidence') or {} for x in reads]
        kind='line_direct_material_readout' if feature.get('source_line_fit') else 'circle_direct_material_readout_no_family_ratio'
    return raw,ev,kind

def _contexts(uv, evidence):
    values=[np.asarray(uv,dtype=float).reshape(1,2)]
    for key in ('query_uv','actual_source_sides_uv','actual_source_side_quantiles_uv'):
        if key in evidence:
            a=np.asarray(evidence[key],dtype=float).reshape(-1,2)
            if not np.isfinite(a).all(): raise ValueError('Nonfinite source profile context')
            values.append(a)
    return np.vstack(values)

def _between(points):
    points=np.asarray(points,float).reshape(-1,2)
    if len(points)<2:return points
    # All original profile contexts lie on the profile normal. Pick the farthest
    # pair without assuming quantiles are a sigma or a new acceptance bound.
    distances=np.sum((points[:,None,:]-points[None,:,:])**2,axis=2)
    i,j=np.unravel_index(int(np.argmax(distances)),distances.shape)
    n=max(1,int(np.ceil(np.sqrt(distances[i,j])/.25)))
    return points[i]+np.linspace(0.,1.,n+1)[:,None]*(points[j]-points[i])

def _sample_sites(uv,normal,operator,sides,evidence,corner=False):
    uv=np.asarray(uv,float);n=np.asarray(normal,float)
    if corner:
        offsets=np.asarray([(x,y) for y in range(-3,4) for x in range(-3,4)],float)
        return np.vstack([uv+offsets+d for d in ([.5,0],[-.5,0],[0,.5],[0,-.5])])
    if n.shape!=(2,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-8:
        return np.empty((0,2))
    n=n/np.linalg.norm(n)
    seeds=_points(sides) if operator=='paint_center' else uv[None,:]
    sites=[q+np.arange(-3.,3.0001,.25)[:,None]*n for q in seeds]
    # Recorded original support uses its own reader stencil below; do not
    # pretend quantiles are the range of the original profile.
    return np.vstack(sites)

def support_pixel_audit(sites,mask,local_records=(),*,gaussian_radius=3):
    """Exact finite stencil of v1 Gaussian(.8, truncate4)+order-1 sampling.

    This evaluates the reader's existing stencil, not an expanded mask/ROI.
    Gaussian reflect boundary conditions and bilinear zero weights are explicit.
    """
    sites=np.asarray(sites,float).reshape(-1,2);h,w=mask.shape
    if not len(sites):return {'sites':0,'status':'missing_reader_support','person_mask_pixels':0,'static_exclusion_ids':[]}
    if not np.isfinite(sites).all():raise ValueError('Nonfinite reader sites')
    outside=bool(((sites<0)|(sites>np.array([w-1,h-1]))).any())
    q=np.clip(sites,[0,0],[w-1,h-1]);floor=np.floor(q).astype(int)
    # ceil==floor on integer coordinates, avoiding falsely including zero-weight taps.
    ceil=np.ceil(q).astype(int)
    centers=np.vstack([np.column_stack((xs,ys)) for xs in (floor[:,0],ceil[:,0]) for ys in (floor[:,1],ceil[:,1])])
    centerids=np.unique(centers[:,1]*w+centers[:,0]);centers=np.column_stack((centerids%w,centerids//w))
    off=np.asarray([(x,y) for y in range(-gaussian_radius,gaussian_radius+1) for x in range(-gaussian_radius,gaussian_radius+1)])
    pixels=(centers[:,None,:]+off[None,:,:]).reshape(-1,2)
    for axis,size in ((0,w),(1,h)):
        z=pixels[:,axis]%(2*size);pixels[:,axis]=np.where(z<size,z,2*size-1-z)
    ids=np.unique(pixels[:,1]*w+pixels[:,0]);pixels=np.column_stack((ids%w,ids//w))
    hit=pixels[mask[pixels[:,1],pixels[:,0]]!=0]
    local=[]
    for r in local_records:
        box=np.asarray(r['source_roi_xyxy']);inside=((pixels>=box[:2])&(pixels<box[2:])).all(axis=1)
        if inside.any():local.append(r.get('id',r.get('exclusion_id',r['source_roi_xyxy'])))
    return {'sites':len(sites),'native_pixels':len(pixels),'outside_native_image':outside,
        'person_mask_pixels':len(hit),'first_person_mask_pixel':hit[0].tolist() if len(hit) else None,
        'static_exclusion_ids':local,'status':'masked_reader_support' if len(hit) or local else 'reader_support_clear'}

def original_source_profile_audit(uv,evidence,mask,local_records=()):
    """Recorded source readout footprint, distinct from the v1 +/-3 worker.

    sample_chroma uses normal offsets at 1 px and tangent integer -3..3,
    float32 1/32 remap quantization, with raw chroma/luminance (no 2-D
    Gaussian). Full recorded interval is source support, not just q25/q75.
    """
    n=np.asarray(evidence.get('profile_normal',[]),float)
    q=np.asarray(evidence.get('query_uv',[]),float)
    interval=None;interval_key=None
    for key in ('whole_transition_native_interval_px','complete_band_native_interval_px','complete_band_native_interval'):
        if key in evidence:
            interval=np.asarray(evidence[key],float);interval_key=key;break
    if interval is not None:
        if n.shape!=(2,) or q.shape!=(2,) or interval.shape!=(2,) or not np.isfinite(np.r_[n,q,interval]).all() or interval[1]<interval[0]:
            raise ValueError('Invalid recorded full native source profile')
        if not np.isclose(np.linalg.norm(n),1.,rtol=0,atol=2e-6):raise ValueError('Recorded profile normal is not unit')
        offsets=interval[0]+np.arange(int(np.floor(interval[1]-interval[0]+.25))+1)
        tangent=np.array([-n[1],n[0]])
        sites=q[None,None,:]+offsets[None,:,None]*n+np.arange(-3,4)[:,None,None]*tangent
        sites=np.rint(sites.astype(np.float32)*32)/32
        guard=support_pixel_audit(sites.reshape(-1,2),mask,local_records,gaussian_radius=0)
        guard.update(stencil='recorded_sample_chroma_full_interval_tangent7_cv2_bilinear32',interval_key=interval_key,normal_interval_px=interval.tolist(),profile_tangent_offsets_px=[-3,3])
    else:
        # Preserve the declared two-side/quantile context even where a source
        # did not record the entire historical search interval. No invented ROI.
        guard=support_pixel_audit(_between(_contexts(uv,evidence)),mask,local_records,gaussian_radius=0)
        guard.update(stencil='recorded_context_continuous_interval_only',full_historical_reader_interval_recorded=False)
    return guard

def guarded_remeasure(uv,normal,channels,operator,sides,mask,*,evidence=None,
                      local_records=(),manual=False,corner=False):
    evidence=evidence or {}
    if manual:
        result=_remeasure(uv,normal,channels,operator,sides,True,corner)
        result['support_guard']={'status':'exact_manual_click_no_invented_profile'}
        return result
    sites=_sample_sites(uv,normal,operator,sides,evidence,corner)
    guard=support_pixel_audit(sites,mask,local_records)
    original=original_source_profile_audit(uv,evidence,mask,local_records)
    hits=profile_roi_hits({'uv':uv,'source_evidence':evidence},local_records) if local_records else []
    guard['recorded_profile_exclusion_ids']=[r.get('id',r.get('exclusion_id',r['source_roi_xyxy'])) for r in hits]
    result=_remeasure(uv,normal,channels,operator,sides,False,corner)
    if guard.get('person_mask_pixels') or original.get('person_mask_pixels') or guard.get('static_exclusion_ids') or original.get('static_exclusion_ids') or hits:
        result['unqualified_current_uv']=result['current_uv']
        result['current_uv']=None
        result['status']='actual_reader_profile_intersects_person_mask' if guard['person_mask_pixels'] or original['person_mask_pixels'] else 'actual_reader_profile_intersects_static_ice_exclusion'
    elif guard.get('outside_native_image') or original.get('outside_native_image'):
        result['unqualified_current_uv']=result['current_uv'];result['current_uv']=None
        result['status']='actual_reader_profile_outside_native_image'
    result['support_guard']=guard
    result['original_source_profile_guard']=original
    return result

def _support_segments(indices,audits,mask,local_records):
    by={a['raw_index']:a for a in audits};segments=[];breaks=[]
    for i in indices:
        if not segments:segments.append([i]);continue
        prev=segments[-1][-1]
        if i!=prev+1:segments.append([i]);continue
        a=by[prev]['current_uv'];b=by[i]['current_uv']
        guard=support_pixel_audit(_between([a,b]),mask,local_records)
        if guard['person_mask_pixels'] or guard['static_exclusion_ids'] or guard.get('outside_native_image'):
            breaks.append({'between_raw_indices':[prev,i],'reason':'source_visibility_link_intersects_existing_foreground_support','support_guard':guard});segments.append([i])
        else:segments[-1].append(i)
    return segments,breaks

def audit_sources(normalized_dir,native_dir,output_dir,project_root,*,locked_roles=None, shot_ids=None, native_pts=None, static_exclusions=None):
 normalized=Path(normalized_dir);native=Path(native_dir);out=Path(output_dir);project=Path(project_root)
 if out.exists():raise ValueError('Source audit requires a new immutable output generation')
 manifest=read_json(native/'manifest.json');records={(f['shot_id'],int(f['native_pts'])):f for f in manifest['frames']}
 rows=list(read_jsonl(normalized/'observations.jsonl.gz'));groups=list(read_jsonl(normalized/'groups.jsonl.gz'))
 if shot_ids is not None: rows=[r for r in rows if r['shot_id'] in set(shot_ids)]
 if native_pts is not None: rows=[r for r in rows if r['native_pts'] in set(native_pts)]
 scope={(r['shot_id'],r['native_pts']) for r in rows};records={k:v for k,v in records.items() if k in scope}
 exclusions=read_json(static_exclusions)['records'] if static_exclusions else []
 gi={g['group_id']:g for g in groups};byframe=defaultdict(list);bygroup=defaultdict(list)
 for r in rows:byframe[(r['shot_id'],r['native_pts'])].append(r);bygroup[r['provenance_group_id']].append(r)
 if set(byframe)!=set(records):raise ValueError('Decoded native frame scope differs from normalized inputs')
 roles=read_json(locked_roles) if locked_roles else {};roles=roles.get('rows',roles)
 if isinstance(roles,list):roles={r['source_row_id']:r for r in roles}
 out.mkdir(parents=True);write_json(out/'policy.json',POLICY);policy_sha=file_sha256(out/'policy.json')
 regs={};model_refs=[]
 for sid in sorted({r['shot_id'] for r in rows}):
  nr=int(sid[1:]);dirname='shot2_8a28_r2_20261014' if nr==2 else f'shot{nr}_8a28_20261014'
  rp=project/'events/ankara_2026/models'/dirname/'feature_registry.json';regs[sid]=read_json(rp)['targets']
  model_refs.append({'shot_id':sid,'path':str(rp),'sha256':file_sha256(rp),'role':'typed material descriptor; numerical geometry initializer only'})
 supports=[];audit_rows=[];material_members=defaultdict(list);counts=Counter();frames=[]
 mask_records={}
 data_root=next(x for x in native.parents if x.name=='RinkSpace_Data')
 for sid in regs:
  nr=int(sid[1:]);mb=data_root/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014'
  mp=mb/'r2/frame_manifest.json' if nr==2 else mb/'frame_manifest.json'
  for mf in read_json(mp)['frames']:
   mask_records[(sid,int(mf.get('native_pts',mf.get('source_pts'))))]=mf
 for (sid,pts),fr in sorted(records.items(),key=lambda x:x[0][1]):
  png=native/fr['native_png']
  if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Native image hash changed')
  rgb=np.asarray(Image.open(png).convert('RGB'));channels=_channels(rgb)
  mask_record=mask_records[(sid,pts)];mask_path=Path(mask_record.get('mask_path') or mask_record.get('source',{}).get('mask_path'))
  expected_mask_sha=mask_record.get('mask_sha256') or mask_record.get('source',{}).get('mask_sha256')
  if expected_mask_sha and file_sha256(mask_path)!=expected_mask_sha:raise ValueError(f'Authoritative mask hash changed: {mask_path}')
  mask=np.asarray(Image.open(mask_path).convert('L')) if mask_path.is_file() else None
  authority=mask_record.get('source',mask_record)
  expected_rgb=authority.get('decoded_RGB_sha256') or mask_record.get('decoded_native_RGB_sha256')
  if not expected_rgb or hashlib.sha256(rgb.tobytes()).hexdigest()!=expected_rgb:raise ValueError('Decoded source RGB differs from pinned source authority')
  local=source_records(authority,exclusions)
  local_ice=lambda row: local if row.get('category')=='ice_curve' or row.get('physical_surface')=='ice' else []
  if mask is None:raise ValueError(f'Missing current source person mask: {mask_path}')
  if rgb.shape!=(1080,1920,3):raise ValueError('Non-native source dimensions')
  frame_counts=Counter();frrows=byframe[(sid,pts)]
  for gid in dict.fromkeys(r['provenance_group_id'] for r in frrows):
   members=bygroup[gid];group=gi[gid];feature=gi.get(group.get('feature_source_group_id'),{}); target=members[0].get('target_id')
   raw,raw_ev,raw_kind=_raw_support(group,feature,target);manual=any(r.get('manual_authoritative') for r in members)
   indices=_visible_indices(raw,members,manual);retained=[];raw_audits=[]
   definition=measurement_definition(members[0]);operator=definition['operator'];family=_family(group,feature)
   for i in indices:
    e=raw_ev[i] if len(raw_ev)==len(raw) else {};n=_normal(raw,int(i),e)
    rr=guarded_remeasure(raw[i],n,channels,operator,e.get('actual_source_sides_uv'),mask,evidence=e,local_records=local_ice(members[0]))
    if rr['current_uv'] is not None:
     pu,pv=np.rint(rr['current_uv']).astype(int)
     if not (0<=pu<1920 and 0<=pv<1080):rr['status']='current_readout_outside_image';rr['current_uv']=None
     elif mask[pv,pu]:rr['status']='current_readout_inside_person_mask';rr['current_uv']=None
     else:retained.append(int(i))
    raw_audits.append({'raw_index':int(i),'raw_uv':raw[i].tolist(),'original_profile_evidence':e,**rr})
   support_id='native:'+gid
   support={'schema':'rinkfit.native_support_family.v1','statistical_input_id':support_id,'source_family_id':family,'source_group_id':gid,'shot_id':sid,'native_pts':pts,'source_packet':group['source_packet'],'source_packet_sha256':group.get('packet_sha256') or members[0]['source_packet_sha256'],'raw_kind':raw_kind,'original_raw_count':len(raw),'visibility_intersection_indices':indices.tolist(),'current_support_indices':retained,'actual_source_audits':raw_audits,'representative_row_ids':[r['source_row_id'] for r in members],'count_raw_and_derived_representatives_together':False,'manual_authority_preserved':True,'source_flags_modified':False,'geometry_radius_ratios_used':False,'source_curve_family_metadata':{'source_circle_fit':feature.get('source_circle_fit'),'source_line_fit':feature.get('source_line_fit'),'joint_group_navigation':feature.get('joint_group_navigation'),'source_flags':group.get('source_flags'),'source_native_transition_iqr_px':group.get('source_native_transition_iqr_px'),'source_native_station_sigma_px':group.get('source_native_station_sigma_px'),'declared_correlation_ids':group.get('declared_correlation_ids'),'circle_family_coupling_hint':group.get('circle_family_coupling_hint')},'native_png':str(png),'native_png_sha256':fr['native_png_sha256']}
   support['current_support_segments'],support['source_visibility_link_rejections']=_support_segments(retained,raw_audits,mask,local_ice(members[0]))
   support['mask_path']=str(mask_path);support['mask_sha256']=expected_mask_sha
   support['source_exclusion_ids']=[r.get('id',r.get('exclusion_id',r['source_roi_xyxy'])) for r in local_ice(members[0])]
   supports.append(support)
   for j,row in enumerate(members):
    definition=measurement_definition(row);operator=definition['operator']
    e=(group.get('point_source_evidence') or [{}]*len(members));e=e[j] if isinstance(e,list) and len(e)==len(members) else {}
    if not isinstance(e,dict):e={}
    # Real direct material reads provide navigation; fitted representatives are
    # not promoted into new independent evidence by this operation.
    uv=np.asarray(row['source_uv'],float);normal=np.asarray(e.get('profile_normal',[np.nan,np.nan]),float);sides=e.get('actual_source_sides_uv')
    if not np.isfinite(normal).all() and len(raw):
     k=int(cKDTree(raw).query(uv)[1]);er=raw_ev[k] if len(raw_ev)==len(raw) else {};normal=_normal(raw,k,er);sides=sides or er.get('actual_source_sides_uv')
    if not np.isfinite(normal).all() and len(members)>1:
     q=_points([r['source_uv'] for r in members]);normal=_normal(q,j,{})
    manualclick=row.get('evidence_class')=='manual_native_click';corner=bool(row.get('point_correspondence'))
    measure=guarded_remeasure(uv,normal,channels,operator,sides,mask,evidence=e,local_records=local_ice(row),manual=manualclick,corner=corner)
    desc,identity_error=_descriptor(row,regs[sid]);reasons=[]
    if identity_error:reasons.append(identity_error)
    if measure['current_uv'] is None:reasons.append(measure['status'])
    if row.get('inside_supplied_person_mask') and not manualclick:reasons.append('supplied_person_mask_overlap')
    if row.get('inside_supplied_person_mask') and manualclick:
     # The sole conflict was checked on the native f1249 pixel and mask crop:
     # mask reaches visible grey ice just left of the hand.  Keep the exact click.
     if row['native_pts']==575744 and np.linalg.norm(uv-[668.,548.0952])<1e-6:
      measure['mask_resolution']='point_level_native_ice_visible_mask_overreach_near_blurred_hand_edge'
      measure['near_person_edge_uncertainty_preserved']=True
     else:reasons.append('manual_click_mask_conflict_retained_pending_mask_recheck')
    if 'exact_duplicate_same_target_uv' in row.get('qa_flags',[]):reasons.append('duplicate_retained_not_independent')
    if definition['conflict']:reasons.append('explicit_source_operator_conflicts_with_material_definition')
    if row.get('category')=='ad_print' and desc:
     # Frame-local component numbers require a demonstrated contour crosswalk;
     # storing a typed descriptor is insufficient to establish that crosswalk.
     key=desc.get('contour_key') or ''
     if any(x in key for x in ('component','current_','shot2_')):reasons.append('frame_local_print_component_requires_shape_crosswalk')
    status='source_supported_identity_candidate' if not reasons else 'auto_source_followup'
    role=roles.get(row['source_row_id'],{}).get('fit_role','development')
    result={**row,'schema':'rinkfit.joint9.native_remeasurement_row.v1','current_uv':measure['current_uv'],'measurement_operator_resolved':operator,'measurement_definition':definition,'joint_material_id':semantic_key({**row,'measurement_operator':operator}),'source_family_id':family,'statistical_input_id':support_id,'source_measurement_audit':measure,'typed_material_descriptor':desc,'physical_identity_status':'typed_identity_current_pixel_supported' if not identity_error and measure['current_uv'] is not None else 'pending_source_identity','source_support_status':status,'source_review_reasons':reasons,'fit_role':role,'numeric_statistics_ready':False,'solver_eligible':False,'original_flags_preserved':True,'manual_reference_preserved':manualclick,'native_png':str(png),'native_png_sha256':fr['native_png_sha256'],'policy_sha256':policy_sha,'raw_source_measurement_sha256':object_sha256({'native_pts':row['native_pts'],'source_uv':row['source_uv'],'source_row_id':row['source_row_id']})}
    audit_rows.append(result);counts[status]+=1;frame_counts[status]+=1
    if manualclick:counts['manual_clicks_preserved']+=1
    if corner:counts['fixed_entity_corner_rows']+=1
    material_members[result['joint_material_id'] or 'unbound:'+sid].append(result)
  frames.append({'shot_id':sid,'native_pts':pts,'native_png':str(png),'native_png_sha256':fr['native_png_sha256'],'rows':len(frrows),'decisions':dict(frame_counts),'original_source_rows_deleted':0})
  print(json.dumps({'shot_id':sid,'native_pts':pts,'rows_complete':sum(x['rows'] for x in frames)},separators=(',',':')),flush=True)
 write_jsonl(out/'relocated_rows.jsonl.gz',audit_rows);write_jsonl(out/'retained_support.jsonl.gz',supports)
 registry=[]
 for key,members in sorted(material_members.items()):
  first=members[0];good=[r for r in members if r['source_support_status']=='source_supported_identity_candidate'];shots=sorted({r['shot_id'] for r in members});times=sorted({r['native_pts'] for r in members})
  registry.append({'id':key,'source_target':first.get('target_id'),'surface':first.get('physical_surface'),'measurement_operator':first['measurement_operator_resolved'],'boundary_role':first.get('boundary_role'),'physical_entity':(first.get('target_id') or '').split('.print.')[0],'source_row_count':len(members),'current_pixel_supported_identity_candidate_rows':len(good),'shots':shots,'native_pts':times,'descriptor_candidates':list({object_sha256(r['typed_material_descriptor']):r['typed_material_descriptor'] for r in members if r['typed_material_descriptor']}.values()),'identity_accepted':False,'identity_status':'current_native_support_and_typed_descriptor_audited_pending_multiview_shape_crosswalk','numeric_geometry_status':'reference_or_new_shared_geometry_initializer','geometry_frozen':False})
 write_json(out/'joint_material_registry.json',{'schema':'rinkfit.joint9.source_material_registry.v1','materials':registry,'model_descriptor_sources':model_refs,'raw_source_modified':False,'circle_role_authority':'current user instruction: C0 and A1-A4 are finite-width circular lines; F1-F4/R1-R4 are targets'})
 write_json(out/'frames.json',frames)
 summary={'schema':'rinkfit.S01_S04.native_source_audit.v2','rows_total':len(rows),'frames_total':len(frames),'source_support_families':len(supports),'counts':dict(counts),'materials':len(registry),'native_manifest_sha256':file_sha256(native/'manifest.json'),'policy_sha256':policy_sha,'raw_source_rows_deleted':0,'source_flags_modified':False,'fit_residuals_used':False,'statistics_and_multiview_identity_promotion_pending':True,'source_pixel_remeasurement_complete':True,'static_exclusions':{'path':str(static_exclusions),'sha256':file_sha256(static_exclusions)} if static_exclusions else None,'selected_shots':sorted({r['shot_id'] for r in rows}),'manual_clicks_have_no_automatic_profile':True}
 write_json(out/'summary.json',summary);return summary
