"""Native source remeasurement and lossless typed assembly for shots 2--9.

Source identity and pixel support are separate decisions.  This worker never
changes historical source flags or uses model residuals as a deletion rule.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import argparse, gzip, hashlib, json, math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256, object_sha256
from .materials import measurement_definition, semantic_key

POLICY = {
 'schema':'rinkfit.joint9.native_source_policy.v1',
 'measurement_authority':'official current native pixels; manual clicks unchanged',
 'circular_width_lines':['C0','A1','A2','A3','A4'],
 'target_rings':['F1','F2','F3','F4','R1','R2','R3','R4'],
 'normal_search_radius_px':3., 'normal_step_px':.25,
 'minimum_peak_gradient_8bit_per_px':.75,
 'minimum_profile_contrast_8bit':4.,
 'manual_deleted_support':'intersection with final segment visibility, no bridging singleton fragments',
 'source_statistics':'actual retained raw support; derived representative coordinates excluded from independent raw statistics',
 'identity_rule':'typed finite surface descriptor plus native material transition and repeated source shape; local component suffix is not a physical identity',
 'fit_residuals_used':False, 'raw_source_modified':False,
}

def _points(x):
 try: a=np.asarray(x,dtype=float)
 except (TypeError,ValueError): return np.empty((0,2))
 return a if a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all() else np.empty((0,2))

def _sample(a,uv):
 uv=np.asarray(uv).reshape(-1,2)
 return np.stack([map_coordinates(c,[uv[:,1],uv[:,0]],order=1,mode='nearest',prefilter=False) for c in a],axis=1)

def _channels(rgb):
 a=rgb.astype(np.float32)
 return gaussian_filter(np.stack((.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2], a[:,:,0]-a[:,:,1], a[:,:,2]-.5*(a[:,:,0]+a[:,:,1]))),sigma=(0,.8,.8))

def _remeasure(uv,normal,channels,operator,sides=None,manual=False,corner=False):
 uv=np.asarray(uv,dtype=float); n=np.asarray(normal,dtype=float)
 if manual:
  return {'current_uv':uv.tolist(),'status':'manual_native_click_preserved','normal':None,'source_basis':'manual_native_click','moved_px':0.}
 if corner:
  offsets=np.asarray([(x,y) for y in range(-3,4) for x in range(-3,4)],float)
  x=_sample(channels,uv+offsets+[.5,0])-_sample(channels,uv+offsets+[-.5,0]); y=_sample(channels,uv+offsets+[0,.5])-_sample(channels,uv+offsets+[0,-.5])
  c=int(np.hypot(x,y).max(axis=0).argmax()); eig=np.linalg.eigvalsh([[np.mean(x[:,c]**2),np.mean(x[:,c]*y[:,c])],[np.mean(x[:,c]*y[:,c]),np.mean(y[:,c]**2)]])
  ratio=float(max(0,eig[0])/max(eig[1],1e-12)); ok=ratio>=.025 and float(np.hypot(x,y).max())>=.75
  return {'current_uv':uv.tolist(),'status':'entity_corner_two_direction_support' if ok else 'entity_corner_requires_source_geometry_review','normal':None,'corner_second_direction_ratio':ratio,'moved_px':0.,'entity_station_may_slide':False}
 if n.shape!=(2,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-8:
  return {'current_uv':None,'status':'missing_current_source_normal','normal':None}
 n=n/np.linalg.norm(n)
 def edge(q):
  off=np.arange(-3.,3.0001,.25); a=_sample(channels,np.asarray(q)[None,:]+off[:,None]*n)
  g=np.gradient(a,.25,axis=0); c=int(np.max(np.abs(g),axis=0).argmax()); val=np.abs(g[:,c]); k=int(val.argmax())
  # Finite search at supplied actual source interface, never global nearest geometry.
  contrast=float(np.ptp(a[:,c])); peak=float(val[k]); good=peak>=.75 and contrast>=4. and k not in (0,len(off)-1)
  shift=float(off[k])
  if 0<k<len(off)-1:
   v=val[k-1:k+2]; den=v[0]-2*v[1]+v[2]
   if abs(den)>1e-9: shift+=float(np.clip(.5*(v[0]-v[2])/den,-1,1))*.25
  return (np.asarray(q)+shift*n),{'peak_gradient':peak,'contrast':contrast,'offset_px':shift,'channel':c,'polarity':int(np.sign(g[k,c])),'valid':good}
 sides=_points(sides)
 if operator=='paint_center':
  if len(sides)!=2 or np.linalg.norm(sides[1]-sides[0])<.25:
   return {'current_uv':None,'status':'width_midline_missing_two_distinct_material_edges','normal':n.tolist()}
  left,la=edge(sides[0]);right,ra=edge(sides[1]); distance=float(np.linalg.norm(right-left)); good=la['valid'] and ra['valid'] and distance>=.25
  current=.5*(left+right)
  return {'current_uv':current.tolist() if good else None,'status':'two_current_material_edges_midline' if good else 'width_midline_weak_or_unresolved_flank','normal':n.tolist(),'flank_uv':[left.tolist(),right.tolist()],'flank_profiles':[la,ra],'width_px':distance,'moved_px':float(np.linalg.norm(current-uv))}
 current,a=edge(uv)
 return {'current_uv':current.tolist() if a['valid'] else None,'status':'current_native_material_interface' if a['valid'] else 'weak_or_search_boundary_source','normal':n.tolist(),'profile':a,'moved_px':float(np.linalg.norm(current-uv))}

def _raw_support(group,feature,target):
 raw=_points(group.get('source_native_readout_uv')); ev=[];kind='segment_native_readout'
 if not len(raw):
  reads=(feature.get('source_circle_fit') or {}).get('direct_readouts') or []
  role=target.rsplit(':',1)[-1] if target and ':' in target else None
  if role and any(x.get('role')==role for x in reads): reads=[x for x in reads if x.get('role')==role]
  reads=[x for x in reads if len(_points([x.get('uv')]))]
  if reads:
   raw=_points([x['uv'] for x in reads]);ev=[x.get('source_evidence') or {} for x in reads];kind='circle_direct_material_readout_no_family_ratio'
 return raw,ev,kind

def _visible_indices(raw,rows,manual):
 if not len(raw): return np.empty(0,dtype=int)
 if not manual: return np.arange(len(raw))
 # Final manually edited segments are the visibility authority.  Endpoint spans
 # are intersected independently; singleton fragments retain only their own
 # nearest real reading and cannot bridge a removed span.
 reps=_points([r['source_uv'] for r in rows]); nearest=cKDTree(raw).query(reps)[1]
 if len(nearest)==1: return np.unique(nearest)
 return np.arange(int(nearest.min()),int(nearest.max())+1)

def _normal(raw,i,evidence):
 n=np.asarray(evidence.get('profile_normal',[np.nan,np.nan]),float)
 if n.shape==(2,) and np.isfinite(n).all() and np.linalg.norm(n)>1e-8:return n/np.linalg.norm(n)
 if len(raw)>1:
  d=raw[min(i+1,len(raw)-1)]-raw[max(i-1,0)]; norm=np.linalg.norm(d)
  if norm>1e-8:return np.array([-d[1],d[0]])/norm
 return np.array([np.nan,np.nan])

def _descriptor(row,registry):
 target=row.get('target_id');d=(registry or {}).get(target)
 if not target:return None,'unbound_static_material_identity_not_yet_resolved'
 if not d:return None,'missing_typed_material_descriptor'
 kind=d.get('kind')
 if kind not in {'ice_curve','board_edge','ad_edge','ad_corner','ad_print'}:return d,'unsupported_typed_material_kind'
 if kind=='ad_print' and not d.get('advertisement_id'):return d,'print_missing_physical_ad_panel'
 return d,None

def _family(group,feature):
 # Native circle fits couple layers through old radius ratios; their real edge
 # readings are retained separately and never inherit those fitted ratios.
 return group.get('feature_source_group_id') or group['group_id']

def audit_sources(normalized_dir,native_dir,output_dir,project_root,*,locked_roles=None):
 normalized=Path(normalized_dir);native=Path(native_dir);out=Path(output_dir);project=Path(project_root)
 if out.exists():raise ValueError('Source audit requires a new immutable output generation')
 manifest=read_json(native/'manifest.json');records={(f['shot_id'],int(f['native_pts'])):f for f in manifest['frames']}
 rows=list(read_jsonl(normalized/'observations.jsonl.gz'));groups=list(read_jsonl(normalized/'groups.jsonl.gz'))
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
    rr=_remeasure(raw[i],n,channels,operator,e.get('actual_source_sides_uv'))
    if rr['current_uv'] is not None:
     pu,pv=np.rint(rr['current_uv']).astype(int)
     if not (0<=pu<1920 and 0<=pv<1080):rr['status']='current_readout_outside_image';rr['current_uv']=None
     elif mask[pv,pu]:rr['status']='current_readout_inside_person_mask';rr['current_uv']=None
     else:retained.append(int(i))
    raw_audits.append({'raw_index':int(i),'raw_uv':raw[i].tolist(),'original_profile_evidence':e,**rr})
   support_id='native:'+gid
   support={'schema':'rinkfit.native_support_family.v1','statistical_input_id':support_id,'source_family_id':family,'source_group_id':gid,'shot_id':sid,'native_pts':pts,'source_packet':group['source_packet'],'source_packet_sha256':group.get('packet_sha256') or members[0]['source_packet_sha256'],'raw_kind':raw_kind,'original_raw_count':len(raw),'visibility_intersection_indices':indices.tolist(),'current_support_indices':retained,'actual_source_audits':raw_audits,'representative_row_ids':[r['source_row_id'] for r in members],'count_raw_and_derived_representatives_together':False,'manual_authority_preserved':True,'source_flags_modified':False,'geometry_radius_ratios_used':False,'source_curve_family_metadata':{'source_circle_fit':feature.get('source_circle_fit'),'joint_group_navigation':feature.get('joint_group_navigation'),'source_flags':group.get('source_flags'),'source_native_transition_iqr_px':group.get('source_native_transition_iqr_px'),'source_native_station_sigma_px':group.get('source_native_station_sigma_px'),'declared_correlation_ids':group.get('declared_correlation_ids'),'circle_family_coupling_hint':group.get('circle_family_coupling_hint')},'native_png':str(png),'native_png_sha256':fr['native_png_sha256']}
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
    measure=_remeasure(uv,normal,channels,operator,sides,manualclick,corner)
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
 summary={'schema':'rinkfit.joint9.native_source_audit.v1','rows_total':len(rows),'frames_total':len(frames),'source_support_families':len(supports),'counts':dict(counts),'materials':len(registry),'native_manifest_sha256':file_sha256(native/'manifest.json'),'policy_sha256':policy_sha,'raw_source_rows_deleted':0,'source_flags_modified':False,'fit_residuals_used':False,'statistics_and_multiview_identity_promotion_pending':True,'source_pixel_remeasurement_complete':True}
 write_json(out/'summary.json',summary);return summary

def assemble_s09(relocated_rows,output_dir):
 """A real 62-frame merged packet with all full support and 500-point displays."""
 out=Path(output_dir)
 if out.exists():raise ValueError('New s09 assembly generation required')
 rows=[r for r in read_jsonl(relocated_rows) if r['shot_id']=='s09'];by=defaultdict(list)
 for r in rows:by[r['native_pts']].append(r)
 if len(by)!=62:raise ValueError('s09 selected frame count differs from authoritative inventory')
 out.mkdir(parents=True);(out/'frames').mkdir();manifest=[]
 for pts,members in sorted(by.items()):
  # Deduplicate current same target/native UV only in display membership.  All
  # originals remain in the support ledger with explicit alias provenance.
  unique={};dups=[]
  for r in members:
   key=(r.get('target_id') or r.get('unbound_candidate_id'),tuple(r['source_uv']))
   if key in unique:dups.append({'source_row_id':r['source_row_id'],'same_source_as':unique[key]['source_row_id']})
   else:unique[key]=r
  pool=list(unique.values());strata=defaultdict(list)
  for r in pool:strata[r.get('target_id') or 'unbound:'+str(r.get('unbound_candidate_id'))].append(r)
  selected=[]
  # One per finite named material first, preserving entity corners; fill by
  # stratified source order, never selecting only visually attractive residuals.
  for name,a in sorted(strata.items()):
   for r in a:
    if r.get('point_correspondence'):selected.append(r)
   if not any(r.get('point_correspondence') for r in a):selected.append(a[0])
  seen={r['source_row_id'] for r in selected}
  remaining=[r for r in pool if r['source_row_id'] not in seen]
  if len(selected)>500:selected=selected[:500]
  elif len(selected)+len(remaining)>500:
   ii=np.linspace(0,len(remaining)-1,500-len(selected)).astype(int);selected.extend(remaining[i] for i in ii)
  else:selected.extend(remaining)
  idx=members[0]['programme_frame_index_1based'];packet={'schema':'rinkfit.joint9.s09.assembled_source_packet.v1','shot_id':'s09','programme_frame_index_1based':idx,'native_pts':pts,'time_base':'1/12800','source_packets':sorted({(r['source_packet'],r['source_packet_sha256']) for r in members}),'full_source_row_ids':[r['source_row_id'] for r in members],'full_rows':len(members),'named_rows':sum(bool(r.get('target_id')) for r in members),'unbound_rows':sum(not r.get('target_id') for r in members),'display_row_ids':[r['source_row_id'] for r in selected],'display_point_count':len(selected),'display_ceiling':500,'support_compression_is_not_measurement_deletion':True,'duplicate_membership':dups,'typed_bindings':[{'source_row_id':r['source_row_id'],'target_id':r.get('target_id'),'point_correspondence':r['point_correspondence'],'station_fixed_for_entity':bool(r['point_correspondence']),'joint_material_id':r.get('joint_material_id'),'source_support_status':r['source_support_status']} for r in members if r.get('target_id')],'source_review_generation':{'path':str(Path(relocated_rows).resolve()),'sha256':file_sha256(relocated_rows)},'numerical_geometry_claim':'not fitted by assembly','source_flags_modified':False}
  fp=out/'frames'/f'{idx:06d}.json';write_json(fp,packet);manifest.append({'programme_frame_index_1based':idx,'native_pts':pts,'packet':str(fp),'sha256':file_sha256(fp),'full_rows':len(members),'display_rows':len(selected)})
 result={'schema':'rinkfit.joint9.s09.assembly_manifest.v1','frames':manifest,'frame_count':len(manifest),'rows_total':len(rows),'named_rows':sum(bool(r.get('target_id')) for r in rows),'unbound_rows':sum(not r.get('target_id') for r in rows),'original_rows_deleted':0,'formal_source_packet_assembled':True,'all_rows_solver_admitted':False,'old_zero_frame_model_used_as_completion':False}
 write_json(out/'manifest.json',result);return result

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--normalized',type=Path,required=True);ap.add_argument('--native',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--locked-roles',type=Path)
 a=ap.parse_args();print(json.dumps(audit_sources(a.normalized,a.native,a.out,a.project,locked_roles=a.locked_roles),ensure_ascii=False));print(json.dumps(assemble_s09(a.out/'relocated_rows.jsonl.gz',a.out/'s09_packet'),ensure_ascii=False))
