"""Consume complete generic native texture rereads, including off-centre points.

Source mean corrections, confirmed wrong-identity pending and additional-context
unknowns are distinct. No world model or calibrated source precision is issued.
"""
import hashlib
from collections import Counter
import numpy as np
from .io_utils import read_json,read_jsonl

def need(ok,s):
 if not ok:raise ValueError(s)
def h64(v):return hashlib.sha256(np.asarray(v,dtype='<f8').tobytes()).hexdigest()
def refs(v,verify):
 if isinstance(v,dict):
  if 'sha256'in v and'file'in v:verify(v);return
  for x in v.values():refs(x,verify)
 elif isinstance(v,list):
  for x in v:refs(x,verify)

def _template_origin(q):
 if 'template_origin_native_uv'in q:return np.asarray(q['template_origin_native_uv'])
 need(q.get('reference_kind')=='previous_direct_source_verified_exposure_same_physical_track','Actual template origin is missing for non-chain reader')
 return np.asarray(q['birth_uv'])-48.

def _verify_reference_graph(records,old,frames,params):
 by={r['source_row_id']:r for r in records};need(len(by)==len(records),'Repeated adjacent source graph node')
 for rid,q in by.items():
  need(rid in old and q['track_id']==old[rid]['physical_track_id']and q['native_pts']==old[rid]['native_pts']and q['current_RGB_sha256']==old[rid]['current_rgb_sha256'],'Adjacent source graph original lineage differs')
  root=by.get(q['root_birth_row_id']);need(root is not None and root['reference_kind']=='original_native_source_birth_reference'and root['status']=='original_birth_reference'and root['track_id']==q['track_id']and root['native_pts']==q['root_birth_native_pts']and root['current_uv']==q['root_birth_uv'],'Adjacent source graph root differs')
  if q['reference_kind']=='original_native_source_birth_reference':need(q['native_pts']==q['birth_native_pts']and q['source_row_id']==q['root_birth_row_id']and q['current_uv']==old[rid]['source_uv'],'Adjacent root invented a source point');continue
  if q['reference_kind']=='no_legal_adjacent_reference_available':need('reference_source_row_id'not in q and q['status']=='unknown_reference_gap'and not q['source_UV_changed'],'Failed adjacent reference gap propagated');continue
  need(q['reference_kind']=='previous_direct_source_verified_exposure_same_physical_track','Unknown adjacent source reference kind');previous=by.get(q['reference_source_row_id']);need(previous is not None and previous['status']in('original_birth_reference','wide_context_direct_source_supported')and previous['track_id']==q['track_id']and previous['native_pts']==q['native_pts']-256==q['reference_native_pts']==q['birth_native_pts']and previous['current_uv']==q['reference_uv']==q['birth_uv']and h64(q['reference_uv'])==q['reference_uv_binary64_sha256']and previous['current_RGB_sha256']==q['birth_RGB_sha256'],'Adjacent source reference is not the actual prior legal native readout')
  if q['status']=='wide_context_direct_source_supported':need(np.max(abs(_verify_affine(q,frames,params)-np.asarray(q['current_uv'])))<=1e-12,'Adjacent source graph current UV not actual warp')
 return by

def _verify_affine(q,frames,params):
 F=q['forward'];B=q['reverse'];A=np.asarray(F['warp'],float);Z=np.asarray(B['warp'],float);anchored=q.get('evidence_reader')=='anchored_161x65';shape=q['template_shape']
 if anchored:
  need(shape==params['secondary_template_shape']and shape==[161,65],'Unknown actual anchored source template');point=np.asarray(q['birth_source_point_local_uv'],float);anchor=np.asarray(F['template_anchor_local_uv'],float);origin=np.asarray(F['template_origin_birth_native_uv'],float);revpoint=np.asarray(q['reverse_source_point_local_uv'],float);local=np.array([[0,0],[0,64],[160,0],[160,64]],float);fncc=F['NCC'];bncc=B['NCC'];fc=F['translation_NCC'];bc=B['translation_NCC'];fo=F['other_branch_NCC'];bo=B['other_branch_NCC'];fm=F['branch_identity_margin'];bm=B['branch_identity_margin']
  need(np.array_equal(origin+point,q['birth_uv'])and np.all(point>=0)and point[0]<=160 and point[1]<=64,'Source point is not inside actual anchored birth template');forward_anchor=np.asarray(F['search_origin'])+A@np.r_[anchor,1.];reverse_origin=forward_anchor-anchor;need(np.max(abs(forward_anchor-np.asarray(F['registered_uv'])))<=1e-10,'Anchored warp anchor differs')
 else:
  need(shape==[97,97],'Unknown primary texture template');point=np.array([48.,48.]);origin=_template_origin(q);revpoint=point;local=np.array([[0,0],[96,0],[96,96],[0,96]],float);fncc=F['registered_NCC'];bncc=B['registered_NCC'];fc=F['translation_peak_NCC'];bc=B['translation_peak_NCC'];fo=F['other_branch_peak_NCC'];bo=B['other_branch_peak_NCC'];fm=q['forward_branch_identity_margin'];bm=q['reverse_branch_identity_margin'];need(F['branch_exclusion_halfwidth_xy']==B['branch_exclusion_halfwidth_xy']==params['primary_branch_exclusion_halfwidth_xy'],'Primary source branch policy changed');need(np.array_equal(origin+point,q['birth_uv']),'Primary fixed birth template origin differs')
 fuv=np.asarray(F['search_origin'])+A@np.r_[point,1.]
 if anchored:need(np.max(abs((fuv-reverse_origin)-revpoint))<=1e-10,'Off-centre reverse source point was replaced by anchor')
 else:reverse_origin=fuv-point
 back=np.asarray(B['search_origin'])+Z@np.r_[revpoint,1.];corners=local@A[:,:2].T+A[:,2]+np.asarray(F['search_origin']);returned=(corners-reverse_origin)@Z[:,:2].T+Z[:,2]+np.asarray(B['search_origin']);cycle=np.linalg.norm(returned-(local+origin),axis=1)
 need(F['status']==B['status']=='registered'and min(fncc,bncc)>=params['affine_NCC_min']and cycle.max()<=params['whole_patch_cycle_max_px']and abs(cycle.max()-q['affine_patch_cycle_max_px'])<1e-9,'Generic reciprocal affine/full-corner evidence failed')
 need(fc-fo-(1-fc)>0 and bc-bo-(1-bc)>0 and abs(fc-fo-(1-fc)-fm)<1e-10 and abs(bc-bo-(1-bc)-bm)<1e-10,'Wider source context does not disambiguate repeated letters')
 support={x['direction']:x for x in q['full_affine_context_support']};need(set(support)=={'forward','reverse'},'Missing either actual native full-affine support')
 for direction,W in [('forward',F),('reverse',B)]:
  C=local@np.asarray(W['warp'])[:,:2].T+np.asarray(W['warp'])[:,2]+np.asarray(W['search_origin']);pr=support[direction];pts=q['native_pts']if direction=='forward'else q['birth_native_pts'];fr=frames[pts];lo=np.floor(C.min(0)-1).astype(int);hi=np.ceil(C.max(0)+1).astype(int)
  need(pr['native_pts']==pts and pr['actual_RGB_sha256']==fr['RGB_sha256']and pr['mask_sha256']==fr['mask_sha256']and pr['excluded_pixels_in_bbox']==0 and pr['status']=='pass_full_affine_context_support'and pr['conservative_affine_support_bbox_plus_1px_halo']==[lo.tolist(),hi.tolist()]and np.max(abs(C-np.asarray(pr['registered_corners_native_uv'])))<=1e-10,'Generic readout leaves actual full affine RGB/mask support')
  if 'actual_BGR_sha256'in pr:need(pr['actual_BGR_sha256']==fr['BGR_sha256'],'Actual anchored BGR exposure differs')
 return fuv

def source_generic_readout_rows(reference,verify):
 g=read_json(verify(reference));need(g['schema']=='rinkfit.generic_source_current_wide_affine_generation.v3','Generic actual source generation required');refs(g,verify);c=read_json(verify(g['input_contract']));refs(c,verify)
 need(c['schema']=='rinkfit.generic_source_wide_context_input_contract.v3'and c['model_or_projection_input']is None and g['no_model_or_interpolated_UV']and not g['statistical_covariance_issued']and g['full_batch_not_success_subset'],'Generic readout introduced model UV/precision or selected success subset')
 need(g['reader']['sha256']==c['reader']['sha256']and g['full_affine_support_reader']['sha256']==c['full_affine_support_reader']['sha256'],'Actual reader/support strategy changed')
 allold=list(read_jsonl(verify(c['original_typed_source_rows'])));selected=read_json(verify(c['selected_original_rows']));oldby={r['source_row_id']:r for r in allold};selectedids={r['source_row_id']for r in selected};cohorts={(r['shot_id'],r['surface_instance_id'])for r in selected}
 expected={r['source_row_id']for r in allold if(r['shot_id'],r['surface_instance_id'])in cohorts};need(selectedids==expected and len(selected)==len(expected)==c['primary_denominator'],'Primary source cohort is not its complete original shot/entity denominator');pendingids=set(c.get('supplemental_pending_rowIDs',[]));ids=selectedids|pendingids;need(not selectedids.intersection(pendingids)and ids<=set(oldby),'Supplemental pending source selection invalid');old={k:oldby[k]for k in ids}
 rows=list(read_jsonl(verify(g['source_rows'])));records=list(read_jsonl(verify(g['readout_records'])));recs={r['source_row_id']:r for r in records};patch=read_json(verify(g['correction_patch']));ds={r['source_row_id']:r for r in patch['rows']};semantic=read_json(verify(g['semantic_visual_review']));refs(semantic,verify);frames={r['native_pts']:r for r in read_json(verify(g['frame_receipt']))}
 if 'anchored_frame_receipt'in c:frames.update({r['native_pts']:r for r in read_json(verify(c['anchored_frame_receipt']))})
 for key in ('anchored_all_frame_receipt','adjacent_all_frame_receipt'):
  if key in c:frames.update({r['native_pts']:r for r in read_json(verify(c[key]))})
 need(len(old)==len(rows)==len(records)==len(recs)==len(ds)==g['all_source_rows']==c['row_count']and set(recs)==set(ds)==set(old)and len({r['source_row_id']for r in rows})==len(old),'Generic readout loses raw IDs or applies successful subset')
 params=c['parameters'];need(params['primary_template_shape']==[97,97]and params['mean_change_only_over_1px'],'Actual frozen primary operator changed');counts=Counter();changedtracks=set()
 if 'adjacent_all_readout_records'in c:
  adjacent=_verify_reference_graph(list(read_jsonl(verify(c['adjacent_all_readout_records']))),old,frames,params)
  for q in records:
   if q.get('reference_kind')=='previous_direct_source_verified_exposure_same_physical_track':
    need(q['source_chain_conditional']and q['not_independent_source_statistic']and q['reference_readout_records']==c['adjacent_all_readout_records'],'Applied source chain is detached from complete source graph')
    priorchain=adjacent[q['source_row_id']];need(all(q[k]==priorchain[k]for k in ('source_row_id','track_id','native_pts','current_uv','birth_native_pts','birth_uv','reference_source_row_id','reference_uv','reference_uv_binary64_sha256','root_birth_row_id','root_birth_native_pts','root_birth_uv','forward','reverse','full_affine_context_support','status')),'Applied mean does not consume its actual complete adjacent source graph')
 for r in rows:
  rid=r['source_row_id'];prior=old[rid];q=recs[rid];delta=ds[rid];kind=r['source_readout_class'];counts[kind]+=1
  need(r['physical_track_id']==q['track_id']==prior['physical_track_id']and r['native_pts']==q['native_pts']==prior['native_pts']and r['current_rgb_sha256']==q['current_RGB_sha256']==prior['current_rgb_sha256']and r['surface_instance_id']==q['surface_instance_id']==prior['surface_instance_id']and r['shot_id']==q['shot_id']==prior['shot_id'],'Original generic track/entity/clock/RGB lineage changed')
  need(q['original_uv']==prior['source_uv']==r['original_source_uv']==delta['old_source_uv']and q['original_uv_binary64_sha256']==delta['old_source_uv_binary64_sha256']==h64(prior['source_uv']),'Original generic binary64 UV changed')
  need(not q['model_or_world_projection_used']and not r['source_readout_has_calibrated_precision']and not r['new_human_review']and r['source_mean_changed']==q['source_UV_changed']==delta['source_UV_changed'],'Generic reader changed actual mean provenance or covariance')
  if q['status']=='wide_context_direct_source_supported':
   fuv=_verify_affine(q,frames,params)
   if q['source_UV_changed']:need(np.max(abs(fuv-np.asarray(r['current_uv'])))<=1e-12 and np.linalg.norm(fuv-np.asarray(prior['source_uv']))>1 and kind=='corrected_direct_source_supported','Applied mean is not direct actual source-point warp');changedtracks.add(r['physical_track_id'])
   else:need(h64(r['current_uv'])==h64(prior['source_uv'])and kind=='supported_original_mean','Supported unchanged mean must retain original bits')
  else:need(not q['source_UV_changed']and h64(r['current_uv'])==h64(prior['source_uv']),'Unknown source cannot issue a new mean')
  pending=kind=='source_wrong_identity_pending';need(bool(r.get('source_only_pending',False))==pending and bool(delta['source_only_pending'])==pending and r['source_geometry_eligible']==(not pending),'Source wrong identity vs extra-review unknown conflated')
  if pending:need(rid in semantic.get('confirmed_source_pending',{})and q.get('source_only_pending')and q.get('precise_current_native_uv_replacement_available')is False and r['source_readout_unknown_reason'],'Source pending lacks explicit actual wrong-identity evidence')
  need(r['source_uv']==r['current_uv']==q['current_uv']==delta['current_uv']and h64(r['current_uv'])==r['source_mean_binary64_sha256']==q['current_uv_binary64_sha256']==delta['current_uv_binary64_sha256']and r['source_readout_status']==q['status']and r['source_readout_generation_id']==g['observation_generation_id']and delta['readout_class']==kind,'Final generic source row/patch/record disagree')
 need(dict(counts)==g['class_counts']and counts['corrected_direct_source_supported']==g['source_uv_updated_rows']and counts['source_wrong_identity_pending']==g['source_wrong_identity_pending_rows'],'Actual complete generation effect differs')
 return {r['source_row_id']:r for r in rows}
