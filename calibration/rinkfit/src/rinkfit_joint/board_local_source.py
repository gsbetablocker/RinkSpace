"""Localize outer material only inside an independently source-reviewed board object.

The source object context is navigation/scope, never a pixel boundary. No whole
image blue/yellow extrema establish board identity. Historical color-local mean
is reused only after a current native-board context excludes stands and rail.
"""
from pathlib import Path
import hashlib
import numpy as np
from .io_utils import read_json,file_sha256,object_sha256
from .measurement_contract import EventMeasurementContract

CONFIG=dict(version='actual_board_instance_local_outer_v2_explicit_model_search_window',whole_image_color_object_selection=False,old_UV_or_W_or_physical_pass_inherited=False,instance_ROI_is_boundary_UV=False,source_material_remeasured_in_current_native_RGB=True)

def read_current_board_instance_outer(rgb,mask,context):
 from .board_outer_source import source_outer_boundary_rows_untrusted_v2
 rgb=np.asarray(rgb);mask=np.asarray(mask,bool);h,w=mask.shape
 if rgb.shape!=(h,w,3):raise ValueError('Actual RGB and exclusions differ')
 required=('native_pts','shot_id','actual_native_RGB_sha256','source_review')
 if any(k not in context for k in required):raise ValueError('Source-bound actual board object context required')
 if hashlib.sha256(rgb.tobytes()).hexdigest()!=context['actual_native_RGB_sha256']:raise ValueError('Board object context is not this exposure')
 ref=context['source_review'];path=Path(ref['path'])
 if file_sha256(path)!=ref['sha256']:raise ValueError('Actual current object review changed')
 review=read_json(path);frame=next((q for q in review['frames'] if int(q['native_pts'])==int(context['native_pts'])),None)
 if frame is None or frame['shot_id']!=context['shot_id']:raise ValueError('Review does not bind actual PTS/shot')
 schema=review.get('schema')
 if schema not in ('rinkfit.joint9.current_RGB_board_instance_navigation_review.v1','rinkfit.joint9.source_correspondence_board_instance_navigation.v1','rinkfit.joint9.same_native_readout_local_board_navigation.v1','rinkfit.joint9.current_joint_model_board_navigation.v1'):raise ValueError('Unsupported actual source object context schema')
 if hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest()!=frame['decoded_native_BGR_sha256']:raise ValueError('Actual RGB object proof mismatch')
 if schema=='rinkfit.joint9.current_RGB_board_instance_navigation_review.v1':
  if frame.get('physical_semantic_authority')!='user_blue_upper_yellow_lower_definition; Codex_current_exposure_actual_RGB_visual_review' or not frame.get('ROI_is_search_domain_only'):raise ValueError('Typed labels or logits alone are not actual board evidence')
 elif schema=='rinkfit.joint9.same_native_readout_local_board_navigation.v1':
  nr=frame['source_navigation_index']
  if file_sha256(nr['path'])!=nr['sha256']:raise ValueError('Exact-exposure navigation source changed')
  anchors=frame['source_navigation_anchors']
  if any(int(a['native_pts'])!=int(frame['native_pts']) or a['shot_id']!=frame['shot_id'] or not a['navigation_only'] for a in anchors):raise ValueError('Readout navigation is not this exact source exposure')
  if frame.get('physical_semantic_authority')!='current_native_actual_board_material_sequence_with_same_exposure_readout_navigation_only' or not frame.get('source_object_identity_not_proved_by_navigation'):raise ValueError('Legacy source UV/identity cannot be promoted without current material rereading')
 elif schema=='rinkfit.joint9.current_joint_model_board_navigation.v1':
  mr=frame['source_model_navigation']
  if file_sha256(mr['path'])!=mr['sha256']:raise ValueError('Model search proposal changed')
  if frame.get('physical_semantic_authority')!='current_joint_model_projection_search_only; current_native_material_identity_unproved' or not frame.get('source_object_identity_not_proved_by_navigation') or not frame.get('ROI_is_search_domain_only'):raise ValueError('Model navigation cannot establish source identity')
  if frame.get('fit_UV_created',False):raise ValueError('World projection is navigation only')
 else:
  ar=frame['anchor_board_review'];ap=Path(ar['path'])
  if file_sha256(ap)!=ar['sha256']:raise ValueError('Actual source-board anchor review changed')
  anchor=next(q for q in read_json(ap)['frames']if int(q['native_pts'])==int(frame['anchor_native_pts']))
  if anchor['shot_id']!=frame['shot_id'] or anchor['native_png_sha256']!=frame['anchor_native_png_sha256'] or file_sha256(anchor['native_png'])!=anchor['native_png_sha256']:raise ValueError('Actual source object template changed')
  A=np.asarray(frame['actual_inlier_source_uv'],float);B=np.asarray(frame['actual_inlier_current_uv'],float);H=np.asarray(frame['source_to_current_navigation_H'],float);lo=np.asarray(anchor['board_instance_roi_xyxy'][:2]);hi=np.asarray(anchor['board_instance_roi_xyxy'][2:])
  if A.shape!=B.shape or A.ndim!=2 or A.shape[1]!=2 or len(A)<30 or not np.all((A>=lo)&(A<hi)):raise ValueError('Insufficient actual board-body source correspondences')
  q=np.c_[A,np.ones(len(A))]@H.T;pred=q[:,:2]/q[:,2,None]
  if np.quantile(np.linalg.norm(pred-B,axis=1),.95)>2.5 or frame['source_orientation_determinant']<=0:raise ValueError('Current RGB correspondences do not support this object navigation')
  if frame.get('physical_semantic_authority')!='current_actual_board_object_RGB_correspondence_to_direct_current_review; source_band_profiles_still_required' or not frame.get('homography_is_navigation_only') or frame.get('pixels_from_model'):raise ValueError('Current algorithmic context cannot claim direct human review or world-derived UV')
  refcap=frame['same_top_band_navigation_source']
  if file_sha256(refcap['path'])!=refcap['sha256'] or refcap.get('source_mean_transferred_as_FIT_UV') is not False:raise ValueError('Actual same-cap navigation source changed')
 instances=[dict(instance_id='actual_board/'+frame['shot_id']+'/'+str(frame['native_pts']),search_roi_xyxy=frame['board_instance_roi_xyxy'],blue_upper_visible=frame['top_blue_upper_visible'],yellow_lower_visible=frame['contact_yellow_lower_visible'],top_search=frame['top_blue_upper_search_y_range'],contact_search=frame['contact_yellow_lower_search_y_range'])]
 rows=[];certificates=[]
 for instance in instances:
  box=np.asarray(instance['search_roi_xyxy'],float)
  if box.shape!=(4,)or not np.isfinite(box).all():raise ValueError('Actual finite object ROI required')
  x0,y0,x1,y1=box.astype(int)
  if not(0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Board ROI outside source')
  localmask=np.ones_like(mask);localmask[y0:y1,x0:x1]=mask[y0:y1,x0:x1]
  source_rows,source_certs=source_outer_boundary_rows_untrusted_v2(rgb,localmask,known_board_scope=True)
  # A separate blue-filled advertisement is not another top strip. Only the
  # same actually observed coherent cap family may navigate missing columns.
  top_paths=[(i,np.asarray(c['current_color_run_samples'],float))for i,c in enumerate(source_certs)if c['material_role']=='board.top']
  top_paths=[(i,a[(a[:,2]>=instance['top_search'][0])&(a[:,2]<=instance['top_search'][1])])for i,a in top_paths]
  top_paths=[(i,a)for i,a in top_paths if len(a)>=4]
  raw_readout_hint=np.asarray(frame.get('source_top_readout_navigation_uv',[]),float).reshape(-1,2)
  transferred_hint=np.asarray(frame.get('source_top_band_navigation_hint_uv',[]),float).reshape(-1,2)
  if len(raw_readout_hint)>=1 and schema!='rinkfit.joint9.current_joint_model_board_navigation.v1':
   order=np.argsort(raw_readout_hint[:,0]);rh=raw_readout_hint[order]
   top_paths=[(i,a)for i,a in top_paths if np.median(a[:,1]-np.interp(a[:,0],rh[:,0],rh[:,1]))<=8. and np.median(a[:,1]-np.interp(a[:,0],rh[:,0],rh[:,1]))>=-48.]
  if len(transferred_hint)>=4:
   navfit=np.polyfit(transferred_hint[:,0],transferred_hint[:,1],1)
   top_paths=[(i,a)for i,a in top_paths if np.median(np.abs(a[:,1]-np.polyval(navfit,a[:,0])))<=12.]
  band_guide=None
  if top_paths:
   dominant_index,actual=max(top_paths,key=lambda v:np.ptp(v[1][:,0]));fit=np.polyfit(actual[:,0],actual[:,1],1);spread=float(np.quantile(np.abs(actual[:,1]-np.polyval(fit,actual[:,0])),.95));band_guide=dict(actual_source_certificate_index=dominant_index,actual_same_band_navigation_samples=actual[:,:2].tolist(),line_for_search_only=fit.tolist(),source_residual_q95_px=spread,search_half_width_px=max(8.,min(16.,3*spread+4.)),source_fit_pixels_never_output=True)
  used={}
  for r in source_rows:
   ci=r['source_definition_certificate_index'];cert=source_certs[ci];q=np.asarray(r['current_uv']);nav=np.asarray(r['navigation_seed_uv']);samples=cert['current_color_run_samples'];sample=min(samples,key=lambda p:abs(p[0]-nav[0]));x,seed,ba,bb,ya,yb,core,*_=sample
   # Both actually observed bands must belong to this same reviewed object,
   # not a spectator rail paired with a distant real yellow kickplate.
   if not(x0<=x<x1 and y0<=ba<=bb<ya<=core<y1):continue
   if not(instance['top_search'][0]<=ba<=instance['top_search'][1] and instance['contact_search'][0]<=core<=instance['contact_search'][1]):continue
   if not(x0<=q[0]<x1 and y0<=q[1]<y1):continue
   if schema in ('rinkfit.joint9.same_native_readout_local_board_navigation.v1','rinkfit.joint9.current_joint_model_board_navigation.v1'):
    colour=np.median(rgb[int(ba)+1:int(bb),max(0,int(x)-1):min(w,int(x)+2)],axis=(0,1)).astype(float)if bb-ba>=2 else np.zeros(3)
    R,G,B=colour
    if not(B-G>12. and B-R>20. and G-R>3.):continue
    yellow_colour=np.median(rgb[max(0,int(core)-1):min(h,int(core)+2),max(0,int(x)-1):min(w,int(x)+2)],axis=(0,1)).astype(float);YR,YG,YB=yellow_colour
    r['source_measurement_audit']=dict(r['source_measurement_audit'],actual_blue_band_core_RGB=colour.tolist(),actual_yellow_strip_core_RGB=yellow_colour.tolist(),warm_RGB_core_is_diagnostic_only=True,red_letter_exclusion_not_established_by_colour=True)
    if r['material_role']=='board.top'and len(raw_readout_hint)>=1 and schema!='rinkfit.joint9.current_joint_model_board_navigation.v1':
     order=np.argsort(raw_readout_hint[:,0]);rh=raw_readout_hint[order];navy=float(np.interp(q[0],rh[:,0],rh[:,1]))
     if not(-48.<=q[1]-navy<=8.):continue
   if schema=='rinkfit.joint9.current_joint_model_board_navigation.v1'and r['material_role']=='board.top'and not(instance['top_search'][0]<=q[1]<=instance['top_search'][1]):continue
   if r['material_role']=='board.top'and not instance.get('blue_upper_visible',False):continue
   if r['material_role']=='board.top':
    if band_guide is None:continue
    if abs(q[1]-np.polyval(band_guide['line_for_search_only'],q[0]))>band_guide['search_half_width_px']:continue
   if r['material_role']=='board.contact'and not instance.get('yellow_lower_visible',False):continue
   if r['material_role']=='board.contact'and not(instance['contact_search'][0]<=q[1]<=instance['contact_search'][1]):continue
   if ci not in used:
    used[ci]=len(certificates);certificates.append(dict(cert,current_board_instance_id=instance['instance_id'],actual_current_board_object_context=dict(source_review=ref,source_frame_RGB_sha256=context['actual_native_RGB_sha256'],native_pts=context['native_pts'],shot_id=context['shot_id'],search_roi_xyxy=box.tolist()),object_source_semantics='current actual board body between its own blue and yellow material bands; stands/rail/reflection outside reviewed instance',old_global_physical_pass_or_W_inherited=False,current_same_material_band_navigation=band_guide,configuration_local_scope=CONFIG,source_navigation_type=schema,model_navigation_search_policy='explicit current-exposure top/contact windows; current same-cap coherence and own blue/body/yellow evidence required; no inherited one-sided readout offset'if schema=='rinkfit.joint9.current_joint_model_board_navigation.v1'else'actual-source readout or independently reviewed instance navigation'))
   r=dict(r,source_definition_certificate_index=used[ci],actual_current_board_instance_id=instance['instance_id'],actual_current_board_context_sha256=object_sha256(context),source_object_scope_review=ref)
   r['source_measurement_audit']=dict(r['source_measurement_audit'],source_definition_certificate_index=used[ci],actual_current_board_object_context=certificates[used[ci]]['actual_current_board_object_context'],global_color_rule_used_as_object_identity=False)
   rows.append(r)
 # Object navigation and warm colour never sign a physical material role.
 # The caller must supply the actual decode receipt and a new immutable audit.
 if not context.get('native_video_decode_receipt'):
  for r in rows:r['source_measurement_audit']['physical_role_status']='unresolved_missing_actual_decode_requalification'
  return [],certificates
 from .board_source_requalification import reevaluate_current_board_source_rows
 candidates=[dict(r,source_row_id=f"board_candidate/{context['shot_id']}/{context['native_pts']}/{i}",
  native_pts=context['native_pts'],shot_id=context['shot_id'],measurement_operator_resolved=r['measurement_operator'])for i,r in enumerate(rows)]
 audit=reevaluate_current_board_source_rows(rgb,mask,candidates,context)
 eligible=[]
 for r,q in zip(rows,audit['physical_membership_rows']):
  r['source_material_role_requalification']=q
  r['source_measurement_audit']['physical_role_status']=q['status']
  if q['status']!='pass':continue
  r['source_measurement_audit'].update(actual_outer_material_order_verified=True,
   reflection_excluded=True,internal_interface_used=False,
   actual_material_requalification_recipe_sha256=audit['source_recipe_sha256'])
  eligible.append(r)
 return eligible,certificates
