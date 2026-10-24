"""Signed current material readout of an independently identified finite target layer.

No radius ratios or world geometry select an interface. The caller supplies an
actual target-object/layer certificate; world projection is a search proposal.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from .outer_cubic_source import prepare_cubic_channels,replay_outer_cubic

CONFIG=dict(version='actual_declared_target_layer_signed_cubic_v1',normal_navigation_halfwidth_px=20.,normal_step_px=.25,tangent_offsets_px=[-2.,-1.,0.,1.,2.],same_material_minimum_signed_contrast=4.,fixed_luminance_channel=0,source_mask_required=True,profile_operator_halfwidth_px=3.,old_world_radius_or_ratio_used_for_pixel_measurement=False)

def unique_source_interface_candidate(candidates):
 """Select only a unique already-qualified native interface.

 The input has already passed the caller's unchanged RGB, signed-polarity,
 support and exclusion gates. Neither proximity to a navigation curve nor
 a smooth sequence can turn several legal interfaces into one layer identity.
 This function establishes uniqueness only; it does not certify the named
 physical ring layer. All failed candidates remain available to the caller.
 """
 count=len(candidates)
 if count==1:
  return candidates[0],dict(status='one_legal_native_interface',
      actual_interface_candidates=1,nearest_navigation_used_to_resolve_identity=False)
 return None,dict(status='current_declared_target_layer_profile_ambiguous_or_not_visible',
     actual_interface_candidates=count,qualification_issued=False,
     source_identity_decision='unknown',nearest_navigation_used_to_resolve_identity=False)


def qualify_existing_target_interfaces(rgb,mask,records,*,binding):
 """Recheck retained native ring rows without changing UV or source identity.

 Used by the event's actual final-source issuer. This reuses the original
 bilinear q50 operator and its existing local gates, rather than substituting
 the separate cubic reader's pixels. Missing source queries stay unknown.
 No world model, projected ring or historical parent file is read here.
 """
 import re
 from rinkspace.vision_programme_ice_review import _batched_actual_profiles,_actual_profile_candidates
 from rinkspace.vision_source_shapes import _MaterialProfileFrame
 image=np.asarray(rgb);excluded=np.asarray(mask,bool)
 if image.ndim!=3 or image.shape[2]!=3 or excluded.shape!=image.shape[:2]:
  raise ValueError('Same-frame native RGB and exclusion mask required')
 result=[None]*len(records);selected=[];queries=[];normals=[];intervals=[]
 h,w=excluded.shape
 for index,row in enumerate(records):
  key=row.get('curve_id',row.get('feature_id','').removeprefix('ice.'))
  if not re.fullmatch(r'[FR][1-4]:(outer|middle|inner|center)',key):continue
  if row.get('user_confirmed')is True and row.get('source_kind')=='preserved_user_manual_native_image_click':
   result[index]=dict(status='preserved_explicit_manual_layer_reading',source_uv_modified=False)
   continue
  ps=row.get('pixel_support',{});normal=ps.get('normal',ps.get('profile_normal'))
  shift=ps.get('native_shift_px');transition=ps.get('material_transition')
  if normal is None or shift is None or transition not in ('light_to_dark','dark_to_light'):
   result[index]=dict(status='original_ring_query_or_signed_layer_definition_missing',
       source_identity_decision='unknown',qualification_issued=False,source_uv_modified=False)
   continue
  u=np.asarray(row['uv'],float);normal=np.asarray(normal,float)
  if u.shape!=(2,)or normal.shape!=(2,)or not np.isfinite(np.r_[u,normal,shift]).all()or abs(np.linalg.norm(normal)-1)>1e-6:
   result[index]=dict(status='original_ring_query_nonfinite_or_invalid',
       source_identity_decision='unknown',qualification_issued=False,source_uv_modified=False)
   continue
  query=u-float(shift)*normal;interval=np.array([-36.,36.])
  for axis,limit in ((0,w-1),(1,h-1)):
   if abs(normal[axis])>1e-12:
    first,second=sorted(((4-query[axis])/normal[axis],(limit-4-query[axis])/normal[axis]))
    interval[0]=max(interval[0],first);interval[1]=min(interval[1],second)
  qi=np.rint(query).astype(int)
  if interval[1]-interval[0]<20 or(qi<0).any()or(qi>=[w,h]).any()or excluded[qi[1],qi[0]]:
   result[index]=dict(status='original_ring_query_support_not_currently_visible',
       source_identity_decision='unknown',qualification_issued=False,source_uv_modified=False)
   continue
  selected.append(index);queries.append(query);normals.append(normal);intervals.append(interval)
 if not selected:return result
 query=np.asarray(queries);normal=np.asarray(normals);interval=np.asarray(intervals)
 frame=_MaterialProfileFrame(image,binding)
 off,profiles,smooth,valid=_batched_actual_profiles(frame,query,normal,interval,np.array(['dark']*len(selected)))
 for local,index in enumerate(selected):
  row=records[index];ps=row['pixel_support'];good=valid[local];a=off[local,good];pr=profiles[local,good];sm=smooth[local,good];q=query[local];n=normal[local]
  values=_actual_profile_candidates(q,n,a,pr,sm,operator='material_interface',signal='dark',transition=ps['material_transition'],with_quantiles=True)
  legal=[]
  for uv,edges,bad,quantiles in values:
   if bad:continue
   tangent=np.array([-n[1],n[0]]);pieces=[uv[None],edges,quantiles.reshape(-1,2)]
   for point in [uv,*edges]:pieces.append(point+np.arange(-3,4)[:,None]*tangent)
   pixels=np.rint(np.vstack(pieces)).astype(int)
   if(pixels<4).any()or(pixels>[w-5,h-5]).any()or excluded[pixels[:,1],pixels[:,0]].any():continue
   support=(quantiles-q)@n;contrast=[abs(float(np.interp(s[2],a,sm)-np.interp(s[0],a,sm)))for s in support]
   iqr=float(np.max(abs(support[:,2]-support[:,0])));shift=float((uv-q)@n)
   if min(contrast)<1.5 or iqr>22 or abs(shift)>22:continue
   legal.append(dict(uv=uv.tolist(),shift_px=shift,
       minimum_signed_interface_IQR_contrast=min(contrast),interface_IQR_px=iqr))
  chosen,verdict=unique_source_interface_candidate(legal)
  verdict.update(legal_native_candidates=legal,source_uv_modified=False,
      source_row_id=row.get('source_row_id'),current_RGB_and_exclusion_used=True)
  if chosen is not None:
   difference=float(np.linalg.norm(np.asarray(chosen['uv'])-np.asarray(row['uv'])))
   verdict['retained_original_uv_difference_px']=difference
   if difference>128*np.finfo(float).eps*max(1.,float(np.linalg.norm(row['uv']))):
    verdict.update(status='unique_current_interface_differs_from_preserved_source_uv',
        source_identity_decision='unknown',qualification_issued=False)
   else:
    # A unique signed edge proves a local measurement, not which nested target
    # layer supplied it. This API has no independent object/layer authority.
    # Keep the measured candidate available while withholding named fitting.
    verdict.update(status='unique_native_interface_named_layer_not_independently_confirmed',
        local_interface_status='one_legal_native_interface',
        physical_layer_identity_established=False,
        source_identity_decision='unknown',qualification_issued=False)
  result[index]=verdict
 return result


def read_target_interface(rgb,mask,seed_uv,normal,polarity,*,coefficients=None,layer_role=None,actual_object_layer_certificate=None):
 if not actual_object_layer_certificate:return dict(current_uv=None,status='actual_target_object_and_layer_identity_certificate_required')
 seed=np.asarray(seed_uv,float);n=np.asarray(normal,float);mask=np.asarray(mask,bool);n=n/np.linalg.norm(n);t=np.array([-n[1],n[0]]);off=np.arange(-20.,20.0001,.25);toff=np.array(CONFIG['tangent_offsets_px']);xy=seed+toff[:,None,None]*t+off[None,:,None]*n;h,w=mask.shape
 if np.any(xy[:,:,0]<1)or np.any(xy[:,:,0]>w-2)or np.any(xy[:,:,1]<1)or np.any(xy[:,:,1]>h-2):return dict(current_uv=None,status='current_target_search_outside_native_image')
 qi=np.rint(xy).astype(int)
 if mask[qi[:,:,1],qi[:,:,0]].any():return dict(current_uv=None,status='actual_target_profile_actor_graphics_or_other_source_exclusion')
 coeff=prepare_cubic_channels(rgb)if coefficients is None else coefficients
 values=np.stack([map_coordinates(c,[xy[:,:,1],xy[:,:,0]],order=3,mode='nearest',prefilter=False)for c in coeff],axis=-1);g=int(polarity)*np.gradient(values[2,:,0],.25);peaks,_=find_peaks(g,height=.8,prominence=.6);candidates=[]
 for k in peaks:
  if k<25 or k>len(off)-26:continue
  before=values[:,k-24:k-8];after=values[:,k+8:k+24];delta=np.median(after,axis=1)-np.median(before,axis=1);contrast=int(polarity)*delta[:,0]
  if np.min(contrast)<4:continue
  if layer_role in ('outer','middle')and np.any(np.linalg.norm(delta[:,1:],axis=1)>np.maximum(8.,.6*contrast)):continue
  found=[];okay=True
  for line in values:
   gg=int(polarity)*np.gradient(line[:,0],.25);lo=max(1,k-6);hi=min(len(off)-1,k+7);pp,_=find_peaks(gg[lo:hi],height=.8,prominence=.25)
   if len(pp)!=1:okay=False;break
   found.append(float(off[lo+int(pp[0])]))
  if not okay or np.ptp(found)>1.5:continue
  frozen=dict(navigation_seed_uv=(seed+n*float(off[k])).tolist(),normal=n.tolist(),source_measurement_audit=dict(profile=dict(channel=0,polarity=int(polarity))))
  measured=replay_outer_cubic(rgb,frozen,coefficients=coeff)
  if not measured['numeric_valid']or measured['synthetic_window_boundary_count']:continue
  current=np.asarray(measured['current_uv']);idx=np.rint(current).astype(int)
  if not(0<=idx[0]<w and 0<=idx[1]<h)or mask[idx[1],idx[0]]:continue
  candidates.append(dict(current_uv=current.tolist(),normal=n.tolist(),navigation_seed_uv=seed.tolist(),status='current_native_declared_target_material_interface',source_material_receipt=dict(operator_version=CONFIG['version'],configuration=CONFIG,layer_role=layer_role,expected_outward_luminance_polarity=int(polarity),frozen_point_operator=frozen,actual_point_reading=measured,original_navigation_seed_uv=seed.tolist(),actual_five_profile_offsets=off.tolist(),actual_tangent_offsets=toff.tolist(),actual_five_profile_channels=values.tolist(),actual_signed_material_contrast=contrast.tolist(),actual_target_object_layer_certificate=actual_object_layer_certificate,finite_material_identity_proved_by_peak_search=False,whole_exposure_RGB_replay_domain=True,channel_and_polarity_frozen_for_all_replays=True,source_mean_UV_from_world_projection=False)))
 selected,selection=unique_source_interface_candidate(candidates)
 if selected is None:return dict(current_uv=None,**selection)
 return selected
