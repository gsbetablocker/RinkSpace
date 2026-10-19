"""Signed current material readout of an independently identified finite target layer.

No radius ratios or world geometry select an interface. The caller supplies an
actual target-object/layer certificate; world projection is a search proposal.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from .outer_cubic_source import prepare_cubic_channels,replay_outer_cubic

CONFIG=dict(version='actual_declared_target_layer_signed_cubic_v1',normal_navigation_halfwidth_px=20.,normal_step_px=.25,tangent_offsets_px=[-2.,-1.,0.,1.,2.],same_material_minimum_signed_contrast=4.,fixed_luminance_channel=0,source_mask_required=True,profile_operator_halfwidth_px=3.,old_world_radius_or_ratio_used_for_pixel_measurement=False)

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
 if len(candidates)!=1:return dict(current_uv=None,status='current_declared_target_layer_profile_ambiguous_or_not_visible',actual_interface_candidates=len(candidates))
 return candidates[0]
