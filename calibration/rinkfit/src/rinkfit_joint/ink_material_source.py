"""Current white-ink/purple-substrate edge with explicit finite glyph identity.

The caller supplies same physical finite component evidence. Pixel selection is
local and signed; an arbitrary gradient cannot establish the material identity.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from .outer_cubic_source import prepare_cubic_channels,replay_outer_cubic,CONFIG as POINT_CONFIG

CONFIG=dict(version='finite_white_ink_purple_substrate_signed_cubic_v1',
 fixed_channel=0,channel_definition='native RGB luminance after sigma0.8 Gaussian',
 profile_halfwidth_native_px=3.,tangent_offsets_native_px=[-.75,0.,.75],
 native_palette_samples=[1.25,1.75,2.25],
 minimum_current_luminance_contrast=12.,minimum_purple_red_minus_green=12.,
 maximum_white_chroma_absolute=30.,
 same_physical_finite_glyph_required=True,
 material_identity_not_proved_by_gradient=True,source_noise_quality_censoring=False)

def _sample(coeff,points):
 p=np.asarray(points,float);shape=p.shape[:-1];return np.stack([map_coordinates(c,[p[...,1].ravel(),p[...,0].ravel()],order=3,mode='nearest',prefilter=False).reshape(shape)for c in coeff],axis=-1)

def _palette(coeff,q,n):
 a=np.asarray(CONFIG['native_palette_samples']);xy=q+a[:,None]*n;yx=q-a[:,None]*n;v=_sample(coeff,np.stack([yx,xy]));s=np.median(v,axis=1);L,R=s;pol=1 if R[0]>L[0]else -1;white=R if pol>0 else L;purple=L if pol>0 else R
 okay=bool(white[0]-purple[0]>=12 and max(abs(white[1]),abs(white[2]))<=30 and purple[1]>=12 and purple[2]>=-2)
 return okay,pol,dict(before_channels=L.tolist(),after_channels=R.tolist(),white_side_normal_sign=pol,white_channels=white.tolist(),purple_channels=purple.tolist(),white_minus_purple_luminance=float(white[0]-purple[0]))

def read_white_purple_ink_edge(rgb,mask,seed_uv,normal,*,actual_finite_material_certificate=None,coefficients=None):
 if not actual_finite_material_certificate:return dict(current_uv=None,status='actual_finite_print_component_identity_certificate_required')
 rgb=np.asarray(rgb);mask=np.asarray(mask,bool);seed=np.asarray(seed_uv,float);n=np.asarray(normal,float);norm=float(np.linalg.norm(n))
 if not norm>1e-8:return dict(current_uv=None,status='source_navigation_tangent_not_defined')
 n=n/norm;t=np.array([-n[1],n[0]]);off=np.asarray(POINT_CONFIG['normal_offsets']);ts=np.asarray(CONFIG['tangent_offsets_native_px']);xy=seed+ts[:,None,None]*t+off[None,:,None]*n;h,w=mask.shape
 if np.any(xy[...,0]<1)or np.any(xy[...,0]>w-2)or np.any(xy[...,1]<1)or np.any(xy[...,1]>h-2):return dict(current_uv=None,status='source_profile_outside_native_image')
 ii=np.rint(xy).astype(int)
 if mask[ii[...,1],ii[...,0]].any():return dict(current_uv=None,status='source_profile_actor_or_graphics_excluded')
 coeff=prepare_cubic_channels(rgb)if coefficients is None else coefficients;okay,pol,palette=_palette(coeff,seed,n)
 if not okay:return dict(current_uv=None,status='source_navigation_window_does_not_establish_white_ink_purple_substrate',actual_palette=palette)
 frozen=dict(navigation_seed_uv=seed.tolist(),normal=n.tolist(),source_measurement_audit=dict(profile=dict(channel=0,polarity=pol)))
 result=replay_outer_cubic(rgb,frozen,coefficients=coeff)
 if not result['numeric_valid']or result['synthetic_window_boundary_count']:return dict(current_uv=None,status='actual_material_peak_numeric_or_window_pending',actual_point_reading=result)
 q=np.asarray(result['current_uv']);okay2,pol2,after=_palette(coeff,q,n)
 if not okay2 or pol2!=pol:return dict(current_uv=None,status='relocated_profile_not_same_ordered_white_purple_material',actual_palette=after)
 # Test local continuity using the same selected channel and signed transition.
 # No argmax over channels, polarity, target identity or whole-frame objects.
 shifts=[]
 for dt in ts:
  f={**frozen,'navigation_seed_uv':(seed+dt*t).tolist()};a=replay_outer_cubic(rgb,f,coefficients=coeff)
  if not a['numeric_valid']or a['synthetic_window_boundary_count']:return dict(current_uv=None,status='same_finite_material_tangent_profile_window_pending')
  shifts.append(a['normal_offset_px'])
 if np.ptp(shifts)>1.5:return dict(current_uv=None,status='same_finite_material_current_local_continuity_pending')
 profile=_sample(coeff,xy)
 return dict(current_uv=q.tolist(),normal=n.tolist(),navigation_seed_uv=seed.tolist(),status='current_native_finite_white_purple_ink_boundary',source_material_receipt=dict(operator_version=CONFIG['version'],configuration=CONFIG,frozen_point_operator=frozen,actual_point_reading=result,actual_material_palette=after,initial_material_palette=palette,actual_tangent_offsets_native_px=ts.tolist(),actual_profile_normal_offsets=off.tolist(),actual_profile_channels=profile.tolist(),actual_tangent_normal_offsets_px=shifts,actual_finite_material_certificate=actual_finite_material_certificate,source_mean_from_model_projection=False,measurement_operator_material_identity_reselected_in_replay=False,whole_exposure_RGB_replay_domain=True))
