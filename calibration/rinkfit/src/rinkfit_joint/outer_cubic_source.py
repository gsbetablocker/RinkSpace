"""Continuous cubic localization of an already selected actual outer material.

The current-RGB blue-upper/yellow-lower selection and source role are inherited
unchanged. This changes only the frozen within-material localization operator.
No detection, channel/polarity choice or geometry projection is repeated.
"""
import numpy as np
from scipy.ndimage import spline_filter,map_coordinates
from scipy.interpolate import CubicSpline
from .source_worker import _channels

CONFIG=dict(version='fixed_outer_material_continuous_cubic_peak_v1',
 gaussian_sigma_px=.8,gaussian_boundary='reflect',channel=2,
 channel_definition='B-(R+G)/2',interpolation_order=3,
 normal_offsets=np.linspace(-3.,3.,49).tolist(),
 location='maximum signed derivative of cubic intensity spline; continuous stationary roots and endpoints',
 exact_flat_maximum_convention='middle of maximal flat interval, recorded as nonregular',
 material_channel_polarity_and_search_seed_frozen=True,
 synthetic_source_quality_censoring=False,source_identity_reselected=False,
 calibrated_sensor_noise_claimed=False)

def prepare_cubic_channels(rgb):
 channels=_channels(np.asarray(rgb));return np.asarray([spline_filter(c,order=3) for c in channels])

def replay_outer_cubic(rgb,receipt,*,coefficients=None):
 """Pure float-RGB replay; no source-quality gate or material reselection."""
 coeff=prepare_cubic_channels(rgb) if coefficients is None else coefficients
 seed=np.asarray(receipt['navigation_seed_uv'],float);n=np.asarray(receipt['normal'],float);profile=receipt['source_measurement_audit']['profile'];channel=int(profile['channel']);pol=int(profile['polarity']);off=np.asarray(CONFIG['normal_offsets']);uv=seed+off[:,None]*n
 intensity=map_coordinates(coeff[channel],[uv[:,1],uv[:,0]],order=3,mode='nearest',prefilter=False);spline=CubicSpline(off,intensity);d=spline.derivative();dd=d.derivative();roots=dd.roots(extrapolate=False);roots=roots[np.isfinite(roots)&(roots>off[0])&(roots<off[-1])];candidates=np.unique(np.r_[off[[0,-1]],roots]);scores=pol*d(candidates);best=float(np.max(scores));machine=64*np.finfo(float).eps*max(1.,abs(best));plateaux=[]
 # Exact constant derivative intervals have no unique peak. Their explicit
 # half-interval convention is preserved for nonlinear noise diagnostics.
 for j in range(len(off)-1):
  if abs(spline.c[0,j])<=machine and abs(spline.c[1,j])<=machine and abs(pol*spline.c[2,j]-best)<=machine:plateaux.append([float(off[j]),float(off[j+1])])
 maximal=candidates[np.abs(scores-best)<=machine]
 if plateaux:
  a=min(v[0] for v in plateaux);b=max(v[1] for v in plateaux);shift=.5*(a+b)
 else:shift=float(np.mean(maximal))
 point=seed+shift*n;boundary=bool(abs(shift-off[0])<=1e-9 or abs(shift-off[-1])<=1e-9)
 return dict(current_uv=point.tolist(),normal=n.tolist(),numeric_valid=bool(np.isfinite(point).all()),normal_offset_px=float(shift),signed_peak_gradient=best,synthetic_window_boundary_count=int(boundary),flat_peak_intervals=plateaux,maximal_stationary_roots=maximal.tolist(),actual_profile_normal_offsets=off.tolist(),actual_profile_intensity=intensity.tolist(),material_reselected=False,source_quality_censoring_applied=False,operator_version=CONFIG['version'])

def acquire_outer_cubic(rgb,row,*,coefficients=None):
 """Requalify the same actual colour edge at the new continuous location."""
 coeff=prepare_cubic_channels(rgb) if coefficients is None else coefficients;r=replay_outer_cubic(rgb,row,coefficients=coeff);q=np.asarray(r['current_uv']);n=np.asarray(row['normal']);xy=q+np.array([-5.,-3.,3.,5.])[:,None]*n;v=map_coordinates(coeff[2],[xy[:,1],xy[:,0]],order=3,mode='nearest',prefilter=False);a=float(v[:2].mean());b=float(v[2:].mean());role=row.get('material_role') or row['target_id'];correct=(b>8 and b-a>6) if role=='board.top' else (a<-8 and b-a>6 and abs(b)<.6*abs(a));valid=bool(r['numeric_valid'] and not r['synthetic_window_boundary_count'] and correct)
 return {**r,'base_source_material_qualified':valid,'actual_before_after_material_chroma':[a,b],'configuration':CONFIG,'same_actual_user_outer_role_retained':correct,'status':'current_native_user_outer_continuous_cubic_material' if valid else 'current_outer_cubic_material_or_window_source_pending'}
