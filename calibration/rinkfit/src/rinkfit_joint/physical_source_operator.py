"""Semantic current-source readers; the historic abs-gradient reader is excluded.

Material selection is performed once on the actual exposure and recorded. A
replay freezes channel, polarity and windows, so RGB perturbations cannot swap
to an adjacent material. Source identity remains a separate finite-topology test.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import map_coordinates,spline_filter
from .source_worker import _channels
from .measurement_contract import EventMeasurementContract
from .board_outer_source import source_outer_boundary_rows
from .io_utils import object_sha256

CONFIG=dict(version='one_actual_stroke_two_ordered_material_flanks_v1',
 normal_search_px=24.,normal_step_px=.25,tangent_offsets_px=[-2.,-1.,0.,1.,2.],
 minimum_material_contrast_8bit=4.,source_mask_required=True,
 adjacent_line_or_crossing_is_not_a_width_flank=True,independent_channel_selection_per_flank=False)

def _sample(channels,uv):
 q=np.asarray(uv).reshape(-1,2)
 return np.stack([map_coordinates(c,[q[:,1],q[:,0]],order=3,mode='nearest',prefilter=False) for c in channels],axis=1)

def _fail(reason,**kw):return dict(current_uv=None,status=reason,configuration=CONFIG,**kw)

def measure_paint_midline(rgb,mask,seed_uv,normal,*,confirmed_signal=None,channels=None,spline_channels=None):
 """Acquire two flanks of one visible stroke; never relabel one ring interface.

 A source-family caller must additionally prove the finite curve's physical
 identity. This function proves only exposure-local material/width topology.
 """
 rgb=np.asarray(rgb);mask=np.asarray(mask,dtype=bool);seed=np.asarray(seed_uv,float);n=np.asarray(normal,float)
 if rgb.shape[:2]!=mask.shape or n.shape!=(2,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-8:return _fail('actual_source_normal_or_mask_missing')
 n=n/np.linalg.norm(n);t=np.array([-n[1],n[0]]);c=_channels(rgb) if channels is None else channels
 coeff=np.asarray([spline_filter(q,order=3) for q in c]) if spline_channels is None else spline_channels
 off=np.arange(-24.,24.0001,.25);toff=np.asarray(CONFIG['tangent_offsets_px']);coords=seed+toff[:,None,None]*t+off[None,:,None]*n
 h,w=mask.shape
 if np.any(coords[:,:,0]<1) or np.any(coords[:,:,0]>w-2) or np.any(coords[:,:,1]<1) or np.any(coords[:,:,1]>h-2):return _fail('two_flank_source_search_outside_native_image',normal=n.tolist())
 nearest=np.rint(coords).astype(int)
 if mask[nearest[:,:,1],nearest[:,:,0]].any():return _fail('two_flank_source_search_actor_graphics_or_excluded_region',normal=n.tolist())
 values=_sample(coeff,coords.reshape(-1,2)).reshape(5,len(off),3);central=values[2];candidates=[]
 choices=[('red',1,1),('blue',2,1),('dark',0,-1)]
 if confirmed_signal is not None:choices=[v for v in choices if v[0]==confirmed_signal]
 for signal,ch,sign in choices:
  y=sign*central[:,ch];g=np.gradient(y,.25);a,_=find_peaks(g,height=.75,prominence=.5);b,_=find_peaks(-g,height=.75,prominence=.5)
  for ka in a:
   for kb in b:
    if not ka+4<=kb or off[ka]>3 or off[kb]<-3:continue
    interior=y[ka+1:kb];left=y[max(0,ka-12):max(1,ka-3)];right=y[min(len(y)-1,kb+3):min(len(y),kb+12)]
    if min(len(interior),len(left),len(right))<2:continue
    inside=float(np.median(interior));outside=np.array([np.median(left),np.median(right)]);contrast=inside-outside
    if np.min(contrast)<4 or abs(outside[0]-outside[1])>.5*float(np.min(contrast)):continue
    # A red/blue material must have the corresponding actual interior chroma;
    # a neutral dark stroke cannot use a coloured ring boundary as its side.
    region=central[ka+1:kb];ends=np.vstack((central[max(0,ka-12):max(1,ka-3)],central[min(len(y)-1,kb+3):min(len(y),kb+12)]))
    if signal=='red' and not(np.median(region[:,1])>4 and np.median(region[:,1])-np.median(ends[:,1])>4):continue
    if signal=='blue' and not(np.median(region[:,2])>4 and np.median(region[:,2])-np.median(ends[:,2])>4):continue
    if signal=='dark' and np.linalg.norm(np.median(region[:,1:],axis=0)-np.median(ends[:,1:],axis=0))>.4*float(np.min(contrast)):continue
    widths=[];centers=[];windows=[];ok=True
    # Five real tangent profiles must see the same ordered transitions. A
    # crossing, adjacent stroke, hole, or layer change does not form this pair.
    for line in values:
     gg=np.gradient(sign*line[:,ch],.25);positions=[];one=[]
     for k,pol in ((ka,1),(kb,-1)):
      lo=max(1,k-6);hi=min(len(off)-1,k+7);ix=np.arange(lo,hi);v=pol*gg[ix];peaks,_=find_peaks(v,height=.75,prominence=.25)
      if len(peaks)!=1:ok=False;break
      kk=int(ix[peaks[0]]);vv=pol*gg[kk-1:kk+2];den=vv[0]-2*vv[1]+vv[2];shift=0. if abs(den)<1e-10 else float(np.clip(.5*(vv[0]-vv[2])/den,-1,1))*.25
      positions.append(float(off[kk]+shift));one.append(dict(normal_window_offsets=[float(off[lo]),float(off[hi-1])],channel=ch,polarity=int(sign*pol),peak_index=kk))
     if not ok:break
     widths.append(positions[1]-positions[0]);centers.append(.5*sum(positions));windows.append(one)
    if not ok or min(widths)<=.5 or np.ptp(widths)>max(1.,.3*np.median(widths)) or np.ptp(centers)>2.:continue
    # Both same-material plateaux and the ordering are recorded for replay.
    candidates.append(dict(signal=signal,channel=ch,flank_polarity=[sign,-sign],offsets=[float(off[ka]),float(off[kb])],current_center_offset=centers[2],widths=widths,centers=centers,windows=windows,inside=inside,outside=outside.tolist(),values=values[:,:,ch]))
 if len(candidates)!=1:return _fail('no_unique_same_paint_two_flank_material_pair' if not candidates else 'multiple_current_paint_material_pairs_ambiguous',normal=n.tolist(),candidate_count=len(candidates),confirmed_material_signal=confirmed_signal)
 a=candidates[0];center=seed+a['current_center_offset']*n;flanks=[seed+(a['current_center_offset']+q*.5*a['widths'][2])*n for q in (-1,1)]
 receipt=dict(operator_version=CONFIG['version'],configuration=CONFIG,original_navigation_seed_uv=seed.tolist(),measured_source_mean=center.tolist(),normal=n.tolist(),tangent=t.tolist(),material_signal=a['signal'],channel=a['channel'],flank_polarity=a['flank_polarity'],frozen_flank_windows=a['windows'],normal_offsets=off.tolist(),tangent_offsets=toff.tolist(),actual_five_profiles=a['values'].tolist(),actual_same_paint_plateau=a['inside'],actual_outside_plateaux=a['outside'],same_material_both_flanks=True,opposite_ordered_transitions=True,material_channel_frozen_for_replay=True,raw_source_uv_changed=False,finite_material_identity_proved_by_this_reader=False,local_signal_bbox_xyxy=[float(coords[:,:,0].min()-4),float(coords[:,:,1].min()-4),float(coords[:,:,0].max()+4),float(coords[:,:,1].max()+4)],exact_replay_pixel_domain='whole same-exposure RGB; cubic spline prefilter has nonlocal support; local bbox is not a claim of independent pixels')
 return dict(current_uv=center.tolist(),flank_uv=[q.tolist() for q in flanks],normal=n.tolist(),width_px=float(a['widths'][2]),status='current_native_same_paint_two_ordered_flanks_midline',source_material_receipt=receipt,moved_px=float(np.linalg.norm(center-seed)))

def replay_paint_midline(rgb,receipt):
 """Float-RGB replay of the acquired material, with no material reselection.

 Source-quality conditions are never applied to synthetic trials. Window
 boundary/domain changes are returned for the statistician to diagnose.
 """
 if receipt.get('operator_version')!=CONFIG['version']:raise ValueError('Wrong frozen paint operator version')
 seed=np.asarray(receipt['original_navigation_seed_uv'],float);n=np.asarray(receipt['normal'],float);t=np.asarray(receipt['tangent'],float);off=np.asarray(receipt['normal_offsets'],float);toff=np.asarray(receipt['tangent_offsets'],float);ch=int(receipt['channel']);c=_channels(np.asarray(rgb));coeff=np.asarray([spline_filter(q,order=3) for q in c]);coords=seed+toff[:,None,None]*t+off[None,:,None]*n;values=_sample(coeff,coords.reshape(-1,2)).reshape(len(toff),len(off),3);positions=[];boundary=0
 for side,window in enumerate(receipt['frozen_flank_windows'][2]):
  lo,hi=window['normal_window_offsets'];ix=np.flatnonzero((off>=lo-1e-8)&(off<=hi+1e-8));g=int(window['polarity'])*np.gradient(values[2,:,ch],.25);k=int(ix[np.argmax(g[ix])]);boundary+=int(k in (ix[0],ix[-1]));shift=0.
  if 0<k<len(off)-1:
   v=g[k-1:k+2];den=v[0]-2*v[1]+v[2]
   if abs(den)>1e-10:shift=float(np.clip(.5*(v[0]-v[2])/den,-1,1))*.25
  positions.append(float(off[k]+shift))
 uv=seed+.5*sum(positions)*n
 return dict(current_uv=uv.tolist(),numeric_valid=bool(np.isfinite(uv).all()),synthetic_window_boundary_count=boundary,material_reselected=False,source_quality_censoring_applied=False)

def prepare_current_source(rgb,mask,project):
 """One exposure's immutable photometry; no cross-exposure or model source."""
 rgb=np.asarray(rgb);mask=np.asarray(mask,dtype=bool);ch=_channels(rgb)
 contract=EventMeasurementContract(project,'ankara_2026')
 outer,cert=source_outer_boundary_rows(rgb,mask)
 return dict(rgb=rgb,mask=mask,channels=ch,spline_channels=np.asarray([spline_filter(q,order=3) for q in ch]),contract=contract,outer_rows=outer,outer_certificates=cert)

def remeasure_physical_source(context,row,seed_uv,normal=None,flanks=None,*,frozen_operator=None):
 """Only declared physical operators; absent material proof returns pending.

 ``flanks`` is navigation metadata, never proof that they bound one stroke.
 Existing typed names, old peak channels or old W cannot bypass acquisition.
 """
 contract=context['contract'];expected=contract.expected(row.get('target_id'),row.get('category'))
 if expected is None or expected.get('operator') is None:return _fail('physical_material_definition_unclosed_against_existing_user_facts')
 op=expected['operator'];seed=np.asarray(seed_uv,float)
 if op=='paint_center':
  r=replay_paint_midline(context['rgb'],frozen_operator) if frozen_operator is not None else measure_paint_midline(context['rgb'],context['mask'],seed,normal,confirmed_signal=expected.get('signal'),channels=context['channels'],spline_channels=context['spline_channels'])
 elif op in ('board_upper_material_edge','ice_board_interface'):
  role='board.top' if op=='board_upper_material_edge' else 'board.contact'
  candidates=[q for q in context['outer_rows'] if q['material_role']==role and np.linalg.norm(np.asarray(q['current_uv'])-seed)<=24]
  if not candidates:return _fail('actual_user_defined_outer_board_boundary_not_visible_near_navigation_seed')
  q=min(candidates,key=lambda q:np.linalg.norm(np.asarray(q['current_uv'])-seed));r=dict(q['source_measurement_audit']);r['source_material_receipt']=context['outer_certificates'][q['source_definition_certificate_index']]
 elif op=='physical_corner':
  return _fail('fixed_entity_requires_same_actual_two_material_edges_and_frozen_corner_source_operator')
 else:
  # These interfaces need their actual current finite contour/layer material
  # certificate. The former strongest-absolute-gradient reader is not a fallback.
  return _fail('declared_finite_material_layer_or_print_contour_certificate_not_yet_rebuilt',expected_operator=op)
 r['measurement_operator']=op;r['measurement_contract_sha256']=contract.digest;r['physical_role']=expected['physical_role']
 r['source_local_operator_passed']=r.get('current_uv') is not None
 r['finite_physical_identity_and_source_artifacts_still_required']=True
 return r
