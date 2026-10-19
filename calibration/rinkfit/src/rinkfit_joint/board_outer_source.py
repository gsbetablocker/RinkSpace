"""One exposure-local reader of blue UPPER / yellow LOWER actual exteriors."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from .source_worker import _channels
from .measurement_contract import EventMeasurementContract
from .outer_cubic_source import prepare_cubic_channels,replay_outer_cubic,CONFIG as CUBIC_CONFIG
CONFIG=dict(version='actual_blue_upper_yellow_lower_v2_independent_visibility',column_step=8,minimum_contiguous_columns=16,maximum_adjacent_change_px=12,blue_chroma_min=8.,yellow_chroma_max=-8.,plateau_contrast_min=6.,background_brightness_difference_is_not_a_user_fact=True,bright_background_additional_topology='thin coherent highest blue band with the same observed yellow strip/body',external_background_darker_than_body_not_required=True,interpolation='continuous_cubic',reflections_are_not_boundaries=True,model_predictions_used=False,internal_band_body_interfaces_used_as_measurements=False,independent_top_and_contact_visibility=True)
@lru_cache(maxsize=1)
def _contract():return EventMeasurementContract(Path(__file__).resolve().parents[4],'ankara_2026')
def _runs(a):
 d=np.diff(np.r_[False,a,False].astype(np.int8));return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)-1))
def _paths(columns,*,known_board_scope=False):
 active=[];complete=[]
 for x,values in columns:
  used=set();nxt=[]
  for old in active:
   options=[(abs(v[0]-old[-1][1]),j,v) for j,v in enumerate(values) if j not in used and abs(v[0]-old[-1][1])<=12]
   if options:
    _,j,v=min(options,key=lambda q:q[0]);used.add(j);nxt.append(old+[(x,*v)])
   elif len(old)>=16:complete.append(old)
  nxt.extend([[(x,*v)] for j,v in enumerate(values) if j not in used]);active=nxt
 complete.extend(a for a in active if len(a)>=16);kept=[]
 for path in sorted(complete,key=lambda q:np.median([v[1] for v in q])):
  xs={v[0] for v in path}
  if any(len(xs&{v[0] for v in old})>=.8*len(xs) for old in kept):continue
  # In a bright external background the actual upper band must itself be
  # narrow and coherent. Brightness is recorded, never declared user truth.
  bright=[v for v in path if v[7]>v[8]-12]
  if not known_board_scope and bright and (len(path)<24 or np.median([v[3]-v[2]+1 for v in bright])>16):continue
  kept.append(path)
 return kept
def source_outer_boundary_rows_untrusted_v2(rgb,mask,*,known_board_scope=False):
 rgb=np.asarray(rgb);mask=np.asarray(mask,bool)
 if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.shape[:2]!=mask.shape:raise ValueError('Current actual RGB and same-exposure body mask required')
 contract=_contract();channels=_channels(rgb);blue,luma=channels[2],channels[0];h,w=mask.shape;topcols=[];lowercols=[]
 for x in range(8,w-8,8):
  b=np.median(blue[:,x-1:x+2],axis=1);l=np.median(luma[:,x-1:x+2],axis=1);g=np.gradient(b);br=[(a,z) for a,z in _runs(b>8) if 2<=z-a+1<=48 and 12<a<z<h-12];yr=[(a,z) for a,z in _runs(b<-8) if 2<=z-a+1<=(128 if known_board_scope else 48) and 12<a<z<h-12];tops=[];bottoms=[]
  for ba,bb in br:
   for ya,yb in yr:
    if not 24<=ya-bb<=600:continue
    body=b[bb+4:ya-4];light=l[bb+4:ya-4]
    if len(body)<10 or np.median(light)<=60:continue
    above=b[ba-7:ba-2];inside=b[ba+1:min(ba+5,bb+1)];yellow=b[ya:yb+1]
    if not len(inside) or not len(yellow) or np.median(inside)-np.median(above)<6 or abs(np.median(above))>20:continue
    background=float(np.median(l[ba-7:ba-2]));bodylight=float(np.median(light));peaks,_=find_peaks(g[max(0,ba-3):min(bb+1,ba+8)],height=.75,prominence=.5)
    if not len(peaks):continue
    upper=max(0,ba-3)+int(peaks[0]);core=ya+int(np.argmin(yellow));meta=(ba,bb,ya,yb,core,background,bodylight)
    # Visibility is evaluated separately: a skater hiding the lower edge does
    # not remove the genuinely visible blue exterior.
    if not mask[ba-7:bb+8,x-2:x+3].any():tops.append((float(upper),*meta))
    if mask[max(0,core-3):min(h,yb+8),x-2:x+3].any():continue
    peaks,_=find_peaks(g[core:yb+3],height=.75,prominence=.5)
    if not len(peaks):continue
    lower=core+int(peaks[0]);main=float(g[lower]);later=g[lower+4:yb+4]
    if later.size and float(np.max(later))>=main*.8:continue
    # Thin strips have an observed core, even if there are no two pixels at
    # arbitrary offsets -5/-3. Use that actual core and the first neutral ice.
    before=float(np.min(b[max(ya,core-1):min(yb+1,core+2)]));after_values=b[lower+2:lower+5]
    if not len(after_values):continue
    after=float(np.median(after_values))
    if before>=-8 or after-before<6 or abs(after)>.6*abs(before) or np.median(l[lower+2:lower+5])<30:continue
    bottoms.append((float(lower),*meta))
  topcols.append((x,tops));lowercols.append((x,bottoms))
 refs={k:v for k,v in contract.references.items() if k in ('user_board_edge_semantics_20261010.json','ice_contact_material_roles_user_confirmed_20261006.json','visual_boundary_roles_user_confirmed_20261006.json')};coeff=prepare_cubic_channels(rgb);rows=[];certs=[]
 for role,operator,columns in [('board.top','board_upper_material_edge',topcols),('board.contact','ice_board_interface',lowercols)]:
  for path in _paths(columns,known_board_scope=known_board_scope):
   ci=len(certs);cert=dict(path_index=ci,material_role=role,measurement_contract_sha256=contract.digest,authority_references=refs,configuration=CONFIG,current_color_run_samples=[[float(q) for q in v] for v in path],upper_material_order='external background -> BLUE strip -> board body',lower_material_order='board body -> YELLOW strip -> neutral actual ice',reflection_exclusion='first significant return after observed solid yellow core; later comparable reflected return pending',same_current_board_body=False,physical_role_status='unresolved_untrusted_RGB_candidates',independent_visible_boundary=True,physical_support=dict(top='existing common perimeter at H',contact='existing common perimeter at 0'),missing_or_occluded_boundary_is_not_invented=True);start=len(rows)
   for j in range(1,len(path)-1,2):
    v=path[j];seed=np.array(v[:2],float);t=np.array([path[j+1][0]-path[j-1][0],path[j+1][1]-path[j-1][1]],float);n=np.array([-t[1],t[0]]);n/=np.linalg.norm(n)
    if n[1]<0:n=-n
    receipt=dict(navigation_seed_uv=seed.tolist(),normal=n.tolist(),source_measurement_audit=dict(profile=dict(channel=2,polarity=1)));a=replay_outer_cubic(rgb,receipt,coefficients=coeff)
    if not a['numeric_valid'] or a['synthetic_window_boundary_count']:continue
    q=np.asarray(a['current_uv']);qi=np.rint(q).astype(int)
    if not(0<=qi[0]<w and 0<=qi[1]<h) or mask[qi[1],qi[0]]:continue
    profile=dict(channel=2,channel_definition='B-(R+G)/2',polarity=1,normal_offsets=a['actual_profile_normal_offsets'],intensity=a['actual_profile_intensity'],peak_gradient=a['signed_peak_gradient'],offset_px=a['normal_offset_px'],interpolation='continuous_cubic',valid=True,configuration=CUBIC_CONFIG);audit=dict(status='current_native_user_defined_outer_board_boundary',current_uv=q.tolist(),normal=n.tolist(),profile=profile,continuous_cubic_material_profile=a,source_definition_certificate_index=ci,actual_outer_material_order_verified=False,reflection_excluded=False,internal_interface_used=None,physical_role_status='unresolved_untrusted_RGB_candidates',new_material_requalification_required=True,visibility_independent_from_other_boundary=True)
    rows.append(dict(material_role=role,target_id=role,measurement_operator=operator,source_uv=q.tolist(),current_uv=q.tolist(),navigation_seed_uv=seed.tolist(),normal=n.tolist(),source_measurement_audit=audit,measurement_contract_sha256=contract.digest,source_definition_certificate_index=ci))
   cert['actual_measured_rows']=len(rows)-start;certs.append(cert)
 contract.verify_unchanged();return rows,certs


def source_outer_boundary_rows(rgb,mask,*,current_board_context=None):
 """Fail closed until the existing trained/annotated board reader binds scope.

 The historical global colour selector is retained solely for exact diagnostic
 replay. A demonstrated spectator-rail counterexample withdrew its physical
 board certificate. It must never be a fallback for new training or frozen QA.
 Current actual board-instance/navigation context will be supplied by the
 existing SourceRinkReader/BoardProgrammeSourceReader adapter.
 """
 if np.asarray(rgb).shape[:2]!=np.asarray(mask).shape:raise ValueError('Current RGB/mask shape differs')
 if current_board_context is not None:
  from .board_local_source import read_current_board_instance_outer
  return read_current_board_instance_outer(rgb,mask,current_board_context)
 return [],[]
