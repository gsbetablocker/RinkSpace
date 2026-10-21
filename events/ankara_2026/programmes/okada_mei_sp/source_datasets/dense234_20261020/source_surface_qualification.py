"""Current source surface/graphics qualification, using the accepted first-shot timer reader."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,cv2
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(P))
from calibration.program_tracking.extract_first_shot_ice import current_broadcast_exclusion

def current_domains(bgr,board):
 graphics,timer=current_broadcast_exclusion(bgr)
 # This source's technical-score panel below the two permanent score rows.
 # The source coordinates only locate the layout; current colored header,
 # green execution field and neutral body must all be present.
 header=bgr[94:124,72:395].astype(float);sat=header.max(2)-header.min(2)
 header_fraction=float(((sat>55)&(header.max(2)>100)).mean())
 execution=bgr[187:220,234:396].astype(float)
 green_fraction=float(((execution[:,:,1]>execution[:,:,0]+30)&(execution[:,:,1]>execution[:,:,2]+30)&(execution[:,:,1]>120)).mean())
 body=bgr[125:185,72:395].astype(float)
 neutral_fraction=float(((body.min(2)>155)&(body.max(2)-body.min(2)<55)).mean())
 panel=header_fraction>.25 and green_fraction>.15 and neutral_fraction>.35
 if panel:graphics[90:257,60:400]=True
 q=np.asarray(board['roles']['board.contact']['trace_uv'],float);q=q[np.isfinite(q).all(1)];q=q[np.argsort(q[:,0])]
 if len(q)<2:raise ValueError('Actual current board-contact trace unavailable')
 axis=np.arange(1920,dtype=float);border=np.interp(axis,q[:,0],q[:,1]);supported=np.zeros(1920,bool)
 spans=board['roles']['board.contact']['spans']
 for sp in spans:
  a,z=sp['x_range'];supported|=(axis>=a)&(axis<=z)
 # Occluded/missing columns are exclusion-only, never source observations.
 missing=np.flatnonzero(~supported);intervals=[]
 if len(missing):
  cuts=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1)
  for part in cuts:
   a,z=int(part[0]),int(part[-1]);flanks=[border[k]for k in(a-1,z+1)if 0<=k<1920 and supported[k]]
   if flanks:border[a:z+1]=max(flanks)
   intervals.append([a,z])
 receipt=dict(timer=timer,technical_panel=dict(present=bool(panel),header_saturated_fraction=header_fraction,execution_green_fraction=green_fraction,body_neutral_fraction=neutral_fraction,source_layout_roi_only=True,rectangle_xyxy=[60,90,400,257]if panel else None),source_board_contact_file=board.get('anchored_file'),source_board_contact_SHA=board.get('anchored_file_sha256'),ice_domain_from_actual_source_contact=True,unsupported_contact_columns_only_conservative_exclusion=intervals,no_domain_boundary_emitted_as_point=True)
 return graphics,border,receipt

def source_support_positions(r):
 q=[r['uv']]
 if r.get('paired_edge_uv')is not None:q.extend(np.asarray(r['paired_edge_uv'],float).reshape(-1,2).tolist())
 profile=r.get('pixel_support',{}).get('side_interface_quartiles_uv')
 if profile is not None:q.extend(np.asarray(profile,float).reshape(-1,2).tolist())
 return np.asarray(q,float)

def rejection_reason(r,part,graphics,border):
 q=source_support_positions(r);finite=np.isfinite(q).all(1);inside=finite&(q[:,0]>=0)&(q[:,0]<=1919)&(q[:,1]>=0)&(q[:,1]<=1079)
 if part=='ice'and not inside.all():return 'ice_material_support_outside_native_visible_domain'
 q=q[inside]
 if len(q)==0:return 'source_material_support_unavailable'
 xy=np.rint(q).astype(int)
 if graphics[xy[:,1],xy[:,0]].any():return 'actual_current_broadcast_graphics_occlusion'
 if part=='ice'and(q[:,1]<=np.interp(q[:,0],np.arange(1920),border)).any():return 'above_actual_current_board_contact_not_visible_ice_surface'
 return None
