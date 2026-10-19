"""Whole-band and nested-house native material readers for framewise repair.

Search geometry never supplies output pixels. Two ice-side colour transitions
must authenticate a wide line; house layers share one projected world circle
centre and the scene's radius ordering.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares,isotonic_regression
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d,binary_closing
from scipy.signal import find_peaks

def narrow_dark_paint(point,normal,offsets,profile,valid,maximum_width=75):
 """Two local, current ice-to-paint flanks of a narrow unfilled circle.

 A full search strip can cross another painted row. Its outermost endpoints
 are not the background of every stroke in that strip. Locate local valleys
 and read each valley's two own ice-side shoulders instead.
 """
 keep=valid;off=offsets[keep];raw=profile[keep]
 if len(off)<15:return []
 y=gaussian_filter1d(raw,1.2);peaks,_=find_peaks(-y,prominence=3.,distance=5)
 out=[]
 for peak in peaks:
  flank_span=max(50,int(np.ceil(maximum_width*.7)))
  lo=max(0,peak-flank_span);hi=min(len(y)-1,peak+flank_span)
  left=lo+int(np.argmax(y[lo:peak]));right=peak+1+int(np.argmax(y[peak+1:hi+1]))
  if peak-left<3 or right-peak<3:continue
  bottom=float(np.median(y[max(0,peak-1):peak+2]));l=float(y[left]);r=float(y[right])
  if min(l-bottom,r-bottom)<4:continue
  # Monotone pooling localizes measured source level crossings; it never
  # copies the query or a fitted-circle coordinate into a material readout.
  ql=_crossings(off[left:peak+1],y[left:peak+1],bottom+(l-bottom)*np.array([.25,.5,.75]),False)
  qr=_crossings(off[peak:right+1],y[peak:right+1],bottom+(r-bottom)*np.array([.25,.5,.75]),True)
  if ql is None or qr is None:continue
  ql=np.sort(ql);qr=np.sort(qr);width=qr[1]-ql[1]
  if width<1.5 or width>maximum_width:continue
  quantiles=point[None,None]+np.array([ql,qr])[:,:,None]*normal[None,None]
  sides=quantiles[:,1];uv=sides.mean(0)
  sigma=float(np.max(np.linalg.norm(quantiles[:,2]-quantiles[:,0],axis=1))/1.349)
  if sigma>22:continue
  evidence=dict(query_uv=point.tolist(),profile_normal=normal.tolist(),native_profile_shift_px=float((uv-point)@normal),
   actual_source_sides_uv=sides.tolist(),actual_source_side_quantiles_uv=quantiles.tolist(),signed_interface_iqr_contrast=[(l-bottom)*.5,(r-bottom)*.5],
   transition_sigma_px=max(1.,sigma),measurement_origin='current_native_narrow_circle_two_local_ice_side_shoulders_and_monotone_half_level_crossings',
   material_signal='dark_narrow_circular_paint',narrow_paint_width_px=float(width),filled_house_band_is_not_an_outline=True,
   complete_band_native_interval_px=[float(off[left]),float(off[right])],actual_source_support_outside_person_and_broadcast_masks=True)
  out.append(dict(uv=np.round(uv,4).tolist(),sigma_px=max(1.,sigma),source_evidence=evidence))
 return out

def _crossings(offsets,values,levels,increasing):
 fit=isotonic_regression(values,increasing=increasing).x
 if not increasing:fit=-fit;levels=-np.asarray(levels)
 unique,idx=np.unique(fit,return_index=True)
 if len(unique)<3 or min(levels)<unique[0] or max(levels)>unique[-1]:return None
 return np.interp(levels,unique,offsets[idx])

def whole_colour_bands(point,normal,offsets,profile,valid,*,maximum_width=175,minimum_width=7,maximum_blur_ratio=.42):
 """Read outer ice/paint transitions of complete colour bands, not internal ridges."""
 if len(offsets)<40:return []
 flank=min(18,len(profile)//10)
 background=np.linspace(np.median(profile[:flank]),np.median(profile[-flank:]),len(profile))
 raw=profile-background;smooth=gaussian_filter1d(raw,2.)
 noise=max(.12,float(1.4826*np.median(abs(np.diff(smooth,n=2)-np.median(np.diff(smooth,n=2))))))
 peak=float(np.quantile(smooth,.98))
 if peak<max(2.,6*noise):return []
 support=smooth>max(1.,3*noise,.13*peak)
 # Combed internal colour is one painted strip if it remains enclosed by
 # the same two ice-side transitions. Closing only proposes that full search.
 support=binary_closing(support,structure=np.ones(15))
 starts=np.flatnonzero(support&~np.r_[False,support[:-1]])
 ends=np.flatnonzero(support&~np.r_[support[1:],False]);out=[]
 for a,b in zip(starts,ends):
  if b-a<max(3,minimum_width-1) or b-a>maximum_width+35:continue
  lo=max(0,a-22);hi=min(len(profile)-1,b+22)
  if lo==0 or hi==len(profile)-1 or not valid[lo:hi+1].all():continue
  x=offsets[lo:hi+1];y=raw[lo:hi+1]
  amp=float(np.quantile(smooth[a:b+1],.8));width=float(offsets[b]-offsets[a])
  if amp<max(2.,6*noise):continue
  center=float((offsets[a]+offsets[b])*.5)
  def prediction(z):
   base,slope,amplitude,left,right,blur=z
   return base+slope*(x-center)+amplitude*.5*(erf((x-left)/(np.sqrt(2)*blur))-erf((x-right)/(np.sqrt(2)*blur)))
  lower=np.array([-amp*.5,-.12,amp*.35,offsets[lo]+3,center+2,.65])
  upper=np.array([amp*.5,.12,amp*2.8,center-2,offsets[hi]-3,24.])
  if (upper<=lower).any():continue
  initial=np.clip([0.,0.,amp,offsets[a]+2,offsets[b]-2,3.],lower+1e-6,upper-1e-6)
  fit=least_squares(lambda z:(prediction(z)-y),initial,
    bounds=(lower,upper),loss='soft_l1',f_scale=max(1.,amp*.12),max_nfev=70)
  base,slope,amplitude,left,right,blur=fit.x
  if right-left<minimum_width or right-left>maximum_width or blur>(right-left)*maximum_blur_ratio:continue
  residual=float(np.median(abs(prediction(fit.x)-y)))
  if residual>max(1.8,amplitude*.24):continue
  # Locate actual source q25/q50/q75 level crossings after monotone pooling
  # inside each complete transition. The fitted top-hat only locates those
  # two transitions and estimates their local paint/ice endpoints.
  corrected=raw-base-slope*(offsets-center)
  quantiles=[]
  for edge,increasing in ((left,True),(right,False)):
   half=max(7.,min((right-left)*.45,blur*3.5+3))
   ids=(offsets>=edge-half)&(offsets<=edge+half)
   levels=amplitude*np.array([.25,.5,.75])
   q=_crossings(offsets[ids],corrected[ids],levels,increasing)
   if q is None:break
   quantiles.append(point[None]+np.sort(q)[:,None]*normal[None])
  if len(quantiles)!=2:continue
  quantiles=np.asarray(quantiles);sides=quantiles[:,1];uv=sides.mean(0)
  if abs(((uv-point)@normal)-center)>max(6.,width*.15):continue
  out.append(dict(uv=uv,sides=sides,quantiles=quantiles,width_px=float(np.linalg.norm(sides[1]-sides[0])),
    amplitude=amplitude,transition_blur_px=float(blur),whole_band_residual_median=residual,
    complete_band_native_interval=[float(offsets[lo]),float(offsets[hi])],
    actual_profile_offsets=offsets.copy(),actual_chromatic_profile=profile.copy(),local_background=background.copy(),
    method='complete_coloured_band_two_ice_side_transitions_joint_top_hat_navigation_then_native_monotone_q50'))
 return out

def sample_chroma(rgb,queries,normals,intervals,signal,field=None):
 size=int(np.ceil(np.max(intervals[:,1]-intervals[:,0])))+1
 offsets=intervals[:,:1]+np.arange(size)[None];tangent=np.c_[-normals[:,1],normals[:,0]]
 samples=queries[:,None,None]+normals[:,None,None]*offsets[:,None,:,None]+tangent[:,None,None]*np.arange(-3,4)[None,:,None,None]
 xy=np.rint(samples.astype(np.float32)*32)/32
 if field is None:
  color=rgb.astype(np.float32)
  field=(color[...,2]-(color[...,0]+color[...,1])*.5 if signal=='blue' else
         .2126*color[...,0]+.7152*color[...,1]+.0722*color[...,2] if signal=='dark' else color[...,0]-(color[...,1]+color[...,2])*.5)
 native=cv2.remap(field,xy[...,0].reshape(-1,size),xy[...,1].reshape(-1,size),cv2.INTER_LINEAR).reshape(samples.shape[:3])
 profiles=np.median(native,axis=1)
 valid=((samples>=[0,0]).all(-1)&(samples<=[1919,1079]).all(-1)).all(1)&(offsets<=intervals[:,1:]+.25)
 return offsets,profiles,valid

def native_single_colour_edge(point,normal,offsets,profile,valid,sign):
 """An observed material interface with independently visible inner/outer flanks."""
 keep=valid;offsets=offsets[keep];profile=profile[keep]
 if len(offsets)<9:return None
 flank=max(2,min(5,len(offsets)//4));first=float(np.median(profile[:flank]));last=float(np.median(profile[-flank:]))
 contrast=sign*(last-first);s=gaussian_filter1d(profile,1.)
 noise=max(.10,float(1.4826*np.median(abs(np.diff(s,n=2)-np.median(np.diff(s,n=2))))))
 if contrast<max(1.6,5*noise):return None
 levels=first+(last-first)*np.array([.25,.5,.75])
 q=_crossings(offsets,s,levels,sign>0)
 if q is None:return None
 q=np.sort(q)
 if q[0]<offsets[1] or q[-1]>offsets[-2] or q[-1]-q[0]>max(2.,len(offsets)*.65):return None
 uv=point+normal*q[1];quantile=point[None]+q[:,None]*normal[None]
 return dict(uv=np.round(uv,4).tolist(),sigma_px=max(1.,float((q[-1]-q[0])/1.349)),
  source_evidence=dict(query_uv=point.tolist(),profile_normal=normal.tolist(),native_profile_shift_px=float(q[1]),
   actual_source_sides_uv=[uv.tolist(),uv.tolist()],actual_source_side_quantiles_uv=[quantile.tolist(),quantile.tolist()],
   signed_interface_iqr_contrast=[contrast*.5]*2,transition_sigma_px=max(1.,float((q[-1]-q[0])/1.349)),
   measurement_origin='current_native_declared_house_material_interface_with_visible_inner_and_outer_flanks',
   material_signal='current_dark_fill_light_ring_material_sequence',actual_inside_outside_material_levels=[first,last],
   whole_transition_native_interval_px=[float(offsets[0]),float(offsets[-1])],
   actual_source_support_outside_person_and_broadcast_masks=True))

class NestedHouseReader:
 """One ground circle centre and ordered radius ratios for all rings of a house."""
 def __init__(self,scene,paths,guides,cameras,motion_path):
  from calibration.program_tracking.projection import project_fast
  self.project=project_fast;self.cameras=cameras;self.groups={};self.state={}
  for feature in scene['ice_features']:
   if feature.get('kind')!='circle':continue
   fid=feature['id'];group_paths=[p for p in paths if p['feature_id']==fid]
   if not group_paths:continue
   definitions=feature.get('rings') or [dict(role='outer',center_xyz=feature['center_xyz'],radius=feature['radius'])]
   outline=not bool(feature.get('rings'))
   outer=next(r for r in definitions if r['role']=='outer');center=np.asarray(outer['center_xyz'])[:2];radius=float(outer['radius']);anchor=201;authority='8a28_circle_with_one_current_camera_ground_plane_projection'
   choices=[g for g in guides if g['curve_id']==fid+':outer' and '_world_xyz' in g]
   if choices:
    centers=[];radii=[]
    for g in choices:
     xy=np.asarray(g['_world_xyz'])[:,:2];c=xy.mean(0);centers.append(c);radii.append(np.median(np.linalg.norm(xy-c,axis=1)))
    center=np.median(centers,axis=0);radius=float(np.median(radii));anchor=max(201,int(choices[len(choices)//2]['frame']));authority='fixed_human_outer_circle_anchor_shared_world_centre_and_all_scene_radius_ratios'
   else:
    # Choose one fixed visible source clock for the entire track. It never
    # switches on a block boundary or whichever manual anchor is nearest.
    best=None
    for i in range(201,1249,8):
     xyz=np.asarray(group_paths[0]['xyz']);uv,depth=self.project(cameras[i],xyz)
     valid=(uv>=[0,0]).all(1)&(uv<=[1919,1079]).all(1)&(depth>.05)
     if valid.sum()<8:continue
     score=float(valid.sum())+min(200.,float(np.ptp(uv[valid,0])))*.05
     if best is None or score>best[0]:best=(score,i)
    if best:anchor=best[1]
   radii={r['role']:radius*float(r['radius'])/float(outer['radius']) for r in definitions if r['role']!='center'}
   self.groups[fid]=dict(center=center,radii=radii,anchor=anchor,authority=authority,paths=group_paths,outline=outline)
 def projection(self,group,index,role,theta,parameters=None):
  center=group['center'];radius=group['radii'][role]
  xyz=np.c_[center+radius*np.c_[np.cos(theta),np.sin(theta)],np.zeros(len(theta))]
  uv,depth=self.project(self.cameras[index],xyz)
  return self.source_transform(uv,parameters),depth
 def source_transform(self,uv,parameters):
  if parameters is None:return uv
  # One invertible image-plane affine correction of the current ground-plane
  # projection is shared by the entire physical circle family. This corrects
  # source shape without allowing independent ring centres, radii or layers.
  tx,ty,xx,xy,yx,yy=parameters
  return (uv-[960.,540.])@np.array([[1+xx,yx],[xy,1+yy]])+np.array([960+tx,540+ty])
 def queries(self,group,index,role,parameters,spacing=35):
  theta=np.linspace(0,2*np.pi,513,endpoint=False);uv,depth=self.projection(group,index,role,theta,parameters)
  before=self.projection(group,index,role,theta-.002,parameters)[0];after=self.projection(group,index,role,theta+.002,parameters)[0]
  tangent=after-before;tangent/=np.maximum(1e-12,np.linalg.norm(tangent,axis=1))[:,None];normal=np.c_[tangent[:,1],-tangent[:,0]]
  center=group['center']
  cp=self.source_transform(self.project(self.cameras[index],np.r_[center,0.][None])[0],parameters)[0]
  normal[np.einsum('ij,ij->i',normal,uv-cp)<0]*=-1
  margin=75 if group['outline'] else 25
  visible=(uv>=[-margin,-margin]).all(1)&(uv<=[1919+margin,1079+margin]).all(1)&(depth>.05)
  ids=[];last=None
  for k in np.flatnonzero(visible):
   if last is None or np.linalg.norm(uv[k]-uv[last])>=spacing:ids.append(k);last=k
  if not ids:return np.empty(0),np.empty((0,2)),np.empty((0,2)),np.empty((0,2))
  ids=np.array(ids);theta=theta[ids];uv=uv[ids];normal=normal[ids]
  # The permitted source strip ends before the adjacent physical layer's
  # halfway boundary. The same shared world projection supplies every gap.
  radii=group['radii'];radius=radii[role];interval=np.tile([-75.,75.] if group['outline'] else [-25.,25.],(len(ids),1))
  for other,r in radii.items():
   if other==role:continue
   neighbour=self.projection(group,index,other,theta,parameters)[0];gap=np.einsum('ij,ij->i',neighbour-uv,normal)
   if r<radius:interval[:,0]=np.maximum(interval[:,0],gap*.43)
   else:interval[:,1]=np.minimum(interval[:,1],gap*.43)
  if role=='center':
   # Read the white central disk's own interface. An inward profile cannot
   # cross its opposite boundary and use another ring as its inside material.
   inward=np.einsum('ij,ij->i',cp-uv,normal)
   interval[:,0]=np.maximum(interval[:,0],inward*.65)
  return theta,uv,normal,interval
 def measurements(self,group,index,native,parameters):
  rows=[];diagnostics={}
  for role in group['radii']:
   theta,q,n,interval=self.queries(group,index,role,parameters)
   if not len(q):continue
   # Scene schematic colours are not a claim about broadcast RGB. These
   # houses visibly use dark fills separated by a light ring in this source.
   signal='dark';sign=-1 if role in ('middle','center') else 1
   permitted=interval[:,1]-interval[:,0]>=8
   theta,q,n,interval=theta[permitted],q[permitted],n[permitted],interval[permitted]
   if not len(q):continue
   key='repair_whole_chroma_'+signal
   if key not in native.frame.material_signals:
    color=native.frame.rgb.astype(np.float32)
    native.frame.material_signals[key]=.2126*color[...,0]+.7152*color[...,1]+.0722*color[...,2]
   offsets,profiles,validity=sample_chroma(native.frame.rgb,q,n,interval,signal,native.frame.material_signals[key])
   # Padding belongs to no material profile; each native endpoint interval
   # remains independently valid in the batched remap.
   accepted=[]
   for j in range(len(q)):
    if group['outline']:
     choices=narrow_dark_paint(q[j],n[j],offsets[j],profiles[j],validity[j])
     choices.sort(key=lambda r:abs((np.asarray(r['uv'])-q[j])@n[j]))
     row=choices[0] if choices else None
    else:row=native_single_colour_edge(q[j],n[j],offsets[j],profiles[j],validity[j],sign)
    if row is None:continue
    if group['outline'] and next(iter(group['paths']))['feature_id']!='C0':
     ev=row['source_evidence'];width=ev['narrow_paint_width_px']
     # A broad asymmetric filled-house/reflection trough is not evidence of
     # an additional narrow outline, even when a model circle passes there.
     if min(ev['signed_interface_iqr_contrast'])<4 or width>30 or row['sigma_px']>min(7.,.65*width):continue
    evidence=row['source_evidence'];context=np.asarray(evidence['actual_source_side_quantiles_uv']).reshape(-1,2)
    evidence_interval=evidence.get('complete_band_native_interval_px',interval[j])
    flank=q[j]+n[j]*np.asarray(evidence_interval)[:,None];context=np.vstack([context,flank,np.asarray(row['uv'])[None]])
    if (context<[0,0]).any() or (context>[1919,1079]).any():continue
    ip=np.rint(context).astype(int)
    if native.mask[ip[:,1],ip[:,0]].any():continue
    offsets_used=np.arange(np.ceil(evidence_interval[0]),np.floor(evidence_interval[1])+1)
    tangent=np.array([-n[j,1],n[j,0]])
    full_context=q[j]+offsets_used[:,None,None]*n[j]+np.arange(-3,4)[None,:,None]*tangent
    full_context=full_context.reshape(-1,2)
    if (full_context<[0,0]).any() or (full_context>[1919,1079]).any():continue
    fp=np.rint(full_context).astype(int)
    if native.mask[fp[:,1],fp[:,0]].any():continue
    row.update(role=role,theta=float(theta[j]),query_index=j)
    evidence.update(query_index=j,material_signal=signal,physical_layer=role,
      declared_inside_outside_materials=['light_ice','narrow_circular_paint','light_ice'] if group['outline'] else {'outer':['dark_outer_house_fill','light_ice'],'middle':['light_middle_house_ring','dark_outer_house_fill'],'inner':['dark_inner_house_fill','light_middle_house_ring'],'center':['light_center_house_disk','dark_inner_house_fill']}[role],
      material_state_order_verified_from_current_native_luminance_not_scene_style_colours=True)
    accepted.append(row);rows.append(row)
   diagnostics[role]=dict(query_count=len(q),eligible_native_material_readouts=len(accepted))
  return rows,diagnostics
 def curve_distance(self,curve,points):
  center,axes,angle=cv2.fitEllipse(np.asarray(curve,np.float32))
  angle=np.deg2rad(angle);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  matrix=rotation@np.diag(1/(np.asarray(axes)*.5)**2)@rotation.T
  delta=np.asarray(points)-center;gradient=2*delta@matrix
  return (np.einsum('ij,ij->i',delta@matrix,delta)-1)/np.maximum(1e-9,np.linalg.norm(gradient,axis=1))
 def conic(self,curve):
  center,axes,angle=cv2.fitEllipse(np.asarray(curve,np.float32));angle=np.deg2rad(angle)
  rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  return np.asarray(center),rotation@np.diag(1/(np.asarray(axes)*.5)**2)@rotation.T
 def transformed_conic_distance(self,conic,points,z):
  center,matrix=conic;tx,ty,xx,xy,yx,yy=z
  inverse=np.linalg.inv(np.array([[1+xx,xy],[yx,1+yy]]))
  delta=(np.asarray(points)-[960+tx,540+ty])@inverse.T+[960,540]-center
  gradient=2*(delta@matrix)@inverse
  return (np.einsum('ij,ij->i',delta@matrix,delta)-1)/np.maximum(1e-9,np.linalg.norm(gradient,axis=1))
 def transformed_conic_exact_distance(self,conic,points,z):
  center,matrix=conic;tx,ty,xx,xy,yx,yy=z;M=np.array([[1+xx,xy],[yx,1+yy]]);inverse=np.linalg.inv(M)
  source_center=(center-[960,540])@M.T+[960+tx,540+ty]
  source_matrix=inverse.T@matrix@inverse;values,vectors=np.linalg.eigh(source_matrix)
  axes=2/np.sqrt(values);angle=np.rad2deg(np.arctan2(vectors[1,0],vectors[0,0]))
  points=np.asarray(points,float);nearest=cv2.getClosestEllipsePoints((source_center.tolist(),axes.tolist(),float(angle)),points.astype(np.float32)).reshape(-1,2)
  delta=points-source_center;sign=np.sign(np.einsum('ij,ij->i',delta@source_matrix,delta)-1)
  return np.linalg.norm(nearest-points,axis=1)*sign
 def read(self,fid,index,native):
  group=self.groups[fid];initial=np.zeros(6)
  # A standalone re-read and a full replay use identical current-frame
  # initialization. No stale previous-circle state can switch identities.
  bound=np.array([90.,60.,.25,.18,.06,.25]) if fid=='C0' else np.array([40.,30.,.06,.05,.025,.08])
  fitted=initial;diagnostics={};rows=[]
  base_curves={role:self.projection(group,index,role,np.linspace(0,2*np.pi,513),None)[0] for role in group['radii']}
  conics={role:self.conic(curve) for role,curve in base_curves.items()}
  for iteration in range(4):
   rows,diagnostics=self.measurements(group,index,native,fitted)
   if len(rows)<(3 if fid=='C0' else 6):return {},diagnostics
   source_by_role={role:np.asarray([r['uv'] for r in rows if r['role']==role]).reshape(-1,2) for role in group['radii']}
   prior_scale=np.array([60.,40.,.12,.10,.045,.15])
   def residual(unit):
    z=unit*bound
    values=[]
    for role,observed in source_by_role.items():
     if len(observed):values.extend(self.transformed_conic_exact_distance(conics[role],observed,z))
    return np.r_[values,z/prior_scale*.25]
   def jacobian(unit):
    step=.001;columns=[]
    for k in range(6):
     direction=np.eye(6)[k]*step
     columns.append((residual(unit+direction)-residual(unit-direction))/(2*step))
    return np.asarray(columns).T
   fit=least_squares(residual,fitted/bound,jac=jacobian,bounds=(-np.ones(6),np.ones(6)),loss='soft_l1',f_scale=1.5,max_nfev=50)
   change=np.max(abs(fit.x*bound-fitted)/bound);fitted=fit.x*bound
   if change<.008:break
  rows,diagnostics=self.measurements(group,index,native,fitted);output={}
  for role in group['radii']:
   selected=[r for r in rows if r['role']==role]
   if not selected:continue
   curve=self.source_transform(base_curves[role],fitted)
   errors=abs(self.transformed_conic_exact_distance(conics[role],np.array([r['uv'] for r in selected]),fitted))
   selected=[r for r,error in zip(selected,errors) if error<=3.0]
   if len(selected)<3:continue
   if group['outline'] and fid!='C0':
    selected=[r for r in selected if min(r['source_evidence']['signed_interface_iqr_contrast'])>=4. and r['source_evidence']['narrow_paint_width_px']<=30]
    if len(selected)<6:continue
   # Emit correlated sparse representatives of this frame's jointly fitted
   # physical curve. Preserve every native material readout separately. The
   # fit is never used to fill a station without current source material.
   ellipse=cv2.fitEllipse(np.asarray(curve,np.float32))
   actual=np.asarray([row['uv'] for row in selected],np.float32)
   nearest=cv2.getClosestEllipsePoints(ellipse,actual).reshape(-1,2)
   exact_errors=np.linalg.norm(nearest.astype(float)-actual.astype(float),axis=1)
   precise=[]
   for row,representative,error in zip(selected,nearest,exact_errors):
    if error>3.:continue
    row['representative_uv']=np.round(representative.astype(float),4).tolist()
    row['representative_to_actual_material_px']=float(error);precise.append(row)
   selected=precise
   if len(selected)<3:continue
   output[role]=dict(rows=selected,curve=curve,
      joint_group_navigation=dict(kind='shared_current_source_ground_circle_family_affine_projection_correction',house_id=fid,
       source_clock_anchor=group['anchor'],source_identity_authority=group['authority'],
       common_world_circle_center=group['center'].tolist(),common_radius_scale=1.,
       shared_source_affine_parameters=fitted.tolist(),source_projection_correction_fitted_only_to_current_native_material=True,
       scene_radius_ratios={k:v/group['radii']['outer'] for k,v in group['radii'].items()},
       no_ring_independent_center_or_scale=True,maximum_normal_material_to_shared_circle_px=float(errors[errors<=3].max()),
       raw_material_residuals_before_rejection_px=errors.tolist(),
       maximum_true_euclidean_material_to_fitted_curve_px=float(max(r['representative_to_actual_material_px'] for r in selected)),
       inferred_projected_uv_not_emitted=True))
  return output,diagnostics
