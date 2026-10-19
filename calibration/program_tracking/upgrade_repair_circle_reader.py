from pathlib import Path
p=Path('calibration/program_tracking/native_ice_material_repair.py')
s=p.read_text()
a=s.index(' def projection(self,group,index,role,theta,parameters=None):')
b=s.index(' def queries(',a)
s=s[:a]+''' def projection(self,group,index,role,theta,parameters=None):
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
''' +s[b:]
s=s.replace("center=group['center']+(parameters[:2] if parameters is not None else 0)","center=group['center']")
s=s.replace("cp=self.project(self.cameras[index],np.r_[center,0.][None])[0][0]","cp=self.source_transform(self.project(self.cameras[index],np.r_[center,0.][None])[0],parameters)[0]")
s=s.replace("visible=(uv>=[-25,-25]).all(1)&(uv<=[1944,1104]).all(1)&(depth>.05)","margin=75 if group['outline'] else 25\n  visible=(uv>=[-margin,-margin]).all(1)&(uv<=[1919+margin,1079+margin]).all(1)&(depth>.05)")
s=s.replace("interval=np.tile([-25.,25.],(len(ids),1))","interval=np.tile([-75.,75.] if group['outline'] else [-25.,25.],(len(ids),1))")
a=s.index("    if group['outline']:\n",s.index(' def measurements'))
b=s.index('    else:row=native_single_colour_edge',a)
s=s[:a]+'''    if group['outline']:
     bands=whole_colour_bands(q[j],n[j],offsets[j],-profiles[j],validity[j],
       maximum_width=50,minimum_width=3,maximum_blur_ratio=2.)
     choices=[]
     for band in bands:
      point,sides,quantiles=band['uv'],band['sides'],band['quantiles']
      if band['amplitude']<4:continue
      sigma=float(np.max(np.linalg.norm(quantiles[:,2]-quantiles[:,0],axis=1))/1.349)
      evidence=dict(query_uv=q[j].tolist(),profile_normal=n[j].tolist(),native_profile_shift_px=float((point-q[j])@n[j]),
       actual_source_sides_uv=sides.tolist(),actual_source_side_quantiles_uv=quantiles.tolist(),signed_interface_iqr_contrast=[band['amplitude']*.5]*2,
       transition_sigma_px=max(1.,sigma),measurement_origin='current_native_narrow_circular_paint_two_complete_ice_side_interfaces',
       material_signal='dark_narrow_circular_paint',actual_source_support_outside_person_and_broadcast_masks=True,
       narrow_paint_width_px=float(band['width_px']),filled_house_band_is_not_an_outline=True,
       complete_band_native_interval_px=band['complete_band_native_interval'],whole_band_profile_residual_median=band['whole_band_residual_median'])
      choices.append((abs((point-q[j])@n[j]),dict(uv=np.round(point,4).tolist(),sigma_px=max(1.,sigma),source_evidence=evidence)))
     choices.sort(key=lambda x:x[0])
     row=choices[0][1] if choices and not(len(choices)>1 and choices[1][0]-choices[0][0]<3) else None
''' +s[b:]
# Only require the actual measured paint-transition context, not all 150
# navigation pixels. Intervening material interval is checked separately.
s=s.replace("flank=q[j]+n[j]*interval[j,:,None];context=np.vstack([context,flank,np.asarray(row['uv'])[None]])", "evidence_interval=evidence.get('complete_band_native_interval_px',interval[j])\n    flank=q[j]+n[j]*np.asarray(evidence_interval)[:,None];context=np.vstack([context,flank,np.asarray(row['uv'])[None]])")
a=s.index(' def read(self,fid,index,native):')
s=s[:a]+''' def read(self,fid,index,native):
  group=self.groups[fid];initial=np.zeros(6)
  previous=self.state.get(fid)
  if previous is not None:initial=previous.copy()
  bound=np.array([90.,60.,.16,.12,.045,.20]) if group['outline'] else np.array([40.,30.,.06,.05,.025,.08])
  initial=np.clip(initial,-bound+1e-8,bound-1e-8)
  fitted=initial;diagnostics={};rows=[]
  for iteration in range(3):
   rows,diagnostics=self.measurements(group,index,native,fitted)
   if len(rows)<6:return {},diagnostics
   # Every residual below is a real current RGB interface measurement.
   # A common affine image correction preserves the nested physical family
   # under one ground-plane projectivity, with no independent circle fit.
   projected=[];observed=[];normals=[]
   for role in group['radii']:
    selected=[r for r in rows if r['role']==role]
    if not selected:continue
    theta=np.array([r['theta'] for r in selected])
    projected.extend(self.projection(group,index,role,theta,None)[0]);observed.extend(r['uv'] for r in selected)
    normals.extend(r['source_evidence']['profile_normal'] for r in selected)
   projected=np.asarray(projected);observed=np.asarray(observed);normals=np.asarray(normals)
   prior_scale=np.array([60.,40.,.08,.07,.035,.10])
   def residual(z):
    return np.r_[np.einsum('ij,ij->i',self.source_transform(projected,z)-observed,normals),z/prior_scale*.5]
   fit=least_squares(residual,fitted,bounds=(-bound,bound),loss='soft_l1',f_scale=1.5,max_nfev=35)
   change=np.max(abs(fit.x-fitted)/bound);fitted=fit.x
   if change<.01:break
  rows,diagnostics=self.measurements(group,index,native,fitted);output={}
  for role in group['radii']:
   selected=[r for r in rows if r['role']==role]
   if not selected:continue
   predicted=self.projection(group,index,role,np.array([r['theta'] for r in selected]),fitted)[0]
   normal=np.array([r['source_evidence']['profile_normal'] for r in selected]);observed=np.array([r['uv'] for r in selected])
   errors=abs(np.einsum('ij,ij->i',predicted-observed,normal))
   selected=[r for r,error in zip(selected,errors) if error<=3.0]
   if len(selected)<(5 if group['outline'] else 3):continue
   # Extra unlabelled outline circles require an independently discernible
   # arc. A two-point grey sliver near a predicted A-circle is no identity.
   if group['outline'] and fid!='C0':
    keep=[]
    for r in selected:
     evidence=r['source_evidence']
     if min(evidence['signed_interface_iqr_contrast'])>=4. and evidence['narrow_paint_width_px']<=30:keep.append(r)
    selected=keep
    if len(selected)<6:continue
   curve=self.projection(group,index,role,np.linspace(0,2*np.pi,513),fitted)[0]
   output[role]=dict(rows=selected,curve=curve,
      joint_group_navigation=dict(kind='shared_current_source_ground_circle_family_affine_projection_correction',house_id=fid,
       source_clock_anchor=group['anchor'],source_identity_authority=group['authority'],
       common_world_circle_center=group['center'].tolist(),common_radius_scale=1.,
       shared_source_affine_parameters=fitted.tolist(),source_projection_correction_fitted_only_to_current_native_material=True,
       scene_radius_ratios={k:v/group['radii']['outer'] for k,v in group['radii'].items()},
       no_ring_independent_center_or_scale=True,maximum_normal_material_to_shared_circle_px=float(errors[errors<=3].max()),
       raw_material_residuals_before_rejection_px=errors.tolist(),
       inferred_projected_uv_not_emitted=True))
  if output:self.state[fid]=fitted
  return output,diagnostics
'''
p.write_text(s)
