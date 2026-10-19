from pathlib import Path
p=Path('calibration/program_tracking/native_ice_material_repair.py');s=p.read_text()
a=s.index(' def read(self,fid,index,native):')
s=s[:a]+''' def curve_distance(self,curve,points):
  center,axes,angle=cv2.fitEllipse(np.asarray(curve,np.float32))
  angle=np.deg2rad(angle);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  matrix=rotation@np.diag(1/(np.asarray(axes)*.5)**2)@rotation.T
  delta=np.asarray(points)-center;gradient=2*delta@matrix
  return (np.einsum('ij,ij->i',delta@matrix,delta)-1)/np.maximum(1e-9,np.linalg.norm(gradient,axis=1))
 def read(self,fid,index,native):
  group=self.groups[fid];initial=np.zeros(6)
  # A standalone re-read and a full replay use identical current-frame
  # initialization. No stale previous-circle state can switch identities.
  bound=np.array([90.,60.,.25,.18,.06,.25]) if group['outline'] else np.array([40.,30.,.06,.05,.025,.08])
  fitted=initial;diagnostics={};rows=[]
  base_curves={role:self.projection(group,index,role,np.linspace(0,2*np.pi,513),None)[0] for role in group['radii']}
  for iteration in range(4):
   rows,diagnostics=self.measurements(group,index,native,fitted)
   if len(rows)<6:return {},diagnostics
   source_by_role={role:np.asarray([r['uv'] for r in rows if r['role']==role]).reshape(-1,2) for role in group['radii']}
   prior_scale=np.array([60.,40.,.12,.10,.045,.15])
   def residual(z):
    values=[]
    for role,observed in source_by_role.items():
     if len(observed):values.extend(self.curve_distance(self.source_transform(base_curves[role],z),observed))
    return np.r_[values,z/prior_scale*.25]
   fit=least_squares(residual,fitted,bounds=(-bound,bound),loss='soft_l1',f_scale=1.5,max_nfev=50,diff_step=1e-4)
   change=np.max(abs(fit.x-fitted)/bound);fitted=fit.x
   if change<.008:break
  rows,diagnostics=self.measurements(group,index,native,fitted);output={}
  for role in group['radii']:
   selected=[r for r in rows if r['role']==role]
   if not selected:continue
   curve=self.source_transform(base_curves[role],fitted)
   errors=abs(self.curve_distance(curve,np.array([r['uv'] for r in selected])))
   selected=[r for r,error in zip(selected,errors) if error<=3.0]
   if len(selected)<(5 if group['outline'] else 3):continue
   if group['outline'] and fid!='C0':
    selected=[r for r in selected if min(r['source_evidence']['signed_interface_iqr_contrast'])>=4. and r['source_evidence']['narrow_paint_width_px']<=30]
    if len(selected)<6:continue
   output[role]=dict(rows=selected,curve=curve,
      joint_group_navigation=dict(kind='shared_current_source_ground_circle_family_affine_projection_correction',house_id=fid,
       source_clock_anchor=group['anchor'],source_identity_authority=group['authority'],
       common_world_circle_center=group['center'].tolist(),common_radius_scale=1.,
       shared_source_affine_parameters=fitted.tolist(),source_projection_correction_fitted_only_to_current_native_material=True,
       scene_radius_ratios={k:v/group['radii']['outer'] for k,v in group['radii'].items()},
       no_ring_independent_center_or_scale=True,maximum_normal_material_to_shared_circle_px=float(errors[errors<=3].max()),
       raw_material_residuals_before_rejection_px=errors.tolist(),
       inferred_projected_uv_not_emitted=True))
  return output,diagnostics
''';p.write_text(s)
