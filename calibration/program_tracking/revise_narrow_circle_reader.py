from pathlib import Path
p=Path('calibration/program_tracking/native_ice_material_repair.py');s=p.read_text();a=s.index("    if group['outline']:\n",s.index(' def measurements'));b=s.index('    else:row=native_single_colour_edge',a)
s=s[:a]+'''    if group['outline']:
     from rinkspace.vision_programme_ice_review import _actual_profile_candidates
     valid=validity[j];off=offsets[j,valid];profile=profiles[j,valid]
     if len(off)<20:continue
     smooth=gaussian_filter1d(profile,1.)
     candidates=_actual_profile_candidates(q[j],n[j],off,profile,smooth,operator='paint_center',signal='dark',transition=None,with_quantiles=True)
     choices=[]
     for point,sides,bad,quantiles in candidates:
      width=np.linalg.norm(sides[1]-sides[0]);sigma=float(np.max(np.linalg.norm(quantiles[:,2]-quantiles[:,0],axis=1))/1.349)
      qo=(quantiles-q[j])@n[j];contrasts=[abs(float(np.interp(v[2],off,smooth)-np.interp(v[0],off,smooth))) for v in qo]
      if bad or width<1.5 or width>50 or sigma>18 or min(contrasts)<2.5:continue
      evidence=dict(query_uv=q[j].tolist(),profile_normal=n[j].tolist(),native_profile_shift_px=float((point-q[j])@n[j]),
       actual_source_sides_uv=sides.tolist(),actual_source_side_quantiles_uv=quantiles.tolist(),signed_interface_iqr_contrast=contrasts,
       transition_sigma_px=max(1.,sigma),measurement_origin='current_native_narrow_circular_paint_two_complete_ice_side_interfaces',
       material_signal='dark_narrow_circular_paint',actual_source_support_outside_person_and_broadcast_masks=True,
       narrow_paint_width_px=float(width),filled_house_band_is_not_an_outline=True,
       complete_band_native_interval_px=[float(qo.min()-4),float(qo.max()+4)])
      choices.append((abs((point-q[j])@n[j]),dict(uv=np.round(point,4).tolist(),sigma_px=max(1.,sigma),source_evidence=evidence)))
     choices.sort(key=lambda x:x[0])
     row=choices[0][1] if choices and not(len(choices)>1 and choices[1][0]-choices[0][0]<3) else None
''' +s[b:];p.write_text(s)
