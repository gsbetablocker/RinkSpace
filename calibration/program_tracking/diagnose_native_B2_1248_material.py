"""Read-only RGB/physical-width evidence for the user-corrected1248 B2."""
from calibration.program_tracking.extract_first_shot_ice import *
from calibration.program_tracking.physical_ice_width import projected_width_profile,sample_physical_width_field,restore_physical_profile_readout
from calibration.program_tracking.native_ice_material_repair import narrow_dark_paint
from scipy.ndimage import gaussian_filter1d

p=DATA/'repair_20261014';out=p/'wide_quantitative_diagnostics';out.mkdir(exist_ok=True)
e=Extractor(p,ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014');path=next(v for v in e.paths if v['feature_id']=='B2');camera=e.cameras[1248]
source=DATA/'frames/001248.png';raw=cv2.imread(str(source));rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
old_packet=read(p/'wide_line_candidates/001248.json');feature=next(f for f in old_packet['features'] if f['target_id']=='ice.B2:segment0')
annotation=Path('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00009.png');user=cv2.imread(str(annotation));yy,xx=np.nonzero((user[:,:,0]<75)&(user[:,:,1]>240)&(user[:,:,2]>240)&(np.indices(user.shape[:2])[1]>1500)&(np.indices(user.shape[:2])[0]>450)&(np.indices(user.shape[:2])[0]<610));slope,intercept=np.polyfit(xx,yy,1);scale=np.array([1920/user.shape[1],1080/user.shape[0]]);user_slope=slope*scale[1]/scale[0];user_intercept=intercept*scale[1]
records=[];canvas=np.full((1100,1900,3),245,np.uint8);crop=raw[400:610,1460:1920];display=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST);overlay=display.copy();lo=np.array([1460,400])
def mark(uv,color,radius=4):cv2.circle(overlay,tuple(np.rint((np.asarray(uv)-lo)*2).astype(int)),radius,color,-1)
def line(points,color,width=1):cv2.polylines(overlay,[np.rint((np.asarray(points)-lo)*2).astype(int)],False,color,width)
for row in feature['source_line_fit']['direct_readouts']:
 qi=row['source_evidence']['query_index'];profile=projected_width_profile(camera,path,row['source_evidence']['query_uv'],radius_px=90,step_px=1.)
 fields={signal:sample_physical_width_field(rgb,profile,signal) for signal in ('dark','blue')}
 choices=narrow_dark_paint(np.zeros(2),np.array([1.,0.]),profile['offsets'],fields['dark'],profile['valid']);choices.sort(key=lambda r:abs(r['uv'][0]));assert choices
 grey=restore_physical_profile_readout(camera,profile,choices[0]);evidence=grey['source_evidence'];sides=np.asarray(evidence['actual_source_sides_uv']);center=np.asarray(grey['uv']);width=np.linalg.norm(sides[1]-sides[0])
 records.append(dict(query_index=qi,old_chromatic_readout=row,actual_physical_width_query_direction_uv=profile['query_width_direction_uv'].tolist(),whole_grey_blue_body_readout=grey,
  grey_body_side_distances_image_px=width,grey_body_width_ground_units=float(np.linalg.norm(np.diff(evidence['actual_measured_sides_ground_xy'],axis=0))),
  profile=dict(offsets=profile['offsets'].tolist(),native_uv=profile['source_uv'].tolist(),luminance=fields['dark'].tolist(),blue_chroma=fields['blue'].tolist()),
  grey_ground_midpoint_to_user_yellow_guide_normal_px=float((center[1]-user_slope*center[0]-user_intercept)/np.sqrt(1+user_slope**2)),
  pure_chroma_is_not_complete_material_boundary=True))
 mark(row['uv'],(0,0,255),6);line(row['source_evidence']['actual_source_sides_uv'],(0,0,255),2)
 mark(center,(0,140,0),6);line(sides,(0,170,0),2)
 for side in sides:mark(side,(255,0,255),4)
 # Native source sample values, shown without normalization of either signal.
 chart=np.full((420,650,3),255,np.uint8);cv2.putText(chart,f'q{qi}: Luma(green); blue chroma+180(blue)',(10,22),0,.5,(0,0,0),1)
 for value in (160,180,200,220):
  y=int(370-(value-150)*4);cv2.line(chart,(45,y),(630,y),(220,220,220),1);cv2.putText(chart,str(value),(5,y+4),0,.4,(0,0,0),1)
 for off in (-80,-40,0,40,80):
  x=int(45+(off+90)*3.2);cv2.line(chart,(x,40),(x,370),(235,235,235),1);cv2.putText(chart,str(off),(x-9,390),0,.4,(0,0,0),1)
 for values,color in ((fields['dark'],(0,150,0)),(fields['blue']+180,(255,90,0))):
  xy=np.c_[45+(profile['offsets']+90)*3.2,370-(values-150)*4];cv2.polylines(chart,[np.rint(xy).astype(int)],False,color,2)
 for offset in np.asarray(choices[0]['source_evidence']['actual_source_sides_uv'])[:,0]:
  x=int(45+(offset+90)*3.2);cv2.line(chart,(x,45),(x,370),(220,0,220),1)
 cv2.putText(chart,f'Whole body sides {sides[:,0].round(2).tolist()}; center {center.round(2).tolist()}',(10,410),0,.35,(0,0,0),1)
 xbase=20 if qi==3 else 750;canvas[640:1060,xbase:xbase+650]=chart
line([r['whole_grey_blue_body_readout']['uv'] for r in records],(0,150,0),1)
canvas[50:470,20:940]=display;canvas[50:470,970:1890]=overlay
cv2.putText(canvas,'1248 B2 native raw (left); old chroma red / whole grey-body edges magenta, center green',(20,30),0,.6,(0,0,0),1)
centers=np.array([r['whole_grey_blue_body_readout']['uv'] for r in records]);new_slope=(centers[1,1]-centers[0,1])/(centers[1,0]-centers[0,0]);old_centers=np.array([r['old_chromatic_readout']['uv'] for r in records]);old_slope=np.diff(old_centers[:,1])[0]/np.diff(old_centers[:,0])[0]
cv2.putText(canvas,f'User guide slope {user_slope:.5f}; grey-body midpoints slope {new_slope:.5f}; old chroma slope {old_slope:.5f}',(20,515),0,.65,(0,0,0),1)
cv2.putText(canvas,'Physical transverse sections here are nearly horizontal. Old image-normal chroma sees internal hue fringes.',(20,550),0,.6,(0,0,0),1)
cv2.putText(canvas,'These are independent current-source diagnostic measurements; no fitting candidate is changed.',(20,585),0,.6,(0,0,0),1)
cv2.imwrite(str(out/'001248_B2_material_profiles.png'),canvas)
write(out/'001248_B2_material_profiles.json',dict(schema='rinkspace.readonly-current-native-wide-material-evidence.v1',source_native_path=str(source),source_native_sha256=sha(source),official_native_pts=575488,source_time='00:00:44.960',user_annotation=dict(path=str(annotation),sha256=sha(annotation),native_navigation_line=dict(slope=user_slope,intercept=user_intercept),not_emitted_as_native_observation=True),records=records,comparison=dict(old_pure_chromatic_midpoints_slope=float(old_slope),actual_grey_blue_body_midpoints_slope=float(new_slope),user_navigation_slope=float(user_slope)),diagnostic_png=str(out/'001248_B2_material_profiles.png'),limitations=['Half-level crossings localize visible blurred grey-blue body, not a claim of subpixel latent pigment edges before motion blur.','Two profiles do not establish whole-shot correctness; root must verify shared physical strip and temporal source evidence over the whole source sequence.'],candidate_or_model_modified=False))
print(json.dumps(dict(records=[dict(q=r['query_index'],center=r['whole_grey_blue_body_readout']['uv'],sides=r['whole_grey_blue_body_readout']['source_evidence']['actual_source_sides_uv'],yellow_error=r['grey_ground_midpoint_to_user_yellow_guide_normal_px']) for r in records],slopes=[user_slope,new_slope,old_slope]),indent=2))
