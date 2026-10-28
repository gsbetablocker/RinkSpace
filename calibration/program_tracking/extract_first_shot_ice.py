"""Native ice material measurements with human-labelled shape navigation.

Human traces and 8a28 cameras place searches only. Every accepted measurement
has current-frame material profiles. Lines share a source estimate; circles
retain measured samples belonging to a common projected physical circle.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from copy import deepcopy
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rinkspace.packed_assets import materialize_project_asset, read_project_json
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.optimize import least_squares
from calibration.program_tracking.projection import project_fast,ground_xy
from calibration.program_tracking.extract_single_frame_ice_source import simplify,ellipse_distance
from calibration.program_tracking.extract_continuation_ice_source import broadcast_exclusion,blank_feature
from calibration.program_tracking.apply_ice_source_review_amendment import runs
from rinkspace.vision_programme_ice_review import ice_curve_inventory,_source_queries,_batched_actual_profiles,_actual_profile_candidates
from rinkspace.vision_source_shapes import _MaterialProfileFrame

BASE=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013'
MODEL=ROOT/'events/ankara_2026/models/framewise_8a28_20261013'
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps')
def read(p):return read_project_json(p, project_root=ROOT)
def sha(p):
 with materialize_project_asset(p, project_root=ROOT).open("rb") as stream:
  return hashlib.file_digest(stream, "sha256").hexdigest()

def circle_geometry_digest(scene):
 circles=sorted([f for f in scene['ice_features'] if f.get('kind')=='circle'],key=lambda f:f['id'])
 return hashlib.sha256(json.dumps(circles,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()

def qualify_extra_circle_material(features):
 from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts
 rejected=conflicts([f for f in features.values() if f['point_count']])
 contaminated={row['target_id'] for row in rejected}
 for feature in features.values():
  if feature['target_id'] not in contaminated:continue
  archive={k:deepcopy(feature[k]) for k in ('segments','source_circle_fit','geometric_interpretation','joint_group_navigation') if k in feature}
  feature['rejected_material_candidate']=archive
  feature['segments']=[];feature['point_count']=0
  for name in ('source_circle_fit','source_evidence_group_id','geometric_interpretation','joint_group_navigation'):feature.pop(name,None)
  reason='current extra-outline circle estimate contaminated by independently identified filled-house material band; physical ownership of remaining estimate not established'
  feature['source_material_identity_rejections']=[r for r in rejected if r['target_id']==feature['target_id']]
  feature['extraction_diagnostic'].update(physical_identity_rejected=True,missing_reason=reason)
 return features
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def warp(uv,h):return cv2.perspectiveTransform(np.asarray(uv,np.float64)[None],h)[0]
def ellipse_points(e,n=513):
 center,axes,angle=e;t=np.linspace(0,2*np.pi,n);a=np.deg2rad(angle)
 r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 return np.c_[np.cos(t),np.sin(t)]*(np.asarray(axes)/2)@r.T+center

def current_broadcast_exclusion(bgr):
 mask,legacy=broadcast_exclusion(bgr)
 # The official broadcast timer stays in this source rectangle throughout
 # this shot. Its translucent grey-white background and grey digits can fail
 # both of the older absolute white/black component thresholds.
 rectangle=[1692,944,1864,1002]
 gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(float)
 body=gray[948:997,1697:1859];glyph=gray[956:994,1720:1848]
 background=float(np.quantile(body,.85));dark=glyph<min(190.,background-22)
 dark_count=int(dark.sum());yy,xx=np.nonzero(dark)
 span=[int(np.ptp(xx))+1,int(np.ptp(yy))+1] if len(xx) else [0,0]
 present=dark_count>=100 and span[0]>=55 and span[1]>=16
 if present:
  x0,y0,x1,y1=rectangle;mask[y0-4:y1+5,x0-4:x1+5]=True
 return mask,dict(actual_native_rectangle_xyxy_inclusive=rectangle if present else legacy['actual_native_rectangle_xyxy_inclusive'],
   current_source_timer_present=present or legacy['current_source_timer_present'],edge_halo_px=4,
   method='same-shot fixed source timer region confirmed by current grey digit mass and rectangle background; union with legacy components',
   current_source_glyph_dark_pixel_count=dark_count,current_source_glyph_extent_px=span,current_source_timer_background_q85=background,
   legacy_component_detector=legacy,applied_to_source_readouts_and_line_visibility=True)

def current_ice_visible_domain(index):
 packet=read(DATA/f'board_candidates/{index:06d}.json')
 contact=[f for f in packet['features'] if f['target_id'].startswith('board.contact')]
 points=np.asarray([q for f in contact for s in f['segments'] for q in s.get('source_native_readout_uv',s['uv'])],float)
 if len(points)<8:raise ValueError('Current source board-contact support missing for ice-visible domain '+str(index))
 # Preserve each currently measured straight/corner part. The continuous
 # board can turn through an arc; it cannot be flattened to one source line.
 border=np.full(1920,np.nan);axis=np.arange(1920);part_records=[]
 for f in contact:
  for si,s in enumerate(f['segments']):
   uv=np.asarray(s.get('source_native_readout_uv',s['uv']),float)
   if len(uv)<2:continue
   uv=uv[np.argsort(uv[:,0])];xx,inv=np.unique(uv[:,0],return_inverse=True)
   yy=np.array([uv[inv==j,1].max() for j in range(len(xx))])
   if len(xx)<2:continue
   visible=(axis>=xx[0])&(axis<=xx[-1]);values=np.interp(axis[visible],xx,yy)
   border[visible]=np.fmax(border[visible],values)
   part_records.append(dict(target_id=f['target_id'],segment_index=si,native_readout_count=len(uv),source_x_interval=[float(xx[0]),float(xx[-1])]))
 supported=np.isfinite(border)
 if supported.sum()<50:raise ValueError('Insufficient measured contact outline '+str(index))
 supported_ids=np.flatnonzero(supported);filled=np.interp(axis,supported_ids,border[supported])
 # Across missing/occluded intervals, use the lower of the measured flanks
 # as a conservative exclusion boundary. These coordinates are never points.
 unsupported=[]
 for first,last in runs(np.flatnonzero(~supported)):
  flank=[border[q] for q in (first-1,last+1) if 0<=q<1920 and supported[q]]
  if flank:filled[first:last+1]=max(flank)
  unsupported.append([int(first),int(last)])
 mask=np.arange(1080)[:,None]<=filled[None]
 record=dict(method='piecewise_current_native_board_contact_material_polylines_lower_envelope_with_conservative_occluded_gaps',
  board_contact_candidate_path=str(DATA/f'board_candidates/{index:06d}.json'),native_source_pts=packet['source']['native_pts'],
  current_source_contact_readout_count=len(points),current_source_contact_material_uv_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
  source_contact_parts=part_records,unsupported_x_intervals_used_only_as_conservative_exclusion=unsupported,
  representatives_and_all_signed_interface_context_must_be_below_source_contact=True,
  model_projection_used_for_domain=False,straight_and_arc_parts_preserved=True,exclusion_gap_coordinates_emitted_as_observations=False)
 return mask,record

ANNOTATIONS={
 300:('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00010.png',
      [[18,653,2019,589],[18,704,2019,640],[18,772,2019,698],[18,857,2019,778],[18,966,2019,875],[18,1125,2019,1011]],'B2:segment0',[18,583,2019,1053]),
 400:('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00011.png',
      [[25,671,2025,667],[25,714,2025,714],[25,769,2025,769],[25,845,2025,845],[25,936,2025,936],[25,1070,2025,1078]],'C:segment0',[775,1125,982,566]),
 500:('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00012.png',
      [[25,710,2025,764],[25,757,2025,814],[25,814,2025,878],[25,886,2025,965],[25,971,2025,1062],[25,1104,430,1125]],'B1:segment0',[25,1022,1848,650])}
LINE_NAMES=['lane4:edge2','lane3:edge0','house_connection_F3_R3:segment0','lane3:edge2','lane2:edge0','house_connection_F2_R2:segment0']
LONG_LINE_ORDER=['lane4:edge0','house_connection_F4_R4:segment0','lane4:edge2','lane3:edge0',
 'house_connection_F3_R3:segment0','lane3:edge2','lane2:edge0','house_connection_F2_R2:segment0',
 'lane2:edge2','lane1:edge0','house_connection_F1_R1:segment0','lane1:edge2']

def image_line(f):
 z=f['source_line_fit'];o=np.asarray(z['point_origin']);v=np.asarray(z['unit_tangent']);n=np.asarray(z['unit_normal']);q=z['normal_offset_coefficients']
 direction=v+n*q[0];point=o+n*q[1]
 if abs(direction[0])<1e-8:return None
 m=float(direction[1]/direction[0]);y=float(point[1]+m*(960-point[0]))
 uv=np.asarray([r['uv'] for r in z['direct_readouts']]);span=float(np.ptp(uv[:,0]))
 return dict(slope=m,y960=y,readout_span_px=span,readout_count=len(uv),residual_px=z['residual_max_px'])

def sparse_representative_cap(features,maximum=170):
 """Remove only redundant collinear representatives, never material readouts."""
 removed=[]
 while sum(f['point_count'] for f in features)>maximum:
  choices=[]
  for fi,f in enumerate(features):
   if 'source_line_fit' not in f:continue
   for si,s in enumerate(f['segments']):
    uv=np.asarray(s['uv'])
    for pi in range(1,len(uv)-1):
     choices.append((float(np.linalg.norm(uv[pi+1]-uv[pi-1])),fi,si,pi))
  if not choices:raise ValueError('No redundant straight representatives remain under requested cap')
  _,fi,si,pi=min(choices);f=features[fi];s=f['segments'][si]
  removed.append(dict(target_id=f['target_id'],segment_index=si,representative_index_before_removal=pi,uv=s['uv'][pi],
    reason='collinear_representative_redundant_same_native_line_fit; material_readouts_and_visible_segment_endpoints_retained'))
  for key in ('uv','sigma_px','source_evidence'):s[key].pop(pi)
  s['vertex_count']-=1;f['point_count']-=1
 return removed

def prepare_guides():
 guides=[];receipts=[]
 sift=cv2.SIFT_create(nfeatures=4000)
 for index,(filename,lines,cross,crossline) in ANNOTATIONS.items():
  annotated=cv2.imread(filename);native=cv2.imread(str(DATA/f'frames/{index:06d}.png'))
  if annotated is None or native is None:raise ValueError('Missing anchor frame '+str(index))
  ka,da=sift.detectAndCompute(cv2.cvtColor(annotated,cv2.COLOR_BGR2GRAY),None)
  kb,db=sift.detectAndCompute(cv2.cvtColor(native,cv2.COLOR_BGR2GRAY),None)
  matches=cv2.BFMatcher().knnMatch(da,db,k=2);good=[a for a,b in matches if a.distance<.7*b.distance]
  aa=np.array([ka[m.queryIdx].pt for m in good]);bb=np.array([kb[m.trainIdx].pt for m in good])
  h,keep=cv2.findHomography(aa,bb,cv2.RANSAC,2.)
  if h is None or int(keep.sum())<25:raise ValueError('Annotation source registration failed')
  residual=np.linalg.norm(warp(aa,h)-bb,axis=1)[keep[:,0]>0]
  receipts.append(dict(frame=index,annotation=filename,annotation_sha256=sha(filename),screenshot_to_native_homography=h.tolist(),
                       static_image_registration_inliers=int(keep.sum()),median_registration_px=float(np.median(residual))))
  for name,line in zip(LINE_NAMES+[cross],lines+[crossline]):
   guides.append(dict(frame=index,curve_id=name,uv=warp(np.asarray(line).reshape(2,2),h).tolist(),kind='line',authority='user_manual_annotation_20261014',use='identity_and_search_only'))
  # Small coloured circles are the user's drawn trace, separate from labels.
  if index in (300,400):
   b,g,r=cv2.split(annotated.astype(float))
   color=((g>r*1.35)&(g>b*1.2)&(g>80)) if index==400 else ((r>b*.45)&(b>r*1.1)&(b>g*1.35)&(r>g*1.2))
   color[:700 if index==400 else 510]=False
   if index==300:color[620:]=False
   n,labels,stats,centres=cv2.connectedComponentsWithStats(color.astype(np.uint8),8)
   points=[centres[k] for k in range(1,n) if 8<=stats[k,4]<=100 and 3<=stats[k,2]<=13 and 3<=stats[k,3]<=13]
   if len(points)>=8:
    uv=warp(points,h);e=cv2.fitEllipse(uv.astype(np.float32))
    guides.append(dict(frame=index,curve_id='C0:outer' if index==400 else 'R4:outer',uv=uv.tolist(),ellipse=[list(e[0]),list(e[1]),float(e[2])],kind='circle',authority='user_manual_annotation_20261014',use='identity_and_search_only'))
 # Previously approved R3/R4 material layers supply a local shape pattern.
 p=read(BASE/'continuation_101_200_50fps/frames/000200.json')
 for f in p['features']:
  if not f['id'].startswith('ice.R') or f['point_count']<8:continue
  uv=np.asarray([q for s in f['segments'] for q in s['uv']],np.float32)
  if len(uv)<8:continue
  e=cv2.fitEllipse(uv)
  guides.append(dict(frame=200,curve_id=f['world_curve'],uv=uv.tolist(),ellipse=[list(e[0]),list(e[1]),float(e[2])],kind='circle',authority='user_accepted_frame_200',use='identity_and_search_only'))
 output=dict(schema='rinkspace.human_shape_navigation.v1',guides=guides,registrations=receipts,source_pixels_used_as_final_observations=False)
 write(DATA/'user_annotations/native_shape_navigation.json',output)
 print(json.dumps({'guides':len(guides),'registration':receipts}),flush=True)

class NativeProfiles:
 def __init__(self,rgb,source,exclusion):
  self.frame=_MaterialProfileFrame(rgb,dict(source_sha256=source['source_video_sha256'],source_pts=source['native_pts'],time_base=source['time_base'],native_hw=[1080,1920],shot_id=1))
  self.mask=exclusion;self.archive={}
 def sample(self,path,guide,radius=28,spacing=48):
  if path['feature_id'] in ('B1','B2','C'):
   return self.sample_complete_colour_band(path,guide,spacing)
  local=dict(path)
  if path.get('_circle'):local['_projected_center']=np.asarray(cv2.fitEllipse(np.asarray(guide,np.float32))[0])
  support_radius=radius
  if path.get('_separate_profile_support'):
   # Identity bounds the centre search, not the extent needed to see both
   # material flanks and their ice plateaus on a motion-blurred exposure.
   support_radius+=.5*path.get('_maximum_width',60)+2*path.get('_maximum_sigma',15)
  query,normal,interval=_source_queries(local,guide,np.ones(len(guide)),(1080,1920),spacing,support_radius)
  if not len(query):return [],dict(query_count=0)
  profiles=_batched_actual_profiles(self.frame,query,normal,interval,np.full(len(query),path['material_signal']))
  rows=[];offsets,values,smooth,valid=profiles
  for k in range(len(query)):
   ok=valid[k];p=values[k,ok];s=smooth[k,ok];off=offsets[k,ok]
   candidates=_actual_profile_candidates(query[k],normal[k],off,p,s,operator=path['operator'],signal=path['material_signal'],transition=path['material_transition'],with_quantiles=True)
   eligible=[]
   for uv,sides,bad,quantiles in candidates:
    qo=(quantiles-query[k])@normal[k];sigma=float(np.max(np.abs(qo[:,2]-qo[:,0]))/1.349)
    contrast=[abs(float(np.interp(o[2],off,s)-np.interp(o[0],off,s))) for o in qo]
    shift=float((uv-query[k])@normal[k]);width=float(np.linalg.norm(sides[1]-sides[0]))
    context=np.vstack([uv[None],sides,quantiles.reshape(-1,2)])
    ip=np.rint(context).astype(int)
    if bad or sigma>path.get('_maximum_sigma',15) or abs(shift)>radius-4 or min(contrast)<path.get('_minimum_contrast',3.0):continue
    if path['operator']=='paint_center' and (width<1.5 or width>path.get('_maximum_width',60)):continue
    if (context<[0,0]).any() or (context>[1919,1079]).any() or self.mask[ip[:,1],ip[:,0]].any():continue
    evidence=dict(query_index=k,query_uv=query[k].tolist(),profile_normal=normal[k].tolist(),native_profile_shift_px=shift,
       actual_source_sides_uv=sides.tolist(),actual_source_side_quantiles_uv=quantiles.tolist(),signed_interface_iqr_contrast=contrast,
       transition_sigma_px=max(1.,sigma),measurement_origin='current_native_signed_RGB_material_profiles',material_signal=path['material_signal'],
       actual_source_support_outside_person_and_broadcast_masks=True,
       center_search_radius_px=radius,material_profile_support_radius_px=support_radius)
    eligible.append((abs(shift),dict(uv=np.round(uv,4).tolist(),sigma_px=max(1.,sigma),source_evidence=evidence)))
   eligible.sort(key=lambda r:r[0])
   if path.get('_collect_line_candidates'):rows.extend(v[1] for v in eligible)
   elif eligible and not(len(eligible)>1 and eligible[1][0]-eligible[0][0]<2):rows.append(eligible[0][1])
  return rows,dict(query_count=len(query),measured_count=len(rows))
 def sample_complete_colour_band(self,path,guide,spacing):
  from calibration.program_tracking.native_ice_material_repair import sample_chroma,whole_colour_bands
  radius=path.get('_whole_band_profile_radius',185)
  query,normal,interval=_source_queries(path,guide,np.ones(len(guide)),(1080,1920),spacing,radius)
  if not len(query):return [],dict(query_count=0)
  offsets,profiles,valid=sample_chroma(self.frame.rgb,query,normal,interval,path['material_signal'])
  rows=[]
  for k in range(len(query)):
   # Remap padding beyond the native image is not an ice background. Read
   # only the contiguous in-frame profile before estimating its two flanks.
   visible=valid[k]
   candidates=whole_colour_bands(query[k],normal[k],offsets[k,visible],profiles[k,visible],np.ones(int(visible.sum()),bool))
   for b in candidates:
    uv,sides,quantiles=b['uv'],b['sides'],b['quantiles'];context=np.vstack([uv[None],sides,quantiles.reshape(-1,2)])
    ip=np.rint(context).astype(int)
    if (context<[0,0]).any() or (context>[1919,1079]).any() or self.mask[ip[:,1],ip[:,0]].any():continue
    # Both complete flanks must be clear current ice. A body/board/timer
    # occlusion within the coloured band cannot authenticate its centre.
    off=offsets[k];lo,hi=b['complete_band_native_interval'];used=(off>=lo)&(off<=hi)
    support=query[k]+off[used,None]*normal[k];sp=np.rint(support).astype(int)
    if self.mask[sp[:,1],sp[:,0]].any():continue
    sigma=float(np.max(np.linalg.norm(quantiles[:,2]-quantiles[:,0],axis=1))/1.349)
    evidence=dict(query_index=k,query_uv=query[k].tolist(),profile_normal=normal[k].tolist(),
      native_profile_shift_px=float((uv-query[k])@normal[k]),actual_source_sides_uv=sides.tolist(),
      actual_source_side_quantiles_uv=quantiles.tolist(),signed_interface_iqr_contrast=[b['amplitude']*.5]*2,
      transition_sigma_px=max(1.,sigma),measurement_origin='current_native_whole_chromatic_band_ice_side_level_crossings',
      material_signal=path['material_signal'],whole_band_method=b['method'],complete_band_native_interval_px=[lo,hi],
      whole_band_profile_residual_median=b['whole_band_residual_median'],joint_band_width_px=b['width_px'],
      internal_chromatic_ridges_not_used_as_outer_edges=True,actual_source_support_outside_person_and_broadcast_masks=True)
    rows.append(dict(uv=np.round(uv,4).tolist(),sigma_px=max(1.,sigma),source_evidence=evidence))
  return rows,dict(query_count=len(query),measured_count=len(rows),operator='whole_native_colour_band_two_ice_side_material_interfaces')

def final_support_line_fit(stations, normal_offsets, support_weights):
 """Estimate the emitted mean from its final native support, without a guide prior.

 Family direction and robust hypotheses may select a material branch. Once
 that support is fixed, the same weighted linear estimator defines its mean
 and native-readout influence; clipping or a median must not survive here.
 """
 t=np.asarray(stations,float);d=np.asarray(normal_offsets,float);w=np.asarray(support_weights,float)
 if (t.ndim!=1 or d.shape!=t.shape or w.shape!=t.shape or len(t)<2
     or not np.isfinite(t).all() or not np.isfinite(d).all() or not np.isfinite(w).all()
     or (w<=0).any() or np.ptp(t)<=0):
  raise ValueError('Final native line support requires separated finite stations and positive weights')
 center=float(np.mean(t));span=float(np.ptp(t));root=np.sqrt(w/np.max(w))
 design=np.c_[(t-center)/span,np.ones(len(t))]
 scaled=np.linalg.lstsq(design*root[:,None],d*root,rcond=None)[0]
 return np.array([scaled[0]/span,scaled[1]-scaled[0]*center/span])


class Extractor:
 def __init__(self,candidate_output_root=None,candidate_model_root=None):
  self.output_root=Path(candidate_output_root) if candidate_output_root else DATA
  self.model_root=Path(candidate_model_root) if candidate_model_root else MODEL
  self.scene=read(self.model_root/'scene.json');self.paths=ice_curve_inventory(self.scene)
  self.circle_geometry_sha256=circle_geometry_digest(self.scene)
  circles={f['id'] for f in self.scene['ice_features'] if f['kind']=='circle'}
  for p in self.paths:p['_circle']=p['feature_id'] in circles
  self.paths=[p for p in self.paths if p['world_curve_role']!='center' and p['operator']!='unresolved' and p['curve_id']!='curling_centerline_row3:segment0']
  a=np.load(DATA/'camera_seeds.npz');self.cameras={int(i):c for i,c in zip(a['programme_indices'],a['camera25'])}
  old=np.load(DATA.parent/'continuation_101_200_50fps/camera_seeds.npz');self.cameras[200]=old['camera25'][-1]
  self.guides=read(DATA/'user_annotations/native_shape_navigation.json')['guides'];self.previous={};self.previous_index=None
  recovered=DATA/'user_annotations/recovered_ice_guides.json'
  if recovered.exists():
   ledger=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/history/tracking_runs/source_cadence_comparison/O6j8L8TmfKo/full_programme_alignment/native60_to_official50.jsonl')
   mapping=[json.loads(s) for s in ledger.read_text(encoding='utf-8').splitlines()]
   mapping=[r for r in mapping if r['shot_index']==1 and r.get('accepted_official_pts') is not None]
   for oldguide in read(recovered)['guides']:
    match=min(mapping,key=lambda r:abs(r['full_pts']-oldguide['source_legacy_pts']))
    if abs(match['full_pts']-oldguide['source_legacy_pts'])>512:continue
    index=1+(match['accepted_official_pts']-256256)//256
    if index not in self.cameras:continue
    cid=oldguide['target_id'].removeprefix('ice.').replace('C:0','C:segment0')
    uv=np.asarray(oldguide['native_source_uv'],np.float32)
    guide=dict(frame=index,curve_id=cid,uv=uv.tolist(),authority='recovered_user_hand_trace:'+str(oldguide['source_legacy_pts']),use='legacy_trace_identity_and_search_only_native_current_pixels_reread')
    if cid.endswith(':outer'):
     e=cv2.fitEllipse(uv);guide.update(kind='circle',ellipse=[list(e[0]),list(e[1]),float(e[2])])
    else:
     direction=np.linalg.svd(uv-uv.mean(0),full_matrices=False)[2][0];t=(uv-uv.mean(0))@direction
     guide.update(kind='line',uv=[(uv.mean(0)+direction*q).tolist() for q in (t.min(),t.max())])
    self.guides.append(guide)
  manifest=self.output_root/'frame_manifest.json'
  self.rows={r['index']:r for r in read(manifest if manifest.exists() else DATA/'frame_manifest.json')['frames']}
  qualification=self.output_root/'temporal_diagnostics/extra_ice_unqualified_query_domains.json'
  self.query_qualification=read(qualification) if qualification.exists() else None
  self.query_qualification_path=qualification
  self.hcache={}
  # Work in the undistorted ground plane for transport. Fitting a single
  # homography to distorted projections of arbitrary off-screen/behind-camera
  # stations can send a later visible line millions of pixels away.
  for g in self.guides:
   p=next((p for p in self.paths if p['curve_id']==g['curve_id']),None)
   if p is None:continue
   xy,valid=ground_xy(self.cameras[g['frame']],np.asarray(g['uv'],float));xy=xy[valid]
   if len(xy)<2:continue
   if g['kind']=='circle' and len(xy)>=5:
    z=np.linalg.lstsq(np.c_[xy,np.ones(len(xy))],-(xy**2).sum(1),rcond=None)[0]
    center=-z[:2]/2;radius2=float(center@center-z[2])
    if radius2<=0:continue
    theta=np.linspace(0,2*np.pi,513)
    world=np.c_[center[0]+np.sqrt(radius2)*np.cos(theta),center[1]+np.sqrt(radius2)*np.sin(theta),np.zeros(len(theta))]
   elif g['kind']=='line':
    center=xy.mean(0);direction=np.linalg.svd(xy-center,full_matrices=False)[2][0]
    extent=(np.asarray(p['xyz'])[:,:2]-center)@direction
    # Model supplies the finite physical extent, human source pixels supply
    # the search-line position/direction. Neither becomes emitted observation.
    stations=np.linspace(extent.min(),extent.max(),257)
    world=np.c_[center+stations[:,None]*direction,np.zeros(len(stations))]
   else:continue
   g['_world_xyz']=world
  from calibration.program_tracking.native_ice_material_repair import NestedHouseReader
  motion_path=self.output_root/'temporal_diagnostics/source_ice_motion_all.npz'
  self.house_reader=NestedHouseReader(self.scene,self.paths,self.guides,self.cameras,motion_path) if motion_path.exists() else None
  from calibration.program_tracking.native_circle_identity_review import install_reviewed_circle_material_domains
  install_reviewed_circle_material_domains(self.house_reader,self.output_root)
  from calibration.program_tracking.native_ice_line_continuity import NativeLineIdentity
  quality_path=self.output_root/'temporal_diagnostics/all_source_motion_baseline.json'
  self.line_identity=NativeLineIdentity(quality_path,self.guides) if quality_path.exists() else None
  if self.line_identity is not None:
   self.line_identity.seed_accepted_frame(BASE/'continuation_101_200_50fps/frames/000200.json',BASE/'continuation_101_200_50fps/user_acceptance.json')
 def homography(self,a,b):
  key=(a,b)
  if key not in self.hcache:
   def plane(c):
    r=cv2.Rodrigues(c[3:6])[0];k=np.array([[np.exp(c[6]),c[10],c[8]],[0,np.exp(c[7]),c[9]],[0,0,1.]])
    return k@np.column_stack((r[:,:2],-r@c[:3]))
   self.hcache[key]=plane(self.cameras[b])@np.linalg.inv(plane(self.cameras[a]))
  return self.hcache[key]
 def guide(self,p,index):
  model,depth=project_fast(self.cameras[index],p['xyz']);guide=model;origin='8a28_full_model_shape'
  choices=[g for g in self.guides if g['curve_id']==p['curve_id']]
  if choices:
   g=min(choices,key=lambda g:abs(g['frame']-index))
   if '_world_xyz' in g:
    guide,depth=project_fast(self.cameras[index],g['_world_xyz'])
   origin=g['authority']+':'+str(g['frame'])+':distortion_aware_ground_transport'
  elif self.previous_index==index-1 and p['curve_id'] in self.previous:
   xy,valid=ground_xy(self.cameras[index-1],self.previous[p['curve_id']]);world=np.c_[xy[valid],np.zeros(valid.sum())]
   if len(world)>=2:guide,depth=project_fast(self.cameras[index],world)
   origin='previous_native_shape_distortion_aware_ground_transport'
  good=np.isfinite(guide).all(1)&(depth>0.05)
  guide=guide[good]
  if not p['_circle'] and len(guide)>=2:
   near=(guide>=[-150,-150]).all(1)&(guide<=[2069,1229]).all(1)
   if near.sum()>=2:
    points=guide[near];center=points.mean(0);v=np.linalg.svd(points-center,full_matrices=False)[2][0]
    extent=(guide-center)@v;lo=float(extent.min());hi=float(extent.max())
    # Clip before resampling: uniform stations over a pole's remote extension
    # otherwise leave only a handful of samples inside the source frame.
    for axis,bounds in enumerate(((-120,2039),(-120,1199))):
     if abs(v[axis])>1e-12:
      ends=sorted((np.asarray(bounds)-center[axis])/v[axis]);lo=max(lo,float(ends[0]));hi=min(hi,float(ends[1]))
    guide=np.linspace(center+v*lo,center+v*hi,max(2,int(max(0,hi-lo)/8)+1)) if hi>lo else np.empty((0,2))
  return guide,origin
 def line(self,p,guide,native,family_slope=None):
  f=blank_feature(p);a,b=guide[0],guide[-1];v=b-a;length=np.linalg.norm(v)
  if length<10:return f,[],None
  v/=length;n=np.array([-v[1],v[0]])
  local=dict(p,_collect_line_candidates=True,_minimum_contrast=1.6)
  broad=p['feature_id'] in ('B1','B2','C')
  if broad:local.update(_maximum_width=160,_maximum_sigma=24)
  from calibration.program_tracking.native_ice_line_continuity import thin_material_width_limit,record_thin_material_width,current_material_intervals
  width_policy=thin_material_width_limit(p,native)
  if width_policy:local['_maximum_width']=width_policy['maximum_width_px']
  policy=p.get('_line_identity_policy',{})
  rows,diagnostic=native.sample(local,guide,radius=100 if broad else policy.get('search_radius_px',32),
    spacing=p.get('_source_query_spacing_px',55))
  from calibration.program_tracking.native_ice_line_continuity import identity_filter
  rows=identity_filter(rows,guide,policy)
  f['native_static_line_identity_policy']=policy
  if width_policy:f['current_thin_paint_material_width_reference']=width_policy
  f['extraction_diagnostic']=dict(**diagnostic,eligible_native_material_readouts=len(rows),
    acceptance='current source two-sided material contrast plus coherent straight source family; absence is not filled from a guide')
  # A short visible source segment may carry only two separated readouts.
  # It still requires two-sided material support; navigation creates no point.
  native_visible=(guide>=[0,0]).all(1)&(guide<=[1919,1079]).all(1)
  visible_length=float(np.ptp((guide[native_visible]-a)@v)) if native_visible.sum()>=2 else 0.
  short=0<visible_length<500 or bool(policy.get('current_identity_constrained'))
  minimum=2 if short else 4;minimum_span=35 if short else 140
  if len(rows)<minimum:return f,[],None
  uv=np.asarray([r['uv'] for r in rows]);t=(uv-a)@v;d=(uv-a)@n
  qi=np.asarray([r['source_evidence']['query_index'] for r in rows]);unique=np.unique(qi)
  # Competing broad-band shoulders are ranked by their measured two-sided
  # colour signal. Numerous weak reflections must not outvote the actual
  # painted strip. This changes branch selection, not the visibility gate.
  signal=np.asarray([min(r['source_evidence']['signed_interface_iqr_contrast']) for r in rows])
  support_weight=(signal/max(1.,float(np.median(signal))))**2 if broad else np.ones(len(rows))
  # A nearby circle or reflection can dominate a least-squares fit. Choose
  # one coherent straight material trace across separated source stations;
  # multiple candidates of one profile do not count as extra evidence.
  hypotheses=[(0.,float(q)) for q in np.unique(np.round(d))]
  candidates=np.unique(np.linspace(0,len(rows)-1,min(35,len(rows))).astype(int))
  for first in candidates:
   for last in candidates:
    if t[last]-t[first]<(35 if short else 250):continue
    slope=(d[last]-d[first])/(t[last]-t[first])
    if abs(slope)<=(.35 if broad else .08):hypotheses.append((float(slope),float(d[first]-slope*t[first])))
  expected_q=None
  if family_slope is not None:
   expected_q=float((v[1]-family_slope*v[0])/(family_slope*n[0]-n[1]))
   hypotheses.extend((expected_q,float(q)) for q in np.unique(np.round(d-expected_q*t)))
  best=None
  for slope,intercept in hypotheses:
   if expected_q is not None and abs(slope-expected_q)*length>7:continue
   residual=abs(d-(slope*t+intercept));selected=[]
   for key in unique:
    ids=np.flatnonzero(qi==key);k=ids[np.argmin(residual[ids])]
    if residual[k]<=3.5:selected.append(k)
   if len(selected)<minimum:continue
   selected=np.asarray(selected,int);span=float(np.ptp(t[selected]))
   if span<minimum_span:continue
   mid=float(np.median(t[selected]));offset=abs(slope*mid+intercept)
   score=float(support_weight[selected].sum())+span/500.-(.015 if broad else .20)*offset-(5 if broad else 40)*abs(slope)
   if best is None or score>best[0]:best=(score,selected)
  if best is None:return f,[],None
  active=np.zeros(len(rows),bool);active[best[1]]=True
  for _ in range(3):
   fit=np.polyfit(t[active],d[active],1,w=np.sqrt(support_weight[active]))
   if expected_q is not None:
    fit[0]=np.clip(fit[0],expected_q-7/length,expected_q+7/length)
    fit[1]=float(np.median(d[active]-fit[0]*t[active]))
   res=d-np.polyval(fit,t);selected=[]
   for key in unique:
    ids=np.flatnonzero(qi==key);k=ids[np.argmin(abs(res[ids]))]
    if abs(res[k])<4.0:selected.append(k)
   active[:]=False;active[selected]=True
   if active.sum()<minimum:return f,[],None
  if active.sum()<minimum or np.ptp(t[active])<minimum_span:return f,[],None
  # The last selection can change support after the third candidate fit.
  # Emit one unconstrained final-support WLS mean, matching its source D.
  fit=final_support_line_fit(t[active],d[active],support_weight[active])
  res=d-np.polyval(fit,t)
  if abs(fit[0])>(.35 if broad else .08):return f,[],None
  rows=[r for r,ok in zip(rows,active) if ok];uv=uv[active];t=t[active]
  # Restrict material support to the current-frame readout span. Extend a
  # small sampling cell only; never fill a missing half-frame from a prior.
  lo=max(0.,t.min());hi=min(length,t.max())
  stations=np.arange(lo,hi+1,1.);points=a+stations[:,None]*v+np.polyval(fit,stations)[:,None]*n
  ip=np.rint(points).astype(int);inside=(points>=[0.,0.]).all(1)&(points<=[1919.,1079.]).all(1)
  visible=inside.copy();visible[inside]&=~native.mask[ip[inside,1],ip[inside,0]]
  # Long unsupported intervals remain gaps, not inferred observations.
  visible &= current_material_intervals(stations,t)
  group='native_line_'+str(native.frame.binding['source_pts'])+'_'+p['curve_id'];inferred=[]
  for first,last in runs(np.flatnonzero(visible)):
   if last-first<20:continue
   take=np.unique(np.rint(np.linspace(first,last,max(2,int(np.ceil((last-first)/220))+1))).astype(int))
   f['segments'].append(dict(uv=np.round(points[take],4).tolist(),vertex_count=len(take),closed=False,
      sigma_px=[max(1.,float(np.std(res[active])))]*len(take),measurement_origin='same_frame_joint_straight_edge_fit',
      representative_points_share_line_estimate=True,source_evidence_group_id=group,pixels_measured_in_this_frame=False,
      source_evidence_measured_in_this_frame=True,individual_native_pixel_readouts=False,hidden_span_filled=False,
      source_evidence=[dict(kind='shared_current_source_line_representative',source_evidence_group_id=group,individual_native_pixel_readout=False) for _ in take]))
  f.update(point_count=sum(s['vertex_count'] for s in f['segments']),source_evidence_group_id=group,
     source_line_fit=dict(point_origin=a.tolist(),unit_tangent=v.tolist(),unit_normal=n.tolist(),normal_offset_coefficients=fit.tolist(),current_frame_only=True,
      direct_readouts=rows,actual_native_readout_count=len(rows),residual_max_px=float(abs(res[active]).max()),
      parallel_family_direction_used_for_candidate_disambiguation=family_slope is not None,
      parallel_family_source_slope=family_slope,
      estimator_contract=dict(schema='rinkspace.fixed-final-native-support-line.v1',
       kind='weighted_linear_final_native_support',
       weights_policy='minimum_two_interface_contrast_squared' if broad else 'uniform_current_native_line_support',
       final_support_refit_unconstrained=True,family_direction_used_for_candidate_disambiguation_only=True,
       source_sigma_coordinates='recorded_navigation_normal_coverage_px',
       derivative_normal_projection_factor_required=True,representative_UV_quantization_digits=4)),
     geometric_interpretation=dict(kind='straight_ice_paint_centerline',inference_only=True,current_frame_native_measurement=True),
     note_zh='当前帧原图的双边材料读数共同确定直线；均匀稀疏代表点共享同一估计，人物和字幕遮挡不补点。')
  if f['point_count']:record_thin_material_width(p,native,rows)
  refined=np.array([a+v*q+n*np.polyval(fit,q) for q in (0,length)])
  return f,inferred,np.linspace(*refined,257)
 def circle(self,p,guide,native):
  f=blank_feature(p)
  if not np.isfinite(guide).all():return f,None
  e=cv2.fitEllipse(np.asarray(guide,np.float32));axes=np.asarray(e[1]);center=np.asarray(e[0]);a=np.deg2rad(e[2])
  if min(axes)<5 or max(axes)>50000:return f,None
  local=dict(p)
  if p['operator']=='paint_center':local.update(_maximum_width=20,_minimum_contrast=1.6,_maximum_sigma=10.)
  rows,diag=native.sample(local,guide,radius=42 if p['operator']=='paint_center' else 28,spacing=38)
  f['extraction_diagnostic']=dict(**diag,eligible_native_material_readouts=len(rows),
    acceptance='current source material readouts belonging to a single projected physical circle; absent material is not filled')
  if len(rows)<5:return f,None
  uv=np.asarray([r['uv'] for r in rows]);q=np.asarray([r['source_evidence']['query_uv'] for r in rows]);n=np.asarray([r['source_evidence']['profile_normal'] for r in rows])
  off=np.einsum('ij,ij->i',uv-q,n)
  design=np.c_[n,np.einsum('ij,ij->i',q-center,n)]
  sol=least_squares(lambda z:np.r_[design@z-off,z[0]/12,z[1]/12,z[2]/.015],np.zeros(3),bounds=([-24,-24,-.025],[24,24,.025]),loss='soft_l1',f_scale=2.5)
  e2=(tuple(center+sol.x[:2]),tuple(axes*(1+sol.x[2])),e[2]);refined=ellipse_points(e2)
  # Re-measure after aligning a complete-circle hypothesis to this source.
  rows,diag=native.sample(local,refined,radius=32 if p['operator']=='paint_center' else 18,spacing=38)
  if not rows:return f,None
  uv=np.asarray([r['uv'] for r in rows]);dist=ellipse_distance(uv,e2)
  # Curvature is identity evidence. Interior marks cannot become arbitrary
  # extra branches merely because they are near the search strip.
  keep=dist<=5.0
  rows=[r for r,k in zip(rows,keep) if k]
  if len(rows)<3:return f,None
  groups=[];part=[];last=None
  for row in rows:
   qi=row['source_evidence']['query_index']
   if part and (qi-last>2 or np.linalg.norm(np.asarray(row['uv'])-part[-1]['uv'])>100):groups.append(part);part=[]
   part.append(row);last=qi
  if part:groups.append(part)
  for part in groups:
   if len(part)<2:continue
   points=np.asarray([r['uv'] for r in part]);take=simplify(points,tolerance=2.4,maximum_gap=220,closed=False)
   selected=[part[k] for k in take]
   f['segments'].append(dict(uv=[r['uv'] for r in selected],vertex_count=len(selected),closed=False,sigma_px=[r['sigma_px'] for r in selected],
      source_evidence=[r['source_evidence'] for r in selected],measurement_origin='current_native_signed_RGB_material_profiles',
      pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True,individual_native_pixel_readouts=True,hidden_span_filled=False))
  f.update(point_count=sum(s['vertex_count'] for s in f['segments']),geometric_interpretation=dict(kind='projected_physical_circle_ellipse',inference_only=True,
    center_uv=list(e2[0]),axes_px=list(e2[1]),angle_degrees=e2[2],maximum_readout_to_ellipse_px=float(dist[keep].max()),inferred_uv_not_emitted_as_source_points=True),
    note_zh='按同一实体圆的投影椭圆识别材料轮廓；坐标均为本帧原片剖面读数，不复制模型投影或前帧点。')
  return f,refined
 def nested_house_features(self,index,native):
  output={};diagnostic={}
  if self.house_reader is None:return output,diagnostic
  cache=self.output_root/f'circle_candidates/{index:06d}.json'
  algorithm_sha=sha(ROOT/'calibration/program_tracking/native_ice_material_repair.py')
  if getattr(self,'use_circle_cache',True) and cache.exists():
   saved=read(cache);source=self.rows[index]['source']
   if (saved.get('algorithm_sha256')==algorithm_sha and saved['source']['native_pts']==source['native_pts']
       and saved['source']['mask_sha256']==sha(source['mask_path'])
       and saved.get('circle_geometry_sha256',circle_geometry_digest(read(MODEL/'scene.json')) if saved['model_scene_sha256']==sha(MODEL/'scene.json') else None)==self.circle_geometry_sha256):
    for feature in saved['features_by_curve'].values():
     feature['source_schematic_material_signal']=feature.get('material_signal')
     feature['material_signal']='dark'
     feature['extraction_diagnostic']['acceptance']='current native narrow-paint midpoint or declared filled-house material interface, jointly consistent with one physical circle family'
    return qualify_extra_circle_material(saved['features_by_curve']),saved['diagnostics']
  for fid,group in self.house_reader.groups.items():
   results,counts=self.house_reader.read(fid,index,native);diagnostic[fid]=counts
   if not group['outline'] and not results and 2<=counts.get('outer',{}).get('eligible_native_material_readouts',0)<=5:
    from calibration.program_tracking.native_ice_short_arc_repair import short_native_outer_arc
    results,counts=short_native_outer_arc(self.house_reader,fid,index,native,output_root=self.output_root);diagnostic[fid]=counts
   for path in group['paths']:
    role=path['world_curve_role'];f=blank_feature(path)
    f.update(material_signal='dark',source_schematic_material_signal=path['material_signal'])
    f['extraction_diagnostic']=dict(counts.get(role,dict(query_count=0,eligible_native_material_readouts=0)),
      acceptance='current native narrow-paint midpoint or declared filled-house material interface, jointly consistent with one physical circle family')
    result=results.get(role)
    if result:
     rows=result['rows'];parts=[];part=[];last=None
     group_id='native_circle_'+str(native.frame.binding['source_pts'])+'_'+path['curve_id']
     for row in rows:
      representative=np.asarray(row['representative_uv']);ip=np.rint(representative).astype(int)
      if (representative<[0,0]).any() or (representative>[1919,1079]).any() or native.mask[ip[1],ip[0]]:continue
      qi=row['query_index']
      if part and (qi-last>1 or np.linalg.norm(np.asarray(row['uv'])-np.asarray(part[-1]['uv']))>120):parts.append(part);part=[]
      part.append(row);last=qi
     if part:parts.append(part)
     for part in parts:
      uv=np.array([r['representative_uv'] for r in part]);take=simplify(uv,tolerance=2.0,maximum_gap=160,closed=False);selected=[part[k] for k in take]
      f['segments'].append(dict(uv=[r['representative_uv'] for r in selected],vertex_count=len(selected),closed=False,
        sigma_px=[r['sigma_px'] for r in selected],source_evidence=[dict(r['source_evidence'],kind='shared_current_source_circle_representative',actual_native_readout_uv=r['uv'],
         representative_to_actual_material_px=r['representative_to_actual_material_px'],source_evidence_group_id=group_id,individual_native_pixel_readout=False) for r in selected],
        measurement_origin='same_frame_joint_physical_circle_family_fit',pixels_measured_in_this_frame=False,
        representative_points_share_circle_family_estimate=True,source_evidence_group_id=group_id,
        source_evidence_measured_in_this_frame=True,individual_native_pixel_readouts=False,hidden_span_filled=False))
     curve=result['curve'];valid=np.isfinite(curve).all(1)
     e=cv2.fitEllipse(np.asarray(curve[valid],np.float32))
     f.update(joint_group_navigation=result['joint_group_navigation'],source_evidence_group_id=group_id,
       source_circle_fit=dict(current_frame_only=True,direct_readouts=rows,actual_native_readout_count=len(rows),
         shared_physical_family_projection=True,material_readouts_are_not_model_or_temporal_pixels=True),
       geometric_interpretation=dict(kind='projected_physical_circle_ellipse',inference_only=True,
        center_uv=list(e[0]),axes_px=list(e[1]),angle_degrees=float(e[2]),
        maximum_readout_to_ellipse_px=result['joint_group_navigation']['maximum_normal_material_to_shared_circle_px'],
        inferred_uv_not_emitted_as_direct_material_readouts=True),
       navigation_origin=result['joint_group_navigation']['source_identity_authority'],
       note_zh='各层共同使用同一冰平面投影下的实体圆心及场景半径比例；各代表点只在本帧已确认材料站位取样并共享本帧源圆拟合；原始材料读数单独保留，遮挡/缺失不补，独立圈心/尺度及靶圈填充色带中点禁止。')
    f['point_count']=sum(s['vertex_count'] for s in f['segments']);output[path['curve_id']]=f
  return qualify_extra_circle_material(output),diagnostic
 def family_audit(self,index,features,searches,native):
  """Disambiguate material branches using the parallel physical line family.

  Six human-labelled lines establish order and a same-frame source direction.
  A camera vanishing direction stabilises the estimate only when too few
  source rows are visible. Reflections cannot acquire a physical target merely
  by occurring near a model search strip.
  """
  byid={f['world_curve']:f for f in features if 'source_line_fit' in f and f['world_curve'] in LONG_LINE_ORDER}
  # Other shots may establish different finite source identities. Do not
  # silently reuse S01's six human-labelled references for every camera.
  reference_ids=set(getattr(self,'source_parallel_reference_ids',LINE_NAMES)) & set(LONG_LINE_ORDER)
  photographic_references=hasattr(self,'source_parallel_reference_ids')
  labelled=[(cid,image_line(f)) for cid,f in byid.items() if cid in reference_ids]
  samples=[(cid,z) for cid,z in labelled if z and z['readout_span_px']>=400 and z['readout_count']>=4]
  if not samples:samples=[(cid,z) for cid,z in labelled if z and z['readout_span_px']>=180 and z['readout_count']>=3]
  if not byid:return features,[],dict(status='no_long_parallel_line_points_this_frame')
  if hasattr(self,'current_native_parallel_direction'):
   direction=self.current_native_parallel_direction(index)
   if direction is None:
    source_pairs=[(a,b)for a in range(len(samples))for b in range(a)if abs(samples[a][1]['y960']-samples[b][1]['y960'])>100]
    if not source_pairs:
     return features,[],dict(status='native_parallel_direction_unavailable_and_source_family_insufficient',
       actual_material_readouts_preserved=True,parallel_family_not_certified=True)
    camera_a=float(np.median([(samples[a][1]['slope']-samples[b][1]['slope'])/(samples[a][1]['y960']-samples[b][1]['y960'])for a,b in source_pairs]))
    camera_b=float(np.median([z['slope']-camera_a*z['y960']for _,z in samples]))
   else:camera_a,camera_b=direction
   camera_authority='current_native_projection_parallel_direction'
  else:
   c=self.cameras[index];r=cv2.Rodrigues(c[3:6])[0]
   k=np.array([[np.exp(c[6]),c[10],c[8]],[0,np.exp(c[7]),c[9]],[0,0,1.]])
   path=next(p for p in self.paths if p['curve_id']=='lane3:edge0');d=np.asarray(path['xyz'])[-1]-np.asarray(path['xyz'])[0]
   vp=k@r@d
   camera_a=float(vp[2]/(960*vp[2]-vp[0]));camera_b=float(-vp[1]/(960*vp[2]-vp[0]))
   camera_authority='8a28_camera_parallel_direction'
  ys=np.asarray([z['y960'] for _,z in samples]);ms=np.asarray([z['slope'] for _,z in samples])
  pairs=[(ms[j]-ms[h])/(ys[j]-ys[h]) for j in range(len(ys)) for h in range(j) if abs(ys[j]-ys[h])>100]
  if (len(samples)>=3 and np.ptp(ys)>=180 and pairs) or (photographic_references and len(samples)>=2 and pairs):
   aa=float(np.median(pairs));origin='current_shot_photographic_parallel_line_family' if photographic_references else 'robust_current_source_human_labelled_parallel_line_family'
  elif samples:aa=camera_a;origin='camera_parallel_direction_stabilised_by_current_source_human_labelled_lines'
  else:aa=camera_a;origin=camera_authority+'_identity_check_no_reliable_current_labelled_family_reference'
  if photographic_references and samples and 'human_labelled' in origin:origin=origin.replace('human_labelled','photographic_identity')
  bb=float(np.median(ms-aa*ys)) if samples else camera_b
  rejected=[];out=[];refined_out={}
  report=dict(status='checked',reference_origin=origin,slope_vs_y960=[aa,bb],reference_target_ids=['ice.'+cid for cid,_ in samples],
   rule='long lane boundaries and F-R centres share one source vanishing point and ordered distinct source rows; B/C cross-lines excluded')
  for f in features:
   cid=f['world_curve'];z=image_line(f) if cid in byid else None
   if z is None:out.append(f);continue
   expected=aa*z['y960']+bb
   error=float(abs(z['slope']-expected)*z['readout_span_px']/np.sqrt(1+expected**2))
   audit=dict(target_id=f['target_id'],source_line=z,expected_family_image_slope=expected,orientation_departure_over_readout_span_px=error)
   if error>8:
    p,guide,origin0=searches[cid]
    replacement,_,refined=self.line(p,guide,native,family_slope=expected)
    replacement['navigation_origin']=origin0
    rz=image_line(replacement) if replacement['point_count'] else None
    error2=float(abs(rz['slope']-(aa*rz['y960']+bb))*rz['readout_span_px']/np.sqrt(1+expected**2)) if rz else None
    rejected.append(dict(**audit,reason='current_material_branch_conflicts_with_source_parallel_family',
      original_candidate=f,replacement_source_readout_count=rz['readout_count'] if rz else 0,
      replacement_orientation_error_px=error2))
    if rz is None or error2>8:continue
    f=replacement;z=rz;error=error2;refined_out[cid]=refined
   f['parallel_family_geometry_check']=dict(reference_origin=origin,expected_image_slope=float(aa*z['y960']+bb),
      orientation_departure_over_readout_span_px=error,current_frame_material_support_required=True)
   out.append(f)
  # The model fixes physical row spacing; current source rows fix its image
  # projectivity. This is an identity check only, never an emitted model point.
  source_rows=[f for f in out if f['world_curve'] in reference_ids and 'source_line_fit' in f and image_line(f)['readout_span_px']>=400]
  coordinate={p['curve_id']:float(np.mean(np.asarray(p['xyz'])[:,1])) for p in self.paths if p['curve_id'] in LONG_LINE_ORDER}
  if len(source_rows)>=4:
   rr=np.asarray([coordinate[f['world_curve']] for f in source_rows]);yy=np.asarray([image_line(f)['y960'] for f in source_rows])
   rowfit=np.linalg.lstsq(np.c_[rr,np.ones(len(rr)),-rr*yy],yy,rcond=None)[0]
   residual=yy-(rowfit[0]*rr+rowfit[1])/(rowfit[2]*rr+1)
   row_tolerance=max(10.,2*float(np.quantile(abs(residual),.9)))
   report.update(source_row_projectivity_coefficients=rowfit.tolist(),source_row_fit_residual_max_px=float(abs(residual).max()),
      source_row_identity_tolerance_px=row_tolerance)
   filtered=[]
   for f in out:
    cid=f['world_curve']
    if cid not in LONG_LINE_ORDER or cid in reference_ids or 'source_line_fit' not in f:filtered.append(f);continue
    z=image_line(f);r0=coordinate[cid];pred=float((rowfit[0]*r0+rowfit[1])/(rowfit[2]*r0+1))
    nearest=float(np.clip(r0,rr.min(),rr.max()));den=1+rowfit[2]*r0;nearest_den=1+rowfit[2]*nearest
    # Near foreground rows enlarge source errors under projectivity. Their
    # extrapolation cannot use the same pixel tolerance as the distant rows.
    tolerance=row_tolerance*max(1.,float((nearest_den/den)**2)) if abs(den)>1e-6 else row_tolerance
    if abs(z['y960']-pred)>tolerance:
     rejected.append(dict(target_id=f['target_id'],reason='source_stroke_outside_corresponding_physical_row_of_current_source_family',
       measured_y_at_x960_px=z['y960'],family_row_identity_navigation_y_at_x960_px=pred,tolerance_px=tolerance,original_candidate=f))
    else:filtered.append(f)
   out=filtered
  # Different physical rows cannot bind the same actual material stroke.
  # Explicit human-labelled rows win a conflict with unlabelled model searches.
  ordered=sorted([f for f in out if f['world_curve'] in LONG_LINE_ORDER and 'source_line_fit' in f],
    key=lambda f:(f['world_curve'] not in reference_ids,-f['source_line_fit']['actual_native_readout_count']))
  keep=[];remove=set()
  for f in ordered:
   cid=f['world_curve'];rank=LONG_LINE_ORDER.index(cid);z=image_line(f);conflict=None
   for other in keep:
    oid=other['world_curve'];oz=image_line(other);orank=LONG_LINE_ORDER.index(oid)
    separation=z['y960']-oz['y960']
    if (rank-orank)*separation<=0 or abs(separation)<5:
     conflict=(other,oz,separation);break
   if conflict:
    other,oz,separation=conflict;remove.add(cid)
    rejected.append(dict(target_id=f['target_id'],reason='distinct_parallel_physical_targets_conflict_in_source_order_or_bind_same_stroke',
      conflicting_target_id=other['target_id'],separation_at_x960_px=float(separation),original_candidate=f))
   else:keep.append(f)
  out=[f for f in out if f['world_curve'] not in remove]
  report['rejected_candidate_count']=len(rejected)
  return out,rejected,dict(**report,remeasured_navigation_shapes=refined_out)
 def extract(self,index):
  started=time.perf_counter();row=self.rows[index];source=deepcopy(row['source'])
  bgr=cv2.imread(str(DATA/f'frames/{index:06d}.png'));maskfile=Path(source.get('mask_path',str(DATA/f'masks_current/{index:06d}.png')))
  person=cv2.imread(str(maskfile),0)
  if person is None:raise ValueError('Missing current foreground mask '+str(index))
  broadcast,bcast=current_broadcast_exclusion(bgr);outside_ice,domain=current_ice_visible_domain(index)
  mask=(person>0)|broadcast|outside_ice
  native=NativeProfiles(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source,mask)
  features=[];reports=[];current={};inferred=[];searches={}
  nested,_=self.nested_house_features(index,native)
  for p in self.paths:
   p=dict(p)
   unqualified=next((r for r in (self.query_qualification or {}).get('records',[])
      if r['target_id']=='ice.'+p['curve_id'] and r['programme_frame_range'][0]<=index<=r['programme_frame_range'][1]
      and not r['allowed_as_fitting_observation']),None)
   if unqualified:
    reports.append(dict(target_id=unqualified['target_id'],points=0,
      missing_reason=unqualified['rejection_reason'],query_qualification_path=str(self.query_qualification_path),
      physical_model_target_removed=False))
    continue
   guide,origin=self.guide(p,index)
   if not p['_circle'] and self.line_identity is not None:
    guide,origin,policy=self.line_identity.select_guide(p,index,guide,origin);p['_line_identity_policy']=policy
   visible=np.isfinite(guide).all(1)&(guide>=[-32,-32]).all(1)&(guide<=[1951,1111]).all(1)
   if not visible.any() and p['curve_id'] not in nested:continue
   searches[p['curve_id']]=(p,guide,origin)
   if p['curve_id'] in nested:f=nested[p['curve_id']];refined=None;origin=f.get('navigation_origin','shared_nested_world_circle_family')
   elif p['_circle']:f,refined=self.circle(p,guide,native)
   else:f,hidden,refined=self.line(p,guide,native);inferred.extend(hidden)
   f['navigation_origin']=origin
   if f['point_count']:
    features.append(f)
    if refined is not None:current[p['curve_id']]=refined
   reports.append(dict(target_id=f['target_id'],points=f['point_count'],navigation_origin=origin,
     native_material_diagnostic=f.get('extraction_diagnostic'),
     missing_reason=None if f['point_count'] else f.get('extraction_diagnostic',{}).get('missing_reason','no_coherent_unmasked_current_native_material_readouts_under_line_or_circle_acceptance')))
  features,rejected,family=self.family_audit(index,features,searches,native)
  if self.line_identity is not None:self.line_identity.record(index,features)
  for cid,refined in family.pop('remeasured_navigation_shapes',{}).items():current[cid]=refined
  accepted={f['world_curve'] for f in features};current={cid:uv for cid,uv in current.items() if cid in accepted}
  counts={f['target_id']:f['point_count'] for f in features}
  for report in reports:
   if report['target_id'] not in counts and report['points']:
    reasons=[r['reason'] for r in rejected if r['target_id']==report['target_id']]
    report.update(points=0,source_candidate_rejected=True,rejection_reason=reasons,
     missing_reason='measured_candidate_rejected: '+', '.join(reasons))
   elif report['target_id'] in counts:report['points']=counts[report['target_id']]
  cap_removed=sparse_representative_cap(features)
  final_counts={f['target_id']:f['point_count'] for f in features}
  for report in reports:
   if report['target_id'] in final_counts:report['points']=final_counts[report['target_id']]
  self.previous=current;self.previous_index=index
  source.update(mask_path=str(maskfile),mask_sha256=sha(maskfile),broadcast_timer_exclusion=bcast,ice_visible_domain=domain)
  packet=dict(schema='rinkspace.single-frame-source-ice-features.v1',programme_frame_index_1based=index,source=source,
      status='awaiting_user_review',user_accepted=False,features=features,inferred_segments=inferred,
      point_count=sum(f['point_count'] for f in features),optimization_performed=False,model_scene_sha256=sha(self.model_root/'scene.json'),
      circle_geometry_sha256=self.circle_geometry_sha256,
      per_feature_coverage=reports,elapsed_seconds=time.perf_counter()-started,
      parallel_family_geometry_audit=family,representative_cap_removals=cap_removed,
      rejected_material_candidate_report_path=str(self.output_root/f'ice_diagnostics/rejected_candidates/{index:06d}.json') if rejected else None,
      source_correlated_groups=[dict(group_id=f['source_evidence_group_id'],target_id=f['target_id'],native_source_pts=source['native_pts'],
        representative_point_count=f['point_count'],correlation_reason='same current-frame measured physical curve estimate; representatives are not independent measurements',
        native_individual_readout_count=f.get('source_line_fit',f.get('source_circle_fit',{}))['actual_native_readout_count'])
        for f in features if 'source_line_fit' in f or 'source_circle_fit' in f])
  from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts,apply_rejections
  circle_rejections=conflicts(packet['features'])
  apply_rejections(packet,circle_rejections)
  # A fit influenced by a reused house band must not navigate the next frame.
  for cid in {r['target_id'].removeprefix('ice.') for r in circle_rejections}:self.previous.pop(cid,None)
  write(self.output_root/f'ice_candidates/{index:06d}.json',packet)
  if rejected:write(self.output_root/f'ice_diagnostics/rejected_candidates/{index:06d}.json',dict(programme_frame_index_1based=index,source=source,
      note='Historical failed material branches retained for diagnosis only; never admitted as source observations',rejected_candidates=rejected))
  return packet

def main():
 p=argparse.ArgumentParser();p.add_argument('--guides',action='store_true');p.add_argument('--start',type=int,default=201);p.add_argument('--stop',type=int,default=1248);p.add_argument('--indices');p.add_argument('--overwrite',action='store_true');p.add_argument('--candidate-output-root');p.add_argument('--candidate-model-root');args=p.parse_args()
 cv2.setNumThreads(1)
 if args.guides:prepare_guides();return
 extractor=Extractor(args.candidate_output_root,args.candidate_model_root);indices=list(map(int,args.indices.split(','))) if args.indices else range(args.start,args.stop+1)
 for index in indices:
  dest=extractor.output_root/f'ice_candidates/{index:06d}.json'
  if dest.exists() and not args.overwrite:continue
  result=extractor.extract(index)
  if index%10==0 or args.indices:print(json.dumps({'frame':index,'points':result['point_count'],'seconds':result['elapsed_seconds'],'features':[(x['target_id'],x['point_count']) for x in result['features']]}),flush=True)
if __name__=='__main__':main()
