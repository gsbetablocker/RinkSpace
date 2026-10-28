"""Find extra outline arcs that instead measure an already labelled house band."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import cv2
from calibration.program_tracking.extract_single_frame_ice_source import ellipse_distance

def ellipse(f):
 g=f.get('geometric_interpretation',{})
 return (g['center_uv'],g['axes_px'],g['angle_degrees']) if g.get('kind')=='projected_physical_circle_ellipse' else None

def conflicts(features):
 houses={f['target_id']:f for f in features if f['target_id'].startswith(('ice.F','ice.R')) and ellipse(f)}
 pairs=[]
 for tid,f in houses.items():
  if not tid.endswith(':outer'):continue
  base=tid.rsplit(':',1)[0]
  for a,b in (('outer','middle'),('middle','inner')):
   first,second=houses.get(base+':'+a),houses.get(base+':'+b)
   if first and second:pairs.append((base,a,b,ellipse(first),ellipse(second)))
 rejected=[]
 for f in features:
  if not f['target_id'].startswith('ice.A') or f['measurement_operator']!='paint_center':continue
  for si,s in enumerate(f['segments']):
   for vi,e in enumerate(s.get('source_evidence',[])):
    sides=np.asarray(e['actual_source_sides_uv']);width=float(np.linalg.norm(sides[1]-sides[0]))
    if width<9:continue
    for base,a,b,ea,eb in pairs:
     da=ellipse_distance(sides,ea);db=ellipse_distance(sides,eb)
     err=min(max(da[0],db[1]),max(da[1],db[0]))
     if err<=6:
      rejected.append(dict(target_id=f['target_id'],segment_index=si,source_vertex_index=vi,uv=s['uv'][vi],
        conflicting_house_band=base+':'+a+'-'+b,paired_interface_width_px=width,
        maximum_interface_to_current_house_contour_px=float(err),
        actual_native_source_evidence=e,original_circle_shape_inference=f['geometric_interpretation'],
        reason='extra_outline_arc_reads_midpoint_of_already_identified_house_filled_paint_band_not_independent_thin_outline'))
      break
 return rejected


def observed_house_fill(features, shape=(1080,1920)):
 """Join corresponding *observed* interfaces into local paint patches.

 No fitted ellipse or projected circle fills missing arcs. Both interfaces
 must have actual ordered source support over the same angular interval.
 The returned mask constrains other material searches; it supplies no UV.
 """
 mask=np.zeros(shape,np.uint8);patches=[]
 for house in ('F1','F2','F3','F4','R1','R2','R3','R4'):
  for outer,inner in (('outer','middle'),('inner','center')):
   first=features.get(f'ice.{house}:{outer}',{}).get('rows',[])
   second=features.get(f'ice.{house}:{inner}',{}).get('rows',[])
   if len(first)<3 or len(second)<3 or any('theta'not in r for r in first+second):continue
   def ordered(rows):
    angle=np.asarray([r['theta']for r in rows]);uv=np.asarray([r['uv']for r in rows]);ix=np.argsort(angle);angle,uv=angle[ix],uv[ix]
    return np.r_[angle[-1]-2*np.pi,angle,angle[0]+2*np.pi],np.vstack([uv[-1],uv,uv[0]])
   a,p=ordered(first);b,q=ordered(second)
   da=np.diff(a);db=np.diff(b)
   limit_a=min(np.pi/4,3*float(np.median(da[da>0])))
   limit_b=min(np.pi/4,3*float(np.median(db[db>0])))
   for j in range(1,len(a)-1):
    t=a[j:j+2]
    if t[1]-t[0]>limit_a or np.linalg.norm(p[j+1]-p[j])>105:continue
    k=np.searchsorted(b,t,side='right')-1
    if (k<0).any()or(k>=len(b)-1).any()or np.any(np.diff(b)[k]>limit_b):continue
    if any(np.linalg.norm(q[n+1]-q[n])>105 for n in k):continue
    v=np.c_[np.interp(t,b,q[:,0]),np.interp(t,b,q[:,1])]
    quad=np.array([p[j],p[j+1],v[1],v[0]],np.float32)
    if cv2.contourArea(quad)<1:continue
    cv2.fillPoly(mask,[np.rint(quad).astype(np.int32)],1)
    patches.append(dict(house=house,band=f'{outer}-{inner}',actual_source_interface_quad=quad.tolist()))
 return mask.astype(bool),patches


def centreline_house_conflicts(features, entities, *, include_crosslines=False):
 """Detect material ownership conflicts for all current-native centrelines.

 The old outline-only audit missed Lane and connection readers. This check
 also covers additions from gap recovery and packaging fallback paths.
 """
 mask,patches=observed_house_fill(features)
 interior=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,3)>2
 cross=np.zeros_like(mask,dtype=np.uint8)
 crossline_names={'ice.B1:segment0','ice.B2:segment0','ice.C:segment0'}
 if include_crosslines:
  # A coloured transverse stripe is not evidence for a separate thin lane
  # or house connection at a crossing. Use only locally observed band sides;
  # neither native projections nor fitted lines create exclusion pixels.
  for name in crossline_names:
   rows=[r for r in features.get(name,{}).get('rows',[])if r.get('source_evidence',{}).get('actual_source_sides_uv')]
   if len(rows)<2:continue
   uv=np.asarray([r['uv']for r in rows]);_,_,axes=np.linalg.svd(uv-uv.mean(0),full_matrices=False)
   order=np.argsort((uv-uv.mean(0))@axes[0]);rows=[rows[i]for i in order]
   for first,second in zip(rows,rows[1:]):
    gap=np.linalg.norm(np.asarray(first['uv'])-second['uv'])
    if gap<1 or gap>105:continue
    a=np.asarray(first['source_evidence']['actual_source_sides_uv']);b=np.asarray(second['source_evidence']['actual_source_sides_uv'])
    if np.sum((a-b[::-1])**2)<np.sum((a-b)**2):b=b[::-1]
    quad=np.array([a[0],b[0],b[1],a[1]],np.float32)
    if cv2.contourArea(quad)>=1:cv2.fillPoly(cross,[np.rint(quad).astype(np.int32)],1)
 cross_interior=cv2.distanceTransform(cross,cv2.DIST_L2,3)>2
 failures=[]
 for name,feature in features.items():
  path=entities.get(feature['native_path_id'],{})
  if path.get('measurement')!='centreline':continue
  for i,row in enumerate(feature['rows']):
   x,y=np.rint(row['uv']).astype(int)
   if 0<=y<interior.shape[0]and 0<=x<interior.shape[1]and interior[y,x]:
    failures.append(dict(target_id=name,source_vertex_index=i,source_row_id=row['source_row_id'],uv=row['uv'],
     reason='centreline_profile_inside_currently_measured_house_fill_requires_independent_material_support',
     source_evidence=row.get('source_evidence',{})))
   elif name not in crossline_names and 0<=y<cross.shape[0]and 0<=x<cross.shape[1]and cross_interior[y,x]:
    failures.append(dict(target_id=name,source_vertex_index=i,source_row_id=row['source_row_id'],uv=row['uv'],
     reason='thin_centreline_profile_reads_observed_red_or_blue_crossline_without_independent_material_support',
     source_evidence=row.get('source_evidence',{})))
 return failures,mask,patches

def apply_rejections(packet,bad):
 look={(r['target_id'],r['segment_index'],r['source_vertex_index']) for r in bad}
 kept=[]
 for f in packet['features']:
  segments=[]
  for si,s in enumerate(f['segments']):
   ids=[j for j in range(len(s['uv'])) if (f['target_id'],si,j) not in look]
   # Break preview connectivity at rejected material; never bridge it.
   groups=[]
   for j in ids:
    if groups and j==groups[-1][-1]+1:groups[-1].append(j)
    else:groups.append([j])
   for group in groups:
    part=dict(s)
    for name in ('uv','sigma_px','source_evidence'):
     if name in s:part[name]=[s[name][j] for j in group]
    part['vertex_count']=len(group);part['closed']=False;segments.append(part)
  f['segments']=segments;f['point_count']=sum(s['vertex_count'] for s in segments)
  if f['point_count']:kept.append(f)
 packet['features']=kept;packet['point_count']=sum(f['point_count'] for f in kept)
 packet['extra_circle_material_identity_audit']=dict(status='checked_current_house_band_interface_pairs',rejected_observation_count=len(bad),rejections=bad)
 for row in packet['per_feature_coverage']:
  row['points']=sum(f['point_count'] for f in kept if f['target_id']==row['target_id'])
  if any(r['target_id']==row['target_id'] for r in bad):
   row['source_material_identity_rejections']=[r for r in bad if r['target_id']==row['target_id']]
   if row['points']==0:row['missing_reason']='extra_circle_source_profile_binds_existing_house_fill_band_not_independent_outline'
 return packet

def main():
 from calibration.program_tracking.extract_first_shot_ice import DATA,read,write
 ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
 records=[]
 for index in range(201,1249):
  path=DATA/f'ice_candidates/{index:06d}.json';packet=read(path);bad=conflicts(packet['features'])
  if bad:records.append(dict(frame=index,rejections=bad))
  if not args.apply or not bad:continue
  write(path,apply_rejections(packet,bad))
 write(DATA/'ice_diagnostics/extra_circle_material_identity_audit.json',dict(schema='rinkspace.extra_circle_material_identity_audit.v1',applied=args.apply,frame_count_checked=1048,
   frame_count_with_rejections=len(records),rejected_observation_count=sum(len(r['rejections']) for r in records),records=records,
   scope='A outline arcs whose two actual native interfaces match outer/middle or middle/inner material contours of a currently measured F/R house; no new pixel observations are introduced'))
 print(json.dumps(dict(applied=args.apply,frames=len(records),points=sum(len(r['rejections']) for r in records)),ensure_ascii=False))
if __name__=='__main__':main()
