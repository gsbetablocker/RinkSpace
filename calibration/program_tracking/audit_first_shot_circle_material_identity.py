"""Find extra outline arcs that instead measure an already labelled house band."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
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
