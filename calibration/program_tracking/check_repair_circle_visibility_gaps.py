from calibration.program_tracking.extract_first_shot_ice import *
from collections import defaultdict
p=DATA/'repair_20261014';rows=defaultdict(dict)
for i in range(201,1249):
 a=read(p/f'circle_candidates/{i:06d}.json');ff=qualify_extra_circle_material(a['features_by_curve'])
 for cid,f in ff.items():rows[cid][i]=f['point_count']
report=[]
for cid,v in rows.items():
 for i in range(202,1248):
  if v[i]==0 and v[i-1]>=5 and v[i+1]>=5:report.append(dict(target_id=cid,frame=i,neighbours=[v[i-1],v[i+1]]))
write(p/'circle_diagnostics/short_visibility_gaps.json',dict(gaps=report,meaning='Diagnostic only; source occlusion or ambiguity may explain a real one-frame absence. Not a mandate to fill or create points.'))
print(len(report),report)
