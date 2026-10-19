from calibration.program_tracking.extract_first_shot_ice import *
p=DATA/'repair_20261014';records=[]
for i in range(201,1249):
 a=read(p/f'circle_candidates/{i:06d}.json')
 for fid,d in a['diagnostics'].items():
  if not fid.startswith(('F','R')):continue
  outer=d.get('outer',{}).get('eligible_native_material_readouts',0);total=sum(r.get('eligible_native_material_readouts',0) for r in d.values())
  if not a['features_by_curve'][fid+':outer']['point_count'] and 2<=outer<=5 and total<6:records.append(dict(frame=i,house=fid,outer_readouts=outer,total=total))
write(p/'circle_diagnostics/short_arc_repair_plan.json',dict(records=records,frames=sorted({r['frame'] for r in records})))
print(len(records),len({r['frame'] for r in records}),records)
