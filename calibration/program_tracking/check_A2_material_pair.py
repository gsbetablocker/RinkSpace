from calibration.program_tracking.extract_first_shot_ice import *
from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts
from calibration.program_tracking.extract_single_frame_ice_source import ellipse_distance
p=DATA/'repair_20261014';a=read(p/'circle_candidates/000230.json')['features_by_curve'];print([(cid,f['point_count']) for cid,f in a.items() if f['point_count']]);f=a['A2:outer'];print(f['segments'][0]['uv']);o=a['R3:outer']['geometric_interpretation'];m=a['R3:middle']['geometric_interpretation']
for s in f['segments']:
 for row in s['source_evidence']:
  sides=np.array(row['actual_source_sides_uv']);print('width',np.linalg.norm(sides[1]-sides[0]),'outer',ellipse_distance(sides,(o['center_uv'],o['axes_px'],o['angle_degrees'])),'middle',ellipse_distance(sides,(m['center_uv'],m['axes_px'],m['angle_degrees'])))
