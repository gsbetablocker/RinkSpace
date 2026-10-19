from pathlib import Path
p=Path('calibration/program_tracking/audit_repaired_first_shot_circles.py');s=p.read_text()
s=s.replace("for cid,f in packet['features_by_curve'].items():", "for cid,f in qualify_extra_circle_material(packet['features_by_curve']).items():")
s=s.replace("packet=read(file);source=packet['source'];bgr=", "packet=read(file);source=packet['source'];bgr=")
s=s.replace("features=[f for f in packet['features_by_curve'].values() if f['point_count']];byid=", "identity_rejections=conflicts([f for f in packet['features_by_curve'].values() if f['point_count']]);material_reject+=len(identity_rejections)\n  qualified=qualify_extra_circle_material(packet['features_by_curve'])\n  features=[f for f in qualified.values() if f['point_count']];byid=")
s=s.replace("  material_reject+=len(conflicts(features));counts.append(count);frames+=1", "  counts.append(count);frames+=1")
s=s.replace("if dev>3.2:", "if dev>3.0002:")
s=s.replace("  if sha(source['mask_path'])", "  if packet.get('algorithm_sha256')!=sha(ROOT/'calibration/program_tracking/native_ice_material_repair.py'):issues.append(dict(frame=index,kind='stale_circle_measurement_algorithm'))\n  if sha(source['mask_path'])")
p.write_text(s)
