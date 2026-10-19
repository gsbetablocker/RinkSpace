from pathlib import Path
p=Path('calibration/program_tracking/produce_first_shot_circle_repair.py');s=p.read_text().replace("model_scene_sha256=sha(MODEL/'scene.json'),features_by_curve=features", "model_scene_sha256=sha(ex.model_root/'scene.json'),circle_geometry_sha256=ex.circle_geometry_sha256,features_by_curve=features");p.write_text(s)
p=Path('calibration/program_tracking/audit_repaired_first_shot_circles.py');s=p.read_text().replace("role=f['world_curve_role']", "role=cid.rsplit(':',1)[1]");p.write_text(s)
