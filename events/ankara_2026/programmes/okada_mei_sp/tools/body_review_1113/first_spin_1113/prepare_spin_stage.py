from pathlib import Path
import os

from experiments.skate_video_v1.inputs import atomic_json, digest, read

run=Path(os.environ['RINKSPACE_AUDIT_RUN'])
job=Path(os.environ.get('RINKSPACE_OKADA_BODY_REVIEW', run.parent/'audit_af48_1113')).resolve()/'first_spin_1113'
cuts=job.parent/'cut_continuity_1113'
parent=cuts/'delivered'
source=cuts/'cut_source.json'
doc=read(source)
rep=doc['scoped_candidate_import']
rep.pop('registered_camera_transition',None)
focus=[round(81.14*12800),round(93.18*12800)]
rep.update(parent_sha256=digest(parent/'solve/candidate.npz'),
    parent_evidence=str(source), parent_evidence_sha256=digest(source),
    parent_result=str(parent/'fitted/result.npz'),parent_result_sha256=digest(parent/'fitted/result.npz'),
    focus_pts=[focus],tracks_by_window=[list(range(-1,22))],
    component_ice_pts=[rep['opening_pts'],focus])
surface=Path(read(parent/'retained_surface.json')['surface_path'])
doc['programme_structure'].update(surface_witnesses=str(surface),surface_witnesses_sha256=digest(surface))
doc['first_spin_motion_review']=dict(
    first_last_pts=focus, stable_support_pts=[round(81.62*12800),round(92.72*12800)],
    held_shoe_pts=[1111040,1183488], direction=1,
    source_pts_time_base=[1,12800], phase_change_prior='second differences in a world-Z co-rotating frame',
    same_native_source=True, shoe_assembly_fixed_mount=True)
atomic_json(job/'spin_source.json',doc)
print('spin programme contract',digest(job/'spin_source.json'),flush=True)
