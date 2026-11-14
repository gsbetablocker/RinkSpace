
from pathlib import Path
import os,numpy as np
from experiments.skate_video_v1.inputs import read,atomic_json,atomic_npz,digest
from calibration.program_tracking.body_tracking.rebuild_prepare import load

run=Path(os.environ['RINKSPACE_AUDIT_RUN'])
job=Path(os.environ.get('RINKSPACE_OKADA_BODY_REVIEW', run.parent/'audit_af48_1113')).resolve()/'cut_continuity_1113'
previous=job.parent/'opening_lunge'
parent=previous/'delivered'
src=previous/'opening_source.json'
source=read(src)
bridge=read(job/'camera_bridge.json')
registered=read(bridge['record'])
# The two registered camera/geometry binaries are immutable. The sibling
# task may append status metadata to its programme registration.
record=job/'camera_registration.json'
atomic_json(record,registered)
parent_param=parent/'solve/candidate.npz'
prior_camera=source['programme_skate_evidence']['camera_sha256']
selected=bridge['camera']['sha256']
whole=source['whole_skate_source']
original=Path(whole['path'])
assert digest(original)==whole['sha256']
raw=load(original)
assert str(raw['camera_sha256'])==prior_camera
# All pixels, visibility/role and instance IDs are source-native. Update
# only the projector association; correspondences are recalculated on fit.
roles=job/'whole_skate_roles.npz'
atomic_npz(roles,**dict(raw,camera_sha256=np.array(selected)))
whole.update(path=str(roles),sha256=digest(roles),
             source_pixel_ledger=dict(path=str(original),sha256=digest(original),pixel_arrays_unchanged=True))
skate=source['programme_skate_evidence']
archive=skate['plane_xy_diagnostics']
atomic_json(job/'previous_camera_plane_xy.json',archive)
skate.update(camera_sha256=selected,plane_xy_diagnostics=[],
             previous_camera_plane_diagnostics=dict(path=str(job/'previous_camera_plane_xy.json'),
                sha256=digest(job/'previous_camera_plane_xy.json'),camera_sha256=prior_camera,
                used_in_objective=False))
source['programme_camera_candidate']=dict(record=str(record),record_sha256=digest(record),
    parent_camera_sha256=prior_camera,registered_programme=dict(path=bridge['record'],
         camera_sha256=selected,scene_values_sha256=bridge['scene']['value_sha256']))
source['scene_contract'].update(plane_source='registered shared Ankara scene Z=0',
                               geometry=bridge['scene'],registered_camera_sha256=selected)
surface=Path(read(parent/'retained_surface.json')['surface_path'])
source['programme_structure'].update(surface_witnesses=str(surface),surface_witnesses_sha256=digest(surface))
pts=load(run/'prepared_refined/input.npz')
cuts=np.flatnonzero(pts['shot_ids'][1:]!=pts['shot_ids'][:-1])+1
pts=pts['pts']
spec=source['scoped_candidate_import']
spec.update(parent_sha256=digest(parent_param),parent_evidence=str(src),parent_evidence_sha256=digest(src),
    parent_result=str(parent/'fitted/result.npz'),parent_result_sha256=digest(parent/'fitted/result.npz'),
    registered_camera_transition=dict(parent_camera_sha256=prior_camera,camera_sha256=selected,
         scene_values_sha256=bridge['scene']['value_sha256']),
    focus_pts=[[int(pts[k-12]),int(pts[k+11])] for k in cuts],
    tracks_by_window=[list(range(-1,22)) for k in cuts],
    camera_cut_continuity=dict(cut_pts=pts[cuts].tolist(),half_width_s=.24,
        acceleration_weight_multiplier=55.,
        evidence_role='source-adjacent reviewed live edits with unsupported kinematic spikes',
        source_time_base=[1,12800]))
# Include the unchanged earlier curvature and physical row populations as
# context for their distinct checks, even though no controls are active there.
spec['component_ice_pts']=[spec['opening_pts'],*spec['focus_pts']]
atomic_json(job/'cut_source.json',source)
print('New objective source SHA',digest(job/'cut_source.json'),flush=True)
