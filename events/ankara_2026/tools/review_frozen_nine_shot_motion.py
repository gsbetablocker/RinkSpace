"""Read-only actual camera review on preserved native RGB held tracks.

No feature extraction, source selection, full graph construction, or images.
The finite-board review reuses the production native source factor exactly.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np

spec=importlib.util.spec_from_file_location('preserved_rgb_motion',ROOT/'scripts/diagnostics/external_image_motion_probe_20261007.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--state',required=True,type=Path)
    p.add_argument('--auxiliary',required=True,type=Path)
    p.add_argument('--audit',required=True,type=Path)
    p.add_argument('--typed-board-packet',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();started=time.perf_counter()
    identity=json.loads((a.state.parent/'state_identity.json').read_text(encoding='utf8'))
    if probe.sha(a.state)!=identity['state_npz_sha256']:
        raise ValueError('Immutable complete state changed')
    camera_path=Path(identity['camera_path'])
    if probe.sha(camera_path)!=identity['camera_sha256']:
        raise ValueError('Immutable complete camera changed')
    cameras=dict(np.load(camera_path,allow_pickle=False))
    audit=json.loads(a.audit.read_text(encoding='utf8'))
    tracks_path=a.audit.parent/'actual_RGB_tracks.npz'
    tracks=np.load(tracks_path,allow_pickle=False)
    ix={int(v):i for i,v in enumerate(cameras['pts'])}
    diagnostics=[]
    for number,pair in enumerate(audit['pairs']):
        for material in ('ice','board'):
            key=f'pair{number:02d}_{material}'
            row=dict(**pair,material=material)
            original=next(r for r in audit['experiments'] if r['source_pts']==pair['source_pts'] and r['target_pts']==pair['target_pts'] and r['material']==material)
            if key+'_source' not in tracks:
                row['failure']=original.get('failure','No preserved source tracks')
                diagnostics.append(row);continue
            source=tracks[key+'_source'];target=tracks[key+'_target']
            held=tracks[key+'_held']&tracks[key+'_source_static_keep']&(tracks[key+'_forward_backward_error']<.8)
            ca,cb=[cameras['camera25'][ix[pair[k]]] for k in ('source_pts','target_pts')]
            predicted,valid=probe.model_transfer(ca,cb,source,0. if material=='ice' else None)
            used=held&valid;delta=predicted-target
            row.update(held_points=int(used.sum()),error_native_px=probe.stats(np.linalg.norm(delta[used],axis=1)),
                signed_median_xy_px=None if not used.any() else np.median(delta[used],axis=0).tolist(),
                source_image_H_held_error_px=original['independent_good_held_error_px'],
                previous_models=original['models'],
                transfer='actual_Z0_plane' if material=='ice' else 'rotation_only_without_board_depth_or_lever_parallax')
            diagnostics.append(row)
    from scripts.diagnostics.verify_current_native_primitives_20261007 import state_camera_adapter
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    provider,world,proof=current_complete_source_geometry(a.state,a.auxiliary)
    arrays=dict(np.load(a.state,allow_pickle=False))
    state,error=state_camera_adapter(arrays,cameras,provider)
    from structured_programme_source_candidate.native_board_track_source import NativeBoardTrackFactor
    factor=NativeBoardTrackFactor(state,a.typed_board_packet,len(arrays['q']))
    finite=factor.review(state.evaluation_q(arrays['q']))
    shots=json.loads((ROOT/'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots'][1:10]
    summaries=[]
    for i,shot in enumerate(shots,1):
        held=[r for r in finite['rows'] if r['split']=='holdout' and int(shot['start_pts'])<=r['source_pts']<int(shot['end_pts'])]
        worst=max(held,key=lambda r:r['median_native_px']) if held else None
        ice=[r for r in diagnostics if r['shot_index']==i and r['material']=='ice' and r.get('held_points',0)>=6]
        worstice=max(ice,key=lambda r:r['error_native_px']['median']) if ice else None
        summaries.append(dict(shot_index=i,shot_id=shot['id'],camera_id=shot['camera_id'],source_pts=[shot['start_pts'],shot['end_pts']],
            actual_finite_board_held=dict(pairs=len(held),points=sum(r['points']for r in held),worst=worst,
                pair_median_native_px=probe.stats([r['median_native_px'] for r in held])),
            actual_ice_Z0_held_diagnostic=dict(pairs=len(ice),points=sum(r['held_points']for r in ice),worst=worstice)))
    report=dict(schema='rinkspace.preserved-native-RGB-held-actual-state-review.v1',state=str(a.state),
        state_identity=identity,physical_camera_archive_reproduce_max=error,geometry_proof=proof,
        source_audit=dict(path=str(a.audit),sha256=probe.sha(a.audit)),
        source_tracks=dict(path=str(tracks_path),sha256=probe.sha(tracks_path)),
        source_UV_and_HELD_not_changed=True,source_selection_model_independent=True,
        full_graph_not_loaded=True,images_or_video_generated=False,finite_board_review=finite,
        source_motion_diagnostics=diagnostics,per_shot=summaries,
        limitations=['Relative motion does not prove absolute alignment.',
            'Original ice static navigation can include moving reflections; these remain diagnostic.',
            'Rotation-only board diagnostic ignores depth and lever; actual finite-board rows include both.',
            'Multiple tracks on one advertisement are correlated, not independent calibration evidence.'],
        accuracy_accepted=False,wall_seconds=time.perf_counter()-started)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(output=str(a.output),wall_seconds=report['wall_seconds'],per_shot=summaries),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
