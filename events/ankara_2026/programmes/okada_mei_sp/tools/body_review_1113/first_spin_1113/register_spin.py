from pathlib import Path
from argparse import ArgumentParser

from experiments.skate_video_v1.inputs import PROJECT, ROOT, atomic_json, digest, read

p=ArgumentParser()
p.add_argument('stage',type=Path)
args=p.parse_args()
stage=args.stage.resolve()
run=stage.parent
programme=PROJECT/'events/ankara_2026/programmes/okada_mei_sp'
registered_programme=programme/'fit_v11/provisional_final_1113.json'
registered_scene=PROJECT/'events/ankara_2026/scene/provisional_final_1113.json'
previous=programme/'fit_v11/body_cut_continuity_1113.json'
target=programme/'fit_v11/body_first_spin_1113.json'
for path in (target,target.with_name(target.name+'.pending')):
    assert len(str(path).encode('utf-16-le'))//2<260
    assert all(len(part.encode('utf-16-le'))//2<=255 for part in path.parts)

def bind(path,root=ROOT):
    return dict(path=path.relative_to(root).as_posix(),sha256=digest(path))

scope=read(stage/'first_spin_scope.json')
native=read(stage/'native_review/checks.json')
clock=read(stage/'player_025x_verification.json')
video=read(stage/'video_receipt.json')
slow=read(stage/'first_spin_3D_receipt.json')
media=read(stage/'media_checks.json')
motion=read(run/'spin_motion_candidate.json')
guards=read(stage/'final_metrics.json')['primary_guards']
fit=read(stage/'run_summary.json')
source=read(run/'spin_source.json')
surface=read(stage/'retained_surface.json')
base=read(registered_programme)
scene=read(registered_scene)
assert not target.exists()
assert clock['passed'] and native['passed'] and fit['additional_accepted_steps']==1
assert not any(x.get('rejected') for x in guards.values()) and surface['regressions']==[]
assert video['controls_sha256']==slow['candidate_sha256']==clock['candidate_sha256']==scope['parameter_sha256']
assert video['sha256']==media['overlay_sha256'] and slow['sha256']==media['quarter_speed_sha256']
assert digest(Path(slow['path']))==slow['sha256']
assert scope['native_camera_sha256']==video['camera_sha256']==base['camera']['sha256']
assert scope['scene_values_sha256']==base['scene_values_sha256']==scene['geometry']['value_sha256']
assert scope['exact_outside_unchanged'] and scope['native_frames']==8329
proof=dict(
    solve_state=stage/'solve/candidate.npz',
    fitted_state=stage/'fitted/result.npz',
    source_evidence=run/'spin_source.json',
    main_log=stage/'progress.jsonl',
    review_metrics=run/'spin_motion_candidate.json',
    unchanged_scope=stage/'first_spin_scope.json',
    surface=stage/'retained_surface.json',
    guard_results=stage/'final_metrics.json',
    viewer=stage/'START_REFITTED_3D.cmd',
    viewer_configuration=stage/'viewer3d.json',
    navigation=stage/'navigation.json',
    full_overlay=Path(video['path']),
    quarter_speed_comparison=Path(slow['path']),
    native_viewer_check=stage/'native_review/checks.json',
    slow_clock_check=stage/'player_025x_verification.json',
    media_check=stage/'media_checks.json',
    source_audio_alignment=stage/'audio_review.json')
record=dict(schema='rinkspace.okada_first_spin_candidate.v1',
    programme='okada_mei_sp', candidate_id='okada_first_spin_1113',
    status='engineering_checked_candidate', accuracy_accepted=False, user_accepted=False,
    registered_programme=bind(registered_programme,PROJECT), registered_scene=bind(registered_scene,PROJECT),
    previous_body_candidate=bind(previous,PROJECT), source_video_sha256=source['source_sha256'],
    source_time_base=scope['source_time_base'], source_first_last_pts=video['source_pts'],
    source_exposures=scope['native_frames'], changed_intervals_seconds=scope['edited_intervals_seconds'],
    changed_exposures=scope['changed_native_frames'], exact_outside_unchanged=True,
    native_camera_sha256=scope['native_camera_sha256'], shared_scene_values_sha256=scope['scene_values_sha256'],
    core_support_pts=source['first_spin_motion_review']['stable_support_pts'],
    body_shape_and_skate_geometry_fixed=True,
    metrics=dict(
        entry_torso_peak_deg_per_exposure=[m['entry']['actual_angle_step_deg']['torso']['max'] for m in motion.values()],
        support_co_rotating_com_p95_mps2=[m['support']['co_rotating_com_accel_mps2']['p95'] for m in motion.values()],
        body_pixels_p90=[m['whole']['reprojection_px'][1] for m in motion.values()],
        native_clock_exposures=clock['window']['exposures'], native_checks=len(native['checks'])),
    data_root_artifacts={k:bind(v) for k,v in proof.items()},
    remaining=[dict(pts=1042432,issue='跳入速度尖峰减弱，手臂回投仍偏低。'),
               dict(pts=1143296,issue='支撑刀回投及高举自由鞋仍有轮廓偏差。'),
               dict(pts=1182720,issue='出旋前自由鞋仍有较大的单帧转角。')])
atomic_json(target,record)
print(target,digest(target),flush=True)
