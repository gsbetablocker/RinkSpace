from argparse import ArgumentParser
from pathlib import Path
import os
import shutil

import numpy as np

from calibration.program_tracking.body_tracking.rebuild_prepare import load
from experiments.skate_video_v1.body_skate_fit import NODES
from experiments.skate_video_v1.deliver_body_skate_fit import render_video
from experiments.skate_video_v1.inputs import atomic_json, digest, read
from experiments.skate_video_v1.programme_motion_followup import bind, changed_support, intervals


RUN = Path(os.environ['RINKSPACE_AUDIT_RUN'])
JOB = Path(os.environ.get('RINKSPACE_OKADA_BODY_REVIEW', RUN.parent/'audit_af48_1113')).resolve()/'first_spin_1113'
PARENT = JOB.parent / 'cut_continuity_1113/delivered'
PARENT_EVIDENCE = PARENT.parent / 'cut_source.json'
EVIDENCE = JOB / 'spin_source.json'
BRIDGE = PARENT.parent / 'camera_bridge.json'
PREPARED = RUN / 'prepared_refined'
VISUAL = PREPARED.parent.parent / 'skate_video_v1/whole_overlap_1112'
VERSION = 'okada_first_spin_1113'


def state(stage):
    parent_parameter = PARENT / 'solve/candidate.npz'
    parameter = stage / 'solve/candidate.npz'
    r0, c0, i0, _, z0 = bind(PREPARED, PARENT_EVIDENCE, parent_parameter)
    r1, c1, i1, doc, z1 = bind(PREPARED, EVIDENCE, parameter)
    np.testing.assert_array_equal(r0.p['pts'], r1.p['pts'])
    np.testing.assert_array_equal(z0['x'][:17], z1['x'][:17])
    np.testing.assert_array_equal(z0['relative_skate_rotation_controls'], z1['relative_skate_rotation_controls'])
    bridge = read(BRIDGE)
    assert i1['state_sha256'] == bridge['camera']['sha256']
    assert i1['scene_values_sha256'] == bridge['scene']['value_sha256']
    assert i0['state_sha256'] == i1['state_sha256']
    assert i0['scene_values_sha256'] == i1['scene_values_sha256']
    return r0, r1, c1, i1, doc, z0, z1


def image_problem(camera, info, ref):
    # Use the established source reader/crops and the registered projection clock.
    from experiments.skate_video_v1.body_skate_fit import Problem

    problem = Problem(VISUAL, JOB)
    np.testing.assert_array_equal(problem.i.pts, ref.p['pts'])
    camera.core.motion_projection_threads = 8
    for native in camera.cameras.values():
        native.core.motion_projection_threads = 8
    problem.camera, problem.info = camera, info
    problem.i.camera_sha = info['state_sha256']
    problem.times = ref.times[:, None] + camera.readout_times(np.arange(len(ref.times)))[:, None] * NODES
    return problem


def catalog(stage, r0, r1, info, doc, z0, z1):
    pts = r1.p['pts']
    parameter = stage / 'solve/candidate.npz'
    fitted = stage / 'fitted/result.npz'
    changed = changed_support(z0, z1, r1.times)
    assert changed.any()
    seconds = pts/12800.
    changes = intervals(changed, seconds)
    with np.load(PARENT / 'fitted/result.npz', allow_pickle=False) as old, np.load(fitted, allow_pickle=False) as new:
        for packet in (old, new):
            np.testing.assert_array_equal(packet['source_pts'], pts)
        for key in ('P_world_mid','native22_world_mid','skate_origin_world_mid','skate_R_world_mid',
                    'native_body_local_R_mid','com_world_mid','com_projection_world_mid'):
            np.testing.assert_allclose(old[key][~changed], new[key][~changed], atol=1.e-8, rtol=0)
        np.testing.assert_array_equal(new['com_world_mid'][:, :2], new['com_projection_world_mid'][:, :2])
        np.testing.assert_array_equal(new['com_projection_world_mid'][:, 2], np.zeros(len(pts)))

    metrics = read(stage/'run_summary.json')
    assert metrics['additional_accepted_steps'] >= 1
    assert read(stage/'retained_surface.json')['regressions'] == []
    cfg = read(PARENT / 'viewer3d.json')
    nav = read(PARENT / 'navigation.json')
    exact = doc['first_spin_motion_review']['first_last_pts']
    for start, cue, end, label in (
            (81.14, 81.40, 81.78, '跳入与落冰'),
            (81.78, 84.90, 86.40, '燕式旋与支撑鞋转向'),
            (86.40, 88.84, 92.48, '提刀与高位旋转'),
            (92.48, 92.72, 93.18, '放手与出旋')):
        a, t, b = (int(round(x*12800)) for x in (start,cue,end))
        assert exact[0] <= a <= t <= b <= exact[1]
        nav['events'].append(dict(kind='first_spin', worst_pts=t, start_pts=a, end_pts=b,
                                  label='第一旋转 · '+label))
    nav['events'].sort(key=lambda row: row['start_pts'])
    atomic_json(stage / 'navigation.json', nav)

    previous = cfg['variants'][0]
    bridge = read(BRIDGE)
    camera_info = dict(record=dict(path=bridge['record'], sha256=bridge['record_sha256']),
        state=bridge['camera'], scene=bridge['scene'],
        shot_state_sha256={key: value['state_sha256'] for key, value in info['shots'].items()},
        frozen_camera_sha256=info['state_sha256'])
    current = dict(previous, id=VERSION, candidate_id=VERSION, label='第一旋转 · 全身与鞋刀重拟合',
        result=str(fitted), result_sha256=digest(fitted), parameter=str(parameter), parameter_sha256=digest(parameter),
        camera_binding=camera_info, body_reference_label='跳入、落冰、提刀与出旋联合重拟合',
        body_refinement_intervals_seconds=changes, accuracy_accepted=False, user_accepted=False)
    cfg.update(variants=[current, *cfg['variants']], default_variant=VERSION, latest_review_variant=VERSION,
        session_id=VERSION,
        comparison_variants=[VERSION, previous['id']], source_navigation=str(stage/'navigation.json'),
        initial_source_pts=1038592, initial_view='side', trajectory_reference='com',
        body_window_scope='全节目连续人体与鞋刀，含模型质心和冰面投影',
        body_scope_label='第一旋转的全身与鞋刀连续运动复核',
        review_status='单机位遮挡处及手鞋接触为约束下的三维估计')
    atomic_json(stage/'viewer3d.json', cfg)
    shutil.copyfile(PARENT/'START_REFITTED_3D.cmd', stage/'START_REFITTED_3D.cmd')
    atomic_json(stage/'first_spin_scope.json',dict(
        edited_intervals_seconds=changes, native_frames=len(pts),
        changed_native_frames=int(changed.sum()), exact_outside_unchanged=True,
        source_time_base=[1,12800], parameter_sha256=digest(parameter), result_sha256=digest(fitted),
        native_camera_sha256=info['state_sha256'],scene_values_sha256=info['scene_values_sha256']))
    print('exact source exposures', len(pts), 'changed', int(changed.sum()), changes, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('stage', type=Path)
    parser.add_argument('mode', choices=('catalog','full','proof'))
    args = parser.parse_args()
    stage = args.stage.resolve()
    r0, r1, camera, info, doc, z0, z1 = state(stage)
    if args.mode == 'catalog':
        catalog(stage, r0, r1, info, doc, z0, z1)
    else:
        problem = image_problem(camera, info, r1)
        review = [81.36,81.44,81.52,82.04,83.28,84.90,86.14,86.80,87.80,88.12,88.84,89.32,
                  90.76,92.40,92.64,93.14]
        render_video(problem, stage, proof_only=args.mode == 'proof', references=(r0,r1),
            proof_seconds=review, parameter=stage/'solve/candidate.npz', refit_label='First spin body/skates refit')
