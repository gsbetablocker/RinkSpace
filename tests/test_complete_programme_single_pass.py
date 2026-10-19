from pathlib import Path
import json

import pytest

from calibration.program_tracking.export_complete_programme_review import complete_programme_segment, check_segment


def test_single_pass_command_keeps_first_frame_and_all_cut_windows():
    command = ['python', '-m', 'calibration.program_tracking.render_programme_root_trace',
        '--camera', 'current-camera.npz', '--body', 'current-body.npz',
        '--current-source-points', 'current-pixels.npz', '--video',
        '--output', 'shot_01', '--start-pts', '100', '--end-pts', '102']
    plan = dict(segments=[dict(command=command), dict(command=command)],
        output=str(Path('whole')), source_pts_first=100, source_pts_last=120,
        native_presentations=7, native_pts_sha256='ordered-whole-programme')
    result = complete_programme_segment(plan)
    actual = result['command']
    assert actual[actual.index('--output') + 1] == plan['output']
    assert actual[actual.index('--start-pts') + 1] == '100'
    assert actual[actual.index('--end-pts') + 1] == '120'
    assert actual[actual.index('--current-source-points') + 1] == 'current-pixels.npz'
    assert result['native_frames'] == 7
    assert result['native_pts_sha256'] == 'ordered-whole-programme'
    assert command[-1] == '102'


def test_single_pass_receipt_cannot_skip_first_exposure_overlay(tmp_path):
    camera = dict(path='camera.npz', sha256='camera')
    body = dict(path='body.npz', sha256='body')
    scene = dict(path='scene.json', sha256='scene')
    points = dict(path='points.npz', sha256='points')
    coverage = dict(native_presentations=7, first_native_pts=100, first_frame_points=2,
        maximum_frame_points=5, maximum_frame_points_native_pts=113,
        native_actual_marker_count_rows_sha256='every-native-actual-point-count')
    plan = dict(camera=camera, scene=scene, current_source_points=points,
        single_pass=True, current_source_point_coverage_expected=coverage,
        expected_geometry=dict(source_and_plan_all_geometry=dict(grid=1, marking=65),
                               advertisement_ids=list(range(1, 27))))
    segment = dict(directory=str(tmp_path), native_frames=7,
        source_start_pts=100, source_last_pts_inclusive=120, native_pts_sha256='native')
    receipt = dict(inputs=[camera, body, scene, points], current_source_points=points,
        native_video_frames=7, video_production=dict(source_pts_first=100, source_pts_last=120,
            encoded_input_native_pts_sha256='native', encoded_input_exact_source_order=True,
            current_source_point_draw_coverage=dict(coverage,
                all_native_frames_points_draw_invoked=True, only_current_exposure_points=True,
                missing_exposures_not_filled=True)),
        source_and_plan_all_geometry=dict(grid=10, marking=65),
        advertisement_ids=list(range(1, 27)), display_camera_smoothing=dict(enabled=False),
        trajectory_filtered_in_renderer=False, camera_and_body_exposures_equal=True)
    (tmp_path / 'grid_and_trajectory_WORKING.mp4').touch()
    receipt_path = tmp_path / 'render.json'
    receipt_path.write_text(json.dumps(receipt))
    assert check_segment(plan, segment, body)['native_video_frames'] == 7
    receipt['video_production']['current_source_point_draw_coverage']['first_frame_points'] = 0
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='frame zero'):
        check_segment(plan, segment, body)
