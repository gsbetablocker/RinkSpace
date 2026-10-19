import json
from types import SimpleNamespace

import numpy as np

from calibration.solver_runtime import install
install()
from calibration.program_tracking.current_source_points import CurrentSourcePoints, save_current_source_points
from calibration.program_tracking.render_programme_root_trace import ProgrammeRootView


def test_frame_zero_alias_cut_return_and_missing_exposure_use_own_source_points(tmp_path):
    native = np.array([100, 101, 103, 108], np.int64)
    actual = np.array([100, 100, 103, 108], np.int64)
    camera = tmp_path / 'camera.npz'
    np.savez(camera, pts=native, exposure_representative_pts=actual,
        metadata_json=json.dumps(dict(source_sha256='a' * 64, source_time_base='1/15360')))
    manifest = tmp_path / 'input.json'
    manifest.write_text('{}')
    point_path = tmp_path / 'points.npz'
    save_current_source_points(point_path, camera,
        [(100, [[12., 30.]]), (103, [[52., 71.], [80., 100.]])],
        q_sha256='b' * 64, input_manifest=manifest,
        family_receipts=[dict(complete_active_input=True)])
    view = ProgrammeRootView.__new__(ProgrammeRootView)
    view.current_source_points = CurrentSourcePoints(point_path, camera)
    view.reps = actual
    view.lookup = {int(p): i for i, p in enumerate(native)}
    view.display_camera25 = np.zeros((4, 25))
    view.valid = np.zeros(4, bool)
    view.inferred = np.zeros(4, bool)
    view.trail = np.zeros((1080, 660, 3), np.uint8)
    view.header = np.zeros((70, 2580, 3), np.uint8)
    view.status_patches = dict(unknown=np.zeros((64, 660, 3), np.uint8))
    view.renderer = SimpleNamespace(draw=lambda image, *_args, **_kw: (image.copy(), None))
    view.masks = SimpleNamespace(get=lambda *_args, **_kw: (None, None))
    view.advance = lambda _i: None
    black = np.zeros((1080, 1920, 3), np.uint8)

    first = view.draw(black, 100)
    assert view.last_current_source_point_draw == (100, 100, 1)
    alias = view.draw(black, 101)
    np.testing.assert_array_equal(first[70:1150, :1920], alias[70:1150, :1920])
    across_cut = view.draw(black, 103)
    assert view.last_current_source_point_draw == (103, 103, 2)
    assert across_cut[137:145, 48:57].any()
    assert not across_cut[97:105, 8:17].any()
    missing = view.draw(black, 108)
    assert view.last_current_source_point_draw == (108, 108, 0)
    assert not missing[70:1150, :1920].any()
