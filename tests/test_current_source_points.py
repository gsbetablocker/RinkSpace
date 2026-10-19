import json

import numpy as np
import pytest

from calibration.program_tracking.current_source_points import CurrentSourcePoints, save_current_source_points


def packet(tmp_path):
    camera = tmp_path / 'camera.npz'
    np.savez(camera, pts=np.array([100, 101, 103], np.int64),
             exposure_representative_pts=np.array([100, 100, 103], np.int64),
             metadata_json=json.dumps(dict(source_sha256='a' * 64, source_time_base='1/15360')))
    manifest = tmp_path / 'input.json'
    manifest.write_text('{}')
    path = tmp_path / 'points.npz'
    save_current_source_points(path, camera, [(100, [[12.25, 30.5], [52., 71.]])],
        q_sha256='b' * 64, input_manifest=manifest,
        family_receipts=[dict(family='actual_active_test_input', complete_active_input=True)])
    return path, camera


def test_only_current_exposure_pixels_drawn_and_missing_exposure_not_filled(tmp_path):
    path, camera = packet(tmp_path)
    view = CurrentSourcePoints(path, camera, q_sha256='b' * 64)
    np.testing.assert_array_equal(view.points(100), [[12.25, 30.5], [52., 71.]])
    assert view.points(103).shape == (0, 2)
    image = np.zeros((1080, 1920, 3), np.uint8)
    view.draw(image, 103)
    assert not image.any()
    view.draw(image, 100)
    assert image[27:35, 8:17].any() and image[67:76, 48:57].any()
    # No point numbers, history labels or model projection elsewhere.
    image[27:35, 8:17] = 0
    image[67:76, 48:57] = 0
    assert not image.any()
    with pytest.raises(KeyError):
        view.points(101)  # Presentation aliases must be resolved by the camera.


def test_model_or_camera_mismatch_cannot_silently_draw_other_points(tmp_path):
    path, camera = packet(tmp_path)
    with pytest.raises(ValueError, match='exact saved camera'):
        CurrentSourcePoints(path, camera, q_sha256='c' * 64)
    with np.load(camera, allow_pickle=False) as z:
        values = {key: z[key].copy() for key in z.files}
    values['pts'] = np.array([100, 102, 103], np.int64)
    np.savez(camera, **values)
    with pytest.raises(ValueError, match='exact saved camera'):
        CurrentSourcePoints(path, camera)


@pytest.mark.parametrize('field', ['historical_or_retired_points_included', 'model_projected_points_included',
                                  'nonpixel_placeholder_constraints_included', 'extra_point_labels'])
def test_history_projection_placeholders_and_extra_labels_rejected(tmp_path, field):
    path, camera = packet(tmp_path)
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key].copy() for key in z.files}
    metadata = json.loads(str(values['metadata_json'].item()))
    metadata[field] = True
    values['metadata_json'] = json.dumps(metadata)
    np.savez(path, **values)
    with pytest.raises(ValueError, match='exact saved camera'):
        CurrentSourcePoints(path, camera)
