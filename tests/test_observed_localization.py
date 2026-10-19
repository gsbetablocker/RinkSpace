import numpy as np
import pytest
from test_whole_rink_localization import fixture, sample
from rinkspace.whole_rink_localization import WholeRinkConfig, WholeRinkLocalizer
from rinkspace.observed_localization import register_edge, path_uncertainty


def tracker(tmp_path):
    return WholeRinkLocalizer(tmp_path/'model.json', tmp_path,
        WholeRinkConfig(sample_width=480, sample_height=270, reobserve_every=1000))


def test_observed_frames_continue_after_old_anchor_age_without_blind_prediction(tmp_path):
    image, _ = fixture(tmp_path)
    loc = tracker(tmp_path)
    anchor = loc.push(image, sample())
    assert anchor['status'] == 'model_reobserved'
    for i in range(1, 61):
        row = loc.push(image, sample(i, 100+i*2))
    assert row['status'] == 'image_tracked'
    assert row['anchor_age_s'] == pytest.approx(1.2)
    assert row['last_image_observation_age_s'] == 0
    assert row['local_registration']['H_keyframe_from_current']
    assert row['local_registration']['fit_keyframe_xy']
    assert len(row['anchor_lineage']) == 1
    assert not row['independent_validation']


def test_disconnected_component_can_restart_but_has_no_global_identity(tmp_path, monkeypatch):
    image, _ = fixture(tmp_path)
    loc = tracker(tmp_path)
    assert loc.push(image, sample())['status'] == 'model_reobserved'
    loc.local.last_good_time = 0.
    monkeypatch.setattr(loc, '_reobserve', lambda *a: (None, []))
    row = loc.push(image, sample(1, 102))
    assert row['local_registration']['reference_frame_only']
    assert row['local_restart_reason'] == 'unanchored_component_after_local_observation_gap'
    assert row['projection_chain'] is None and loc.bridge is None


def test_conflicting_model_observations_remain_conflicting_across_following_frames(tmp_path, monkeypatch):
    image, _ = fixture(tmp_path)
    loc = tracker(tmp_path)
    loc.push(image, sample())
    original = loc._reobserve
    def shifted(*args):
        candidate, attempts = original(*args)
        ref, h, fit = candidate
        h = h.copy(); h[0, 2] += 35
        return (ref, h, fit), attempts
    monkeypatch.setattr(loc, '_reobserve', shifted)
    conflict = loc.push(image, sample(1000, 102))
    following = loc.push(image, sample(1001, 104))
    assert conflict['status'] == following['status'] == 'ambiguous'
    assert conflict['model_anchor_candidate']['projection_chain']
    assert following['projection_chain'] is None
    assert following['camera_component_id'] == conflict['camera_component_id']


def test_derived_keys_do_not_reset_engineering_path_error():
    nodes = {}
    register_edge(nodes, {'status':'candidate', 'reference_frame_only':True}, {'sample_index':0}, 960)
    fit = {'status':'candidate', 'inlier_p95_reference_px':2, 'train_inliers':30,
           'support_fraction':.05, 'H_keyframe_from_current':np.eye(3).tolist()}
    for i in range(1, 20):
        register_edge(nodes, {**fit,'edge_keyframe_sample_index':i-1}, {'sample_index':i},960)
    short = path_uncertainty(nodes,0,1,fit,960)
    long = path_uncertainty(nodes,0,19,fit,960)
    assert long['estimated_error_px_at_960'] > short['estimated_error_px_at_960']
    assert not path_uncertainty(nodes,0,99,fit,960)['accepted']
