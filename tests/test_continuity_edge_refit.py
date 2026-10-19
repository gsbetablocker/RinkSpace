import copy

import numpy as np
import pytest

from rinkspace.continuity_edge_refit import (EdgeRefitConfig, fit_frozen_edge,
    refit_frames, regularized_plane_fit, similarity_fit)
from rinkspace.geometry import project


def points():
    xx, yy = np.meshgrid(np.linspace(100, 850, 8), np.linspace(150, 460, 6))
    return np.c_[xx.ravel(), yy.ravel()]


@pytest.mark.parametrize('strategy', ['similarity', 'regularized_plane'])
def test_fits_same_observed_translation_without_resetting_reference(strategy):
    x = points(); y = x+[6., -3.]
    fit = fit_frozen_edge(x, y, [960, 540], strategy)
    assert fit['accepted']
    assert fit['holdout_count'] == len(x)//4
    np.testing.assert_allclose(project(fit['H_keyframe_from_current'], x), y, atol=1e-6)


def test_regularized_plane_has_bounded_extrapolation_under_narrow_noisy_ice_support():
    rng = np.random.default_rng(6)
    x = points()*[.35, .5]+[250, 100]
    y = x+[3., -2.]+rng.normal(0, .3, x.shape)
    h = regularized_plane_fit(x, y, [960, 540], EdgeRefitConfig())
    corners = np.array([[0, 0], [960, 0], [960, 540], [0, 540]])
    assert np.max(np.linalg.norm(project(h, corners)-(corners+[3., -2.]), axis=1)) < 3.
    assert np.linalg.det(h) > 0


def test_holdout_error_rejects_without_refitting_to_withheld_points():
    x = points(); y = x+[3., -2.]
    y[np.arange(len(y)) % 4 == 0] += [10., 0.]
    fit = fit_frozen_edge(x, y, [960, 540], 'similarity')
    assert not fit['accepted'] and fit['holdout_p95_sample_px'] == pytest.approx(10.)
    np.testing.assert_allclose(project(fit['H_keyframe_from_current'], x), x+[3., -2.], atol=1e-8)


def test_refit_keeps_original_pairs_matrices_anchors_and_pts():
    import runpy
    fixture = runpy.run_path('tests/test_continuity_graph.py')['fixture']
    rows, _ = fixture(count=4, anchors=(0, 3))
    frozen = copy.deepcopy(rows)
    result = refit_frames(rows, 'similarity')
    assert rows == frozen
    for before, after in zip(rows, result['frames']):
        assert before['pts'] == after['pts']
        assert before.get('model_anchor_candidate') == after.get('model_anchor_candidate')
        a, b = before['local_registration'], after['local_registration']
        assert a.get('fit_current_xy') == b.get('fit_current_xy')
        assert a.get('fit_keyframe_xy') == b.get('fit_keyframe_xy')
        assert b['original_H_reference_from_current'] == a['H_reference_from_current']
        np.testing.assert_allclose(b['H_reference_from_current'], a['H_reference_from_current'], atol=1e-7)
        assert after['projection_chain'] is None


def test_degenerate_point_cloud_does_not_create_camera_fit():
    x = np.c_[np.arange(20), np.zeros(20)]
    with pytest.raises(ValueError, match='two-dimensional'):
        similarity_fit(x, x)
