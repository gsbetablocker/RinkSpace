"""Synthetic conic contracts; equal ellipse angles are never point identities."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('scipy')

from rinkspace.event_model import conic_distance, ellipse_conic, ellipse_points, fit_curve_map
from rinkspace.geometry import project


def transformed_ellipse(ellipse, image_from_model):
    inverse = np.linalg.inv(image_from_model)
    conic = inverse.T @ ellipse_conic(ellipse) @ inverse
    a, b = conic[:2, :2], conic[:2, 2]
    center = -np.linalg.solve(a, b)
    constant = conic[2, 2] + b @ center
    values, vectors = np.linalg.eigh(a / -constant)
    assert np.all(values > 0), 'Synthetic transform must preserve an ellipse'
    radii = 1 / np.sqrt(values)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    angle = (angle + 90) % 180 - 90
    return [*center.tolist(), *radii.tolist(), float(angle)]


def curves():
    return [[210., 170., 80., 43., 14.], [730., 230., 86., 51., -8.],
            [410., 520., 102., 48., 28.]]


def normal_errors(h, model, image):
    """Fresh curve samples and normals; no same-angle pixel pair residual."""
    forward = np.concatenate([np.abs(conic_distance(project(h, ellipse_points(a, 173)), ellipse_conic(b)))
                              for a, b in zip(model, image)])
    backward = np.concatenate([np.abs(conic_distance(project(np.linalg.inv(h), ellipse_points(b, 181)), ellipse_conic(a)))
                               for a, b in zip(model, image)])
    return forward, backward


def test_three_separated_identified_conics_recover_projective_normal_alignment():
    model = curves()
    truth = np.array([[1.04, .13, 74.], [-.025, .86, 51.], [1.5e-4, -1.2e-4, 1.]])
    image = [transformed_ellipse(ellipse, truth) for ellipse in model]
    recovered, audit = fit_curve_map(model, image)
    forward, backward = normal_errors(recovered, model, image)
    assert np.percentile(forward, 95) < .02
    assert np.percentile(backward, 95) < .02
    assert audit['physical_group_count'] == 3
    assert audit['solver_rank'] == 8
    assert not audit['ellipse_extrema_used_as_correspondences']
    assert not audit['independent_validation']
    assert audit['status'] == 'candidate'


def test_axis_swap_and_angle_periodicity_do_not_create_point_correspondences():
    model = curves()
    truth = np.array([[.92, -.035, 36.], [.025, 1.06, -19.], [8e-5, -3e-5, 1.]])
    image = [transformed_ellipse(ellipse, truth) for ellipse in model]
    # These are exactly the same ellipses but their parameter origins differ.
    image[0][4] += 180
    image[1] = [*image[1][:2], image[1][3], image[1][2], image[1][4] + 90]
    recovered, audit = fit_curve_map(model, image)
    forward, backward = normal_errors(recovered, model, image)
    assert np.percentile(forward, 95) < .02
    assert np.percentile(backward, 95) < .02
    assert audit['curve_normal_fit_p95_sample_px'] < .02


@pytest.mark.parametrize('count', [0, 1, 2])
def test_fewer_than_three_physical_conic_groups_are_rejected(count):
    model = curves()[:count]
    with pytest.raises(ValueError, match='At least three'):
        fit_curve_map(model, model)


def test_mismatched_conic_group_lists_are_rejected():
    with pytest.raises(ValueError, match='At least three'):
        fit_curve_map(curves(), curves()[:2])


@pytest.mark.parametrize('model', [
    [[300., 200., 80., 45., 10.]] * 3,
    [[300., 200., 80., 45., 10.], [300., 200., 60., 33.75, 10.], [300., 200., 40., 22.5, 10.]],
])
def test_repeated_or_concentric_curves_cannot_inflate_physical_group_count(model):
    with pytest.raises(ValueError, match='Distinct spatial physical curves'):
        fit_curve_map(model, model)


def test_three_conics_with_annotation_perturbation_report_candidate_fit_only():
    model = curves()
    truth = np.array([[1.04, .13, 74.], [-.025, .86, 51.], [1.5e-4, -1.2e-4, 1.]])
    image = np.array([transformed_ellipse(ellipse, truth) for ellipse in model])
    image += np.array([[.7, -.5, .2, -.3, .15], [-.4, .3, -.4, .2, -.1], [.3, .5, -.2, .4, .1]])
    recovered, audit = fit_curve_map(model, image.tolist())
    forward, backward = normal_errors(recovered, model, image)
    assert np.percentile(forward, 95) < 1.5
    assert np.percentile(backward, 95) < 1.5
    assert audit['curve_normal_fit_p95_sample_px'] > .05
    assert audit['status'] == 'candidate'
    assert not audit['independent_validation']


def test_conic_normal_distance_is_zero_on_curve_and_measures_small_normal_offset():
    ellipse = [150., 120., 80., 35., 23.]
    points = ellipse_points(ellipse, 151)
    conic = ellipse_conic(ellipse)
    assert np.max(np.abs(conic_distance(points, conic))) < 1e-10
    gradient = (np.c_[points, np.ones(len(points))] @ conic)[:, :2] * 2
    gradient /= np.linalg.norm(gradient, axis=1, keepdims=True)
    errors = conic_distance(points + .05 * gradient, conic)
    assert np.max(np.abs(errors - .05)) < .0002
