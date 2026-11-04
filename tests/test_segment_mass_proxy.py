"""Volume integration, rigid transforms and explicit degenerate-parent merging."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.segment_mass_proxy import (
    tetraVolumeCentroid, prepare_segment_mass_proxy)


def test_segment_volume_centroid_and_native_weight_aggregation():
    tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    volume, centre = tetraVolumeCentroid(tetra)
    np.testing.assert_allclose(volume, 1./6.)
    np.testing.assert_allclose(centre, [.25, .25, .25])
    R = Rotation.from_rotvec([.4, -.2, .8]).as_matrix()
    translation = np.array([3., -4., 5.])
    shifted_volume, shifted_centre = tetraVolumeCentroid(tetra @ R.T + translation)
    np.testing.assert_allclose(shifted_volume, volume)
    np.testing.assert_allclose(shifted_centre, centre @ R.T + translation)
    points = np.tile(tetra, (2620, 1))[:10475].copy()
    weights = np.zeros((10475, 55))
    weights[:, 0] = 1.
    for start, influence, offset in [(0, 22, [0., 3., 0.]),
                                     (4, 25, [-3., 0., 0.]),
                                     (8, 40, [3., 0., 0.])]:
        points[start:start+4] = tetra + offset
        weights[start:start+4, 0] = 0.
        weights[start:start+4, influence] = 1.
    # Two points have no volume: left-foot -> ankle -> knee -> hip -> root.
    weights[12:14, 0] = 0.
    weights[12:14, 10] = 1.
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
               12, 13, 14, 16, 17, 18, 19]
    surface = SimpleNamespace(shaped=points, weights=weights,
                              body=SimpleNamespace(parents=np.array(parents)))
    original_points, original_weights = points.copy(), weights.copy()
    proxy = prepare_segment_mass_proxy(surface)
    np.testing.assert_array_equal(proxy['native_joint_ids'], [0, 15, 20, 21])
    np.testing.assert_allclose(proxy['mass_fractions'], [.25]*4)
    np.testing.assert_allclose(proxy['centres_rest'][1:],
        np.array([[.25, 3.25, .25], [-2.75, .25, .25], [3.25, .25, .25]]))
    assert [(m['native_joint_id'], m['parent_native_joint_id']) for m in proxy['merged_groups']] == [
        (10, 7), (7, 4), (4, 1), (1, 0)]
    assert sum(proxy['vertex_counts']) == 10475
    np.testing.assert_array_equal(surface.shaped, original_points)
    np.testing.assert_array_equal(surface.weights, original_weights)
