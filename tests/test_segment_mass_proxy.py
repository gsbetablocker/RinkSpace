"""Volume integration, rigid transforms and explicit degenerate-parent merging."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.segment_mass_proxy import (
    tetraVolumeCentroid, tetraVolumeProperties, prepare_segment_mass_proxy,
    prepare_body_skate_mass_proxy)


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
    np.testing.assert_array_equal(proxy['initial_native_joint_to_retained_native_joint'][[0, 10, 15, 20, 21]],
                                  [0, 0, 15, 20, 21])
    np.testing.assert_array_equal(proxy['initial_native_joint_to_retained_native_joint'][[1, 7]], [-1, -1])
    # The same original vertex grouping, including each explicitly merged foot,
    # must supply inertia as well as volume/COM. No joint-origin inertia sneaks in.
    for segment, joint in enumerate(proxy['native_joint_ids']):
        if joint == 0:
            selected = (weights[:, 0] > 0) | (weights[:, 10] > 0)
        else:
            selected = weights[:, {15: 22, 20: 25, 21: 40}[joint]] > 0
        volume, centre, inertia = tetraVolumeProperties(points[selected])
        np.testing.assert_allclose(proxy['volumes_rest'][segment], volume)
        np.testing.assert_allclose(proxy['centres_rest'][segment], centre)
        np.testing.assert_allclose(proxy['unit_mass_inertia_rest'][segment], inertia)
    joints = np.zeros((22, 3))
    joints[:, 0] = np.arange(22)*.1
    attached = prepare_body_skate_mass_proxy(proxy, joints, tetra, metres_per_model_unit=2.,
        metres_per_skate_unit=.1, relative_mass_per_skate=[.02, .03], mass_variant_id='synthetic-uniform')
    np.testing.assert_allclose(attached['segment_mass_fractions'], np.r_[proxy['mass_fractions'], .02, .03]/1.05)
    np.testing.assert_allclose(attached['segment_com_rest_local_m'][:4],
                               (proxy['centres_rest']-joints[proxy['native_joint_ids']])*2.)
    np.testing.assert_allclose(attached['segment_com_rest_local_m'][4:], [[.025]*3]*2)
    np.testing.assert_allclose(attached['segment_unit_mass_inertia_local_m2'][:4], proxy['unit_mass_inertia_rest']*4.)
    np.testing.assert_allclose(attached['segment_unit_mass_inertia_local_m2'][4:],
                               np.broadcast_to(tetraVolumeProperties(tetra)[2]*.01, (2, 3, 3)))
    assert attached['mass_variant_id'] == 'synthetic-uniform'
    assert attached['absolute_mass_inferred'] is False
    np.testing.assert_array_equal(surface.shaped, original_points)
    np.testing.assert_array_equal(surface.weights, original_weights)



def test_own_com_inertia_rotation_translation_and_square_scale():
    box = np.array([[x, y, z] for x in (0., 2.) for y in (0., 4.) for z in (0., 6.)])
    volume, centre, inertia = tetraVolumeProperties(box)
    expected = np.diag([(16.+36.)/12., (4.+36.)/12., (4.+16.)/12.])
    np.testing.assert_allclose(inertia, expected, atol=1e-13)
    R = Rotation.from_rotvec([.2, -.5, 1.1]).as_matrix()
    scale = .013
    shifted = box @ R.T * scale + [13., -25., 40.]
    new_volume, new_centre, new_inertia = tetraVolumeProperties(shifted)
    np.testing.assert_allclose(new_volume, volume*scale**3, rtol=1e-12)
    np.testing.assert_allclose(new_centre, centre @ R.T*scale+[13., -25., 40.])
    np.testing.assert_allclose(new_inertia, R @ expected @ R.T*scale**2, atol=1e-15)
