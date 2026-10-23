import copy

import numpy as np
import pytest

from rinkfit_joint.material_path import new_board_path_points
from rinkfit_start.native_scene import NativeScene


def scene(points, closed=True):
    return NativeScene({
        "blocks": [
            {"name": "global", "values": [.5, .1, .1, 2., .02] + [0.] * 27},
            {"name": "ad/test", "values": [.3, .1, .002, .018]},
        ],
        "materials": [{"label": "print", "paths": [{
            "kind": 2, "advertisement_block": 1, "closed": closed, "points": points,
        }]}],
    })


def test_new_closed_kind2_contains_final_edge_in_existing_consumer():
    original = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    untouched = copy.deepcopy(original)
    points = new_board_path_points(original, closed=True)
    assert original == untouched
    assert points == original + [original[0]]
    material = scene(points)
    np.testing.assert_array_equal(material.world(0, 0, 0.), material.world(0, 0, 1.))
    np.testing.assert_allclose(material.world(0, 0, .875), material.board(.3, .01), atol=1e-14)


def test_existing_repeated_closure_and_open_points_keep_station_world_points():
    original = [[0., 0.], [1., 0.], [1., 1.], [0., 0.]]
    stations = np.array([0., .1, .4, .75, 1.])
    points = new_board_path_points(original, closed=True)
    assert points == original
    np.testing.assert_array_equal(scene(points).world(0, 0, stations), scene(original).world(0, 0, stations))
    open_points = original[:-1]
    assert new_board_path_points(open_points, closed=False) == open_points
    assert new_board_path_points([[.2, .8]], closed=True) == [[.2, .8]]


def test_reading_legacy_unclosed_vertices_does_not_silently_migrate_station():
    original = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    material = scene(original)
    # Existing N-1 geometry remains readable at its old station. A migration
    # of the stored station/domain would be required to add its closing edge.
    np.testing.assert_allclose(material.world(0, 0, 1.), material.board(.3, .018), atol=1e-14)
    assert material.materials[0]["paths"][0]["points"] == original


@pytest.mark.parametrize("points", [[], [[0., 1., 2.]], [[0., float("nan")]]])
def test_new_path_refuses_invalid_vertices(points):
    with pytest.raises(ValueError):
        new_board_path_points(points, closed=True)
