"""The display camera must look from inside and above the ice."""
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from rinkspace.session import print_installation


@pytest.mark.parametrize("mid", np.pi * np.array([.25, .75, 1.25, 1.75]))
def test_board_view_observer_is_above_ice_on_all_corners(monkeypatch, tmp_path, mid):
    # A synthetic circular course isolates viewing direction from native model
    # construction. Opposite corners must both present their inward face.
    samples = []

    def board_points(board, stations, heights):
        xyz = np.column_stack((np.cos(stations), np.sin(stations), heights))
        samples.append(xyz)
        return xyz

    core = Mock()
    core.board_points.side_effect = board_points
    core.board_course_chart.return_value = np.linspace(0., 2*np.pi, 9)
    core.board_perimeter.return_value = 2*np.pi
    board = [1., 1., .1, .1, 2., .02]
    stations = [mid-.04, mid+.04]
    data = {
        "instance_id": "print_instance/test",
        "paths": [{
            "id": "test.top_edge", "role": "panel_outer_boundary",
            "sz": [[s, .015] for s in stations],
            "xyz": np.column_stack((np.cos(stations), np.sin(stations), [.015, .015])).tolist(),
        }],
    }
    original = deepcopy(data)
    projected = []
    mapping = print_installation._mapping

    def capture_mapping(points, rect):
        projected.append(np.asarray(points))
        return mapping(points, rect)

    monkeypatch.setattr(print_installation, "_mapping", capture_mapping)
    monkeypatch.setattr(print_installation, "Canvas", Mock())
    print_installation.board_view(data, board, core, tmp_path)

    # Recover the actual display axes from world points and their projection,
    # rather than repeating the implementation's trigonometric formula.
    world = np.vstack(samples[:2])
    coefficients = np.linalg.lstsq(
        np.column_stack((world, np.ones(len(world)))), projected[0], rcond=None,
    )[0]
    screen_right, screen_up = coefficients[:3].T
    toward_observer = np.cross(screen_right, screen_up)
    inward = -np.array([np.cos(mid), np.sin(mid)])
    assert toward_observer[2] > 0, "display observer is below the ice"
    assert toward_observer[:2] @ inward > 0, "display observer is outside the rink"
    assert screen_up[2] > 0, "world height must remain upright"
    assert data == original
