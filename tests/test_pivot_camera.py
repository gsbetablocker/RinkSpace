"""Synthetic checks of a tripod pivot; real-video accuracy is checked separately."""
from __future__ import annotations

import json

import cv2
import numpy as np

from rinkspace.calibration_companion import Camera, Scene, look_at_rvec
from rinkspace.calibration_companion.pivot_camera import TripodPivotJointProblem


def test_rotating_offset_moves_optical_center_and_sparse_derivatives(tmp_path):
    data = {
        "schema_version": "rinkspace.scene3d.v1", "revision": "synthetic-pivot",
        "coordinate_system": {"unit": "synthetic"}, "ice_features": [],
        "parameters": {"bounds": [-5., -3., 5., 3.], "corner_radius": .8, "board_height": .6},
        "ice_boundary": {"points_xyz": [[-5., -3., 0.], [5., -3., 0.],
                                           [5., 3., 0.], [-5., 3., 0.]]},
        "board_surface": {"perimeter_length": 10., "panels": []},
        "meshes": [{"name": "rink_boards", "positions": [
            [-5., -3., 0.], [5., -3., 0.], [-5., -3., .6], [5., -3., .6]]}],
        "advertisements": [],
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    scene = Scene(path)
    pivot = np.array([0., -8., 4.])
    offset = np.array([.3, 0., 1.8])
    points = [[-4., -2., 0.], [4., -2., 0.], [4., 2., 0.],
              [-4., 2., 0.], [-2., 1., 1.], [2., -1., 2.]]
    frames, observations, truth = [], [], {}
    for index, target in enumerate(([0., 0., 0.], [3., .5, 0.])):
        fid = f"view{index}"
        rvec = look_at_rvec(pivot, target)
        R = cv2.Rodrigues(rvec)[0]
        center = pivot + R.T @ offset
        focal = 700. + index * 120.
        camera = Camera(center, rvec,
                        [[focal, 0., 400.], [0., focal, 300.], [0., 0., 1.]], [])
        frames.append({"id": fid, "installation_id": "tripod", "image_size_wh": [800, 600],
                       "principal_point_px": [400., 300.],
                       "initial_rvec_world_to_camera": rvec.tolist(),
                       "initial_focal_px": focal, "focal_bounds_px": [300., 1200.]})
        truth[fid] = (center, rvec, focal)
        for j, xyz in enumerate(points):
            observations.append({"id": f"{fid}_p{j}", "frame_id": fid,
                                 "type": "point", "use": "fit", "sigma_px": 1.,
                                 "target": {"kind": "explicit_point", "xyz": xyz,
                                            "provenance": "synthetic"},
                                 "pixels_uv": camera.project([xyz])[0].tolist()})
    problem = TripodPivotJointProblem(scene, {
        "synthetic": True, "scene_sha256": scene.sha256,
        "geometry_mode": "shared_scene_joint", "scene_parameters": [],
        "installations": [{"id": "tripod", "initial_center_xyz": pivot.tolist(),
                           "fit_center": True,
                           "center_bounds_xyz": [[-2., -10., 2.], [2., -6., 6.]],
                           "tripod_pivot": {
                               "model": "rigid_offset_in_camera_coordinates",
                               "initial_offset_camera_xyz": [0., 0., 1.5],
                               "fit_offset_axes": [True, False, True],
                               "offset_bounds_camera_xyz": [[-2., 0., 0.], [2., 0., 3.]],
                           }}],
        "frames": frames, "observations": observations, "curve_samples": 16,
    })
    names = {name: i for i, name in enumerate(problem.parameter_names)}
    x = problem.x0.copy()
    x[names["installation:tripod:offset_camera_0"]] = offset[0]
    x[names["installation:tripod:offset_camera_2"]] = offset[2]
    for fid, (center, _, _) in truth.items():
        np.testing.assert_allclose(problem.camera(x, fid).center_xyz, center, atol=1e-12)
    assert np.linalg.norm(truth["view0"][0] - truth["view1"][0]) > .1
    np.testing.assert_allclose(problem.residual(x), 0., atol=1e-9)
    sparse = problem.jac_sparsity.toarray()
    for axis in (0, 2):
        column = names[f"installation:tripod:offset_camera_{axis}"]
        a, b = x.copy(), x.copy()
        a[column] += 1e-6
        b[column] -= 1e-6
        derivative = (problem.residual(a) - problem.residual(b)) / 2e-6
        assert np.all(sparse[np.abs(derivative) > 1e-7, column] != 0)
