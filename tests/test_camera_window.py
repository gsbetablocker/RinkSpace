"""Window replay preserves sub-frame camera states and source clock gaps."""
from copy import deepcopy
import os

import numpy as np
import pytest

from rinkspace.core.native import NativeCore
from rinkspace.session.camera_exposure import FAMILY
from rinkspace.session.camera_window import window_controls


def camera_data():
    rig = [1.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1., 1.,0.,0.,
           0.,0.,0., .1,.2,.3, .04,.02,-.01, 100.,2.,
           50.,50.,.01, 2.,3., .01,.002,.001,.0001,.001,.002,.02]
    controls = np.column_stack((np.linspace(-.01,.01,20),
        np.linspace(-.03,.03,20), np.linspace(4.5,4.7,20)))
    camera = {**FAMILY, "rig": rig, "first_control_time": -.04,
              "control_dt": .02, "reference_pts": 2**54,
              "time_base_num": 1, "time_base_den": 12800,
              "first_pts": 2**54, "stop_pts_exclusive": 2**54+4096}
    return camera, controls


def test_window_retains_source_gaps_and_sub_float_clock_precision():
    camera, controls = camera_data()
    pts = np.array([2**54+511, 2**54+512, 2**54+1024], dtype=np.int64)
    before = deepcopy(camera)
    local, coefficients, times = window_controls(camera, controls, pts)
    np.testing.assert_array_equal(times, np.array([511,512,1024])/12800)
    assert len(times) == 3
    assert local["first_pts"] == int(pts[0])
    assert local["stop_pts_exclusive"] == int(pts[-1])+1
    assert camera == before
    start, stop = local["snapshot_origin"]["control_indices"]
    np.testing.assert_array_equal(coefficients, controls[start:stop])


@pytest.mark.parametrize("pts", [[], [True], [1.5], [[2**54]],
    [2**54+256,2**54], [2**54,2**54], [2**54-1], [2**54+4096]])
def test_window_rejects_invalid_identities(pts):
    camera, controls = camera_data()
    with pytest.raises(ValueError):
        window_controls(camera, controls, pts)


@pytest.mark.parametrize("tau", [-.02, 0., .02])
def test_window_preserves_every_readout_state_and_projection(tau):
    library = os.environ.get("RINKSPACE_CORE_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    core = NativeCore(library)
    camera, controls = camera_data()
    camera["rig"][-1] = tau
    pts = np.array([2**54+256,2**54+512,2**54+1024,2**54+1280], dtype=np.int64)
    local, coefficients, times = window_controls(camera, controls, pts)
    row_times = (times[:,None]+tau*np.linspace(-.5,.5,13)).ravel()
    full = core.evaluate_motion_camera(camera["rig"], controls, -.04, .02, row_times)
    saved = core.evaluate_motion_camera(local["rig"], coefficients,
        local["first_control_time"], .02, row_times)
    assert np.all(full["status"] == 0)
    np.testing.assert_array_equal(saved["status"], full["status"])
    np.testing.assert_allclose(saved["values"], full["values"], atol=2e-12, rtol=1e-13)
    for t in times:
        args = (101, t, [[0.,0.,10.],[1.,1.,10.]])
        a = core.project_motion(camera["rig"], controls, -.04, .02, *args)
        b = core.project_motion(local["rig"], coefficients, local["first_control_time"], .02, *args)
        assert np.all(a["status"] == 0)
        np.testing.assert_array_equal(a["status"], b["status"])
        np.testing.assert_allclose(a["uv"], b["uv"], atol=2e-10, rtol=1e-13)
