"""Scene support must survive the Python-to-native factor boundary."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from rinkspace.core.gpu import ResidentResidualProblem


class NativeCall:
    """Record only ABI calls; these tests exercise admission, not GPU math."""
    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return 0


def boundary(station):
    problem = ResidentResidualProblem.__new__(ResidentResidualProblem)
    problem._lib = SimpleNamespace(**{name: NativeCall() for name in (
        "rs_gpu_attach_paths", "rs_g3_attach", "rs_g3_step",
        "rs_g3_export_state", "rs_g3_export_system")})
    problem._handle = ctypes.c_void_p(1)
    problem._codes = np.array([0], np.int32)
    problem._initial_stations = np.array([station])
    problem._local_ids = np.array([0], np.int32)
    problem.linearization_info = {"local_count": 1}
    problem._check = lambda status: None
    return problem


def test_initial_station_cannot_escape_the_selected_scene_support():
    problem = boundary(.1)
    path = {"kind": 0, "parameters": [0., 0., 0., 1., 0., 0.], "support": [.25, .75]}
    with pytest.raises(ValueError, match="outside selected native scene support"):
        problem.configure_geometry([path])
    assert problem._lib.rs_gpu_attach_paths.calls == 0


def test_solver_bounds_cannot_reexpand_a_finite_scene_path():
    problem = boundary(.5)
    problem.configure_geometry([
        {"kind": 0, "parameters": [0., 0., 0., 1., 0., 0.], "support": [.25, .75]},
    ])
    with pytest.raises(ValueError, match="outside selected native scene support"):
        problem.configure_solver(lower=[0.], upper=[1.], device_budget_bytes=2**30,
                                 host_budget_bytes=2**30)
    assert problem._lib.rs_g3_attach.calls == 0
    problem.configure_solver(lower=[.25], upper=[.75], device_budget_bytes=2**30,
                             host_budget_bytes=2**30)
    assert problem._lib.rs_g3_attach.calls == 1


def test_continuous_board_station_keeps_its_unwrapped_domain():
    problem = boundary(3.1)
    problem.configure_geometry([{"kind": 7, "parameters": [1.], "support": None}])
    problem.configure_solver(lower=[-np.inf], upper=[np.inf], device_budget_bytes=2**30,
                             host_budget_bytes=2**30)
    assert problem._lib.rs_g3_attach.calls == 1
