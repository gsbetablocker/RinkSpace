"""CPU-only regressions for honest solver diagnostics and transfer accounting."""
import ctypes as c
from types import SimpleNamespace
import numpy as np
from rinkspace.core.gpu import ResidentResidualProblem


class CallableSymbol:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


def fake_problem(version):
    problem = ResidentResidualProblem.__new__(ResidentResidualProblem)
    problem._handle = c.c_void_p(1)
    problem.count = 2
    problem._out = np.zeros((2, 11))
    problem._status = np.zeros(2, dtype=np.int32)
    problem.diagnostic_download_bytes = 0
    problem.solver_info = {}
    problem._g2_backend = lambda backend: None
    problem._check = lambda status: None if status == 0 else (_ for _ in ()).throw(RuntimeError(status))
    problem.request_stop = lambda: None

    def solve(handle, tol, n, callback, context, out, count):
        summary = np.zeros(32)
        summary[1] = 1.
        summary[15] = 1.
        timings = np.zeros(12)
        callback(summary.ctypes.data_as(c.POINTER(c.c_double)),
                 timings.ctypes.data_as(c.POINTER(c.c_double)), 0., 0., None)
        result = np.ctypeslib.as_array(out, shape=(count,))
        result[:] = 0
        result[10] = 100.
        result[13] = 192.
        return 0

    problem._lib = SimpleNamespace(
        rs_g4_solve=CallableSymbol(solve),
        rs_g1_evaluate=CallableSymbol(lambda *args: 0),
    )
    if version is not None:
        problem._lib.rs_g4_stationarity_version = CallableSymbol(lambda: version)
    return problem


def test_v1_box_projection_semantics_keep_legacy_keys():
    p = fake_problem(1)
    records = []

    def progress(record):
        records.append(record)
        p.evaluate("cuda")

    result = p.solve(progress=progress)
    assert result["projected_gradient_inf"] == result["box_projected_gradient_inf"]
    assert result["stop_reason"] == "projected_gradient_small"  # existing readers
    assert result["stationarity_scope"] == "single_column_rig_boxes_and_station_boxes"
    assert not result["full_nonlinear_kkt_checked"]
    assert not records[0]["full_nonlinear_kkt_checked"]
    assert records[0]["box_projected_gradient_before"] == records[0]["projected_gradient_before"]
    assert result["diagnostic_d2h_bytes_during_solve"] == 2 * (11 * 8 + 4)
    assert result["instrumented_d2h_bytes_during_solve"] == 192 + 2 * (11 * 8 + 4)
    assert result["callback_wall_ms"] >= 0.


def test_older_library_is_never_labelled_as_fixed():
    p = fake_problem(None)
    records = []
    result = p.solve(progress=records.append)
    assert "box_projected_gradient_inf" not in result
    assert "box_projected_gradient_before" not in records[0]
    assert result["stationarity_version"] == 0
    assert result["stationarity_scope"] == "legacy_raw_retained_gradient_and_station_boxes"
    assert not result["full_nonlinear_kkt_checked"]


def test_cpu_evaluation_does_not_claim_a_device_transfer():
    p = fake_problem(None)
    p.evaluate("cpu")
    assert p.diagnostic_download_bytes == 0
    p.evaluate("cuda")
    assert p.diagnostic_download_bytes == p._out.nbytes + p._status.nbytes


def test_v2_coupled_constraint_diagnostic_has_its_own_scope():
    p=fake_problem(2)
    p.stationarity_diagnostics=lambda: {'constraint_projected_gradient_inf':0.,'original_undamped_objective':True}
    records=[];result=p.solve(progress=records.append)
    assert result['stationarity_version']==2
    assert result['full_nonlinear_kkt_checked']
    assert result['stationarity_scope']=='declared_active_physical_tangent_cone'
    assert 'box_projected_gradient_inf' not in result
    assert result['constraint_projected_gradient_inf']==result['projected_gradient_inf']
    assert result['projected_gradient_inf']==0.
    assert 'controller_reported_gradient_inf' in result
    assert records[0]['constraint_projected_gradient_before']==records[0]['projected_gradient_before']
    assert 'box_projected_gradient_before' not in records[0]
