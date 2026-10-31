"""Performance paths must preserve independent equations and state certificates."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import test_g3_gpu as fixture
from rinkspace.core.gpu import ResidentResidualProblem

pytestmark = pytest.mark.skipif(not os.environ.get("RINKSPACE_G4_LIBRARY"), reason="explicit CUDA/cuDSS build required")


def problem_fixture():
    f = fixture.G3GPU()
    f.setUp()
    return f


def test_certificate_cache_restores_descent_after_rejected_trials():
    f = problem_fixture()
    with ResidentResidualProblem(f.core, **f.kw) as base:
        uv = base.evaluate("cpu")["values"][:, 3:5]
    f.kw["observed"] = 50. + 1e5 * (uv - 50.)
    dirs = np.zeros((8, 3, 1)); dirs[:, 2, 0] = .001
    plan = {**f.plan, "directions": dirs, "local_ids": np.full(4, -1, np.int32), "local_scales": np.zeros(0)}
    with f.run_problem(plan) as p:
        f.attach(p)
        first = p.stationarity_diagnostics(include_constraints=True)
        for _ in range(2):
            assert not p.lm_step()["summary"]["accepted"]
            again = p.stationarity_diagnostics(include_constraints=True)
            np.testing.assert_array_equal(again["projected_retained_descent"], first["projected_retained_descent"])
            assert again["constraint_projected_gradient_inf"] == first["constraint_projected_gradient_inf"]
        perf = p.performance_metrics()
        assert perf["stationarity_recomputations"] == 1
        assert perf["stationarity_cache_hits"] == 2


def test_certificate_cache_invalidates_on_acceptance_and_requested_accuracy():
    f = problem_fixture()
    with f.run_problem() as p:
        f.attach(p)
        p.stationarity_diagnostics(tolerance=1e-6)
        p.stationarity_diagnostics(tolerance=1e-6)
        assert p.performance_metrics()["stationarity_recomputations"] == 1
        assert p.lm_step()["summary"]["accepted"]
        after = p.stationarity_diagnostics(tolerance=1e-6)
        assert p.performance_metrics()["stationarity_recomputations"] == 2
        p.stationarity_diagnostics(tolerance=1e-8)
        assert p.performance_metrics()["stationarity_recomputations"] == 3
        assert after["objective"] == pytest.approx(p.linearize("cpu")["summary"]["huber_cost"], abs=1e-10)


def test_cached_and_recomputed_controller_reach_same_solution():
    f = problem_fixture(); answers = []
    for cache in ("0", "1"):
        with patch.dict(os.environ, RINKSPACE_STATIONARITY_CACHE=cache), f.run_problem() as p:
            f.attach(p)
            result = p.solve(gradient_tolerance=1e-6)
            assert result["numerically_converged"]
            assert result["solve_api_wall_ms"] >= result["native_controller_wall_ms"]
            if cache == "1":
                assert result["performance_after"]["stationarity_cache_hits"] >= 1
            answers.append((result, p.export_state()))
    assert answers[0][0]["final_cost"] == pytest.approx(answers[1][0]["final_cost"], abs=1e-12)
    for key in answers[0][1]:
        np.testing.assert_allclose(answers[0][1][key], answers[1][1][key], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("local", [False, True])
def test_rank_two_prediction_matches_dense_component_quadratic(local):
    f = problem_fixture()
    with ResidentResidualProblem(f.core, **f.kw) as p:
        p.configure_pixel_linearization(directions=f.directions,
            local_ids=np.arange(4, dtype=np.int32) if local else np.full(4, -1, np.int32),
            local_scales=np.full(4, .001) if local else np.zeros(0))
        f.attach(p); p.linearize(); blocks = p.export_blocks()
        summary = p.lm_step()["summary"]
        applied = p.export_step_system()["applied"]
        expected = 0.
        for component, columns in zip(p.layout["components"], p.layout["component_columns"]):
            lb, nl, _, _, dim, ho, go = map(int, component)
            qb, nq = map(int, columns)
            ids = np.r_[p.layout["columns"][qb:qb+nq], p._q+p.layout["locals"][lb:lb+nl]].astype(int)
            d = applied[ids]; h = blocks["H"][ho:ho+dim*dim].reshape(dim, dim); g = blocks["g"][go:go+dim]
            expected += -g@d - .5*d@h@d
        assert summary["predicted_decrease"] == pytest.approx(expected, rel=1e-10, abs=1e-12)
        assert p.performance_metrics()["trial_residual_buffer_bytes"] == 0
        assert p.performance_metrics()["profile_restoration_calls"] == 0
        assert summary["trial_cost"] == pytest.approx(.5*np.sum(p.evaluate("cuda")["values"][:, 9:11]**2), abs=1e-10)


def test_correlated_trial_keeps_only_whitened_residuals():
    f = problem_fixture()
    with f.run_problem() as p:
        f.attach(p)
        result = p.lm_step()["summary"]
        assert result["accepted"]
        assert p.performance_metrics()["trial_residual_buffer_bytes"] == 8*sum(f.plan["ranks"])
        assert result["trial_cost"] == pytest.approx(p.linearize("cpu")["summary"]["huber_cost"], abs=1e-10)


def test_curve_refinement_invalidates_cached_certificate():
    f = problem_fixture()
    path = dict(kind=1, parameters=[0., 0., 0., .3, 0., 2*np.pi])
    kw = {**f.kw, "codes": np.zeros(4, np.int32), "stations": np.array([.05, .3, .6, .95])}
    with ResidentResidualProblem(f.core, **kw) as p:
        p.configure_geometry([path]); observed = p.evaluate("cuda")["values"][:, 3:5]
    kw.update(observed=observed, stations=np.array([1., .1, .5, .2]))
    with ResidentResidualProblem(f.core, **kw) as p:
        p.configure_geometry([path])
        p.configure_pixel_linearization(directions=f.directions, local_ids=np.arange(4, dtype=np.int32), local_scales=np.full(4, .001))
        f.attach(p); before = p.stationarity_diagnostics()
        assert p.refine_curve_stations()["changed_rows"] == 4
        after = p.stationarity_diagnostics()
        assert p.performance_metrics()["stationarity_recomputations"] == 2
        assert after["objective"] < before["objective"]
        assert after["objective"] == pytest.approx(p.linearize("cpu")["summary"]["huber_cost"], abs=1e-10)


@pytest.mark.parametrize("windowed", [False, True])
def test_parallel_faces_preserve_serial_certificate_and_face_order(windowed):
    from test_camera_envelope import CameraEnvelopeTests
    camera = CameraEnvelopeTests(); camera.setUp()
    f = problem_fixture(); rig = camera.rig.copy(); rig[37] = .02
    rig[20] = .1+4*(128*np.finfo(float).eps)
    kw = {**f.kw, "rig": rig, "controls": np.tile([0., .15, np.log(100.)], (8, 1)), "codes": np.zeros(4, np.int32)}
    rd = np.zeros((27, 2)); rd[11, 0] = 1.; rd[15, 1] = 1.
    with ResidentResidualProblem(f.core, **kw) as p:
        p.configure_geometry([dict(kind=4, vertices=[[0., 3., 0.]], parameters=[])])
        p.configure_camera_parameters(directions=rd, max_readout=.08)
        p.configure_camera_envelope(forward_bounds=[.1,.3], pupil_bounds=[-.1,.1],
            max_pan_tilt=camera.limits[4], max_tilt_deviation=camera.limits[5],
            elevation_bounds=camera.limits[6:8], roll_bounds=camera.limits[8:10],
            frame_time_intervals=[[-.1,.04],[.08,.2]] if windowed else None)
        p.configure_pixel_linearization(directions=np.zeros((8,3,2)), local_ids=np.full(4,-1), local_scales=[])
        f.attach(p)
        with patch.dict(os.environ, RINKSPACE_STATIONARITY_FACE_BACKEND="serial"):
            serial = p.stationarity_diagnostics(include_constraints=True)
        with patch.dict(os.environ, RINKSPACE_STATIONARITY_FACE_BACKEND="parallel"):
            parallel = p.stationarity_diagnostics(include_constraints=True)
        assert serial["active_constraint_count"] > 0
        for name in ("constraint_records", "normalized_constraint_jacobian", "projected_retained_descent"):
            np.testing.assert_allclose(parallel[name], serial[name], rtol=1e-13, atol=1e-13)
        assert p.performance_metrics()["stationarity_recomputations"] == 2
