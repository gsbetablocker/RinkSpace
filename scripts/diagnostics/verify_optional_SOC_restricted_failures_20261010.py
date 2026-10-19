"""Fault-inject small real QPs to verify strict optional-SOC fallback scope."""
from pathlib import Path
import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import clarabel
from calibration.solver_runtime.optional_soc_runtime import install_optional_SOC_numerical_failure_route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    installation = install_optional_SOC_numerical_failure_route()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    H, g = np.eye(2), np.array([-2., 0.])
    lower = np.full(2, -np.inf)
    upper = -lower
    A = np.array([[1., 0.], [-1., 0.]])
    original_b = np.array([0., -1.])
    tightened_b = np.array([0.1, -0.5])
    original_backend = clarabel.DefaultSolver
    log = io.StringIO()
    with redirect_stdout(log):
        prior_step, _, prior_KKT = qp._linearly_constrained_shared_step(
            H, g, lower, upper, A, original_b)
    checks = []

    class Cancelled(BaseException):
        pass

    def run_case(mode, scope):
        backend_calls = []
        injected = [False]

        class Backend:
            def __init__(self, *args, **kwargs):
                self.original = original_backend(*args, **kwargs)
                self.rows = args[2].shape[0]

            def solve(self):
                backend_calls.append(self.rows)
                if not injected[0]:
                    injected[0] = True
                    if mode == "exception":
                        raise RuntimeError("injected restricted backend numerical exception")
                    if mode == "cancel":
                        raise Cancelled("injected root-owned cancellation")
                    if mode == "status":
                        return SimpleNamespace(status="InsufficientProgress", iterations=1)
                    result = self.original.solve()
                    if mode == "KKT":
                        # A solved candidate with a zeroed dual cannot certify
                        # this active bound; the original full KKT must reject.
                        return SimpleNamespace(status=result.status, iterations=result.iterations,
                                               x=result.x, z=np.zeros_like(result.z))
                    return result
                return self.original.solve()

        clarabel.DefaultSolver = Backend
        prior = SimpleNamespace(step=prior_step.copy(), report={
            "shared_linear_domain_step": dict(prior_KKT)} if scope == "verified_SOC" else {})
        original_copy = prior.step.copy()

        def solve(H, g, diagonal, **options):
            step, _, report = qp._linearly_constrained_shared_step(
                H, g, lower, upper, options["global_step_constraint_matrix"],
                options["global_step_constraint_lower"])
            return SimpleNamespace(step=step, report={"shared_linear_domain_step": report})

        def correction(*args):
            return dict(lower=tightened_b, report=dict(measured_outward_source_faces=True))

        try:
            with redirect_stdout(log):
                if scope == "initial":
                    returned_step, _, returned_KKT = qp._linearly_constrained_shared_step(
                        H, g, lower, upper, A, tightened_b)
                    result = SimpleNamespace(step=returned_step,
                                             report={"shared_linear_domain_step": returned_KKT})
                else:
                    result = joint._sequential_measured_shared_source_curvature(
                        np.zeros(2), prior, np.arange(2), dict(lower=original_b),
                        dict(global_step_constraint_matrix=A, global_step_constraint_lower=original_b),
                        H, g, np.ones(2), correction, solve)
        except Cancelled:
            assert mode == "cancel" and len(backend_calls) == 1
            return dict(mode=mode, scope=scope, root_cancellation_propagated=True,
                        complete_fallback_backend_calls=0)
        finally:
            clarabel.DefaultSolver = original_backend
        SOC = result.report.get("sequential_measured_shared_source_curvature_tightening", {})
        failure = SOC.get("optional_SOC_restricted_numerical_failure")
        if scope == "verified_SOC":
            assert result is prior and np.array_equal(result.step, original_copy)
            assert failure is not None and failure["mathematical_infeasibility_certified"] is False
            expected = {"status": "restricted_numerical_status",
                        "exception": "restricted_numerical_exception",
                        "KKT": "original_full_KKT_failure"}[mode]
            assert failure["restricted_failure_reason"] == expected
            assert len(backend_calls) == 1
            return dict(mode=mode, scope=scope, previous_verified_step_bitexact=True,
                        original_restricted_failure_reason=expected,
                        exact_original_failure_report=failure["original_restricted_failure_report"],
                        complete_fallback_backend_calls=0)
        assert failure is None and len(backend_calls) >= 2
        assert abs(result.step[0] - 0.5) < 1e-12
        assert result.report["shared_linear_domain_step"]["complete_original_nonnegative_IPM_dual_certificate_checked"]
        return dict(mode=mode, scope=scope, complete_fallback_retained=True,
                    accepted_corrected_step=[float(x) for x in result.step],
                    backend_call_count=len(backend_calls))

    for mode in ("status", "exception", "KKT"):
        checks.append(run_case(mode, "verified_SOC"))
        checks.append(run_case(mode, "initial"))
        checks.append(run_case(mode, "unverified_SOC"))
    checks.append(run_case("cancel", "verified_SOC"))
    logs = [json.loads(line) for line in log.getvalue().splitlines()]
    begin = [row for row in logs if row.get("stage") == "QP_backend_begin"]
    end = [row for row in logs if row.get("stage") == "QP_backend_end"]
    assert len(begin) == len(end)
    assert [row["QP_call"] for row in begin] == [row["QP_call"] for row in end]
    assert all("wall_time_ns" in row for row in begin + end)
    result = dict(schema="rinkspace.small-real-QP-optional-SOC-failure-scope-regression.v2",
                  installation=installation, checks=checks,
                  backend_phase_begin_end_logs_paired=True,
                  tested_backend_phases=sorted({row["QP_phase"] for row in begin}),
                  complete_source_graph_or_old_full_QP_baseline_loaded=False,
                  source_observations_or_original_KKT_thresholds_modified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
