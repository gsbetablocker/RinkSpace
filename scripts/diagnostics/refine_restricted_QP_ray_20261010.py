"""Refine only a captured restricted ray; never solve a full QP/source graph."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import linprog
from calibration.solver_runtime.qp_infeasibility_certificate import verify_restricted_infeasibility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    begun = time.perf_counter()
    old_report = json.loads(args.capture.with_suffix(".json").read_text(encoding="utf8"))
    with np.load(args.capture, allow_pickle=False) as saved:
        A, b, lower, upper, selected, scale, row_scale, tolerance, N, d = [saved[name].copy() for name in
            ("constraint_matrix", "constraint_lower", "lower", "upper", "selected",
             "scale", "row_scale", "tolerance", "affine_rows_selected", "affine_targets_selected")]
    n = len(lower)
    eye = np.eye(n)
    rows = np.vstack((A, eye[np.isfinite(lower)], -eye[np.isfinite(upper)]))
    targets = np.r_[b, lower[np.isfinite(lower)], -upper[np.isfinite(upper)]]
    # Solve only a nonnegative certificate equation on original selected
    # rows.  This LP has no source/QP objective and changes no original row.
    equations = np.vstack(((rows[selected] / scale[selected, None]).T,
                           targets[selected] / scale[selected]))
    right = np.r_[np.zeros(n), 1.]
    equation_scale = np.max(np.abs(equations), axis=1)
    nonzero = equation_scale > 0
    equations = equations[nonzero] / equation_scale[nonzero, None]
    right = right[nonzero] / equation_scale[nonzero]
    lp_begin = time.perf_counter()
    solution = linprog(np.zeros(len(selected)), A_eq=equations, b_eq=right,
                       bounds=(0., None), method="highs",
                       options=dict(primal_feasibility_tolerance=1e-9,
                                    dual_feasibility_tolerance=1e-9))
    report = dict(schema="rinkspace.actual-restricted-ray-original-nullspace-refinement.v1",
                  capture=str(args.capture.resolve()), selected_row_count=len(selected),
                  certificate_LP_status=int(solution.status), certificate_LP_message=str(solution.message),
                  certificate_LP_seconds=time.perf_counter() - lp_begin,
                  full_QP_or_source_graph_evaluated=False, original_final_certificate_precision_unchanged=True)
    if solution.success:
        refined = np.maximum(solution.x, 0.)
        support = np.flatnonzero(refined > 0)
        # Only linear equality refinement.  Any negative provisional weight
        # is removed from this candidate certificate, never from the QP.
        for _ in range(3):
            Q = equations[:, support]
            correction = lstsq(Q, right - Q @ refined[support],
                               cond=None, lapack_driver="gelsd")[0]
            refined[support] += correction
            negative = support[refined[support] < 0.]
            refined[negative] = 0.
            support = np.setdiff1d(support, negative)
        # Reconstruct only selected affine rows in their original full-row
        # layout so the same original-unit final certificate gate is used.
        affine_rows = np.zeros((len(rows), n))
        affine_rows[selected] = N
        affine_targets = np.zeros(len(rows))
        affine_targets[selected] = d
        certificate, _ = verify_restricted_infeasibility(
            SimpleNamespace(status="PrimalInfeasible", z=refined), selected,
            rows, targets, scale, affine_rows, affine_targets, row_scale,
            float(tolerance), SimpleNamespace(tol_infeas_abs=old_report["existing_tol_infeas_abs"],
                                             tol_infeas_rel=old_report["existing_tol_infeas_rel"]))
        report["refined_numerical_certificate"] = certificate
        report["refined_support_count"] = int(np.count_nonzero(refined))
        np.savez(args.output.with_suffix(".npz"), selected=selected, refined_ray=refined)
    report["elapsed_seconds"] = time.perf_counter() - begun
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
