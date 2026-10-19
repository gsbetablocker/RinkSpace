"""Verify a captured real restricted-QP ray without a graph or source rebuild."""
from pathlib import Path
import argparse
import hashlib
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
from scipy.linalg import solve_triangular
from calibration.solver_runtime.qp_infeasibility_certificate import verify_restricted_infeasibility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    begun = time.perf_counter()
    original_report = json.loads(args.capture.with_suffix(".json").read_text(encoding="utf8"))
    with args.capture.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != original_report["actual_original_QP_capture"]["sha256"]:
        raise ValueError("Captured actual QP bytes differ from the live receipt")
    with np.load(args.capture, allow_pickle=False) as saved:
        H, g, lower, upper, A, b, selected, z = [saved[name].copy() for name in
            ("matrix", "gradient", "lower", "upper", "constraint_matrix",
             "constraint_lower", "selected", "ray")]
        stored_scale = saved["scale"].copy()
        stored_row_scale = saved["row_scale"].copy()
        stored_tolerance = float(saved["tolerance"])
        stored_affine_rows = saved["affine_rows_selected"].copy()
        stored_affine_targets = saved["affine_targets_selected"].copy()
    n = len(g)
    finite_lower, finite_upper = np.flatnonzero(np.isfinite(lower)), np.flatnonzero(np.isfinite(upper))
    eye = np.eye(n)
    rows = np.vstack((A, eye[finite_lower], -eye[finite_upper]))
    targets = np.r_[b, lower[finite_lower], -upper[finite_upper]]
    objective_scale = max(1., float(np.min(np.diag(H))))
    coordinate_scale = np.sqrt(objective_scale) / np.sqrt(np.diag(H))
    scaled_H = H * coordinate_scale[:, None] * coordinate_scale[None, :] / objective_scale
    factor = np.linalg.cholesky(scaled_H)
    whitened = solve_triangular(factor, (rows * coordinate_scale[None, :]).T,
                               lower=True, check_finite=False).T
    scale = np.sqrt(np.einsum("ij,ij->i", whitened, whitened))
    row_scale = scale / np.sqrt(objective_scale)
    center = coordinate_scale * np.linalg.solve(scaled_H, -g * coordinate_scale / objective_scale)
    affine_rows = whitened / scale[:, None]
    affine_targets = (targets - rows @ center) / scale
    tolerance = 1e-11 * max(1., float(np.max(np.abs((targets - rows @ center) / row_scale))))
    exact_replay = dict(
        reconstructed_row_scale_equal=bool(np.array_equal(row_scale, stored_row_scale)),
        reconstructed_affine_row_scale_equal=bool(np.array_equal(scale, stored_scale)),
        reconstructed_original_tolerance_equal=bool(tolerance == stored_tolerance),
        reconstructed_selected_affine_rows_equal=bool(np.array_equal(affine_rows[selected], stored_affine_rows)),
        reconstructed_selected_affine_targets_equal=bool(np.array_equal(affine_targets[selected], stored_affine_targets)))
    if not all(exact_replay.values()):
        raise ValueError("Live/reconstructed reversible transform differs: " + str(exact_replay))
    settings = SimpleNamespace(tol_infeas_abs=original_report["existing_tol_infeas_abs"],
                               tol_infeas_rel=original_report["existing_tol_infeas_rel"])
    verified, _ = verify_restricted_infeasibility(
        SimpleNamespace(status=original_report["status"], z=z), selected,
        rows, targets, scale, affine_rows, affine_targets, row_scale, tolerance, settings)
    report = dict(schema="rinkspace.independent-real-restricted-QP-ray-replay.v1",
                  capture=dict(path=str(args.capture.resolve()), sha256=actual_sha),
                  original_variable_count=n, original_linear_constraint_count=len(b),
                  exact_reversible_transform_reconstruction=exact_replay,
                  numerical_certificate=verified,
                  source_graph_or_J_rebuilt=False, production_state_or_QP_modified=False,
                  elapsed_seconds=time.perf_counter() - begun)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not verified["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
