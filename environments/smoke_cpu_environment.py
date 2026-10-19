"""Small CPU dependency/CLI check; never opens event data or loads solver backends."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    expected = args.environment.resolve()
    if Path(sys.prefix).resolve() != expected:
        raise SystemExit("Run this check with the explicitly selected independent venv.")
    if not args.out.resolve().is_relative_to(expected.parent):
        raise SystemExit("The check report must stay in the independent Data directory.")
    forbidden = ("/rinkspace/.venv/", "/motiontrail_codex_starter_v1/", "/fsrotation/", "/fspathing/")
    for path in sys.path:
        normalized = str(path).replace("\\", "/").lower().rstrip("/") + "/"
        if any(marker in normalized for marker in forbidden):
            raise SystemExit(f"Unexpected existing runtime on sys.path: {path}")

    # Limit BLAS/OpenMP before binary imports; no parallel project workload is run.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import av
    import clarabel
    import cv2
    import numpy as np
    from PIL import Image
    import quadprog
    from scipy import sparse
    from scipy.spatial.transform import Rotation
    import yaml

    np.testing.assert_array_equal(np.array([[1.0, 2.0]]) @ np.array([[3.0], [4.0]]), [[11.0]])
    np.testing.assert_array_equal(sparse.eye(3, format="csr") @ np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(Rotation.from_rotvec([0.0, 0.0, 0.0]).as_matrix(), np.eye(3), atol=0.0)
    points = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float64)
    np.testing.assert_array_equal(cv2.perspectiveTransform(points, np.eye(3)), points)
    image = Image.new("RGB", (2, 2), (12, 34, 56))
    assert image.getpixel((1, 1)) == (12, 34, 56)
    assert yaml.safe_load("status: smoke\n") == {"status": "smoke"}
    assert callable(quadprog.solve_qp)
    assert hasattr(clarabel, "DefaultSolver")
    assert callable(av.open)

    checks = ["binary_library_imports", "numpy_tiny_matrix_product", "scipy_sparse_product",
              "scipy_identity_rotation", "opencv_identity_perspective", "pillow_in_memory_pixel",
              "yaml_small_mapping", "quadprog_and_clarabel_api_presence"]
    cli_results = {}
    for command in (("--help",), ("doctor",)):
        result = subprocess.run([sys.executable, "-B", "-m", "rinkspace", *command],
                                check=True, capture_output=True, text=True, timeout=20)
        cli_results[command[0]] = json.loads(result.stdout) if command[0] == "doctor" else "passed"
        checks.append("cli_" + command[0].lstrip("-"))
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()}
    report = {
        "schema": "rinkspace.cpu-environment-smoke.v1",
        "status": "passed",
        "python": sys.version,
        "executable": sys.executable,
        "prefix": str(expected),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "checks": checks,
        "cli": cli_results,
        "scope": "Dependency binary imports, tiny synthetic mathematics and CLI help/doctor only.",
        "not_validated": ["event fitting", "native kernel loading", "source video handling",
                          "geometry accuracy", "GPU", "GTSAM", "body models"],
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CPU dependency/CLI smoke passed: {args.out}")


if __name__ == "__main__":
    main()
