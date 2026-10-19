"""Portable launcher; run --help. Does not require an editable installation."""
from pathlib import Path
import os
import sys

# A worker handles independent frames. Avoid nested BLAS pools in that worker.
# Operators may explicitly override these environment settings before launch.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from rinkfit_start.cli import main
    raise SystemExit(main())

