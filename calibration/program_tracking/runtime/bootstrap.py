"""Reuse existing local CV/SciPy installations without changing their environments."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
MAIN_SITE = ROOT / ".venv" / "Lib" / "site-packages"
GPU_SITE = ROOT.parent / "MotionTrail_Codex_Starter_v1" / ".venv-gpu" / "Lib" / "site-packages"


def add_geometry_dependencies() -> tuple[str, ...]:
    """Expose the project's SciPy and the existing GPU OpenCV site when needed.

    Entries are appended only; the active environment's own NumPy, Torch, and
    OpenCV retain precedence.  In the GPU environment its site already comes
    before the appended main-project SciPy site.  In the conda GTSAM environment
    the main-project OpenCV/SciPy work with its own NumPy 1.26.4.
    """
    added: list[str] = []
    for candidate in (MAIN_SITE, GPU_SITE):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))
            added.append(str(candidate))
    return tuple(added)
