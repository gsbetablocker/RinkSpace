"""Keep solver source, event resources and native build products separate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT / "legacy"
PROGRAM_SOURCE_ROOT = SOURCE_ROOT / "snapshot" / "three_frame_refit" / "program_tracking"
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATA_ROOT = Path(
    "D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003"
)
DEFAULT_NATIVE_CACHE_ROOT = Path("D:/CodexData/RinkSpace_Data/native/solver_runtime")


def data_root() -> Path:
    return Path(os.environ.get("RINKSPACE_SOLVER_DATA_ROOT", str(DEFAULT_DATA_ROOT))).resolve()


def data_module_dir(source_file: str | Path) -> Path:
    """Resource directory corresponding to an owned legacy source module."""
    relative = Path(source_file).resolve().relative_to(SOURCE_ROOT)
    return data_root() / relative.parent


def native_build_directory(kernel: str, source: str | Path) -> Path:
    """Find an authenticated existing build by source SHA, without compiling.

    A source-addressed cache is preferred. The explicit legacy build locations
    let this checkout reuse the already verified binaries without copying them.
    If a build is absent, return its cache address so the caller reports its
    normal unavailable-backend diagnostic. No directories or files are created.
    """
    source = Path(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    native_root = Path(os.environ.get("RINKSPACE_SOLVER_NATIVE_CACHE_ROOT", str(DEFAULT_NATIVE_CACHE_ROOT)))
    cache = native_root / kernel / source_sha
    legacy = {
        "nearest_segment": Path("D:/CodexData/RinkSpace_program_tracking_tmp/solver_acceleration_native"),
        "motion": data_root() / "native_motion",
        "csr_scatter": data_root() / "native_scatter",
        "projection_guard": data_root() / "native_projection",
        "projection_value": data_root() / "native_projection_value",
        "motion_cuda": data_root() / "native_motion_cuda",
        "central_brown4": Path("D:/CodexData/RinkSpace_program_tracking_tmp/solver_acceleration/central_brown4_optin"),
    }
    defaults = {
        "nearest_segment": "nearest_segment.dll", "motion": "moving_center_motion.dll",
        "csr_scatter": "fixed_xyz_csr_scatter.dll", "projection_guard": "central_projection_guard.dll",
        "projection_value": "projection_value.dll", "motion_cuda": "moving_center_motion_cuda.dll",
        "central_brown4": "central_brown4.dll",
    }
    if kernel not in legacy:
        raise ValueError(f"Unknown solver native kernel: {kernel}")
    for directory in (cache, legacy[kernel]):
        try:
            manifest = json.loads((directory / "build.json").read_text(encoding="utf8"))
            binary_name = manifest.get("binary", manifest.get("binary_name", defaults[kernel]))
            binary = directory / binary_name
            if (manifest["source_sha256"] == source_sha and binary.is_file()
                    and manifest["dll_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()):
                return directory.resolve()
        except (OSError, ValueError, KeyError):
            continue
    # This kernel predates build.json; its reviewed wrapper pins both hashes.
    if kernel == "central_brown4" and source_sha == "ec467f9aa2a67e41b1dc6417e851f5d29b0313617fd755905ba008391a276473":
        binary = legacy[kernel] / defaults[kernel]
        if binary.is_file() and hashlib.sha256(binary.read_bytes()).hexdigest() == "9d169a82f3f6e6b4641b6ecd8310965bd5124c195943f986348f1460a844b6c2":
            return legacy[kernel]
    return cache.resolve()
