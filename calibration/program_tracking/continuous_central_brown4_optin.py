"""Explicit Brown4 nine-station central-domain backend for source projections.

Only the sampled optical-domain test uses the pinned strict C kernel. Source
rays still use physical_projection._rays (the original cv2 Rodrigues path),
and projection.project_central_valid still returns UV/J from project_fast and
checks the original 1e-5 pixel agreement. Brown14 and numerically ambiguous
Brown4 rays use the original safe_project. No solver state or input UV changes.
"""
from __future__ import annotations

from collections import Counter
import ctypes
import hashlib
import math
from pathlib import Path
from threading import Lock

import numpy as np

try:
    from . import projection as _projection  # also installs geometry paths
except ImportError:
    import projection as _projection
from physical_projection import (_AXIS_STATIONS, _EIGEN_FLOOR, _brown_map_jacobian,
                                 _camera, _rays, safe_project)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "central_brown4_axis_domain.c"
DLL = Path("D:/CodexData/RinkSpace_program_tracking_tmp/solver_acceleration/central_brown4_optin/central_brown4.dll")
SOURCE_SHA256 = "ec467f9aa2a67e41b1dc6417e851f5d29b0313617fd755905ba008391a276473"
DLL_SHA256 = "9d169a82f3f6e6b4641b6ecd8310965bd5124c195943f986348f1460a844b6c2"
AMBIGUOUS_EIGEN_BAND = 1e-7
EXTREME_NORMALIZED_ABS = 1000.
_PDOUBLE = ctypes.POINTER(ctypes.c_double)
_library = None
_stats = Counter()
_stats_lock = Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_kernel():
    global _library
    if _library is not None:
        return _library
    for path, expected in ((SOURCE, SOURCE_SHA256), (DLL, DLL_SHA256)):
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Central Brown4 opt-in dependency missing/changed: {path}")
    library = ctypes.CDLL(str(DLL))
    function = library.brown4_axis_minimum_eigenvalue
    function.argtypes = (_PDOUBLE, ctypes.c_int64, _PDOUBLE, _PDOUBLE)
    function.restype = ctypes.c_int
    _library = library
    return library


def central_mask_from_normalized(normalized_xy, brown4):
    """Return preliminary N-bool central mask and N-bool ambiguity mask.

    Normalized rays must come from the original physical_projection._rays.
    A caller must resolve every ``ambiguous`` ray with original safe_project
    before treating ``valid`` as a final domain verdict. Brown4 coefficient
    order is k1,k2,p1,p2. The denominator is identically one.
    """
    rays = np.ascontiguousarray(np.asarray(normalized_xy, np.float64))
    coefficients = np.ascontiguousarray(np.asarray(brown4, np.float64))
    if (rays.ndim != 2 or rays.shape[1] != 2 or coefficients.shape != (4,)
            or not np.isfinite(coefficients).all()):
        raise ValueError("Expected Nx2 normalized rays and finite Brown4 coefficients")
    minimum = np.empty(len(rays), np.float64)
    function = _verified_kernel().brown4_axis_minimum_eigenvalue
    status = function(rays.ctypes.data_as(_PDOUBLE), len(rays),
                      coefficients.ctypes.data_as(_PDOUBLE),
                      minimum.ctypes.data_as(_PDOUBLE))
    if status:
        raise RuntimeError(f"Central Brown4 strict C kernel failed with status {status}")
    finite_rays = np.isfinite(rays).all(axis=1)
    extreme = np.any(np.abs(rays) > EXTREME_NORMALIZED_ABS, axis=1)
    near_limit = np.isfinite(minimum) & (np.abs(minimum - _EIGEN_FLOOR)
                                         <= AMBIGUOUS_EIGEN_BAND)
    ambiguous = (~finite_rays | extreme | ~np.isfinite(minimum) | near_limit)
    valid = np.isfinite(minimum) & (minimum > _EIGEN_FLOOR) & finite_rays
    return valid, ambiguous


def _brown4_endpoint(normalized, coefficients):
    """Original Brown map's value, without its six unused derivative arrays.

    This is called only for finite, non-extreme Brown4 rays after the sampled
    domain test. Preserve the original multiply/add order (including its
    trailing zero terms); do not use reassociation or fused arithmetic.
    """
    x, y = np.asarray(normalized, np.float64).T
    k1, k2, p1, p2 = coefficients
    r2 = x*x + y*y
    radial = 1. + k1*r2 + k2*r2*r2 + 0.
    return np.column_stack((
        x*radial + 2.*p1*x*y + p2*(r2 + 2.*x*x) + 0. + 0.,
        y*radial + p1*(r2 + 2.*y*y) + 2.*p2*x*y + 0. + 0.))


def safe_project_compatible(camera25, xyz):
    """Drop-in safe_project backend with explicit Brown14/threshold fallback."""
    camera, distortion = _camera(camera25)
    points = np.asarray(xyz, np.float64).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("World coordinates must be finite")
    if (not len(points) or not np.any(np.abs(camera[11:23]) > 1e-14)
            or np.any(camera[15:23] != 0.)):
        with _stats_lock:
            _stats["original_general_calls"] += 1
            if np.any(camera[15:23] != 0.):
                _stats["brown14_fallback_calls"] += 1
                _stats["brown14_fallback_points"] += len(points)
        return safe_project(camera, points)

    uv = np.full((len(points), 2), np.nan)
    normalized, depth, forward = _rays(camera, points)
    valid = np.zeros(len(points), bool)
    indices = np.flatnonzero(forward)
    if not len(indices):
        with _stats_lock:
            _stats["brown4_calls"] += 1
            _stats["brown4_points"] += len(points)
        return uv, depth, valid

    rays = normalized[indices]
    preliminary, ambiguous = central_mask_from_normalized(rays, camera[11:15])
    fast_indices = indices[~ambiguous & preliminary]
    if len(fast_indices):
        # This is the original endpoint map (one station), not an approximation
        # of the returned image: project_central_valid keeps project_fast UV/J.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mapped = _brown4_endpoint(normalized[fast_indices], camera[11:15])
            focal_x, focal_y = math.exp(camera[6]), math.exp(camera[7])
            projected = np.column_stack((
                focal_x * mapped[:, 0] + camera[10] * mapped[:, 1] + camera[8],
                focal_y * mapped[:, 1] + camera[9]))
        finite = np.isfinite(projected).all(axis=1)
        uv[fast_indices[finite]] = projected[finite]
        valid[fast_indices[finite]] = True

    if np.any(ambiguous):
        fallback_indices = indices[ambiguous]
        old_uv, _, old_valid = safe_project(camera, points[fallback_indices])
        uv[fallback_indices] = old_uv
        valid[fallback_indices] = old_valid
    with _stats_lock:
        _stats["brown4_calls"] += 1
        _stats["brown4_points"] += len(points)
        _stats["brown4_forward_points"] += len(indices)
        _stats["brown4_threshold_fallback_points"] += int(np.sum(ambiguous))
    return uv, depth, valid


def backend_stats() -> dict:
    with _stats_lock:
        return dict(_stats)


def enable_projection_backend() -> dict:
    """Explicitly connect the reviewed projection hook; default stays legacy."""
    _verified_kernel()
    current = getattr(_projection, "_central_projection_backend", None)
    if current is None:
        raise RuntimeError("projection.py has no explicit central-backend hook")
    if current not in (safe_project, safe_project_compatible):
        raise RuntimeError("Another central projection backend is already installed")
    _projection._central_projection_backend = safe_project_compatible
    return {
        "schema": "rinkspace.central-brown4-domain-optin.v1",
        "enabled": True, "scope": "Brown4 central-validity only; project_fast UV/J unchanged",
        "nine_axis_stations": list(map(float, _AXIS_STATIONS)),
        "depth_floor": .05, "eigen_floor": _EIGEN_FLOOR,
        "ambiguity_eigen_band": AMBIGUOUS_EIGEN_BAND,
        "extreme_normalized_abs_fallback": EXTREME_NORMALIZED_ABS,
        "brown14_fallback": "original safe_project for any nonzero k3/rational/thin-prism term",
        "threshold_fallback": "original safe_project for near-threshold, nonfinite, or extreme rays",
        "no_silent_kernel_fallback": True,
        "source_path": str(SOURCE), "source_sha256": SOURCE_SHA256,
        "dll_path": str(DLL), "dll_sha256": DLL_SHA256,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": _sha256(Path(__file__)),
        "original_project_fast_uv_j_retained": True,
        "original_uv_agreement_tolerance_px": 1e-5,
        "stats_at_enable": backend_stats(),
    }
