"""Opt-in strict first-board-hit acceleration for existing Brown board factors.

The C DLL replaces only ReferenceBoardPair's ray x panel scan. Original-image
UV, roundtrip Brown rays, global-first/verified-tie policy, finite/ice/print
reasons, target occlusion, seam guards, and 15D derivatives remain inherited.
Source-hit reuse is scoped to one Jacobian callback. Default solver imports
and objects are unchanged; a caller must explicitly call ``upgrade_pairs``.

The pinned DLL is an already-built round5 study artifact. Missing or changed
binary/source or changed inherited callback code is a hard opt-in error: there
is no silent fallback to another objective or implementation.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from threading import local

import cv2
import numpy as np

try:
    from .continuous_brown_board_pair import ContinuousBrownBoardPair
    from .reference_board_pair import BoardHits
except ImportError:
    from continuous_brown_board_pair import ContinuousBrownBoardPair
    from reference_board_pair import BoardHits


STUDY = Path("D:/CodexData/RinkSpace_program_tracking_tmp/solver_acceleration/round5_continuous/derivatives")
DLL = STUDY / "board_first_hit.dll"
SOURCE = STUDY / "board_first_hit.c"
DLL_SHA256 = "0d8c156232fe8c662641e449e9ad39bed67e69d461bbd6265f5e397d54da18ac"
SOURCE_SHA256 = "b715a1de68355b77b373b3ad7835acefc6221e3604837f2d825d33b3845a93e8"
INHERITED_SHA256 = {
    "continuous_brown_board_pair.py": "2a9b52e34df9813fbf99f7c002cf85bdabcd46670bf64ef078101e5a3a5adb0d",
    "native_board_pair.py": "ba95e259756c4a8d39282d829477633a2a287c708e609044913a740d9dc47759",
    "reference_board_pair.py": "1cfa3a2071a0be73f4c00b7e4416b124cdd65279b05eb5928fde0dc3472ccfbf",
}

_PDOUBLE = ctypes.POINTER(ctypes.c_double)
_PUINT8 = ctypes.POINTER(ctypes.c_uint8)
_PINT32 = ctypes.POINTER(ctypes.c_int32)
_library = None


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _verified_library():
    global _library
    if _library is not None:
        return _library
    paths = {DLL: DLL_SHA256, SOURCE: SOURCE_SHA256}
    paths.update({Path(__file__).with_name(name): expected
                  for name, expected in INHERITED_SHA256.items()})
    for path, expected in paths.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Strict board-hit opt-in dependency missing/changed: {path}")
    library = ctypes.CDLL(str(DLL))
    library.board_first_hit.argtypes = (
        _PDOUBLE, _PDOUBLE, _PDOUBLE, _PDOUBLE, _PUINT8,
        ctypes.c_int64, _PDOUBLE, _PDOUBLE, ctypes.c_int64,
        _PDOUBLE, _PINT32, _PDOUBLE, _PUINT8)
    library.board_first_hit.restype = ctypes.c_int
    _library = library
    return library


def _pointer(array: np.ndarray, kind):
    return array.ctypes.data_as(kind)


class StrictFirstHitBrownBoardPair(ContinuousBrownBoardPair):
    """Only the global ray-panel scan and duplicate source hit are accelerated."""

    def _prepare_optin(self) -> None:
        _verified_library()
        self._strict_starts = np.ascontiguousarray(self._starts, np.float64)
        self._strict_edges = np.ascontiguousarray(self._edges, np.float64)
        self._strict_bottom_z = np.ascontiguousarray(self.bottom[:, 2], np.float64)
        self._strict_top_z = np.ascontiguousarray(self.top[:, 2], np.float64)
        self._strict_allowed = np.ascontiguousarray(self._allowed, np.uint8)
        count = len(self.bottom)
        if (self._strict_starts.shape != (count, 2)
                or self._strict_edges.shape != (count, 2)
                or self._strict_bottom_z.shape != (count,)
                or self._strict_top_z.shape != (count,)
                or self._strict_allowed.shape != (count,)):
            raise ValueError("Opt-in board arrays differ from original panel geometry")
        self._source_hit_local = local()

    def _hits_from_rays(self, centre: np.ndarray, rays: np.ndarray) -> BoardHits:
        centre = np.ascontiguousarray(centre, np.float64)
        rays = np.ascontiguousarray(rays, np.float64)
        if centre.shape != (3,) or rays.ndim != 2 or rays.shape[1] != 3:
            raise ValueError("Strict board-hit inputs require one C and Nx3 rays")
        count = len(rays)
        first = np.empty(count, np.float64)
        panel = np.empty(count, np.int32)
        fraction = np.empty(count, np.float64)
        verified = np.empty(count, np.uint8)
        status = _verified_library().board_first_hit(
            _pointer(self._strict_starts, _PDOUBLE),
            _pointer(self._strict_edges, _PDOUBLE),
            _pointer(self._strict_bottom_z, _PDOUBLE),
            _pointer(self._strict_top_z, _PDOUBLE),
            _pointer(self._strict_allowed, _PUINT8),
            len(self._strict_allowed),
            _pointer(centre, _PDOUBLE), _pointer(rays, _PDOUBLE), count,
            _pointer(first, _PDOUBLE), _pointer(panel, _PINT32),
            _pointer(fraction, _PDOUBLE), _pointer(verified, _PUINT8))
        if status:
            raise RuntimeError(f"Strict board-first-hit kernel failed with status {status}")

        xyz = centre[None, :] + first[:, None] * rays
        ray_finite = np.isfinite(rays).all(axis=1)
        first_finite = np.isfinite(first)
        allowed_first = verified.astype(bool)
        invalid_ray = ~ray_finite
        missing_hit = ray_finite & ~first_finite
        outside_arc = ray_finite & first_finite & ~allowed_first
        possible_ice = ray_finite & first_finite & allowed_first
        reason = ["ok"] * count
        for index in np.flatnonzero(invalid_ray):
            reason[int(index)] = "invalid_brown_ray"
        for index in np.flatnonzero(missing_hit):
            reason[int(index)] = "no_board_hit"
        for index in np.flatnonzero(outside_arc):
            reason[int(index)] = "first_hit_outside_verified_print_arc"

        downward = rays[:, 2] < -1e-10
        ground_t = np.full(count, np.inf)
        ground_t[downward] = -centre[2] / rays[downward, 2]
        ground_candidate = (possible_ice & (ground_t > 0.)
                            & (ground_t < first - 1e-7))
        occluded = np.zeros(count, bool)
        for index in np.flatnonzero(ground_candidate):
            ground_xy = centre[:2] + ground_t[index] * rays[index, :2]
            if cv2.pointPolygonTest(self._polygon,
                                    tuple(map(float, ground_xy)), False) >= 0:
                occluded[index] = True
                reason[int(index)] = "ice_occludes_board_hit"
        valid = possible_ice & ~occluded
        xyz[~valid] = np.nan
        panel[~valid] = -1
        fraction[~valid] = np.nan
        return BoardHits(xyz, panel, fraction, valid, tuple(reason))

    def source_hits(self, source_state, source_pixels=None):
        cache = getattr(self._source_hit_local, "cache", None)
        if cache is None or (source_pixels is not None and source_pixels is not self.xy0):
            return super().source_hits(source_state, source_pixels)
        state = np.asarray(source_state)
        key = (state.dtype.str, state.shape, state.tobytes())
        hit = cache.get(key)
        if hit is not None:
            return hit
        hit = super().source_hits(source_state, source_pixels)
        cache[key] = hit
        return hit

    def _raw_jacobians(self, states):
        previous = getattr(self._source_hit_local, "cache", None)
        self._source_hit_local.cache = {}
        try:
            return super()._raw_jacobians(states)
        finally:
            self._source_hit_local.cache = previous


def upgrade_pair(pair: ContinuousBrownBoardPair) -> StrictFirstHitBrownBoardPair:
    """Copy the exact prepared factor data; never reselect or synthesize UV."""
    if type(pair) is not ContinuousBrownBoardPair:
        raise TypeError("Opt-in requires an exact prepared ContinuousBrownBoardPair")
    upgraded = StrictFirstHitBrownBoardPair.__new__(StrictFirstHitBrownBoardPair)
    upgraded.__dict__ = pair.__dict__.copy()
    upgraded._prepare_optin()
    if upgraded.xy0 is not pair.xy0 or upgraded.xy1 is not pair.xy1:
        raise RuntimeError("Opt-in changed original image pixel arrays")
    return upgraded


def upgrade_pairs(pairs: list[tuple[int, int, ContinuousBrownBoardPair]]):
    """Explicit driver hook after source qualification and Brown pair conversion."""
    converted = [(int(a), int(b), upgrade_pair(pair)) for a, b, pair in pairs]
    metadata = {
        "schema": "rinkspace.strict-board-first-hit-optin.v1",
        "enabled": True, "pair_count": len(converted),
        "source_uv_pair_count": sum(len(pair.xy0) for _, _, pair in converted),
        "scope": "dynamic Brown board pair first-hit scan and Jacobian-local source-hit reuse only",
        "invalid_trial_policy": "all original BoardIntersectionError reasons retained; native status hard error; no fallback/drop",
        "native_dll_path": str(DLL), "native_dll_sha256": DLL_SHA256,
        "native_source_path": str(SOURCE), "native_source_sha256": SOURCE_SHA256,
        "inherited_callback_sha256": dict(INHERITED_SHA256),
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": _sha256(Path(__file__)),
        "source_uv_changed": False,
    }
    return converted, metadata
