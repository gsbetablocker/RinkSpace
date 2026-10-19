"""Independent physical-camera adapter to the project's camera25 interface.

This module does not fit cameras or change scene/state files.  Each calibration
belongs to one physical installation and one stable output mode.  Manufacturer
capabilities are not calibration values: zero model coefficients and disabled
effective-image offsets do not imply that field SteadyShot/extender were off.

Column-vector convention: Xc = R @ (X - C), C = T + R.T @ (lever + axial*ez).
Lengths are scene units; an optional declared working scale is not a measured
metric calibration.  Brown coordinates are camera
rays Xc/Zc, Yc/Zc, with the OpenCV 14-component order.  The final camera25 is
[C3, Rodrigues3, log(fx), log(fy), cx, cy, skew_px, Brown14].

Nonzero lever/axial generally moves C during rotation/zoom: the existing pure
fixed-centre relative_kernel is NOT valid for that motion.  Use explicit world
XYZ factors via project_xyz, or a separately typed surface/depth intersection.
Background rays must not silently become ice-plane points.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import cv2
import numpy as np

BROWN14_ORDER = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
                 "s1", "s2", "s3", "s4", "tauX", "tauY")


class UnknownCalibrationError(ValueError):
    """A required identity or calibration fact was explicitly missing/unknown."""


class PhysicalProjectionDomainError(ValueError):
    def __init__(self, invalid_indices):
        self.invalid_indices = np.asarray(invalid_indices, dtype=int)
        super().__init__(f"World points leave the central optical domain at indices {self.invalid_indices.tolist()}")


def _finite_tuple(value, length, label):
    result = tuple(float(v) for v in value)
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} must have {length} finite values")
    return result


def _size(value, label):
    result = tuple(value)
    if len(result) != 2 or any(isinstance(v, (bool, np.bool_)) or
                               not isinstance(v, (int, np.integer)) or v <= 0
                               for v in result):
        raise ValueError(f"{label} must have two positive integer dimensions")
    return tuple(int(v) for v in result)


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise UnknownCalibrationError(f"{label} must be an explicit integer PTS")
    return int(value)


def _coefficients(value, label):
    result = tuple(float(v) for v in value)
    if not 1 <= len(result) <= 3 or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} must be a finite constant/linear/quadratic curve")
    return result


def _polynomial(coefficients, x):
    # Ascending coefficients; no undocumented zoom encoder -> mm conversion.
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * x + coefficient
    if not math.isfinite(result):
        raise ValueError("Calibration curve evaluation is non-finite")
    return result


def _polynomial_derivative(coefficients, x):
    if len(coefficients) == 1:
        return 0.0
    return _polynomial(tuple(i*c for i, c in enumerate(coefficients) if i), x)


@dataclass(frozen=True)
class OutputMode:
    """Known source-domain A, or an unknown chain calibrated in final pixels.

    In source domain q=log(f_source_geometric_px/source_width); source fx/fy
    are scaled once by A.  In final_effective domain q refers to final pixels
    and cannot be interpreted as a measured optical zoom state.  Unknown
    non-unit A is rejected rather than optimized jointly with effective f.
    """

    mode_id: str
    domain: str = "final_effective"
    domain_wh: tuple[int, int] = (1920, 1080)
    final_wh: tuple[int, int] = (1920, 1080)
    source_to_final: tuple[tuple[float, float, float], ...] | None = None
    chain_provenance: str = "unknown"

    def __post_init__(self):
        if not isinstance(self.mode_id, str) or not self.mode_id:
            raise ValueError("Output mode needs a nonempty identity")
        object.__setattr__(self, "domain_wh", _size(self.domain_wh, "domain_wh"))
        object.__setattr__(self, "final_wh", _size(self.final_wh, "final_wh"))
        if self.domain not in ("source", "final_effective"):
            raise ValueError("Output domain must be source or final_effective")
        if self.domain == "final_effective":
            if self.source_to_final is not None or self.domain_wh != self.final_wh:
                raise ValueError("Unknown output chain must use final pixels with no extra A")
        else:
            if self.source_to_final is None or not self.chain_provenance or self.chain_provenance == "unknown":
                raise UnknownCalibrationError("Source-domain A requires explicit transform provenance")
            matrix = np.asarray(self.source_to_final, dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("A must be a finite 3x3 matrix")
            if (not np.array_equal(matrix[2], [0.0, 0.0, 1.0]) or
                    matrix[0, 1] != 0.0 or matrix[1, 0] != 0.0 or
                    matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0):
                raise ValueError("A supports known positive axis-aligned crop/resize/padding only")
            object.__setattr__(self, "source_to_final", tuple(tuple(float(v) for v in row) for row in matrix))

    @classmethod
    def unknown_final(cls, final_wh=(1920, 1080), *, mode_id="unknown_final_output"):
        return cls(mode_id=mode_id, domain_wh=final_wh, final_wh=final_wh)

    @classmethod
    def known_crop_resize(cls, *, mode_id, source_wh, final_wh, sx, sy,
                          crop_xy=(0.0, 0.0), padding_xy=(0.0, 0.0), provenance):
        x0, y0 = _finite_tuple(crop_xy, 2, "crop_xy")
        bx, by = _finite_tuple(padding_xy, 2, "padding_xy")
        sx, sy = _finite_tuple((sx, sy), 2, "scale")
        matrix = ((sx, 0.0, -sx*x0+bx),
                  (0.0, sy, -sy*y0+by), (0.0, 0.0, 1.0))
        return cls(mode_id, "source", source_wh, final_wh, matrix, provenance)

    @property
    def A(self):
        return np.eye(3) if self.source_to_final is None else np.asarray(self.source_to_final, float)

    @property
    def zoom_interpretation(self):
        if self.domain == "final_effective":
            return "conditional_final_effective_proxy_not_identified_optical_zoom"
        return "source_pixel_focal_proxy_not_measured_mm_or_encoder_zoom"


@dataclass(frozen=True)
class ViewState:
    """Four camera-state scalars: Rodrigues R3 and one domain-relative logf.

    effective_pp_offset_px is an optional explicitly enabled image disturbance
    in FINAL pixels.  It is not attributed to OIS, and is not a free crop too.
    """

    rvec_world_to_camera: tuple[float, float, float]
    log_f_ratio: float
    effective_pp_offset_px: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "rvec_world_to_camera", _finite_tuple(self.rvec_world_to_camera, 3, "rvec"))
        object.__setattr__(self, "effective_pp_offset_px", _finite_tuple(self.effective_pp_offset_px, 2, "PP offset"))
        if not math.isfinite(float(self.log_f_ratio)):
            raise ValueError("log_f_ratio must be finite")
        object.__setattr__(self, "log_f_ratio", float(self.log_f_ratio))

    @classmethod
    def from_state4(cls, values, *, effective_pp_offset_px=(0.0, 0.0)):
        values = _finite_tuple(values, 4, "view state")
        return cls(values[:3], values[3], effective_pp_offset_px)

    def state4(self):
        return np.array((*self.rvec_world_to_camera, self.log_f_ratio), dtype=float)


def _unknown_field_settings():
    return {name: "unknown" for name in ("steadyshot", "digital_extender", "focus",
                                         "recording_format", "camera_output_format")}


@dataclass(frozen=True)
class SharedCalibration:
    """One independent physical installation in one declared output mode.

    PP curves are domain pixel fractions (cx/W, cy/H), evaluated at q-q_ref.
    k1/k2 use normalized ray coordinates; p1/p2 are shared constants.  All
    other Brown14 entries are explicitly zero.  axial is in scene units and
    adds to lever.z; its constant coefficient MUST be zero to avoid duplicating
    that lever component.  A source/final mode change requires a reviewed
    calibration-domain conversion, not silent reuse of a fitted zoom curve.
    """

    installation_id: str
    pivot_xyz: tuple[float, float, float]
    lever_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    output_mode: OutputMode = field(default_factory=OutputMode.unknown_final)
    q_reference: float = 0.0
    aspect_fx_over_fy: float = 1.0
    principal_x_fraction: tuple[float, ...] = (0.5,)
    principal_y_fraction: tuple[float, ...] = (0.5,)
    k1: tuple[float, ...] = (0.0,)
    k2: tuple[float, ...] = (0.0,)
    tangential_p1_p2: tuple[float, float] = (0.0, 0.0)
    axial: tuple[float, ...] = (0.0,)
    allow_effective_pp_offset: bool = False
    field_settings: Mapping[str, object] = field(default_factory=_unknown_field_settings)

    def __post_init__(self):
        if not isinstance(self.installation_id, str) or not self.installation_id:
            raise ValueError("Calibration needs a physical installation identity")
        if not isinstance(self.output_mode, OutputMode):
            raise ValueError("Calibration needs an OutputMode")
        for name in ("pivot_xyz", "lever_xyz"):
            object.__setattr__(self, name, _finite_tuple(getattr(self, name), 3, name))
        for name in ("principal_x_fraction", "principal_y_fraction", "k1", "k2", "axial"):
            object.__setattr__(self, name, _coefficients(getattr(self, name), name))
        if self.axial[0] != 0.0:
            raise ValueError("axial constant must be zero; use lever.z for the reference offset")
        object.__setattr__(self, "tangential_p1_p2", _finite_tuple(self.tangential_p1_p2, 2, "p1/p2"))
        object.__setattr__(self, "q_reference", float(self.q_reference))
        object.__setattr__(self, "aspect_fx_over_fy", float(self.aspect_fx_over_fy))
        if not math.isfinite(self.q_reference) or not math.isfinite(self.aspect_fx_over_fy) or self.aspect_fx_over_fy <= 0:
            raise ValueError("q_reference must be finite and aspect must be positive finite")
        if not isinstance(self.allow_effective_pp_offset, bool):
            raise ValueError("allow_effective_pp_offset must be a boolean modelling choice")
        settings = _unknown_field_settings()
        settings.update(dict(self.field_settings))
        object.__setattr__(self, "field_settings", MappingProxyType(settings))

    def camera25(self, view: ViewState):
        if not isinstance(view, ViewState):
            raise ValueError("Camera input must be an explicit ViewState")
        if not self.allow_effective_pp_offset and any(v != 0.0 for v in view.effective_pp_offset_px):
            raise ValueError("Effective PP offset is not enabled for this calibration")
        mode = self.output_mode
        q = view.log_f_ratio - self.q_reference
        try:
            f_domain = mode.domain_wh[0] * math.exp(view.log_f_ratio)
        except OverflowError as error:
            raise ValueError("Domain focal length overflows") from error
        if not math.isfinite(f_domain) or f_domain <= 0:
            raise ValueError("Domain focal length must be positive finite")
        ratio_root = math.sqrt(self.aspect_fx_over_fy)
        k_domain = np.array([[f_domain*ratio_root, 0.0,
                              mode.domain_wh[0]*_polynomial(self.principal_x_fraction, q)],
                             [0.0, f_domain/ratio_root,
                              mode.domain_wh[1]*_polynomial(self.principal_y_fraction, q)],
                             [0.0, 0.0, 1.0]])
        # q is SOURCE f only for a known A, FINAL effective f otherwise.  A is
        # applied exactly once here; it is never another free magnification.
        k_final = mode.A @ k_domain
        k_final[:2, 2] += np.asarray(view.effective_pp_offset_px)
        rotation = cv2.Rodrigues(np.asarray(view.rvec_world_to_camera))[0]
        lever = np.asarray(self.lever_xyz).copy()
        lever[2] += _polynomial(self.axial, q)
        centre = np.asarray(self.pivot_xyz) + rotation.T @ lever
        distortion = np.zeros(14, dtype=float)
        distortion[:4] = (_polynomial(self.k1, q), _polynomial(self.k2, q),
                          *self.tangential_p1_p2)
        result = np.r_[centre, view.rvec_world_to_camera,
                       np.log(k_final[0, 0]), np.log(k_final[1, 1]),
                       k_final[0, 2], k_final[1, 2], k_final[0, 1], distortion]
        if result.shape != (25,) or not np.isfinite(result).all():
            raise ValueError("Final camera25 is not finite")
        return result

    def view_from_camera25(self, camera):
        """Numerical R/f warm start only; does not claim to reproduce old K/D/C.

        For known A, remove A BEFORE deriving the domain geometric focal. This
        prevents re-applying an extender/crop to a final effective warm start.
        Existing independent fx/fy are summarized geometrically, and existing
        per-view principal, centre and Brown remain outside this initialization.
        """
        values = np.asarray(camera, dtype=float)
        if values.shape != (25,) or not np.isfinite(values).all():
            raise ValueError("Warm start must contain 25 finite camera values")
        try:
            fx, fy = math.exp(values[6]), math.exp(values[7])
        except OverflowError as error:
            raise ValueError("Warm-start focal length overflows") from error
        source_fx = fx / self.output_mode.A[0, 0]
        source_fy = fy / self.output_mode.A[1, 1]
        q = 0.5*(math.log(source_fx)+math.log(source_fy)) - math.log(self.output_mode.domain_wh[0])
        return ViewState(tuple(values[3:6]), q)

    def shared_state_names(self):
        """Ordering of conditioned installation parameters, with fixed aspect.

        T, lever and axial use scene_units/100 purely for solver conditioning.
        Curve degrees are explicit in the calibration, not added by this API.
        """
        return (tuple(f"pivot_{axis}_scene_div100" for axis in "xyz") +
                tuple(f"lever_{axis}_scene_div100" for axis in "xyz") +
                tuple(f"axial_degree_{i}_scene_div100" for i in range(1, len(self.axial))) +
                tuple(f"principal_x_fraction_degree_{i}" for i in range(len(self.principal_x_fraction))) +
                tuple(f"principal_y_fraction_degree_{i}" for i in range(len(self.principal_y_fraction))) +
                tuple(f"k1_degree_{i}" for i in range(len(self.k1))) +
                tuple(f"k2_degree_{i}" for i in range(len(self.k2))) + ("p1", "p2"))

    def shared_state(self):
        return np.r_[np.asarray(self.pivot_xyz)/100.0,
                     np.asarray(self.lever_xyz)/100.0, np.asarray(self.axial[1:])/100.0,
                     self.principal_x_fraction, self.principal_y_fraction,
                     self.k1, self.k2, self.tangential_p1_p2]

    def with_shared_state(self, state):
        values = _finite_tuple(state, len(self.shared_state_names()), "shared state")
        offset = 6
        updated = {"pivot_xyz": tuple(100*v for v in values[:3]),
                   "lever_xyz": tuple(100*v for v in values[3:6])}
        count = len(self.axial)-1
        updated["axial"] = (0.0, *(100*v for v in values[offset:offset+count]))
        offset += count
        for name in ("principal_x_fraction", "principal_y_fraction", "k1", "k2"):
            count = len(getattr(self, name))
            updated[name] = values[offset:offset+count]
            offset += count
        updated["tangential_p1_p2"] = values[offset:offset+2]
        return replace(self, **updated)

    def camera25_and_jacobians(self, view):
        """Return (camera25, d_camera25/d_shared_state, d_camera25/d_state4).

        Shared Jacobian columns use shared_state_names(), including scene/100
        conditioning.  View columns are Rodrigues3 and domain log_f_ratio.
        These are mapping derivatives, not projection/domain-validity guards;
        callers must use the project's central-domain projection checks.
        """
        camera = self.camera25(view)
        x = view.log_f_ratio - self.q_reference
        mode = self.output_mode
        rotation, drot = cv2.Rodrigues(np.asarray(view.rvec_world_to_camera))
        rotations_derivative = drot.reshape(3, 3, 3)
        lever = np.asarray(self.lever_xyz).copy()
        lever[2] += _polynomial(self.axial, x)
        shared = np.zeros((25, len(self.shared_state_names())))
        shared[:3, :3] = np.eye(3)*100.0
        shared[:3, 3:6] = rotation.T*100.0
        offset = 6
        for degree in range(1, len(self.axial)):
            shared[:3, offset] = rotation.T[:, 2]*100.0*x**degree
            offset += 1
        for name, row, scale in (("principal_x_fraction", 8, mode.A[0, 0]*mode.domain_wh[0]),
                                  ("principal_y_fraction", 9, mode.A[1, 1]*mode.domain_wh[1]),
                                  ("k1", 11, 1.0), ("k2", 12, 1.0)):
            for degree in range(len(getattr(self, name))):
                shared[row, offset] = scale*x**degree
                offset += 1
        shared[13, offset] = 1.0
        shared[14, offset+1] = 1.0
        per_view = np.zeros((25, 4))
        for i in range(3):
            per_view[:3, i] = rotations_derivative[i].T @ lever
        per_view[3:6, :3] = np.eye(3)
        per_view[:3, 3] = rotation.T[:, 2]*_polynomial_derivative(self.axial, x)
        per_view[6:8, 3] = 1.0
        per_view[8, 3] = mode.A[0, 0]*mode.domain_wh[0]*_polynomial_derivative(self.principal_x_fraction, x)
        per_view[9, 3] = mode.A[1, 1]*mode.domain_wh[1]*_polynomial_derivative(self.principal_y_fraction, x)
        per_view[11, 3] = _polynomial_derivative(self.k1, x)
        per_view[12, 3] = _polynomial_derivative(self.k2, x)
        return camera, shared, per_view

    def project_xyz(self, view, xyz, *, with_jacobian=False, require_valid=True):
        """Project explicitly identified 3D points through existing domain guards.

        Returns (uv, depth, valid), or (uv, depth, J_shared, J_view, valid).
        Pixel Jacobians have 2*N rows with alternating u/v and shared_state /
        state4 columns respectively.  This is a world-point factor interface,
        not a fixed-centre ray transfer; point identity, surface type and
        source provenance remain the caller's inputs. Brown derivatives are
        included rather than accidentally using only the old 11-column J.
        """
        try:
            from .projection import project_central_valid
        except ImportError:
            from projection import project_central_valid
        points = np.asarray(xyz, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Explicit world points must be a finite N x 3 array")
        camera, shared_map, view_map = self.camera25_and_jacobians(view)
        projected = project_central_valid(camera, points, with_jacobian=with_jacobian)
        uv, depth, valid = projected[0], projected[1], projected[-1]
        if require_valid and not np.all(valid):
            raise PhysicalProjectionDomainError(np.flatnonzero(~valid))
        if not with_jacobian:
            return uv, depth, valid
        state_from_camera = np.zeros((11, 25))
        state_from_camera[:11, :11] = np.diag([0.01]*3 + [1.0]*5 + [0.001]*3)
        pixel_map = projected[2] @ state_from_camera
        if len(points):
            rotation = cv2.Rodrigues(camera[3:6])[0]
            intrinsics = np.array([[math.exp(camera[6]), 0.0, camera[8]],
                                   [0.0, math.exp(camera[7]), camera[9]],
                                   [0.0, 0.0, 1.0]])
            _, opencv_jac = cv2.projectPoints(points, camera[3:6], -rotation @ camera[:3],
                                             intrinsics, camera[11:25])
            pixel_map[:, 11:25] = opencv_jac[:, 10:24]
        shared_pixels, view_pixels = pixel_map @ shared_map, pixel_map @ view_map
        if require_valid and (not np.isfinite(shared_pixels).all() or not np.isfinite(view_pixels).all()):
            raise ValueError("Physical-camera pixel Jacobian is non-finite")
        return uv, depth, shared_pixels, view_pixels, valid

    @classmethod
    def from_manifest(cls, record):
        """Reload explicit exported calibration data, not manufacturer metadata."""
        if record.get("world_length_unit") != "scene_units_metric_scale_unverified":
            raise ValueError("Calibration scene-unit declaration is missing/changed")
        mode = OutputMode(record["output_mode_id"], record["focal_domain"],
                          tuple(record["domain_wh"]), tuple(record["final_wh"]),
                          record["A_source_to_final"], record["output_chain_provenance"])
        if record.get("Brown14_order") != list(BROWN14_ORDER) or record.get("other_Brown_terms") != "fixed_zero":
            raise ValueError("Unexpected Brown semantics or unlocked higher terms")
        return cls(record["installation_id"], tuple(record["pivot_xyz"]),
                   tuple(record["lever_xyz"]), mode, record["q_reference"],
                   record["aspect_fx_over_fy_in_domain"],
                   tuple(record["principal_x_fraction_coefficients"]),
                   tuple(record["principal_y_fraction_coefficients"]),
                   tuple(record["k1_coefficients"]), tuple(record["k2_coefficients"]),
                   tuple(record["tangential_p1_p2"]), tuple(record["axial_coefficients"]),
                   record["allow_effective_pp_offset"], record["field_settings"])

    def manifest(self):
        mode = self.output_mode
        return {"installation_id": self.installation_id,
                "world_length_unit": "scene_units_metric_scale_unverified",
                "output_mode_id": mode.mode_id, "focal_domain": mode.domain,
                "domain_wh": list(mode.domain_wh), "final_wh": list(mode.final_wh),
                "A_source_to_final": mode.A.tolist() if mode.domain == "source" else None,
                "output_chain_provenance": mode.chain_provenance,
                "zoom_driver_interpretation": mode.zoom_interpretation,
                "focal_parameter": "log(geometric_f_domain_px/domain_width)",
                "q_reference": self.q_reference, "pivot_xyz": list(self.pivot_xyz),
                "lever_xyz": list(self.lever_xyz), "axial_coefficients": list(self.axial),
                "aspect_fx_over_fy_in_domain": self.aspect_fx_over_fy,
                "principal_x_fraction_coefficients": list(self.principal_x_fraction),
                "principal_y_fraction_coefficients": list(self.principal_y_fraction),
                "k1_coefficients": list(self.k1), "k2_coefficients": list(self.k2),
                "tangential_p1_p2": list(self.tangential_p1_p2),
                "Brown14_order": list(BROWN14_ORDER), "other_Brown_terms": "fixed_zero",
                "field_settings": dict(self.field_settings),
                "allow_effective_pp_offset": self.allow_effective_pp_offset,
                "effective_offset_identified_as_OIS": False,
                "accuracy_accepted": False}


@dataclass(frozen=True)
class Z190StandingTripodHypothesis:
    """Presumed crew setup and a soft C-height prior, not Sony calibration.

    Defaults implement the user's working scale and standing-tripod estimate.
    A nominal lens height of 1.5--1.7 m plus unknown ground/ice offset of about
    +/-0.15 m motivates sigma=0.18 m; these are not hard height limits.  The
    height factor applies to the optical centre C even with nonzero lever.
    """

    camera_id: int
    scene_unit_m: float = 0.3135373
    lens_height_above_ground_m: float = 1.60
    ground_minus_ice_m: float = 0.0
    height_sigma_m: float = 0.18

    def __post_init__(self):
        if (isinstance(self.camera_id, (bool, np.bool_)) or
                not isinstance(self.camera_id, (int, np.integer)) or self.camera_id not in (0, 1, 2)):
            raise ValueError("Working crew camera_id must be 0, 1 or 2")
        object.__setattr__(self, "camera_id", int(self.camera_id))
        for name in ("scene_unit_m", "lens_height_above_ground_m", "ground_minus_ice_m", "height_sigma_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.scene_unit_m <= 0 or self.lens_height_above_ground_m <= 0 or self.height_sigma_m <= 0:
            raise ValueError("Working scale, lens height and height sigma must be positive")

    @property
    def role(self):
        return ("main", "secondary", "low_end")[self.camera_id]

    @property
    def height_target_m(self):
        return self.lens_height_above_ground_m + self.ground_minus_ice_m

    @property
    def height_target_scene(self):
        return self.height_target_m/self.scene_unit_m

    @property
    def height_sigma_scene(self):
        return self.height_sigma_m/self.scene_unit_m

    def profile(self):
        """JSON-compatible declared assumptions, retained by existing manifests."""
        return {"schema_version": "rinkspace.z190-standing-tripod-hypothesis.v1",
                "status": "presumed_not_confirmed", "camera_id": self.camera_id,
                "presumed_role": self.role, "presumed_camera_model": "Sony PXW-Z190",
                "scene_unit_m": self.scene_unit_m,
                "metric_scale_status": "user_working_approximation_not_measured_calibration",
                "lens_height_above_ground_m": self.lens_height_above_ground_m,
                "ground_minus_ice_m": self.ground_minus_ice_m,
                "ground_minus_ice_status": "unknown_working_value",
                "height_sigma_m": self.height_sigma_m,
                "height_prior_type": "soft_gaussian_Cz_ice_z_zero_no_hard_bounds",
                "height_sigma_rationale": "approx_0.18m_from_lens_about_1.5_to_1.7m_and_ground_offset_about_plus_minus_0.15m",
                "default_motion_model": "fixed_C_zero_lever_zero_axial",
                "default_fixed_shared_parameters": {"lever_xyz": [0.0, 0.0, 0.0], "axial": [0.0]},
                "numeric_working_priors": {
                    "camera_height": {"target_m_above_ice": self.height_target_m,
                                      "sigma_m": self.height_sigma_m,
                                      "applies_to": "optical_centre_Cz_ice_z_zero"},
                    "principal_point": {"target_final_image_fraction": [0.5, 0.5],
                                        "sigma_final_px": 32.0},
                    "roll": {"target_degrees": 0.0, "sigma_degrees": 0.5,
                             "parameter": "optical_axis_roll_relative_to_world_vertical_not_Rodrigues_z"},
                    "residual_distortion_at_wide_seed_corner": {
                        "sample_radius_fractions": [0.5, 0.75, 1.0], "sigma_final_px": 8.0,
                        "radius_reference": "fixed_wide_seed_full_image_corner_normalized_ray_radius",
                        "seed_policy": "fixed_per_installation_not_recomputed_with_optimized_focal",
                        "effect": "Brown_radial_pixel_displacement_relative_to_zero_distortion",
                        "status": "modelling_choice_not_Sony_measurement"},
                    "k2": {"fixed_value": 0.0, "fitted": False,
                           "status": "source_diagnostic_modelling_choice_not_Sony_measurement"}},
                "professional_settings_status": "presumed_not_confirmed",
                "professional_settings": {"recording_format": "1920x1080/50p",
                                          "camera_output_format": "1920x1080/50p",
                                          "steadyshot": "off", "digital_extender": "off",
                                          "zoom_control": "manual_operator_servo_zoom",
                                          "focus": "MF", "temporary_focus": "occasional_Push_AF",
                                          "focus_assist": "peaking", "iris_control": "manual",
                                          "aperture_f_number": 4.0, "nd_filter": "Clear"},
                "professional_primary_sources": {
                    "manual": "https://pro.sony/support/res/manuals/4740/1bf6ebbd9def311f789db07687817956/47407001M.pdf",
                    "shutter_guidance": "https://www.sony.co.uk/electronics/support/articles/00268256",
                    "specifications": "https://www.sony.jp/xdcam/products/PXW-Z190/spec.html"},
                "professional_primary_basis": {
                    "focus": "manual_page_30_MF_supports_Push_AF_Full_MF_disables_it",
                    "shutter": "Sony_general_movie_guidance_lists_1_over_100_for_50Hz_lighting",
                    "capability_vs_preset": "official_sources_support_controls_and_capabilities_not_the_presumed_event_settings"},
                "official_exposure_rate_hz": 50.0,
                "container_display_rate_hz": 60.0,
                "exposure_alias_policy": "requires_explicit_source_exposure_identity_no_rate_based_aliases",
                "shutter_seconds": 1.0/100.0, "shutter_status": "presumed_50p_180_degree",
                "rolling_shutter_readout_seconds": None,
                "shutter_equals_readout": False}

    @classmethod
    def from_profile(cls, profile):
        hypothesis = cls(profile["camera_id"], profile["scene_unit_m"],
                         profile["lens_height_above_ground_m"], profile["ground_minus_ice_m"],
                         profile["height_sigma_m"])
        if profile != hypothesis.profile():
            raise ValueError("Unknown/changed standing-tripod working profile semantics")
        return hypothesis

    @classmethod
    def from_calibration(cls, calibration):
        if not isinstance(calibration, SharedCalibration) or "working_hypothesis" not in calibration.field_settings:
            raise UnknownCalibrationError("Calibration has no declared standing-tripod working hypothesis")
        return cls.from_profile(calibration.field_settings["working_hypothesis"])

    def create_calibration(self, xy_scene, *, installation_id=None, output_mode=None):
        """Initialize fixed C at the prior mean; callers still add the soft factor.

        Horizontal location remains an explicit input.  This is a starting
        calibration, not a camera position measurement or automatic solve.
        """
        x, y = _finite_tuple(xy_scene, 2, "installation xy in scene units")
        profile = self.profile()
        settings = {key: {"value": value, "status": "presumed_not_confirmed"}
                    for key, value in profile["professional_settings"].items()}
        settings["shutter_seconds"] = {"value": profile["shutter_seconds"], "status": profile["shutter_status"]}
        settings["working_hypothesis"] = profile
        return SharedCalibration(str(self.camera_id) if installation_id is None else installation_id,
                                 (x, y, self.height_target_scene),
                                 output_mode=OutputMode.unknown_final() if output_mode is None else output_mode,
                                 field_settings=settings)

    def height_residual_and_jacobians(self, calibration, view):
        """Return one standardized soft residual and shared/state4 gradients.

        residual = (Cz*scene_unit_m - (lens_height + ground_minus_ice))/sigma.
        The least-squares factor is residual**2; it does not clamp any state.
        The caller should weight an installation prior once, avoiding repeated
        copies for exposure aliases or frames of a fixed-C installation.
        """
        camera, shared_map, view_map = calibration.camera25_and_jacobians(view)
        scale = self.scene_unit_m/self.height_sigma_m
        residual = (camera[2]*self.scene_unit_m-self.height_target_m)/self.height_sigma_m
        return float(residual), shared_map[2]*scale, view_map[2]*scale


@dataclass(frozen=True)
class CanonicalView:
    installation_id: str
    view: ViewState


@dataclass(frozen=True)
class PhysicalCameraModel:
    calibrations: Mapping[str, SharedCalibration]

    def __post_init__(self):
        values = dict(self.calibrations)
        if not values or any(not isinstance(v, SharedCalibration) or k != v.installation_id
                             for k, v in values.items()):
            raise ValueError("Each calibration key must identify its own physical installation")
        object.__setattr__(self, "calibrations", MappingProxyType(values))

    def camera25(self, canonical: CanonicalView):
        if not isinstance(canonical, CanonicalView) or canonical.installation_id not in self.calibrations:
            raise UnknownCalibrationError("Canonical view has no known physical calibration")
        return self.calibrations[canonical.installation_id].camera25(canonical.view)

    def save_calibration_json(self, path):
        record = {"schema_version": "rinkspace.z190-physical-calibration.v1",
                  "calibrations": {k: v.manifest() for k, v in self.calibrations.items()},
                  "accuracy_accepted": False}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        return destination

    @classmethod
    def load_calibration_json(cls, path):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record.get("schema_version") != "rinkspace.z190-physical-calibration.v1":
            raise ValueError("Unknown physical-calibration schema")
        return cls({k: SharedCalibration.from_manifest(v) for k, v in record["calibrations"].items()})

    def exposure_arrays(self, pts: Sequence[int], representative_pts: Mapping[int, int],
                        canonical_views: Mapping[int, CanonicalView], *,
                        installation_by_pts: Mapping[int, str] | None = None):
        """Expand explicit exposure aliases without averaging or inventing views.

        Every requested source PTS needs a direct representative mapping. A
        representative may be outside a requested subwindow, but its state
        must exist explicitly. Alias chains/cycles and contradictory physical
        identities are refused; no missing exposure is filled from neighbours.
        """
        native_pts = [_integer(value, "source PTS") for value in pts]
        if not native_pts or any(b <= a for a, b in zip(native_pts, native_pts[1:])):
            raise ValueError("Requested PTS must be nonempty and strictly increasing")
        aliases = {_integer(k, "alias PTS"): _integer(v, "representative PTS")
                   for k, v in representative_pts.items()}
        known_views = {_integer(k, "canonical PTS"): v for k, v in canonical_views.items()}
        representatives, cameras, installations, states, offsets, cache = [], [], [], [], [], {}
        for source_pts in native_pts:
            if source_pts not in aliases:
                raise UnknownCalibrationError(f"No explicit exposure identity for PTS {source_pts}")
            representative = aliases[source_pts]
            if representative in aliases and aliases[representative] != representative:
                raise ValueError("Representative must be canonical, not an alias chain/cycle")
            if representative not in known_views:
                raise UnknownCalibrationError(f"No canonical camera input for PTS {representative}")
            canonical = known_views[representative]
            if not isinstance(canonical, CanonicalView):
                raise UnknownCalibrationError("Canonical view is missing/unknown")
            if installation_by_pts is not None and (source_pts not in installation_by_pts or
                    installation_by_pts[source_pts] != canonical.installation_id):
                raise UnknownCalibrationError("Exposure alias crosses/omits a physical camera identity")
            if representative not in cache:
                cache[representative] = self.camera25(canonical)
            # Exact copied canonical output, never an alias's own independent
            # focal/rotation or a mean over repeated display-frame estimates.
            representatives.append(representative)
            cameras.append(cache[representative].copy())
            installations.append(canonical.installation_id)
            states.append(canonical.view.state4())
            offsets.append(canonical.view.effective_pp_offset_px)
        return {"pts": np.asarray(native_pts, dtype=np.int64),
                "camera25": np.asarray(cameras, dtype=np.float64),
                "exposure_representative_pts": np.asarray(representatives, dtype=np.int64),
                "installation_id": np.asarray(installations, dtype=str),
                "view_state4": np.asarray(states, dtype=np.float64),
                "effective_pp_offset_px": np.asarray(offsets, dtype=np.float64)}

    def save_npz(self, path, pts, representative_pts, canonical_views, *,
                 source_sha256, time_base, installation_by_pts=None):
        """Create a numeric adapter NPZ; no overwrite of a historical candidate.

        pts/camera25/exposure_representative_pts are compatible numeric fields.
        Explicit view_state4 and effective_pp_offset_px preserve canonical
        inputs for exact replay; they are not recovered from exported K/C/D.
        This is not the full render_result archive: no statuses/metrics, scene
        SHA or validated full-timeline coverage are synthesized here.
        """
        from fractions import Fraction
        if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in source_sha256):
            raise UnknownCalibrationError("Source SHA-256 must be explicit")
        if Fraction(time_base) <= 0:
            raise ValueError("Source time_base must be positive")
        arrays = self.exposure_arrays(pts, representative_pts, canonical_views,
                                      installation_by_pts=installation_by_pts)
        metadata = {"schema_version": "rinkspace.z190-physical-camera.v2",
                    "source_sha256": source_sha256, "source_time_base": str(Fraction(time_base)),
                    "exposure_policy": "explicit_direct_canonical_no_average_no_fill",
                    "calibrations": {k: v.manifest() for k, v in self.calibrations.items()},
                    "accuracy_accepted": False, "rolling_shutter_readout_seconds": None}
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("Camera archive path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        return destination

    @classmethod
    def load_npz(cls, path):
        """Strict numeric-package replay without deriving state from camera25.

        Return (model, source_pts, direct_representative_mapping,
        canonical_views_by_representative).  States for aliases must be exact
        copies, including optional effective PP offsets and installation IDs.
        This replays the declared numeric candidate, not its accuracy approval
        or the separate full-timeline renderer archive contract.
        """
        from fractions import Fraction
        required = {"pts", "camera25", "exposure_representative_pts", "installation_id",
                    "view_state4", "effective_pp_offset_px", "metadata_json"}
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required or len(archive.files) != len(required):
                raise ValueError("Unexpected/missing numeric camera package fields")
            arrays = {name: archive[name].copy() for name in required}
        metadata_json = arrays.pop("metadata_json")
        if metadata_json.shape != () or metadata_json.dtype.kind != "U":
            raise ValueError("Camera package metadata must be a scalar Unicode JSON string")
        metadata = json.loads(str(metadata_json))
        metadata_fields = {"schema_version", "source_sha256", "source_time_base", "exposure_policy",
                           "calibrations", "accuracy_accepted", "rolling_shutter_readout_seconds"}
        if not isinstance(metadata, dict) or set(metadata) != metadata_fields:
            raise ValueError("Unexpected/missing numeric camera package metadata")
        if (metadata["schema_version"] != "rinkspace.z190-physical-camera.v2" or
                metadata["exposure_policy"] != "explicit_direct_canonical_no_average_no_fill" or
                metadata["accuracy_accepted"] is not False or
                metadata["rolling_shutter_readout_seconds"] is not None):
            raise ValueError("Numeric camera package semantics are unsupported/changed")
        source_sha256 = metadata["source_sha256"]
        if (not isinstance(source_sha256, str) or len(source_sha256) != 64 or
                any(c not in "0123456789abcdefABCDEF" for c in source_sha256)):
            raise UnknownCalibrationError("Numeric camera package needs an explicit source SHA-256")
        time_base = metadata["source_time_base"]
        if not isinstance(time_base, str) or Fraction(time_base) <= 0 or str(Fraction(time_base)) != time_base:
            raise ValueError("Source time_base must be a positive canonical rational string")
        records = metadata["calibrations"]
        if not isinstance(records, dict) or not records:
            raise UnknownCalibrationError("Numeric camera package lacks installation calibrations")
        model = cls({key: SharedCalibration.from_manifest(value) for key, value in records.items()})
        if any(records[key] != calibration.manifest() for key, calibration in model.calibrations.items()):
            raise ValueError("Calibration manifest semantics are unsupported/changed")
        pts = arrays["pts"]
        representatives = arrays["exposure_representative_pts"]
        if pts.dtype != np.dtype("int64") or pts.ndim != 1 or len(pts) == 0 or np.any(pts[1:] <= pts[:-1]):
            raise ValueError("Source PTS must be a nonempty strictly increasing int64 vector")
        count = len(pts)
        if representatives.dtype != np.dtype("int64") or representatives.shape != (count,):
            raise ValueError("Exposure representatives must be an explicit int64 vector")
        for name, shape in (("camera25", (count, 25)), ("view_state4", (count, 4)),
                            ("effective_pp_offset_px", (count, 2))):
            if arrays[name].dtype != np.dtype("float64") or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} must be a finite float64 array with shape {shape}")
        installations = arrays["installation_id"]
        if installations.shape != (count,) or installations.dtype.kind != "U":
            raise ValueError("Installation IDs must be a Unicode vector matching source PTS")
        aliases, canonical_views, installation_by_pts, canonical_rows = {}, {}, {}, {}
        for index, source_pts in enumerate(pts):
            representative = int(representatives[index])
            installation = str(installations[index])
            if installation not in model.calibrations:
                raise UnknownCalibrationError("Numeric camera package references an unknown installation")
            state = arrays["view_state4"][index]
            offset = arrays["effective_pp_offset_px"][index]
            if representative in canonical_rows:
                previous = canonical_rows[representative]
                if (installation != str(installations[previous]) or
                        any(not np.array_equal(arrays[name][index], arrays[name][previous])
                            for name in ("camera25", "view_state4", "effective_pp_offset_px"))):
                    raise ValueError("Repeated exposure representative has conflicting state/installation")
            else:
                canonical_rows[representative] = index
                canonical_views[representative] = CanonicalView(
                    installation, ViewState.from_state4(state, effective_pp_offset_px=offset))
            aliases[int(source_pts)] = representative
            installation_by_pts[int(source_pts)] = installation
        replay = model.exposure_arrays(pts, aliases, canonical_views, installation_by_pts=installation_by_pts)
        if any(not np.array_equal(replay[name], arrays[name]) for name in replay):
            raise ValueError("Stored camera25 does not exactly replay its canonical input/calibration")
        return model, pts.copy(), aliases, canonical_views


@dataclass(frozen=True)
class RowTiming:
    """Read-only row-time interface; unknown readout never becomes shutter time.

    It does not alter camera25 or infer/attribute RS from effective shear.  A
    caller with separately supported readout and scan direction may evaluate
    their own continuous camera motion at the returned row exposure centres.
    """

    readout_seconds: float | None = None
    direction: int | None = None
    provenance: str = "unknown"

    def sample_times(self, mid_time_seconds, rows, image_height):
        if self.readout_seconds is None or self.direction is None or not self.provenance or self.provenance == "unknown":
            raise UnknownCalibrationError("Row timing needs known readout, direction and provenance")
        if (not math.isfinite(self.readout_seconds) or self.readout_seconds < 0 or
                isinstance(self.direction, bool) or self.direction not in (-1, 1)):
            raise ValueError("Readout must be finite nonnegative; scan direction must be +/-1")
        if isinstance(image_height, bool) or not isinstance(image_height, (int, np.integer)) or image_height < 2:
            raise ValueError("Image height must be an integer of at least two rows")
        values = np.asarray(rows, dtype=float)
        if not math.isfinite(float(mid_time_seconds)) or not np.isfinite(values).all() or np.any((values < 0) | (values > image_height-1)):
            raise ValueError("Frame time and requested image rows must be finite/in range")
        return float(mid_time_seconds) + self.direction*(values/(image_height-1)-0.5)*self.readout_seconds
