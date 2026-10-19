"""Continuous optical centres and low-dimensional zoom optics for camera graphs.

All graph nodes are Vector11.  A frame node uses only Rodrigues [3:6] and
mean log focal length [6]; a centre control uses C/100 in [0:3].  One physical
camera group has a lens base node and optionally a zoom slope node.  The base
and slope use log(fx/fy) in [6] and cx, cy, skew divided by 1000 in [8:11].
Unused coordinates remain the caller's responsibility to anchor as dummies.

Brown14 is copied from each supplied camera25 template and is *fixed*, not
certified correct.  A future calibrated zoom-dependent distortion model would
need its own graph variables and projection derivatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np

try:
    from .projection import from_state
except ImportError:
    from projection import from_state


STATE_DIM = 11
DEGREE = 3


def _state(value, label: str) -> np.ndarray:
    result = np.asarray(value, np.float64)
    if result.shape != (STATE_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be one finite Vector11")
    return result


@dataclass(frozen=True)
class ShotSpline:
    """One cut's explicit clamped cubic knot vector in source PTS units.

    There must be ``len(control_keys)+4`` knots.  The first and last four knots
    equal the cut's supported endpoints.  Interior knots are supplied by the
    driver; this module imposes no interval length or maximum control count.
    """

    shot_id: int
    control_keys: tuple[int, ...]
    knot_pts: tuple[float, ...]

    def __post_init__(self):
        keys = tuple(map(int, self.control_keys))
        knots = tuple(map(float, self.knot_pts))
        if len(keys) < DEGREE + 1 or len(set(keys)) != len(keys):
            raise ValueError("Cubic shot spline needs at least four distinct control keys")
        if len(knots) != len(keys) + DEGREE + 1 or not np.isfinite(knots).all():
            raise ValueError("Cubic knot vector must have control_count+4 finite PTS")
        if (any(knots[i] > knots[i + 1] for i in range(len(knots) - 1))
                or knots[0] != knots[3] or knots[-1] != knots[-4]
                or knots[3] >= knots[-4]
                or any(k != knots[0] for k in knots[:4])
                or any(k != knots[-1] for k in knots[-4:])):
            raise ValueError("Expected a nondecreasing, endpoint-clamped cubic knot vector")
        object.__setattr__(self, "control_keys", keys)
        object.__setattr__(self, "knot_pts", knots)

    @lru_cache(maxsize=None)
    def basis_pairs(self, pts: float) -> tuple[tuple[int, float], ...]:
        """Cache the four nonzero cubic basis weights at an exposure PTS."""
        t = float(pts)
        knots = self.knot_pts
        count = len(self.control_keys)
        if not np.isfinite(t) or t < knots[3] or t > knots[-4]:
            raise ValueError(f"PTS {pts} lies outside shot {self.shot_id} spline support")
        span = (count - 1 if t == knots[-4]
                else int(np.searchsorted(knots, t, side="right") - 1))
        values = np.zeros(DEGREE + 1)
        values[0] = 1.
        left = np.zeros(DEGREE + 1)
        right = np.zeros(DEGREE + 1)
        for degree in range(1, DEGREE + 1):
            left[degree] = t - knots[span + 1 - degree]
            right[degree] = knots[span + degree] - t
            saved = 0.
            for local in range(degree):
                denominator = right[local + 1] + left[degree - local]
                portion = values[local] / denominator if denominator > 0 else 0.
                values[local] = saved + right[local + 1] * portion
                saved = left[degree - local] * portion
            values[degree] = saved
        if abs(float(np.sum(values)) - 1.) > 1e-12 or np.any(values < -1e-12):
            raise ArithmeticError("Cubic spline basis failed partition of unity")
        return tuple((self.control_keys[span - DEGREE + local], float(weight))
                     for local, weight in enumerate(values) if abs(weight) > 0.)

    def weights(self, pts: float) -> dict[int, float]:
        """Return a copy of the nonzero basis weights for this PTS."""
        return dict(self.basis_pairs(pts))


@dataclass(frozen=True)
class FrameBinding:
    pts: int
    frame_key: int
    shot_id: int
    camera_id: str
    template25: np.ndarray
    spline_pts: float | None = None


class ContinuousCameraModel:
    """Map frame and shared graph coordinates to complete physical cameras."""

    def __init__(self, shot_splines: Mapping[int, ShotSpline],
                 frames: Mapping[int, FrameBinding] | Sequence[FrameBinding],
                 lens_base_keys: Mapping[str, int],
                 lens_slope_keys: Mapping[str, int] | None = None,
                 lens_zoom_reference: Mapping[str, float] | None = None):
        self.shot_splines = {int(k): v for k, v in shot_splines.items()}
        rows = list(frames.values()) if isinstance(frames, Mapping) else list(frames)
        self.frames = {int(row.pts): row for row in rows}
        if len(self.frames) != len(rows) or not rows:
            raise ValueError("Each frame binding must have one unique PTS")
        self.lens_base_keys = {str(k): int(v) for k, v in lens_base_keys.items()}
        self.lens_slope_keys = {str(k): int(v) for k, v in (lens_slope_keys or {}).items()}
        self.lens_zoom_reference = {str(k): float(v) for k, v in
                                    (lens_zoom_reference or {}).items()}
        all_control_keys = [key for shot in self.shot_splines.values()
                            for key in shot.control_keys]
        all_lens_keys = list(self.lens_base_keys.values()) + list(self.lens_slope_keys.values())
        all_frame_keys = [row.frame_key for row in rows]
        if (len(set(all_control_keys)) != len(all_control_keys)
                or len(set(all_lens_keys)) != len(all_lens_keys)
                or set(all_control_keys) & set(all_lens_keys + all_frame_keys)
                or set(all_lens_keys) & set(all_frame_keys)):
            raise ValueError("Frame, centre-control and lens graph keys must be disjoint")
        if set(self.lens_slope_keys) - set(self.lens_base_keys):
            raise ValueError("A zoom slope needs its camera's lens base node")
        for camera_id in self.lens_slope_keys:
            if (camera_id not in self.lens_zoom_reference
                    or not np.isfinite(self.lens_zoom_reference[camera_id])):
                raise ValueError("Each lens slope needs a finite zoom reference")
        aliases: dict[int, tuple[int, str, float]] = {}
        for pts, row in self.frames.items():
            if int(row.pts) != pts or row.shot_id not in self.shot_splines:
                raise ValueError("Frame PTS or shot spline is missing")
            if row.camera_id not in self.lens_base_keys:
                raise ValueError("Frame has no camera-group lens base")
            template = np.asarray(row.template25, np.float64)
            if template.shape != (25,) or not np.isfinite(template).all():
                raise ValueError("Each frame needs a finite Brown25 template")
            canonical = float(pts if row.spline_pts is None else row.spline_pts)
            self.shot_splines[row.shot_id].weights(canonical)
            identity = (row.shot_id, row.camera_id, canonical)
            if row.frame_key in aliases and aliases[row.frame_key] != identity:
                raise ValueError("Exposure aliases must share shot, camera and spline PTS")
            aliases[row.frame_key] = identity
        self.frame_keys = set(all_frame_keys)

    def dependencies(self, pts: int) -> tuple[int, ...]:
        row = self.frames[int(pts)]
        canonical = row.pts if row.spline_pts is None else row.spline_pts
        keys = [row.frame_key, *(key for key, _ in
                                self.shot_splines[row.shot_id].basis_pairs(canonical))]
        keys.append(self.lens_base_keys[row.camera_id])
        if row.camera_id in self.lens_slope_keys:
            keys.append(self.lens_slope_keys[row.camera_id])
        return tuple(dict.fromkeys(keys))

    def _physical(self, pts: int, states: Mapping[int, np.ndarray],
                  jacobian: bool):
        row = self.frames[int(pts)]
        frame = _state(states[row.frame_key], "frame")
        base_key = self.lens_base_keys[row.camera_id]
        base = _state(states[base_key], "lens base")
        slope_key = self.lens_slope_keys.get(row.camera_id)
        slope = (_state(states[slope_key], "lens slope") if slope_key is not None
                 else np.zeros(STATE_DIM))
        z = float(frame[6])
        dz = z - self.lens_zoom_reference.get(row.camera_id, 0.)
        ratio = base[6] + slope[6] * dz
        physical = np.zeros(STATE_DIM, np.float64)
        physical[3:6] = frame[3:6]
        physical[6] = z + .5 * ratio
        physical[7] = z - .5 * ratio
        physical[8:11] = base[8:11] + slope[8:11] * dz
        if jacobian:
            jf = np.zeros((STATE_DIM, STATE_DIM))
            jf[3:6, 3:6] = np.eye(3)
            jf[6, 6], jf[7, 6] = 1. + .5 * slope[6], 1. - .5 * slope[6]
            jf[8:11, 6] = slope[8:11]
            jacobians = {row.frame_key: jf}
            jb = np.zeros((STATE_DIM, STATE_DIM))
            jb[6, 6], jb[7, 6] = .5, -.5
            jb[8:11, 8:11] = np.eye(3)
            jacobians[base_key] = jb
            if slope_key is not None:
                js = np.zeros((STATE_DIM, STATE_DIM))
                js[6, 6], js[7, 6] = .5 * dz, -.5 * dz
                js[8:11, 8:11] = np.eye(3) * dz
                jacobians[slope_key] = js
        canonical = row.pts if row.spline_pts is None else row.spline_pts
        for key, weight in self.shot_splines[row.shot_id].basis_pairs(canonical):
            control = _state(states[key], "centre control")
            physical[:3] += weight * control[:3]
            if jacobian:
                jc = np.zeros((STATE_DIM, STATE_DIM))
                jc[:3, :3] = weight * np.eye(3)
                jacobians[key] = jc
        return (physical, jacobians) if jacobian else physical

    def physical_frame(self, pts: int, states: Mapping[int, np.ndarray]
                       ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Return physical Vector11 and exact 11x11 Jacobian per graph key."""
        return self._physical(pts, states, True)

    def physical_state(self, pts: int, states: Mapping[int, np.ndarray]) -> np.ndarray:
        """Evaluate the physical frame without constructing Jacobian blocks."""
        return self._physical(pts, states, False)

    def camera25(self, pts: int, states: Mapping[int, np.ndarray]) -> np.ndarray:
        """Use the frame's fixed Brown14 template with its derived physical state."""
        return from_state(self.physical_state(pts, states), self.frames[int(pts)].template25)


def adapt_factor(original_keys: Sequence[int], raw_residual, raw_jacobian,
                 options: Mapping, active_keys: set[int],
                 fixed_states: Mapping[int, np.ndarray], model: ContinuousCameraModel,
                 *, derived_frame_keys: set[int] | None = None):
    """Chain a captured raw factor through physical frame states.

    ``original_keys`` may repeat an exposure key.  Each occurrence contributes
    its own Jacobian to the one returned graph block.  A frame omitted from
    ``derived_frame_keys`` is read literally from ``fixed_states`` when external
    to the active subgraph.  Every required spline/lens dependency must be in
    either ``active_keys`` or ``fixed_states``.  Returns ``None`` for a wholly
    constant factor; otherwise (keys, residual, jacobian_or_None, options).
    """
    original_keys = tuple(map(int, original_keys))
    active = set(map(int, active_keys))
    derived = ({key for key in original_keys if key in model.frames
                and model.frames[key].frame_key in active} if
               derived_frame_keys is None else set(map(int, derived_frame_keys)))
    if derived - (set(model.frames) & set(original_keys)):
        raise ValueError("Derived frame keys must occur in this factor and model")
    kept = []
    fixed_needed = set()
    for key in original_keys:
        dependencies = model.dependencies(key) if key in derived else (key,)
        for dependency in dependencies:
            if dependency in active and dependency not in kept:
                kept.append(dependency)
            elif dependency not in active:
                if dependency not in fixed_states:
                    raise KeyError(f"Missing fixed graph state {dependency}")
                fixed_needed.add(dependency)
    if not kept:
        return None
    kept = tuple(kept)
    options = dict(options)
    rows = int(options["residual_dim"])
    fixed_values = {key: _state(fixed_states[key], "fixed factor state").copy()
                    for key in fixed_needed}

    def expand(q, with_jacobian=False):
        if len(q) != len(kept):
            raise ValueError("Adapted factor received the wrong number of states")
        values = fixed_values.copy()
        values.update(zip(kept, (_state(value, "active factor state") for value in q)))
        expanded, derivatives = [], [] if with_jacobian else None
        for key in original_keys:
            if key in derived:
                if with_jacobian:
                    physical, jacobians = model.physical_frame(key, values)
                    expanded.append(physical)
                    derivatives.append(jacobians)
                else:
                    expanded.append(model.physical_state(key, values))
            else:
                expanded.append(_state(values[key], "unmodified factor state"))
                if with_jacobian:
                    derivatives.append({key: np.eye(STATE_DIM)})
        return expanded, derivatives

    def residual(q):
        return raw_residual(expand(q)[0])

    def jacobian(q):
        expanded, derivatives = expand(q, with_jacobian=True)
        originals = raw_jacobian(expanded)
        if len(originals) != len(original_keys):
            raise ValueError("Raw factor omitted an original Jacobian block")
        result = {key: np.zeros((rows, STATE_DIM)) for key in kept}
        for matrix, chain in zip(originals, derivatives):
            matrix = np.asarray(matrix, np.float64)
            if matrix.shape != (rows, STATE_DIM):
                raise ValueError("Raw factor Jacobian block has the wrong shape")
            for key, derivative in chain.items():
                if key in result:
                    result[key] += matrix @ derivative
        return [result[key] for key in kept]

    return kept, residual, jacobian if raw_jacobian is not None else None, options


def initialize_from_baseline(model: ContinuousCameraModel,
                             baseline_camera25: Mapping[int, np.ndarray]
                             ) -> tuple[dict[int, np.ndarray], dict]:
    """Fit only initializer coordinates to saved cameras; report mismatches.

    The least-squares camera centres are fitted independently per cut.  The
    tiny mean-centre tie only chooses null-space control coordinates when a
    cut has fewer independent exposures than spline controls.  It is not an
    optimization factor.  Existing camera25 Brown templates are unchanged.
    """
    cameras = {}
    for pts in model.frames:
        camera = np.asarray(baseline_camera25[pts], np.float64)
        if camera.shape != (25,) or not np.isfinite(camera).all():
            raise ValueError(f"Baseline camera at PTS {pts} must be Brown25")
        cameras[pts] = camera
    result: dict[int, np.ndarray] = {}
    alias_errors = []
    alias_centres = []
    first_alias_pts = {}
    for pts, row in model.frames.items():
        q = np.zeros(STATE_DIM)
        q[3:6] = cameras[pts][3:6]
        q[6] = np.mean(cameras[pts][6:8])
        if row.frame_key in result:
            alias_errors.append(float(np.max(np.abs(result[row.frame_key] - q))))
            alias_centres.append(float(np.linalg.norm(
                cameras[pts][:3] - cameras[first_alias_pts[row.frame_key]][:3])))
        else:
            result[row.frame_key] = q
            first_alias_pts[row.frame_key] = pts
    shot_report = {}
    for shot_id, spline in model.shot_splines.items():
        representatives = {}
        for pts, row in model.frames.items():
            if row.shot_id == shot_id:
                representatives.setdefault(row.frame_key, pts)
        stamps = list(representatives.values())
        if not stamps:
            continue
        basis = np.zeros((len(stamps), len(spline.control_keys)))
        targets = np.asarray([cameras[pts][:3] / 100. for pts in stamps])
        control_index = {key: i for i, key in enumerate(spline.control_keys)}
        for i, pts in enumerate(stamps):
            row = model.frames[pts]
            canonical = pts if row.spline_pts is None else row.spline_pts
            for key, weight in spline.weights(canonical).items():
                basis[i, control_index[key]] = weight
        tie = 1e-6
        design = np.vstack((basis, tie * np.eye(len(spline.control_keys))))
        rhs = np.vstack((targets, tie * np.tile(np.mean(targets, axis=0),
                                               (len(spline.control_keys), 1))))
        controls = np.linalg.lstsq(design, rhs, rcond=None)[0]
        for key, centre in zip(spline.control_keys, controls):
            q = np.zeros(STATE_DIM)
            q[:3] = centre
            result[key] = q
        errors = np.linalg.norm((basis @ controls - targets) * 100., axis=1)
        shot_report[str(shot_id)] = dict(exposures=len(stamps),
            controls=len(spline.control_keys), data_rank=int(np.linalg.matrix_rank(basis)),
            centre_rms_scene=float(np.sqrt(np.mean(errors**2))),
            centre_max_scene=float(np.max(errors)))
    lens_report = {}
    for camera_id, base_key in model.lens_base_keys.items():
        representatives = {}
        for pts, row in model.frames.items():
            if row.camera_id == camera_id:
                representatives.setdefault(row.frame_key, pts)
        stamps = list(representatives.values())
        if not stamps:
            raise ValueError(f"Lens group {camera_id} has no baseline exposure")
        z = np.asarray([result[model.frames[pts].frame_key][6] for pts in stamps])
        observed = np.asarray([
            np.r_[cameras[pts][6] - cameras[pts][7], cameras[pts][8:11] / 1000.]
            for pts in stamps])
        slope_key = model.lens_slope_keys.get(camera_id)
        if slope_key is None:
            design = np.ones((len(stamps), 1))
            coefficients = np.mean(observed, axis=0, keepdims=True)
        else:
            dz = z - model.lens_zoom_reference[camera_id]
            design = np.column_stack((np.ones(len(stamps)), dz))
            if np.linalg.matrix_rank(design) < 2:
                coefficients = np.vstack((np.mean(observed, axis=0), np.zeros(4)))
            else:
                coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
        base = np.zeros(STATE_DIM)
        base[6], base[8:11] = coefficients[0, 0], coefficients[0, 1:]
        result[base_key] = base
        if slope_key is not None:
            slope = np.zeros(STATE_DIM)
            slope[6], slope[8:11] = coefficients[1, 0], coefficients[1, 1:]
            result[slope_key] = slope
        errors = design @ coefficients - observed
        lens_report[camera_id] = dict(exposures=len(stamps),
            data_rank=int(np.linalg.matrix_rank(design)),
            ratio_rms_log=float(np.sqrt(np.mean(errors[:, 0]**2))),
            ratio_max_log=float(np.max(np.abs(errors[:, 0]))),
            principal_skew_rms_px=np.sqrt(np.mean((errors[:, 1:] * 1000.)**2,
                                                  axis=0)).tolist(),
            principal_skew_max_px=np.max(np.abs(errors[:, 1:] * 1000.),
                                          axis=0).tolist())
    projected = {pts: model.camera25(pts, result) for pts in model.frames}
    centre_errors = [np.linalg.norm(projected[p][:3] - cameras[p][:3])
                     for p in model.frames]
    rotation_errors = [np.linalg.norm(projected[p][3:6] - cameras[p][3:6])
                       for p in model.frames]
    zoom_errors = [abs(np.mean(projected[p][6:8] - cameras[p][6:8]))
                   for p in model.frames]
    ratio_errors = [abs((projected[p][6] - projected[p][7])
                        - (cameras[p][6] - cameras[p][7]))
                    for p in model.frames]
    lens_errors = np.asarray([projected[p][8:11] - cameras[p][8:11]
                              for p in model.frames])
    brown_errors = [np.max(np.abs(projected[p][11:25] - cameras[p][11:25]))
                    for p in model.frames]
    return result, dict(shot_centres=shot_report, camera_lenses=lens_report,
        exposure_alias_max_state_difference=float(max(alias_errors, default=0.)),
        exposure_alias_max_centre_scene=float(max(alias_centres, default=0.)),
        all_exposure_initialization=dict(
            centre_max_scene=float(max(centre_errors)),
            rotation_max_rvec=float(max(rotation_errors)),
            zoom_max_mean_log=float(max(zoom_errors)),
            ratio_max_log=float(max(ratio_errors)),
            principal_skew_max_px=np.max(np.abs(lens_errors), axis=0).tolist(),
            brown_template_max_coefficient_difference=float(max(brown_errors))),
        brown_policy="one fixed Brown14 template per supplied frame")
