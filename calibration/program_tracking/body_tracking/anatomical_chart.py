"""Joint-local swing and twist in caller-calibrated anatomical coordinates.

All angles are radians. The model-space axes and finite envelopes are inputs,
with their interpretation and provenance supplied by the caller.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .absolute_motion_curve import RotationChartDomainError


_SINGULAR_TOL = 1e-7
_BRANCH_TOL = 1e-6


def _proper_rotation(value, name):
    value = np.asarray(value, dtype=float)
    if (value.shape[-2:] != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.swapaxes(-1, -2) @ value, np.eye(3),
                               rtol=0, atol=1e-6)
            or not np.allclose(np.linalg.det(value), 1, rtol=0, atol=1e-6)):
        raise ValueError(f"{name} must contain proper SO(3) matrices")
    return value


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _projection(local_R, rest, basis):
    local_R = _proper_rotation(local_R, "local_R")
    delta = basis.swapaxes(-1, -2) @ rest.swapaxes(-1, -2) @ local_R @ basis
    quat = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_quat()
    axial, y, z, real = np.moveaxis(quat.reshape(*delta.shape[:-2], 4), -1, 0)
    projection = np.hypot(axial, real)
    principal = _wrap(2 * np.arctan2(axial, real))
    return axial, y, z, real, projection, principal


def _indices(values, size, name):
    values = np.asarray(values)
    if (values.ndim != 1 or (values.size and not np.issubdtype(values.dtype, np.integer))
            or np.any(values < 0) or np.any(values >= size)):
        raise ValueError(f"{name} must be a one-dimensional in-range integer index array")
    return values.astype(np.intp, copy=False)


@dataclass(frozen=True)
class JointAngles:
    twist: np.ndarray
    swing_vector: np.ndarray
    swing: np.ndarray
    singular: np.ndarray
    outside_branch: np.ndarray


@dataclass(frozen=True)
class JointChart:
    """An anatomical basis and frozen reference twist for each sample and joint.

    The relative rotation is B.T @ rest.T @ local_R @ B, factored as
    swing @ twist. The columns of B are the authored positive twist axis
    followed by the two swing axes in the local rest frame. rest and basis
    have shape (J, 3, 3), and reference_twist has shape (N, J). The same
    arrays can be saved and passed to this constructor to restore the chart.
    """

    rest: np.ndarray
    basis: np.ndarray
    reference_twist: np.ndarray

    def __post_init__(self):
        rest = _proper_rotation(self.rest, "rest").copy()
        basis = _proper_rotation(self.basis, "basis").copy()
        reference = np.array(self.reference_twist, dtype=float, copy=True)
        if (rest.ndim != 3 or basis.shape != rest.shape or not len(rest)
                or reference.ndim != 2 or reference.shape[1] != len(rest)
                or not len(reference) or not np.isfinite(reference).all()):
            raise ValueError("Expected rest and basis (J, 3, 3), reference_twist (N, J)")
        for name, value in (("rest", rest), ("basis", basis), ("reference_twist", reference)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_reference(cls, local_R, rest, basis, *, initial_twist):
        """Lift the complete ordered reference using its supplied first-sample anchor.

        The reference must resolve adjacent twist changes within pi and avoid the
        swing-twist singularity. A first anchor may be scalar or one angle per
        joint. Branch selection is independent of the joint envelope.
        """
        rest = _proper_rotation(rest, "rest")
        basis = _proper_rotation(basis, "basis")
        local_R = np.asarray(local_R, dtype=float)
        if (rest.ndim != 3 or not len(rest) or basis.shape != rest.shape
                or local_R.ndim != 4 or local_R.shape[1:] != rest.shape or not len(local_R)):
            raise ValueError("Expected complete local_R sequence (N, J, 3, 3)")
        initial = np.broadcast_to(np.asarray(initial_twist, dtype=float), (len(rest),))
        if not np.isfinite(initial).all():
            raise ValueError("Initial twist anchors must be finite")
        *_, projection, principal = _projection(local_R, rest, basis)
        if np.any(projection < _SINGULAR_TOL):
            raise RotationChartDomainError("Reference has a singular swing-twist decomposition")
        first_step = _wrap(principal[0] - initial)
        adjacent_step = _wrap(np.diff(principal, axis=0))
        if (np.any(np.abs(first_step) >= np.pi - _BRANCH_TOL)
                or np.any(np.abs(adjacent_step) >= np.pi - _BRANCH_TOL)):
            raise RotationChartDomainError("Reference has an ambiguous twist branch")
        offset = initial + first_step - principal[0]
        return cls(rest, basis, np.unwrap(principal, axis=0) + offset)

    def evaluate(self, local_R, *, sample_indices=None, joint_indices=None):
        """Evaluate full, sampled, or paired sample/joint rotations.

        Supply local_R already in the requested index order. Its shape is
        (N, J, 3, 3), (len(sample_indices), J, 3, 3), or (M, 3, 3) for
        M explicit pairs of sample_indices and joint_indices.

        At a singular decomposition swing magnitude remains defined, while
        twist and swing_vector are NaN. outside_branch marks the seam opposite
        each frozen anchor, where the angular lift is ambiguous.
        """
        reference = self.reference_twist
        rest, basis = self.rest, self.basis
        if sample_indices is not None:
            rows = _indices(sample_indices, len(reference), "sample_indices")
            reference = reference[rows]
        if joint_indices is not None:
            joints = _indices(joint_indices, len(rest), "joint_indices")
            if sample_indices is None or len(joints) != len(rows):
                raise ValueError("Paired indices require equally many samples and joints")
            reference = self.reference_twist[rows, joints]
            rest, basis = rest[joints], basis[joints]
        local_R = np.asarray(local_R, dtype=float)
        if local_R.shape != (*reference.shape, 3, 3):
            raise ValueError("local_R shape must match the selected sample/joint anchors")

        axial, y, z, real, projection, principal = _projection(local_R, rest, basis)
        singular = projection < _SINGULAR_TOL
        offset = _wrap(principal - reference)
        twist = np.where(singular, np.nan, reference + offset)
        outside_branch = ~singular & (np.abs(offset) >= np.pi - _BRANCH_TOL)

        # q_swing = q @ q_twist^-1. Its scalar part is the projection norm,
        # and its axial part is zero. atan2 retains accuracy near zero swing.
        length = np.hypot(y, z)
        swing = 2 * np.arctan2(length, projection)
        scale = np.divide(swing, length, out=np.full_like(length, 2.), where=length > 0)
        direction = np.stack((real * y - axial * z, real * z + axial * y), axis=-1)
        swing_vector = np.divide(direction * scale[..., None], projection[..., None],
                                 out=np.full_like(direction, np.nan), where=~singular[..., None])
        return JointAngles(twist, swing_vector, swing, singular, outside_branch)


@dataclass(frozen=True)
class JointEnvelope:
    """Finite soft bounds with a flexion-dependent swing cross section.

    swing_max supplies a radius (K,) or two ellipse semi-axes (K, 2) at each
    of the K flex_grid entries. Limits interpolate linearly in twist and
    extend as constants beyond the supplied grid. The two swing axes follow
    JointChart.basis. Residuals are dimensionless after division by the scales.
    """

    flex_min: float
    flex_max: float
    flex_grid: np.ndarray
    swing_max: np.ndarray
    flex_scale: float
    swing_scale: float
    provenance: str

    def __post_init__(self):
        parameters = np.asarray((self.flex_min, self.flex_max, self.flex_scale, self.swing_scale))
        grid = np.array(self.flex_grid, dtype=float, copy=True)
        limits = np.array(self.swing_max, dtype=float, copy=True)
        if (not np.isfinite(parameters).all() or self.flex_min >= self.flex_max
                or self.flex_scale <= 0 or self.swing_scale <= 0
                or not isinstance(self.provenance, str) or not self.provenance.strip()):
            raise ValueError("Finite ordered bounds, positive scales and provenance are required")
        if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
                or np.any(np.diff(grid) <= 0)
                or limits.shape not in (grid.shape, (len(grid), 2))
                or not np.isfinite(limits).all() or np.any(limits < 0)
                or (limits.ndim == 2 and np.any(limits == 0))):
            raise ValueError("Invalid flexion grid or coupled swing limits")
        grid.setflags(write=False)
        limits.setflags(write=False)
        object.__setattr__(self, "flex_grid", grid)
        object.__setattr__(self, "swing_max", limits)

    def signed_residual(self, angles):
        """Unclipped authored boundaries; their positive part is the soft residual."""
        if np.any(angles.singular):
            raise RotationChartDomainError("Trial has a singular swing-twist decomposition")
        if np.any(angles.outside_branch):
            raise RotationChartDomainError("Trial reaches the frozen joint twist branch seam")
        twist, swing = angles.twist, angles.swing
        if self.swing_max.ndim == 1:
            allowed = np.interp(twist, self.flex_grid, self.swing_max)
        else:
            axes = np.stack([np.interp(twist, self.flex_grid, axis)
                             for axis in self.swing_max.T], axis=-1)
            unit = np.divide(angles.swing_vector, swing[..., None],
                             out=np.zeros_like(angles.swing_vector), where=swing[..., None] > 0)
            reciprocal_radius = np.linalg.norm(unit / axes, axis=-1)
            allowed = np.divide(1., reciprocal_radius, out=axes.min(axis=-1),
                                where=reciprocal_radius > 0)
        return np.stack(((self.flex_min - twist) / self.flex_scale,
                         (twist - self.flex_max) / self.flex_scale,
                         (swing - allowed) / self.swing_scale), axis=-1)

    def residual(self, angles):
        """Return the original lower/upper/coupled-swing hinge residuals."""
        return np.maximum(self.signed_residual(angles), 0.)
