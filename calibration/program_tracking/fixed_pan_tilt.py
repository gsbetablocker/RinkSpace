"""Fixed shared pan/tilt orientation, independent of optics and translation.

R maps world to camera. Its two temporal coordinates are pan alpha and tilt
beta: R = exp(beta [b]) R0 exp(alpha [n]), with b.T R0 n = 0.
The world pan axis n and camera tilt axis b are shared unknown installation
directions. No world-vertical axis or zero image roll is presumed.

Shared coordinates are two smooth local coordinates per unit axis. Temporal
angles must be unwrapped and interpolated before evaluating this map; an SO3
short arc between its knots generally does not remain on the fixed-head
surface. This module never changes a camera centre, lens or source record.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(name + ' must have finite shape ' + str(shape))
    return value


def _unit(value):
    value = _finite(value, (3,), 'axis')
    length = np.linalg.norm(value)
    if length == 0.:
        raise ValueError('A nonzero installation axis is required')
    return value / length


def skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _reference(axis):
    return np.eye(3)[int(np.argmin(np.abs(axis)))]


def _tangent(axis, reference):
    value = reference - axis * (axis @ reference)
    length = np.linalg.norm(value)
    if length <= 64. * np.finfo(float).eps:
        raise ValueError('Axis chart needs a new tangent reference')
    return value / length


def _tangent_with_derivative(axis, derivative, reference):
    tangent = _tangent(axis, reference)
    length = np.linalg.norm(reference - axis * (axis @ reference))
    raw = -(axis @ reference) * derivative - np.outer(axis, reference @ derivative)
    return tangent, (np.eye(3) - np.outer(tangent, tangent)) @ raw / length


def _axis_at(origin, basis, coordinates):
    raw = origin + basis @ coordinates
    length = np.linalg.norm(raw)
    axis = raw / length
    return axis, (np.eye(3) - np.outer(axis, axis)) @ basis / length


def _exponential(axis, angle):
    sine, cosine = np.sin(angle), np.cos(angle)
    return cosine * np.eye(3) + (1. - cosine) * np.outer(axis, axis) + sine * skew(axis)


def _exponential_axis_derivative(axis, angle, derivative):
    return np.stack([(1. - np.cos(angle)) * (np.outer(d, axis) + np.outer(axis, d))
                     + np.sin(angle) * skew(d) for d in derivative.T], axis=2)


def _left_inverse(vector):
    angle = np.linalg.norm(vector)
    matrix = skew(vector)
    coefficient = (1. / 12. + angle * angle / 720. if angle < 1e-5
                   else (1. - .5 * angle / np.tan(.5 * angle)) / (angle * angle))
    return np.eye(3) - .5 * matrix + coefficient * matrix @ matrix


def _vee(matrix):
    return .5 * np.array([matrix[2, 1] - matrix[1, 2],
                          matrix[0, 2] - matrix[2, 0],
                          matrix[1, 0] - matrix[0, 1]])


def _rotation_log(rotation):
    """Principal log without OpenCV's near-identity matrix-log dead zone."""
    vector = _vee(rotation)
    sine = np.linalg.norm(vector)
    cosine = np.clip(.5 * (np.trace(rotation) - 1.), -1., 1.)
    angle = np.arctan2(sine, cosine)
    if cosine > 0. and sine < 1e-5:
        return vector * (1. + angle * angle / 6. + 7. * angle ** 4 / 360.)
    if angle < np.pi - 1e-5:
        return vector * (angle / sine)
    import cv2
    return cv2.Rodrigues(rotation)[0].reshape(3)


@dataclass(frozen=True)
class FixedPanTilt:
    """Local shared-axis chart; four shared coordinates and two temporal angles.

    Derivative order throughout is [pan_axis_0, pan_axis_1, tilt_axis_0,
    tilt_axis_1, alpha, beta]. The axis signs and angular zeros are coordinate
    gauges, not measured installation facts. Rechart near a tangent pole.
    """
    pan_axis_world: tuple
    tilt_axis_camera: tuple

    def __post_init__(self):
        n, b = _unit(self.pan_axis_world), _unit(self.tilt_axis_camera)
        object.__setattr__(self, 'pan_axis_world', tuple(n))
        object.__setattr__(self, 'tilt_axis_camera', tuple(b))
        for name, axis in (('_pan', n), ('_tilt', b)):
            reference = _reference(axis)
            first = _tangent(axis, reference)
            object.__setattr__(self, name + '_reference', reference)
            object.__setattr__(self, name + '_basis', np.column_stack((first, np.cross(axis, first))))

    def installation(self, shared_axes=None):
        """Return n, b, R0 and their analytic derivatives in four shared columns."""
        s = np.zeros(4) if shared_axes is None else _finite(shared_axes, (4,), 'shared_axes')
        n, dn2 = _axis_at(np.asarray(self.pan_axis_world), self._pan_basis, s[:2])
        b, db2 = _axis_at(np.asarray(self.tilt_axis_camera), self._tilt_basis, s[2:])
        dn, db = np.zeros((3, 4)), np.zeros((3, 4))
        dn[:, :2], db[:, 2:] = dn2, db2
        t, dt = _tangent_with_derivative(n, dn, self._pan_reference)
        c, dc = _tangent_with_derivative(b, db, self._tilt_reference)
        w = np.column_stack((t, np.cross(n, t), n))
        v = np.column_stack((b, np.cross(c, b), c))
        dw = np.stack([np.column_stack((dt[:, i], np.cross(dn[:, i], t)
                        + np.cross(n, dt[:, i]), dn[:, i])) for i in range(4)], axis=2)
        dv = np.stack([np.column_stack((db[:, i], np.cross(dc[:, i], b)
                        + np.cross(c, db[:, i]), dc[:, i])) for i in range(4)], axis=2)
        r0 = v @ w.T
        derivative = np.stack([dv[:, :, i] @ w.T + v @ dw[:, :, i].T for i in range(4)], axis=2)
        return dict(n=n, b=b, R0=r0, dn=dn, db=db, dR0=derivative,
                    world_basis=w, camera_basis=v)

    def rotation(self, angles, shared_axes=None, *, jacobian=False):
        """Evaluate a world-to-camera R; optional dR has shape (3,3,6)."""
        alpha, beta = _finite(angles, (2,), 'pan_tilt_angles')
        installation = self.installation(shared_axes)
        n, b, r0 = (installation[k] for k in ('n', 'b', 'R0'))
        pan, tilt = _exponential(n, alpha), _exponential(b, beta)
        rotation = tilt @ r0 @ pan
        if not jacobian:
            return rotation
        dpan = _exponential_axis_derivative(n, alpha, installation['dn'])
        dtilt = _exponential_axis_derivative(b, beta, installation['db'])
        derivative = np.empty((3, 3, 6))
        for i in range(4):
            derivative[:, :, i] = (dtilt[:, :, i] @ r0 @ pan
                + tilt @ installation['dR0'][:, :, i] @ pan + tilt @ r0 @ dpan[:, :, i])
        derivative[:, :, 4] = rotation @ skew(n)
        derivative[:, :, 5] = skew(b) @ rotation
        return rotation, derivative

    def rotvec(self, angles, shared_axes=None, *, jacobian=False):
        """Rodrigues interface and analytic (3,6) chain for the existing camera."""
        result = self.rotation(angles, shared_axes, jacobian=jacobian)
        rotation, derivative = result if jacobian else (result, None)
        vector = _rotation_log(rotation)
        if not jacobian:
            return vector
        angular = np.column_stack([_vee(derivative[:, :, i] @ rotation.T) for i in range(6)])
        return vector, _left_inverse(vector) @ angular

    def inverse(self, rotation, shared_axes=None, *, project=False):
        """Recover angles; optional nearest rotation is a warmstart, never an exact parent.

        Projection removes only the shortest angular normal departure from the
        shared fixed-head surface. Report it explicitly; it generally changes
        source predictions. Angles are principal values; unwrap a sequence.
        """
        rotation = _finite(rotation, (3, 3), 'rotation')
        if (np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-9
                or abs(np.linalg.det(rotation) - 1.) > 1e-9):
            raise ValueError('A proper input rotation is required')
        installation = self.installation(shared_axes)
        n, b, r0 = (installation[k] for k in ('n', 'b', 'R0'))
        forward = rotation @ n
        defect = float(np.clip(b @ forward, -1., 1.))
        projected = forward - defect * b
        norm = np.linalg.norm(projected)
        if norm <= 64. * np.finfo(float).eps:
            raise ValueError('Nearest fixed-head warmstart is ambiguous at the normal pole')
        if not project and abs(defect) > 1e-10:
            raise ValueError('Input R is outside the fixed pan/tilt surface')
        projected /= norm
        if project:
            cross = np.cross(forward, projected)
            sine = np.linalg.norm(cross)
            if sine > 0.:
                rotation = _exponential(cross / sine, np.arctan2(sine, forward @ projected)) @ rotation
        c = r0 @ n
        beta = np.arctan2(projected @ np.cross(b, c), projected @ c)
        reduced = r0.T @ _exponential(b, -beta) @ rotation
        t, u = installation['world_basis'][:, :2].T
        alpha = np.arctan2(u @ reduced @ t, t @ reduced @ t)
        angles = np.array([alpha, beta])
        reconstructed = self.rotation(angles, shared_axes)
        return dict(angles=angles, rotation=reconstructed,
                    signed_normal_departure_radians=float(np.arcsin(defect)),
                    projection_used=bool(project),
                    reconstruction_max_abs=float(np.max(np.abs(reconstructed - rotation))))

    def interpolate(self, left, right, fraction, shared_axes=None, *, jacobian=False):
        """Interpolate unwrapped pan/tilt in native PTS, including RS extrapolation.

        Derivatives: shared axes4, left angles2, right angles2, fraction1.
        Caller supplies the existing PTS brackets and row-clock fraction.
        """
        left = _finite(left, (2,), 'left_unwrapped_angles')
        right = _finite(right, (2,), 'right_unwrapped_angles')
        if not np.isfinite(fraction):
            raise ValueError('A finite original PTS/row-clock fraction is required')
        angles = (1. - fraction) * left + fraction * right
        if not jacobian:
            return self.rotvec(angles, shared_axes)
        vector, derivative = self.rotvec(angles, shared_axes, jacobian=True)
        return vector, np.column_stack((derivative[:, :4],
            (1. - fraction) * derivative[:, 4:], fraction * derivative[:, 4:],
            derivative[:, 4:] @ (right - left)))

    def inverse_sequence(self, rotations, shared_axes=None, *, project=False,
                         anchor_angles=None):
        """Warmstart one ordered native-PTS sequence on a common angular branch.

        Adjacent equivalent angles are lifted to the nearest 2pi branch. A
        reference first angle can retain a previous chart's winding. Rotation
        matrices alone cannot determine unseen full turns between samples;
        this gauge choice is neither a speed bound nor a smoothing factor.
        The caller must supply the actual ordered PTS without aliases/reorder.
        """
        rotations = np.asarray(rotations, dtype=float)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or not len(rotations):
            raise ValueError('A nonempty actual-PTS ordered rotation sequence is required')
        recovered = [self.inverse(r, shared_axes, project=project) for r in rotations]
        principal = np.array([r['angles'] for r in recovered])
        unwrapped = np.unwrap(principal, axis=0)
        if anchor_angles is not None:
            anchor = _finite(anchor_angles, (2,), 'previous_first_angles')
            unwrapped += 2. * np.pi * np.rint((anchor - unwrapped[0]) / (2. * np.pi))
        return dict(angles=unwrapped, principal_angles=principal,
            signed_normal_departure_radians=np.array([r['signed_normal_departure_radians'] for r in recovered]),
            projected_rotations=np.array([r['rotation'] for r in recovered]),
            projection_used=bool(project),
            unobserved_full_turns_not_identifiable_from_rotation_samples=True)

    def common_perpendicular(self, angles, shared_axes=None, *, jacobian=False):
        """World h=n cross R.T b; unit only on this orthogonal-axis surface.

        It supplies the general fixed-head direction for an independently
        declared axis-separation centre model. No separation or centre is
        introduced here. Returned derivative order matches rotation().
        """
        result = self.rotation(angles, shared_axes, jacobian=jacobian)
        rotation, derivative = result if jacobian else (result, None)
        installation = self.installation(shared_axes)
        n, b = installation['n'], installation['b']
        direction = np.cross(n, rotation.T @ b)
        if not jacobian:
            return direction
        dn = np.column_stack((installation['dn'], np.zeros((3, 2))))
        db = np.column_stack((installation['db'], np.zeros((3, 2))))
        jac = np.column_stack([np.cross(dn[:, i], rotation.T @ b)
            + np.cross(n, derivative[:, :, i].T @ b + rotation.T @ db[:, i]) for i in range(6)])
        return direction, jac
