"""Exact conditioned rig coordinates for a fixed or eccentric camera mount.

The graph still stores Vector11 frame variables, but their first three values
are unused dummies. Every original image, prior, guard and seam callback sees
the physical frame centre derived from one shared mount. This changes the old
1e-7-state Gaussian rig equality into an exact definition; it is a separate
conditional diagnostic objective, not a bitwise rewrite of that soft factor.
"""
from __future__ import annotations

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

try:
    from .projection import from_state
except ImportError:
    from projection import from_state


def _state(value, name):
    state = np.asarray(value, np.float64)
    if state.shape != (11,) or not np.isfinite(state).all():
        raise ValueError(f'{name} must be one finite conditioned Vector11')
    return state


def physical_frame(frame, mount, mode):
    """Return physical Vector11 and exact Jacobians wrt frame and mount."""
    frame, mount = _state(frame, 'frame'), _state(mount, 'mount')
    if mode not in ('fixed', 'pivot'):
        raise ValueError(f'Unknown rig mode: {mode}')
    physical = frame.copy()
    frame_jac = np.eye(11)
    frame_jac[:3, :3] = 0.
    mount_jac = np.zeros((11, 11))
    mount_jac[:3, :3] = np.eye(3)
    if mode == 'fixed':
        # d is an unused mount coordinate kept under its original prior.
        physical[:3] = mount[:3]
    else:
        rotation, derivative = cv2.Rodrigues(np.ascontiguousarray(frame[3:6]))
        if derivative.shape == (3, 9):
            drot = derivative.reshape(3, 3, 3)
        elif derivative.shape == (9, 3):
            drot = derivative.T.reshape(3, 3, 3)
        else:
            raise ValueError(f'Unexpected Rodrigues derivative shape: {derivative.shape}')
        physical[:3] = mount[:3] + rotation.T @ mount[3:6]
        mount_jac[:3, 3:6] = rotation.T
        for axis in range(3):
            frame_jac[:3, 3+axis] = drot[axis].T @ mount[3:6]
    return physical, frame_jac, mount_jac


def camera25(frame, mount, template, mode):
    """Restore the Brown25 camera with the exact derived optical centre."""
    return from_state(physical_frame(frame, mount, mode)[0], template)


def conditioned_spec(spec, active, states, frame_keys, mount_key, mode):
    """Wrap one captured original factor without dropping centre-sensitive rows.

    Its original prior/between/observation residual and original sigma/Huber
    options are retained. A mount key is appended once when active frames need
    it; all their centre Jacobian contributions accumulate into that one block.
    External states remain the saved constants from the full candidate.
    """
    original_keys = tuple(spec['keys'])
    active = set(active)
    frame_keys = set(frame_keys)
    kept = list(dict.fromkeys(key for key in original_keys if key in active))
    dependent_frames = frame_keys.intersection(kept)
    if dependent_frames:
        if mount_key not in active:
            raise ValueError('Derived frame requires one active shared mount')
        if mount_key not in kept:
            kept.append(mount_key)
    if not kept:
        raise ValueError('Derived conditioned factor has no active key')
    if spec['kind'] == 'prior':
        measurement = np.asarray(spec['args'][1], np.float64)
        sigma = spec['args'][2]
        original_residual = lambda q: q[0]-measurement
        original_jacobian = lambda q: [np.eye(11)]
        options = dict(residual_dim=11, sigma=sigma)
    elif spec['kind'] == 'between':
        delta = np.asarray(spec['args'][2], np.float64)
        sigma = spec['args'][3]
        original_residual = lambda q: q[1]-q[0]-delta
        original_jacobian = lambda q: [-np.eye(11), np.eye(11)]
        options = dict(residual_dim=11, sigma=sigma)
    elif spec['kind'] == 'observation':
        original_residual = spec['args'][1]
        options = spec['kwargs'].copy()
        original_jacobian = options.pop('jac_fn', None)
    else:
        raise ValueError(f"Unsupported captured factor kind: {spec['kind']}")
    position = {key: index for index, key in enumerate(kept)}

    def expand(q):
        values = dict(zip(kept, q))
        mount = values.get(mount_key)
        result = []
        for key in original_keys:
            if key in dependent_frames:
                result.append(physical_frame(values[key], mount, mode)[0])
            else:
                result.append(values[key] if key in values else states[key])
        return result

    def residual(q):
        return original_residual(expand(q))

    def jacobian(q):
        original = original_jacobian(expand(q))
        rows = int(options['residual_dim'])
        if len(original) != len(original_keys):
            raise ValueError('Original factor Jacobian omitted a state block')
        blocks = [np.zeros((rows, 11)) for _ in kept]
        values = dict(zip(kept, q))
        for key, matrix in zip(original_keys, original):
            if key not in active:
                continue
            matrix = np.asarray(matrix, np.float64)
            if matrix.shape != (rows, 11):
                raise ValueError('Original factor Jacobian block changed shape')
            if key in dependent_frames:
                _, frame_jac, mount_jac = physical_frame(values[key], values[mount_key], mode)
                blocks[position[key]] += matrix @ frame_jac
                blocks[position[mount_key]] += matrix @ mount_jac
            else:
                blocks[position[key]] += matrix
        return blocks

    return tuple(kept), residual, jacobian if original_jacobian is not None else None, options


def native_pair(pair, source_key, target_key, mount_key, mode):
    """Adapt unchanged actual UV pair callbacks to the exact shared mount."""
    keys = (source_key, target_key, mount_key)
    if len(set(keys)) != 3:
        raise ValueError('Native pair requires distinct frames and one shared mount')

    def physical(q):
        first, first_jac, first_mount = physical_frame(q[0], q[2], mode)
        second, second_jac, second_mount = physical_frame(q[1], q[2], mode)
        return (first, second, first_jac, second_jac, first_mount, second_mount)

    def residual(q):
        first, second, *_ = physical(q)
        return pair._raw_residual([first, second])

    def jacobian(q):
        first, second, j_first, j_second, m_first, m_second = physical(q)
        a, b = pair._raw_jacobians([first, second])
        return [a @ j_first, b @ j_second, a @ m_first + b @ m_second]

    return keys, residual, jacobian


def add_dummy_center_prior(graph, frame_key):
    """Keep each unused frame-centre slot anchored at zero, separate from data."""
    jacobian = np.zeros((3, 11))
    jacobian[:, :3] = np.eye(3)
    graph.add_observation([frame_key], lambda q: q[0][:3], residual_dim=3,
                          sigma=1., jac_fn=lambda q: [jacobian])
