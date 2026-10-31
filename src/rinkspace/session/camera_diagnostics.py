"""Read-only motion/optics diagnostics of the production native camera.

No camera formula or motion smoothing is implemented here. Lens probes use
the native forward with readout zeroed in a temporary diagnostic copy, so
optical deformation and rolling-shutter motion remain separate quantities.
"""
from __future__ import annotations

import numpy as np


def _extent(values):
    values = np.asarray(values)
    return dict(min=float(np.min(values)), max=float(np.max(values)),
                median=float(np.median(values)), p95_abs=float(np.quantile(abs(values), .95)))


def _peak(values, seconds):
    i = int(np.argmax(abs(values)))
    return dict(signed_value=float(values[i]), source_seconds=float(seconds[i]))


def inspect_shot_motion(core, rig, controls, first, dt, times, source_seconds,
                        metres_per_unit=None):
    """Inspect one uninterrupted shot, including its actual readout halo."""
    times = np.asarray(times, float)
    seconds = np.asarray(source_seconds, float)
    if (times.ndim != 1 or len(times) < 3 or times.shape != seconds.shape
            or not np.all(np.diff(times) > 0)
            or not np.allclose(np.diff(times), np.diff(seconds), rtol=0, atol=1e-10)):
        raise ValueError('One monotonic native shot clock is required')
    state = core.evaluate_motion_camera(rig, controls, first, dt, times)
    k = core.evaluate_motion_kinematics(controls, first, dt, times)
    if np.any(state['status']) or np.any(k['status']):
        raise ValueError('Invalid native camera/kinematics state')
    span = [times[0]-abs(rig[37])/2, times[-1]+abs(rig[37])/2]
    extrema = core.motion_derivative_extrema(controls, first, dt, span)
    channels = {}
    for c, name in enumerate(('pan', 'tilt', 'log_f')):
        factor = 180/np.pi if c < 2 else 1.
        channels[name] = dict(position=_extent(k['position'][:, c]*factor))
        for d, order in enumerate(('speed', 'acceleration', 'jerk')):
            values = k['values'][:, d+1, c]*factor
            channels[name][order] = dict(
                sampled=_extent(values), sampled_peak=_peak(values, seconds),
                continuous_peak=dict(signed_value=float(extrema['value'][d, c]*factor),
                    source_seconds=float(extrema['time'][d, c]+seconds[0]-times[0]),
                    native_cell=int(extrema['cell'][d, c])))
    rotations = state['rotation']
    rel = rotations[1:] @ rotations[:-1].transpose(0, 2, 1)
    angle = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2)-1)/2, -1, 1))
    angular_step = angle*180/np.pi
    centre = state['centre']
    velocity = np.gradient(centre, seconds, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, seconds, axis=0, edge_order=2)
    scale = 1. if metres_per_unit is None else float(metres_per_unit)
    centre_speed = np.linalg.norm(velocity, axis=1)*scale
    centre_acceleration = np.linalg.norm(acceleration, axis=1)*scale
    summary = dict(frame_count=len(times), source_interval=[float(seconds[0]), float(seconds[-1])],
        exposure_interval=[float(span[0]+seconds[0]-times[0]), float(span[1]+seconds[0]-times[0])],
        angular_units='degrees / seconds**derivative_order',
        zoom_units='log(f_pixels) / seconds**derivative_order',
        continuous_extrema_include_signed_readout=True,
        cubic_jerk_is_one_sided_at_knots=True, channels=channels,
        focal_pixels=_extent(state['fx']), focal_max_min_ratio=float(state['fx'].max()/state['fx'].min()),
        focal_frame_relative_step=_extent(np.diff(state['fx'])/state['fx'][:-1]),
        angular_frame_step_deg=_extent(angular_step),
        angular_frame_speed_peak=_peak(angular_step/np.diff(seconds), seconds[1:]),
        centre=dict(units='conditional metres' if metres_per_unit else 'scene units',
            source='native fixed pivot, articulated axes, lever and zoom pupil; no free per-frame translation',
            component_span=(np.ptp(centre, axis=0)*scale).tolist(),
            sampled_speed=_extent(centre_speed), sampled_speed_peak=_peak(centre_speed, seconds),
            sampled_acceleration=_extent(centre_acceleration),
            derivative_method='finite differences of native centres at original exposure times'),
        measured_hardware_speed_limits_available=False)
    arrays = dict(source_seconds=seconds, native_times=times, rotation=rotations, centre=centre,
                  fx=state['fx'], kinematics=k['values'], optics=state['values'][:, 16:24])
    return summary, arrays, state


def inspect_lens_forward(core, rig, controls, first, dt, times, source_seconds,
                         state, width, height, grid_shape=(9, 7), delta_px=.25):
    """Native optical probes over ideal sensor rays at every source exposure.

    Native domain checks include a sufficient no-fold certificate throughout
    the disk to each ray. A failed certificate is NOT evidence of a fold.
    Finite-difference Jacobians check the sampled map; they do not certify
    arbitrary rays outside the sampled field or unsampled zoom times.
    """
    if width < 2 or height < 2 or delta_px <= 0:
        raise ValueError('Invalid image domain or finite-difference increment')
    gx, gy = np.meshgrid(np.linspace(0, width-1, grid_shape[0]),
                         np.linspace(0, height-1, grid_shape[1]))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    offsets = np.array([[0, 0], [delta_px, 0], [-delta_px, 0], [0, delta_px], [0, -delta_px]])
    probes = grid[:, None, :]+offsets
    temporary_rig = np.array(rig, dtype=float, copy=True)
    temporary_rig[37] = 0.
    n = len(times)
    displaced = np.full((n, len(grid)), np.nan)
    determinants = np.full_like(displaced, np.nan)
    singular_min = np.full_like(displaced, np.nan)
    singular_max = np.full_like(displaced, np.nan)
    domain_failures = np.zeros((n, len(grid), len(offsets)), dtype=np.int32)
    for begin in range(0, n, 128):
        stop = min(n, begin+128)
        x = (probes[None, :, :, 0]-state['cx'][begin:stop, None, None])/state['fx'][begin:stop, None, None]
        y = (probes[None, :, :, 1]-state['cy'][begin:stop, None, None])/state['fy'][begin:stop, None, None]
        ray = np.stack((x, y, np.ones_like(x)), axis=-1)
        world = np.einsum('nji,nkpj->nkpi', state['rotation'][begin:stop], ray)
        world += state['centre'][begin:stop, None, None, :]
        t = np.broadcast_to(np.asarray(times)[begin:stop, None, None], world.shape[:-1])
        projected = core.project_motion(temporary_rig, controls, first, dt, height,
                                       t.ravel(), world.reshape(-1, 3))
        uv = projected['uv'].reshape(stop-begin, len(grid), len(offsets), 2)
        status = projected['status'].reshape(stop-begin, len(grid), len(offsets))
        domain_failures[begin:stop] = status
        displaced[begin:stop] = np.linalg.norm(uv[:, :, 0]-grid, axis=-1)
        jac = np.stack(((uv[:, :, 1]-uv[:, :, 2])/(2*delta_px),
                        (uv[:, :, 3]-uv[:, :, 4])/(2*delta_px)), axis=-1)
        valid = np.all(status == 0, axis=-1) & np.isfinite(jac).all(axis=(-1, -2))
        determinants[begin:stop][valid] = np.linalg.det(jac[valid])
        singular = np.linalg.svd(jac[valid], compute_uv=False)
        singular_max[begin:stop][valid] = singular[:, 0]
        singular_min[begin:stop][valid] = singular[:, 1]
    valid = np.isfinite(determinants)
    if not np.any(valid):
        raise ValueError('No valid lens probe')
    at = np.unravel_index(np.nanargmax(displaced), displaced.shape)
    minimum = np.unravel_index(np.nanargmin(determinants), determinants.shape)
    summary = dict(kind='effective optical deformation at exposure midpoints, no rolling shutter',
        temporary_readout_zero_only_for_lens_diagnostic=True,
        domain='ideal sensor rectangle; distorted ray endpoints can fall outside the image',
        grid_shape=list(grid_shape), exposure_count=n, step_pixels=float(delta_px),
        native_domain_failure_probes=int(np.count_nonzero(domain_failures)),
        native_domain_failure_frames=int(np.count_nonzero(np.any(domain_failures, axis=(1, 2)))),
        native_domain_status_counts={str(int(code)):int(np.count_nonzero(domain_failures == code))
                                     for code in np.unique(domain_failures)},
        maximum_displacement_pixels=float(displaced[at]),
        maximum_displacement_at=dict(source_seconds=float(source_seconds[at[0]]), ideal_uv=grid[at[1]].tolist()),
        minimum_sampled_jacobian_determinant=float(determinants[minimum]),
        minimum_jacobian_at=dict(source_seconds=float(source_seconds[minimum[0]]), ideal_uv=grid[minimum[1]].tolist()),
        minimum_sampled_singular_value=float(np.nanmin(singular_min)),
        maximum_sampled_singular_value=float(np.nanmax(singular_max)),
        nonpositive_sampled_jacobians=int(np.count_nonzero(determinants[valid] <= 0)),
        mathematical_optical_domain_certificate_on_all_probes=not bool(np.any(domain_failures)))
    arrays = dict(lens_grid=grid, lens_displacement_px=displaced,
                  lens_jacobian_determinant=determinants, lens_probe_status=domain_failures)
    return summary, arrays
