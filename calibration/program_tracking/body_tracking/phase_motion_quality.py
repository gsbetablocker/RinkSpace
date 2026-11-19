"""Comparable source-clock measurements for a saved continuous motion."""
import numpy as np

from .skate_contact import trapezoid_time_weights


def flight_stencil_rows(pts, episode):
    inside = (pts >= episode['start_pts']) & (pts <= episode['end_pts_inclusive'])
    return np.flatnonzero(inside[:-2] & inside[1:-1] & inside[2:])


def flight_acceleration_metrics(times, pts, acceleration, episode, gravity):
    """Evaluate only stencils wholly inside this one reviewed flight."""
    pts, times = np.asarray(pts), np.asarray(times)
    rows = flight_stencil_rows(pts, episode)
    if len(acceleration) != len(times)-2 or len(pts) != len(times) or not len(rows):
        raise ValueError('Flight evaluation requires aligned complete stencils')
    measured = np.asarray(acceleration)[rows]
    error = np.linalg.norm(measured-gravity, axis=1)
    weight = trapezoid_time_weights(times)[1:-1][rows]
    order = np.argsort(error)
    last = order[-1]
    tail = min(np.searchsorted(np.cumsum(weight[order]), .95*weight.sum()), len(error)-1)
    summary = dict(episode_id=episode['episode_id'], derivative_rows=len(rows),
        rms_gravity_error_mps2=float(np.sqrt(np.average(error**2, weights=weight))),
        p95_gravity_error_mps2=float(error[order[tail]]), max_gravity_error_mps2=float(error[last]),
        worst_source_pts=float(pts[1:-1][rows[last]]), duration_weight_s=float(weight.sum()))
    return summary, dict(source_pts=pts[1:-1][rows].tolist(),
                         acceleration_mps2=measured.tolist(), gravity_error_mps2=error.tolist())


def joint_domain_changes(pts, names, previous, current, *, native_period_pts=256, time_base=(1,12800)):
    """Report source events individually, including equal-count exchanges."""
    old, new = np.asarray(previous), np.asarray(current)
    if old.shape != new.shape or old.shape != (len(pts), len(names), 3):
        raise ValueError('Different anatomical chart or source population')
    result = {}
    for j, name in enumerate(names):
        before, after = old[:, j], new[:, j]
        a, b = np.any(before > 0., axis=1), np.any(after > 0., axis=1)

        def entries(which, values):
            return [dict(pts=int(pts[i]), components=values[i].tolist())
                    for i in np.flatnonzero(which)]

        def spans(which):
            selected=np.flatnonzero(which)
            if not len(selected):
                return []
            runs=np.split(selected,np.flatnonzero(np.diff(np.asarray(pts)[selected])!=native_period_pts)+1)
            return [dict(first_pts=int(pts[r[0]]),last_pts=int(pts[r[-1]]),exposures=len(r),
                         exposure_duration_seconds=len(r)*native_period_pts*time_base[0]/time_base[1]) for r in runs]

        result[str(name)] = dict(before_count=int(a.sum()), after_count=int(b.sum()),
            new=entries(~a & b, after), resolved=entries(a & ~b, before),
            new_spans=spans(~a & b),resolved_spans=spans(a & ~b),
            increased_components=entries(np.any(after > before+1.e-9, axis=1), after-before),
            before_peak=float(before.max(initial=0.)), after_peak=float(after.max(initial=0.)))
    return result
