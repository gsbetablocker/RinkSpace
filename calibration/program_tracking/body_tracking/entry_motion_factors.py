"""Finite source-clock prior on actual root acceleration along the camera depth.

The horizontal direction is frozen from the reference and native camera at
construction. This is an engineering prior, not a speed/contact acceptance rule.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, hstack

from .skate_contact import trapezoid_time_weights


class EntryMotionFactors:
    """Integrate squared actual horizontal depth acceleration on one source phase.

    Config requires ``source_interval_seconds``, ``taper_seconds``,
    ``acceleration_scale_m_s2`` and ``weight``. The taper extends outside the
    interval. Explicit ``camera_centres_solver`` or ``horizontal_view_directions``
    may be supplied in reference-frame order or selected-frame order.
    """
    name = 'entry_actual_depth_acceleration'

    def __init__(self, problem, config):
        self.problem = problem
        self.state = problem.body.state
        self.reference = problem.reference
        self.config = dict(config)
        interval = np.asarray(config['source_interval_seconds'], float)
        taper = float(config['taper_seconds'])
        scale = float(config['acceleration_scale_m_s2'])
        weight = float(config['weight'])
        if (interval.shape != (2,) or not np.isfinite(interval).all()
                or interval[1] <= interval[0] or not np.isfinite(taper) or taper < 0
                or not np.isfinite(scale) or scale <= 0
                or not np.isfinite(weight) or weight < 0):
            raise ValueError('Invalid finite entry acceleration prior configuration')
        self.interval = interval.copy()
        self.ratio = float(problem.binding.ratio)
        if not np.isfinite(self.ratio) or self.ratio <= 0:
            raise ValueError('Entry acceleration requires the declared length export ratio')
        state = self.state
        self.body_size = int(state.size)
        if int(problem.size) != self.body_size + 17:
            raise ValueError('Expected 17 shared shoe parameters followed by body controls')
        active = np.asarray(state.active, int)
        if active.shape != (state.d,) or len(np.unique(active)) != state.d:
            raise ValueError('Body active column mapping must be unique')
        times = np.asarray(self.reference.times, float)
        pts = np.asarray(self.reference.p['pts'])
        source = pts.astype(float) / 12800.
        if (times.ndim != 1 or pts.shape != times.shape or len(times) < 2
                or not np.isfinite(times).all() or not np.isfinite(source).all()
                or np.any(np.diff(times) <= 0) or np.any(np.diff(source) <= 0)
                or not np.allclose(source - times, source[0] - times[0], rtol=0, atol=1e-9)):
            raise ValueError('Entry acceleration requires aligned increasing source-clock samples')
        self.frames = np.flatnonzero((source >= interval[0] - taper)
                                    & (source <= interval[1] + taper))
        if len(self.frames) < 2:
            raise ValueError('Entry acceleration phase needs at least two source samples')
        self.times = times[self.frames].copy()
        self.source_seconds = source[self.frames].copy()
        self.pts = pts[self.frames].copy()
        if taper:
            u = np.clip(np.minimum((self.source_seconds - interval[0] + taper) / taper,
                                   (interval[1] + taper - self.source_seconds) / taper), 0., 1.)
            self.phase_weights = u**3 * (10. - 15.*u + 6.*u*u)
        else:
            self.phase_weights = np.ones(len(self.times))
        for start,stop in config.get('exclude_source_intervals_seconds',()):
            self.phase_weights[(self.source_seconds>=start)&(self.source_seconds<=stop)]=0.
        self.dt = trapezoid_time_weights(self.times)
        self.row_weights = np.sqrt(weight * self.dt * self.phase_weights) / scale
        reference_P = np.asarray(self.reference.at(self.times)['P'], float)
        if reference_P.shape != (len(self.times), 3) or not np.isfinite(reference_P).all():
            raise ValueError('Nonfinite reference root at entry samples')
        if 'horizontal_view_directions' in config:
            h = self._selected(config['horizontal_view_directions'], len(times), 2)
            self.direction_source = 'explicit fixed horizontal reference directions'
        else:
            if 'camera_centres_solver' in config:
                centres = self._selected(config['camera_centres_solver'], len(times), 3)
                self.direction_source = 'explicit camera centres in solver length units'
            elif hasattr(getattr(problem.camera, 'core', None), 'evaluate_motion_camera'):
                from .boot_integration import camera_centres
                centres = np.asarray(camera_centres(problem.camera, self.frames), float)
                self.direction_source = 'frozen native camera_centres at source midpoints'
            else:
                p = getattr(problem.camera, 'p', {})
                if 'camera_centre' not in p:
                    p = self.reference.p
                if 'camera_centre' not in p or 'metres_per_scene_unit' not in p:
                    raise ValueError('Supply bound camera centres or fixed horizontal directions')
                centres = self._selected(p['camera_centre'], len(times), 3) * float(p['metres_per_scene_unit'])
                self.direction_source = 'bound camera_centre scene units converted to solver units'
            if centres.shape != reference_P.shape or not np.isfinite(centres).all():
                raise ValueError('Invalid bound camera centres')
            h = (reference_P - centres)[:, :2]
        norms = np.linalg.norm(h, axis=1)
        if not np.isfinite(h).all() or np.any(norms <= 0):
            raise ValueError('Cannot define horizontal reference-camera depth direction')
        self.h = (h / norms[:, None]).copy()
        B2 = np.asarray(state.basis(self.times, 2), float)
        reference_a = np.asarray(self.reference.root_derivative(self.times, 2), float)
        if (B2.shape != (len(self.times), state.m) or not np.isfinite(B2).all()
                or reference_a.shape != reference_P.shape or not np.isfinite(reference_a).all()):
            raise ValueError('Invalid actual-root second derivatives')
        # Retain the source root: smoothing only the correction would miss the defect.
        self.offset = np.einsum('ni,ni->n', self.h, reference_a[:, :2]) * self.ratio
        frame, knot = np.nonzero(B2)
        rows, columns, values = [], [], []
        for axis in (0, 1):
            location = np.flatnonzero(active == axis)
            if len(location):
                rows.append(frame)
                columns.append(knot * state.d + int(location[0]))
                values.append(B2[frame, knot] * self.h[frame, axis] * self.ratio)
        if rows:
            self.M = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                                shape=(len(self.times), self.body_size)).tocsr()
            self.M.eliminate_zeros()
        else:
            self.M = csr_matrix((len(self.times), self.body_size))
        self.J = hstack((csr_matrix((len(self.times), 17)),
                         diags(self.row_weights) @ self.M), format='csr')
        self.last = None

    def _selected(self, values, reference_length, width):
        values = np.asarray(values, float)
        if values.shape == (reference_length, width):
            return values[self.frames].copy()
        if values.shape == (len(self.frames), width):
            return values.copy()
        raise ValueError('Explicit camera/direction arrays must align with reference or phase samples')

    def evaluate(self, body_x, jacobian=False):
        x = np.asarray(body_x, float)
        if x.shape != (self.body_size,) or not np.isfinite(x).all():
            raise ValueError('Entry acceleration requires finite body controls only')
        acceleration = self.offset + self.M @ x
        residual = self.row_weights * acceleration
        self.last = dict(
            residual=residual, jac=self.J if jacobian else None,
            parts={self.name: float(residual @ residual / 2)},
            block_slices={self.name: slice(0, len(residual))},
            metrics=dict(
                source_interval_seconds=self.interval.tolist(),
                sampled_source_range_seconds=self.source_seconds[[0, -1]].tolist(),
                source_pts=self.pts.tolist(), samples=len(self.times),
                actual_depth_acceleration_m_s2=acceleration.tolist(),
                quadrature_seconds=self.dt.tolist(), phase_weights=self.phase_weights.tolist(),
                horizontal_directions=self.h.tolist(), direction_source=self.direction_source,
                length_export_ratio=self.ratio, vertical_untouched=True,
                finite_only=True, hard_speed_limit=False, contact_classification=False,
                reference_root_acceleration_included=True,
                prior_status='finite engineering prior; not an accuracy or contact acceptance rule'))
        return self.last
