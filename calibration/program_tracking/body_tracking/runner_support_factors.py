"""Finite runner-mode priors on the existing body and fixed skate geometry.

A supported phase can prefer an interior runner witness over walking on a
longitudinal end/pick. This is a conditional mechanical prior, not a measured
contact point, force, flat-blade requirement or calibrated ice-height band.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from .skate_contact import trapezoid_time_weights
from .skate_template import _profile, profile_height
from .skate_physics import batch_features, fk_state
from .runner_interior import runner_interior_band, runner_longitudinal_coordinate, longitudinal_barrier
from .blade_contact_modes import blade_axes_diagnostics, inner_edge_y_sign, source_edge_schedule


def runner_tangent_limits(parameters):
    main, front, _ = _profile(parameters)
    def tangent(arc, theta):
        return -np.sin(theta)*np.asarray(arc['axis_cos'])+np.cos(theta)*np.asarray(arc['axis_sin'])
    rear_t = tangent(main, main['theta_start_rad'])
    front_t = tangent(front, front['theta_end_rad'])
    scale = .5*np.linalg.norm(front_t-rear_t)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Degenerate fixed runner tangent span')
    return rear_t, front_t, float(scale)


def runner_mode_terms(normal, runner_minus_pick_m, rear_t, front_t, normal_scale, gap_scale_m, weights):
    """Three finite hinge rows; bank/roll is free within the runner normal fan.

    The longitudinal endpoints must not be strict minima on an interior-runner
    phase. The third row prefers runner steel no higher than the toe head.
    No world origin/height enters these translation-invariant terms.
    """
    normal = np.asarray(normal, float)
    raw = np.c_[normal@rear_t, -normal@front_t, runner_minus_pick_m]
    scale = np.sqrt(np.asarray(weights, float))[:, None]/np.array([normal_scale, normal_scale, gap_scale_m])
    residual = scale*np.maximum(raw, 0.)
    D = scale*(raw > 0)
    return residual, D, raw


def source_runner_samples(pts, seconds, policy):
    """Use actual native samples only; each source-supported interval owns dt."""
    pts = np.asarray(pts, dtype=np.int64)
    samples = []; seen = set()
    if policy.get('time_base') != [1, 12800]:
        raise ValueError('Runner-mode policy must use the original native time base')
    stride = policy.get('quadrature_stride', 2)
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError('Positive integer quadrature stride required')
    weight = float(policy['objective_weight'])
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Runner-mode prior needs a finite positive weight')
    lookup = set(pts.tolist())
    for row in policy['intervals']:
        if row.get('state') != 'supported' or row.get('role') != 'fit' or not row.get('optimization_enabled', False):
            continue
        if row.get('mode') != 'runner_interior' or row.get('foot') not in (0, 1):
            raise ValueError('Explicit fixed foot and runner mode required')
        if not row.get('source_evidence_pts') or any(p not in lookup for p in row['source_evidence_pts']):
            raise ValueError('Runner-mode prior has no recoverable native source evidence')
        mask = (pts >= row['start_pts']) & (pts <= row['end_pts']) & (pts/12800. >= seconds[0]) & (pts/12800. <= seconds[1])
        selected = np.flatnonzero(mask)[::stride]
        # Add endpoints only when they are real native PTS inside this context.
        selected = np.unique(np.r_[selected, np.flatnonzero(mask & np.isin(pts, [row['start_pts'], row['end_pts']]))])
        if not len(selected):
            continue
        dt = trapezoid_time_weights(pts[selected]/12800., singleton_duration_s=256/12800.)
        for i, w in zip(selected, dt):
            identity = (int(pts[i]), int(row['foot']))
            if identity in seen:
                raise ValueError('Overlapping runner-mode intervals would duplicate an exposure/foot')
            seen.add(identity)
            samples.append((int(i), int(row['foot']), float(w*weight), row['id']))
    return samples


class RunnerSupportFactors:
    def __init__(self, problem, policy):
        if problem.optimize_shared:
            raise ValueError('Runner-mode correction requires frozen shared footwear')
        self.problem = problem; self.body = problem.body; self.ref = problem.reference
        self.binding = problem.binding; self.policy = policy
        document = problem.reference.source_document
        if policy.get('source_id') != document['source_id'] or policy.get('shot_id') != document['shot_id']:
            raise ValueError('Runner-mode policy source/shot identity mismatch')
        samples = source_runner_samples(self.ref.p['pts'], self.body.state.seconds, policy)
        self.frames = np.array([s[0] for s in samples], int)
        self.feet = np.array([s[1] for s in samples], int)
        self.weights = np.array([s[2] for s in samples])
        self.interval_ids = [s[3] for s in samples]
        self.pts = self.ref.p['pts'][self.frames]
        self.times = self.ref.times[self.frames]
        self.base = self.ref.at(self.times) if len(samples) else None
        self.B = self.body.state.basis(self.times) if len(samples) else None
        p = self.binding.packet()['parameters']
        self.rear_t, self.front_t, self.normal_scale = runner_tangent_limits(p)
        self.parameters = p
        self.gap_scale = p.blade_web_height_m*self.binding.ratio
        self.u_band, self.u_scale = runner_interior_band(p, policy.get('interior_margin'))
        self.inner_sides = inner_edge_y_sign(self.ref, self.binding)
        self.edge_schedule = source_edge_schedule(self.pts, self.feet, policy)
        modes = self.edge_schedule['modes']
        known = np.isin(modes, ['inside_edge', 'outside_edge'])
        if np.any(known & (self.inner_sides[self.feet] == 0)):
            raise ValueError('Source edge identity cannot be mapped to the frozen footwear frame')
        self.edge_sides = np.where(modes == 'inside_edge', self.inner_sides[self.feet], np.where(modes == 'outside_edge', -self.inner_sides[self.feet], 0))
        self.block_indices = {name: (np.arange(len(samples))[:, None]*5+columns).ravel() for name, columns in [('normal_fan', np.array([0, 1])), ('gap', np.array([2])), ('u', np.array([3])), ('edge', np.array([4]))]}
        if self.gap_scale <= 0:
            raise ValueError('Positive fixed blade web height required')
        xx = np.array([-p.runner_rear_extension_m, .5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m), p.sole_length_m-p.runner_front_inset_m])
        self.stations = np.c_[xx, np.zeros(3), profile_height(xx, p)]
        self.x_domain = xx[[0, 2]]

    def evaluate(self, x, jacobian=False):
        n = len(self.frames)
        if not n:
            return dict(residual=np.empty(0), jac=csr_matrix((0, self.problem.size)) if jacobian else None, parts={'runner_support_mode': 0.}, metrics={'runner_support_mode': []})
        d = self.body.state.increments(x, self.times)
        s = fk_state(self.ref, self.base, d)
        features = batch_features(self.binding, s['R'], s['O'])
        eligible = set(self.binding.packet()['template']['support_feature_ids']['glide'])
        runner_ids = np.array([i for i, name in enumerate(features['ids']) if name in eligible])
        pick_ids = np.array([i for i, name in enumerate(features['ids']) if features['semantics'][i] == 'toe_pick'])
        ix = np.arange(n); foot = self.feet
        hh = features['height'][ix, foot]
        rb = runner_ids[np.argmin(hh[:, runner_ids], axis=1)]
        pb = pick_ids[np.argmin(hh[:, pick_ids], axis=1)]
        q_runner = features['local_point'][ix, foot, rb]
        q_pick = features['local_point'][ix, foot, pb]
        normal = s['R'][ix, foot, 2, :]
        gap = hh[ix, rb]-hh[ix, pb]
        legacy, D, raw = runner_mode_terms(normal, gap, self.rear_t, self.front_t, self.normal_scale, self.gap_scale, self.weights)
        coordinate = runner_longitudinal_coordinate(normal, self.parameters, q_runner[:, 0])
        ur, u_derivative = longitudinal_barrier(coordinate, self.u_band, self.u_scale, self.weights)
        edge_raw = self.edge_sides*normal[:, 1]
        edge_w = np.sqrt(self.weights)
        er = edge_w*np.maximum(edge_raw, 0.)
        edge_derivative = edge_w*(edge_raw > 0)*self.edge_sides
        r = np.c_[legacy, ur, er]
        J = None
        if jacobian:
            def column(active):
                # Existing local-FK derivative path, then exact spline chain.
                h = 1e-6; a = d.copy(); b = d.copy(); a[:, active] += h; b[:, active] -= h
                sa = fk_state(self.ref, self.base, a); sb = fk_state(self.ref, self.base, b)
                dn = (sa['R'][ix, foot, 2, :]-sb['R'][ix, foot, 2, :])/(2*h)
                dr = np.c_[dn@self.rear_t, -dn@self.front_t, self.binding.ratio*np.sum(dn*(q_runner-q_pick), axis=1)]
                return np.c_[D*dr, np.sum(u_derivative*dn, axis=1), edge_derivative*dn[:, 1]]
            local = np.stack(list(self.body.execution.map_columns(column, self.body.state.active)), -1)
            if hasattr(d, 'chain'):
                body_J = d.chain(local)
            else:
                body_J = csr_matrix(np.einsum('nkd,nm->nkmd', local, self.B).reshape(5*n, self.body.size))
            if callable(getattr(getattr(d, 'owner', None), 'shoe_chain', None)):
                from .sequence_skate_state import perturb_relative
                shoe_columns = []
                for shoe in range(2):
                    for axis in range(3):
                        h = 1e-6
                        a = perturb_relative(d, shoe, axis, h); b = perturb_relative(d, shoe, axis, -h)
                        sa = fk_state(self.ref, self.base, a); sb = fk_state(self.ref, self.base, b)
                        dn = (sa['R'][ix, foot, 2, :]-sb['R'][ix, foot, 2, :])/(2*h)
                        dr = np.c_[dn@self.rear_t, -dn@self.front_t, self.binding.ratio*np.sum(dn*(q_runner-q_pick), axis=1)]
                        shoe_columns.append(np.c_[D*dr, np.sum(u_derivative*dn, axis=1), edge_derivative*dn[:, 1]])
                body_J = body_J+d.owner.shoe_chain(np.stack(shoe_columns, -1), d.parameters, d.sample_times)
            J = hstack((csr_matrix((5*n, 17)), body_J), format='csr')
        station_h = (s['O'][ix, foot, 2, None]+normal@self.stations.T)*self.binding.ratio
        axes = blade_axes_diagnostics(s['R'][ix, foot])
        metrics = []
        for i in range(n):
            u = float(coordinate['actual_u'][i])
            regime = 'runner_front_endpoint' if u >= 1-1e-10 else ('runner_rear_endpoint' if u <= 1e-10 else 'runner_interior')
            conflict = []
            if regime != 'runner_interior':conflict.append(regime)
            if not self.u_band[0] <= u <= self.u_band[1]:conflict.append('outside_finite_interior_margin')
            if edge_raw[i] > 0:conflict.append('source_edge_mode_conflicts_with_bank')
            metrics.append(dict(pts=int(self.pts[i]), foot=int(foot[i]), interval_id=self.interval_ids[i],
                longitudinal_normal_violations=raw[i, :2].tolist(), runner_minus_toe_pick_m=float(gap[i]),
                runner_min_height_m=float(hh[i, rb[i]]), toe_pick_min_height_m=float(hh[i, pb[i]]),
                runner_local_x_fraction=u, stationary_effective_u=float(coordinate['effective_u'][i]),
                endpoint_restoration_u=float(coordinate['endpoint_extension_u'][i]),
                u_target_interval=self.u_band.tolist(), geometric_runner_regime=regime,
                predicted_contact_mode='runner_interior', edge_mode=str(self.edge_schedule['modes'][i]),
                geometric_minimum_edge_side=features['ids'][rb[i]].rsplit('_', 1)[0],
                mode_conflicts=conflict, runner_feature=features['ids'][rb[i]],
                rear_middle_front_height_m=station_h[i].tolist(),
                **{name: np.asarray(value[i]).tolist() for name, value in axes.items()},
                finite_prior_not_contact_truth=True, exact_material_contact='unknown'))
        flat = r.ravel()
        transition_cost = float(self.edge_schedule['transition_cost'])
        if transition_cost > 0:
            flat = np.r_[flat, np.sqrt(2*transition_cost)]
            if jacobian:J = vstack((J, csr_matrix((1, self.problem.size))), format='csr')
        costs = {name: float(flat[indices]@flat[indices]/2) for name, indices in self.block_indices.items()}
        costs['edge_switch'] = transition_cost
        self.last = dict(residual=flat, jac=J, parts={'runner_support_mode': float(flat@flat/2)},
            metrics={'runner_support_mode': metrics, 'runner_support_mode_costs': costs,
                     'runner_support_mode_normal_scale': self.normal_scale, 'runner_support_mode_gap_scale_m': self.gap_scale,
                     'runner_support_mode_u_band': self.u_band.tolist(), 'runner_support_mode_u_scale': self.u_scale,
                     'runner_support_mode_rows': len(flat), 'runner_support_mode_is_conditional_prior': True,
                     'runner_edge_mode_switches': self.edge_schedule['switches'],
                     'runner_edge_mode_switch_cost': self.edge_schedule['transition_cost'],
                     'endpoint_extension_is_optimization_restoration_not_material_point': True})
        return self.last
