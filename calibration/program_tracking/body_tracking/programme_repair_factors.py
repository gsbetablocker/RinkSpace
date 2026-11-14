"""Local source-supported curvature and all-component ice on the main state."""
import numpy as np
from scipy.sparse import csr_matrix, diags, hstack, kron, vstack

from .skate_contact import trapezoid_time_weights
from .skate_physics import (NORMALIZATION, batch_features, body_constraint_jacobian,
                            fk_state)
from .weighted_blocks import block, full_columns, signature


def sampled_windows(reference, state, clock, windows):
    """Use full spline support and the declared native and midpoint clock."""
    pts = reference.p['pts']
    spans = np.asarray(windows, dtype=np.int64)
    rows = np.searchsorted(pts, spans)
    np.testing.assert_array_equal(pts[rows], spans)
    windows = reference.times[rows]
    active = np.any((state.knots[:-4,None] < windows[:,1]) &
                    (state.knots[4:,None] > windows[:,0]), axis=1)
    return np.asarray(state.basis(clock)[:, active].getnnz(axis=1)>0)


def component_ice(binding, pose, *, geometry=False):
    """Raw signed height for the continuous runner, picks and separate solids."""
    features = batch_features(binding, pose['R'], pose['O'], pose)
    template = binding.packet()['template']
    normal, origin = pose['R'][...,2,:], pose['O'][...,2]
    names, heights, points = [], [], []
    ids = np.arange(len(origin))[:,None], np.arange(2)[None,:]
    for name, choose in (
        ('runner', [k == 'runner' for k in features['semantics']]),
        ('toe_pick', [k == 'toe_pick' for k in features['semantics']]),
    ):
        index = np.flatnonzero(choose)
        witness = index[np.argmin(features['height'][...,index], axis=-1)]
        names.append(name)
        heights.append(features['height'][ids + (witness,)])
        points.append(features['local_point'][ids + (witness,)])
    for name, (start, stop) in template['part_ranges'].items():
        if name in ('steel_runner','toe_pick_head'):
            continue
        vertices = template['vertices'][start:stop]
        ix = np.argmin(normal @ vertices.T, axis=-1)
        point = vertices[ix]
        heights.append((origin + np.sum(normal*point, axis=-1))*binding.ratio)
        points.append(point)
        names.append(name)
    names.append('upper')
    heights.append(features['height'][...,-1])
    points.append(features['local_point'][...,-1,:])
    return (tuple(names), np.stack(heights,axis=-1), np.stack(points,axis=-2),
            features['upper_index']) if geometry else (tuple(names),np.stack(heights,axis=-1))


class ProgrammeComponentIce:
    """Finite component residuals, including for a foot with unknown contact."""
    name = 'programme_component_ice'

    def __init__(self, problem, spec):
        self.problem = problem
        self.ref = problem.reference
        self.state = problem.body.state
        clock = problem.physics_times
        keep = sampled_windows(self.ref, self.state, clock, spec.get('component_ice_pts', spec['focus_pts']))
        self.times = clock[keep]
        self.base = self.ref.at(self.times)
        self.measure = trapezoid_time_weights(clock)[keep]
        self.policy = spec['ice']

    def evaluate(self, x, jacobian=False):
        p = self.problem
        d = self.state.increments(x, self.times)
        pose = fk_state(self.ref, self.base, d)
        names, height, points, upper = component_ice(p.binding, pose, geometry=True)
        self.last_height = height
        self.part_names = names
        allowance = np.array([self.policy['tolerance_m'].get(k,self.policy['tolerance_m']['solid']) for k in names])
        scale = self.policy['scale_m']
        weight = np.sqrt(self.measure)[:,None,None]/scale
        excess = -height - allowance
        residual = (np.maximum(excess, 0.)*weight).ravel()
        J = None
        if jacobian:
            frames, feet, parts = np.nonzero(excess>0)
            row = (frames*2+feet)*len(names)+parts
            e = dict(rows=list(zip(frames, feet, points[frames,feet,parts], np.ones(len(row)), strict=True)),
                     times=self.times,
                     row_upper_vertex=np.where(parts==len(names)-1,upper[frames,feet],-1))
            M = body_constraint_jacobian(self.ref,pose,d,self.state.active,
                                         self.state.basis(self.times),e)
            scatter = csr_matrix((-NORMALIZATION*weight[frames,0,0],
                                  (row,np.arange(len(row)))),shape=(len(residual),len(row)))
            J = hstack((csr_matrix((len(residual),17)),scatter@M),format='csr')
        meta = dict(parts=names, tolerance_m=allowance.tolist(), scale_m=scale,
                    samples=len(self.times), minimum_m=height.min(axis=0).tolist())
        b = block(self.name,residual,J,meta,signature(self.times,allowance))
        return dict(residual=residual,jac=J,parts={self.name:float(residual@residual/2)},
                    metrics={self.name:meta},weighted_blocks=[b])


class ProgrammeOpeningLunge:
    """User-specified support and anatomical lead while the skates are out of frame."""

    support_name = 'opening_lunge_two_skate_support'
    lead_name = 'opening_lunge_right_foot_lead'

    def __init__(self, problem, config):
        from .body_clock import time_base

        self.ref = problem.reference
        self.state = problem.body.state
        self.config = config
        if not self.ref.ice_plane_known:
            raise ValueError('Opening lunge requires the calibrated ice reference')
        pts = self.ref.p['pts']
        tb = time_base(self.ref.p)
        if tuple(config['time_base']) != tuple(tb):
            raise ValueError('Opening lunge must use the native source clock')
        scales = np.asarray([config['height_scale_m'], config['lead_scale_m']], float)
        bounds = np.asarray(config['lead_range_m'], float)
        if (not np.isfinite(scales).all() or np.any(scales <= 0) or
                bounds.shape != (2,) or not np.isfinite(bounds).all() or
                not 0 < bounds[0] < bounds[1] or not np.isfinite(config['target_height_m'])):
            raise ValueError('Opening lunge needs positive scales and ordered lead bounds')
        clock = problem.physics_times
        measure = trapezoid_time_weights(clock)
        clock_pts = pts[0] + (clock - self.ref.times[0]) * tb[1] / tb[0]

        def in_span(span):
            if len(span) != 2 or span[0] >= span[1] or not np.isin(span, pts).all():
                raise ValueError('Opening lunge needs ordered native source PTS')
            return (clock_pts >= span[0]-1.e-5) & (clock_pts <= span[1]+1.e-5)

        left = in_span(config['left_support_pts'])
        right = in_span(config['right_support_pts'])
        lead = in_span(config['right_lead_pts'])
        standing = in_span(config['standing_pts'])
        keep = left | right | lead
        self.times = clock[keep]
        self.base = self.ref.at(self.times)
        self.sample, self.foot = np.nonzero(np.c_[left[keep], right[keep]])
        self.standing = standing[keep][self.sample]
        self.lead_samples = np.flatnonzero(lead[keep])
        self.support_weight = np.sqrt(measure[keep][self.sample])/config['height_scale_m']
        self.lead_weight = np.sqrt(measure[keep][self.lead_samples])/config['lead_scale_m']
        model = self.ref.body.J
        forward = np.cross(model[1]-model[2], model[12]-model[0])
        forward[1] = 0.
        self.forward = forward/np.linalg.norm(forward)
        self.association = signature(config, self.times, self.sample, self.foot, self.lead_samples)

    def lead(self, pose):
        sel = self.lead_samples
        forward = (pose['Q'][sel] @ self.forward)[:, :2]
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        ankle = pose['native_world'][sel, 8, :2] - pose['native_world'][sel, 7, :2]
        return np.sum(ankle*forward, axis=1)*self.ref.binding.ratio

    def evaluate(self, x, jacobian=False):
        d = self.state.increments(x, self.times)
        pose = fk_state(self.ref, self.base, d)
        f = batch_features(self.ref.binding, pose['R'], pose['O'])
        sample, foot = self.sample, self.foot
        runner = np.asarray(f['semantics']) == 'runner'
        eligible = runner[None] | (self.standing[:, None] & np.asarray(f['eligible'])[None])
        height = f['height'][sample, foot]
        witness = np.argmin(np.where(eligible, height, np.inf), axis=1)
        ix = np.arange(len(sample))
        height = height[ix, witness]
        support = (height-self.config['target_height_m'])*self.support_weight
        lead = self.lead(pose)
        lower, upper = self.config['lead_range_m']
        lead_residual = (lead-np.clip(lead, lower, upper))*self.lead_weight

        support_J = lead_J = None
        if jacobian:
            points = f['local_point'][sample, foot, witness]
            rows = list(zip(sample, foot, points, np.ones(len(sample)), strict=True))
            D = body_constraint_jacobian(self.ref, pose, d, self.state.active,
                self.state.basis(self.times), dict(rows=rows, times=self.times))
            support_J = diags(self.support_weight*NORMALIZATION) @ D
            # The ankle positions depend on Q, the hip and the knee. The
            # shared root position cancels in the signed fore-aft difference.
            columns = np.r_[3:6, 6:12, 15:21]
            local = np.zeros((len(lead), 1, 69))
            for col in columns:
                eps = 1.e-6
                plus, minus = d.copy(), d.copy()
                plus[:,col] += eps
                minus[:,col] -= eps
                a = self.lead(fk_state(self.ref, self.base, plus))
                b = self.lead(fk_state(self.ref, self.base, minus))
                local[:,0,col] = (a-b)/(2*eps)
            weight = self.lead_weight*((lead < lower) | (lead > upper))
            lead_J = d.chain(local*weight[:,None,None], sample_indices=self.lead_samples)
        meta = dict(self.config, image_contact_observed=False, world_xy_free=True)
        terms = [
            block(self.support_name, support, support_J, meta, self.association),
            block(self.lead_name, lead_residual, lead_J, meta, self.association),
        ]
        terms = [full_columns(term) for term in terms]
        result = dict(height_range_mm=[(float(height[foot==k].min()*1000), float(height[foot==k].max()*1000))
                                       for k in (0,1)],
                      right_lead_range_m=[float(lead.min()), float(lead.max())],
                      image_contact_observed=False)
        return dict(residual=np.r_[support,lead_residual],
                    jac=vstack([term.jacobian for term in terms],format='csr') if jacobian else None,
                    parts={term.name:float(term.residual@term.residual/2) for term in terms},
                    metrics=dict(opening_lunge=result), weighted_blocks=terms)


class ProgrammeRootCurvature:
    """Analytic cubic root acceleration over the opening's full support."""
    name = 'programme_root_curvature'

    def __init__(self, problem, spec):
        self.state = problem.body.state
        self.ref = problem.reference
        clock = problem.physics_times
        keep = sampled_windows(self.ref,self.state,clock,[spec['opening_pts']])
        scale = np.full(len(clock), spec['opening_acceleration_scale_mps2'])
        if 'camera_cut_continuity' in spec:
            from .continuous_motion_factors import reviewed_cut_weight
            cut_weight = reviewed_cut_weight(self.ref,clock)
            active = cut_weight > 1.
            keep |= active
            scale[active] /= np.sqrt(cut_weight[active])
        self.times = problem.physics_times[keep]
        self.weight = (np.sqrt(trapezoid_time_weights(problem.physics_times)[keep])
                       * self.ref.binding.ratio / scale[keep])
        self.D = diags(self.weight) @ self.state.basis(self.times,2)
        self.size = problem.size

    def evaluate(self, x, jacobian=False):
        curve = self.state.current_curve(x)
        residual = (self.D@curve.position_controls).ravel()
        J = None
        if jacobian:
            body = kron(self.D,csr_matrix(np.c_[np.eye(3),np.zeros((3,66))]),format='csr')
            J = hstack((csr_matrix((len(residual),17)),body,
                        csr_matrix((len(residual),self.size-17-body.shape[1]))),format='csr')
        b = block(self.name,residual,J,dict(unit='m/s2',samples=len(self.times)),
                  signature(self.times,self.weight))
        return dict(residual=residual,jac=J,parts={self.name:float(residual@residual/2)},
                    metrics={},weighted_blocks=[b])


def repair_direction_check(problem, primary, x):
    """Check the new residual derivatives on actual contact and opening samples."""
    from copy import copy

    state = problem.body.state
    rng = np.random.default_rng(1113)
    receipt = {}
    for factor in primary.factors:
        if isinstance(factor, ProgrammeComponentIce):
            probe = copy(factor)
            # Check the deepest active witness per foot on the actual
            # native/readout/midpoint clock.
            minimum = factor.last_height.min(axis=-1)
            indices = np.unique(np.argmin(minimum,axis=0))
            probe.times = factor.times[indices]
            probe.base = probe.ref.at(probe.times)
            probe.measure = factor.measure[indices]
        elif isinstance(factor, ProgrammeRootCurvature):
            probe = factor
        else:
            continue
        direction = np.zeros(state.size)
        controls = np.unique(state.basis(probe.times).indices)
        columns = (controls[:,None] * 69 + np.arange(30)).ravel()
        direction[columns] = rng.normal(size=len(columns))
        direction /= np.linalg.norm(direction)
        centre = probe.evaluate(x,True)
        analytic = centre['jac'][:,17:] @ direction
        h = 2e-6
        a = probe.evaluate(x+h*direction)['residual']
        b = probe.evaluate(x-h*direction)['residual']
        numeric = (a-b)/(2*h)
        norm = np.linalg.norm(numeric)
        relative = float(np.linalg.norm(analytic-numeric)/max(norm, 1e-12))
        if not np.isfinite(relative) or norm < 1e-9 or relative > 2e-4:
            raise ValueError(f'Actual {factor.name} directional derivative check failed: {relative}')
        receipt[factor.name] = dict(rows=len(numeric),active_derivative_rows=int(np.count_nonzero(analytic)),
                                    max_abs=float(abs(analytic-numeric).max()),relative=relative,
                                    source_seconds=(probe.times[[0,-1]]+problem.origin).tolist())
    return receipt
