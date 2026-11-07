"""Source-gated step hypotheses and a finite running-blade travel residual."""
import numpy as np
from scipy.sparse import csr_matrix

from .sequence_skate_state import perturb_relative
from .skate_contact import trapezoid_time_weights
from .skate_physics import fk_state
from .source_support_locations import _runner_witness
from .weighted_blocks import block, full_columns


def relation_hypotheses(names, entry, exit, *, turn_sign=None):
    """Check observed relations and retain ties and unknown edges explicitly."""
    result = []
    for name in names:
        if name not in ('glide', 'transfer_glide', 'mohawk', 'choctaw', 'rocker',
                        'counter', 'bracket', 'three_turn', 'twizzle'):
            raise ValueError('Unknown step relation hypothesis: '+name)
        expectations = {
            'same_foot': name not in ('mohawk', 'choctaw', 'transfer_glide'),
            'same_edge': (name in ('mohawk', 'rocker', 'counter')
                          if name not in ('glide', 'transfer_glide', 'twizzle') else None),
            'same_travel': (name in ('glide', 'transfer_glide') if name != 'twizzle' else None),
            'same_curvature': (name in ('mohawk', 'bracket', 'three_turn')
                               if name not in ('glide', 'transfer_glide', 'twizzle') else None),
        }
        relations = {}
        for relation, expected in expectations.items():
            field = relation.removeprefix('same_')
            a, b = entry.get(field), exit.get(field)
            relations[relation] = None if expected is None or a is None or b is None else (a == b) == expected
        if name in ('rocker', 'counter', 'bracket', 'three_turn'):
            curvature = entry.get('curvature')
            relations['turn_with_entry_curve'] = (None if turn_sign is None or curvature is None else
                (turn_sign == curvature) == (name in ('rocker', 'three_turn')))
        result.append(dict(candidate=name, source_relations=relations,
            violations=sum(value is False for value in relations.values())))
    return result


class StepRelationFactors:
    """The material at the central runner witness moves with the bound shoe.

    Test its velocity across adjacent exposures in each supported phase. The
    moving geometric minimum itself is not a material-point trajectory.
    """
    def __init__(self, problem, episode, episodes):
        self.ref = problem.reference
        self.state = problem.body.state
        self.episode = episode
        pts = self.ref.p['pts']
        frames, feet, travel, weights, phases = [], [], [], [], []
        if not episode['step_phases'] or not episode['step_hypotheses']:
            raise ValueError('Step relation needs supported phases and explicit hypotheses')
        last = episode['start_pts'] - 1
        for phase in episode['step_phases']:
            support = episodes[phase['support_episode_id']]
            if support['kind'] != 'support_candidate' or support.get('contact_mode') != 'normal_runner':
                raise ValueError('Step travel requires source-backed normal-runner support')
            if phase['start_pts'] < support['start_pts'] or phase['end_pts_inclusive'] > support['end_pts_inclusive']:
                raise ValueError('A step travel stencil cannot cross a support boundary')
            if (not last < phase['start_pts'] <= phase['end_pts_inclusive'] <= episode['end_pts_inclusive']
                or not 0 < float(phase['reliability']) <= 1):
                raise ValueError('Expected ordered step phases and finite source reliability')
            last = phase['end_pts_inclusive']
            ids = np.flatnonzero((pts >= phase['start_pts']) & (pts <= phase['end_pts_inclusive']))
            centres = ids[1:-1:2]
            if not len(centres) or not np.all(np.diff(ids) == 1):
                raise ValueError('Step travel requires adjacent real source exposures')
            sign = {None: 0, 'forward': 1, 'backward': -1}[phase.get('travel')]
            if sign:
                evidence=phase.get('travel_evidence_pts',[])
                if len(set(evidence))<2 or not np.isin(evidence,pts[ids]).all():
                    raise ValueError('Signed travel requires two or more exact source exposures in its phase')
            frames.extend(centres)
            feet.extend([support['foot']] * len(centres))
            travel.extend([sign] * len(centres))
            phases.append(dict(phase, foot=('left', 'right')[support['foot']]))
            dt = 2 * float(np.median(np.diff(self.ref.times[ids])))
            weights.extend(trapezoid_time_weights(self.ref.times[centres], singleton_duration_s=dt)
                           * float(phase['reliability']))
        self.feet = np.asarray(feet, int)
        self.sign = np.asarray(travel, int)
        self.weights = np.sqrt(weights)
        self.pts = pts[frames]
        ids = np.asarray(frames)[:, None] + np.arange(-1, 2)
        self.times = self.ref.times[ids].ravel()
        self.dt = self.ref.times[ids[:, 2]] - self.ref.times[ids[:, 0]]
        self.base = self.ref.at(self.times)
        self.rows = np.arange(len(self.times)).reshape(-1, 3)
        self.hypotheses = relation_hypotheses(episode['step_hypotheses'], phases[0], phases[-1],
            turn_sign=episode.get('turn_direction_sign'))

    def values(self, delta):
        state = fk_state(self.ref, self.base, delta)
        r = state['R'][self.rows, self.feet[:, None]]
        o = state['O'][self.rows, self.feet[:, None]]
        witness = _runner_witness(self.ref.binding, r[:, 1], o[:, 1])
        motion = o[:, 2] - o[:, 0] + np.einsum('nij,nj->ni', r[:, 2] - r[:, 0], witness['local'])
        velocity = motion[:, :2] * (self.ref.binding.ratio / self.dt)[:, None]
        forward = r[:, 1, :2, 0]
        forward = forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-8)
        lateral = np.column_stack((-forward[:, 1], forward[:, 0]))
        along = np.sum(velocity * forward, axis=1)
        across = np.sum(velocity * lateral, axis=1)
        drift = np.sign(across) * np.maximum(np.abs(across) - .30, 0.)
        reversed_travel = np.maximum(-self.sign * along - .20, 0.)
        residual = np.column_stack((np.arcsinh(drift / .75), np.arcsinh(reversed_travel / .75)))
        return residual * self.weights[:, None], along, across

    def evaluate(self, x, jacobian=False):
        d = self.state.increments(x, self.times)
        residual, along, across = self.values(d)
        J = None
        if jacobian:
            J = csr_matrix((residual.size, self.state.size))
            step = 1.e-6
            columns = np.r_[np.arange(6), *[np.arange(3*j+3, 3*j+6) for j in (1, 2, 4, 5, 7, 8)]]
            for ids in self.rows.T:
                local = np.zeros((len(ids), 2, 69))
                for col in columns:
                    plus, minus = d.copy(), d.copy()
                    plus[ids, col] += step
                    minus[ids, col] -= step
                    local[:, :, col] = (self.values(plus)[0] - self.values(minus)[0]) / (2 * step)
                J += d.chain(local, sample_indices=ids)
                shoe = np.zeros((len(ids), 2, 6))
                other = np.setdiff1d(np.arange(len(d)), ids)
                for foot in (0, 1):
                    for axis in range(3):
                        plus = perturb_relative(d, foot, axis, step)
                        minus = perturb_relative(d, foot, axis, -step)
                        plus.skate_relative_R[other] = d.skate_relative_R[other]
                        minus.skate_relative_R[other] = d.skate_relative_R[other]
                        shoe[:, :, 3*foot+axis] = (self.values(plus)[0] - self.values(minus)[0]) / (2 * step)
                J += self.state.shoe_chain(shoe, d.parameters, d.sample_times, sample_indices=ids)
        signed = self.sign != 0
        parts = [('step_runner_lateral', np.arange(0, residual.size, 2)),
                 ('step_runner_travel', 2*np.flatnonzero(signed)+1)]
        blocks = [full_columns(block(name, residual.ravel()[rows], None if J is None else J[rows],
            dict(source_pts=self.pts[rows//2], unit='m/s scaled', finite_scale_mps=.75)))
            for name, rows in parts if len(rows)]
        from .weighted_blocks import assemble
        values, matrix = assemble(blocks, jacobian)
        return dict(residual=values, jac=matrix, weighted_blocks=blocks,
            parts={b.name: float(b.residual @ b.residual / 2) for b in blocks},
            metrics=dict(source_pts=self.pts.tolist(), foot=self.feet.tolist(),
                along_runner_mps=along.tolist(), across_runner_mps=across.tolist(),
                source_travel_sign=self.sign.tolist(), hypotheses=self.hypotheses,
                material_witness_policy='central rocker witness with material coordinate held for velocity',
                edge_identity_is_source_truth=False, pose_and_trajectory_continuous=True))
