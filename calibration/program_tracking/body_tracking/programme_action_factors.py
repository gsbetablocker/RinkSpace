"""Source-supported rotation paths on the existing programme control curves."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from .absolute_motion_curve import _log
from .absolute_motion_state import inverse_right_jacobian
from .fullbody_fit import right_jacobian
from .skate_physics import fk_state
from .weighted_blocks import block, full_columns

FACING_ANGLES = {'front': 0., 'screen_right': .5*np.pi,
                 'back': np.pi, 'screen_left': -.5*np.pi}
UP = np.array([0., 0., 1.])
TORSO_COLUMNS = tuple(range(6)) + tuple(
    6+3*(joint-1)+axis for joint in (3, 6, 9) for axis in range(3))


def active_episode_rows(pts, episode):
    """The source episode endpoints are inclusive native exposures."""
    rows = np.flatnonzero((pts >= episode['start_pts']) &
                          (pts <= episode['end_pts_inclusive']))
    if not len(rows) or pts[rows[0]] != episode['start_pts'] or pts[rows[-1]] != episode['end_pts_inclusive']:
        raise ValueError('Episode endpoints outside the source clock: '+episode['episode_id'])
    return rows


def action_policy(document):
    return document.get('programme_action_physics', {})


def supported_phase_mask(document, pts, foot):
    episodes = document.get('source_episode_review', {}).get('episodes', [])
    support = np.zeros(len(pts), bool)
    lifted = np.zeros(len(pts), bool)
    for episode in episodes:
        kind = episode['kind']
        if kind == 'flight_visible_candidate' or (kind == 'free_foot_off_ice_candidate' and episode.get('foot') == foot):
            lifted |= (pts >= episode['start_pts']) & (pts <= episode['end_pts_inclusive'])
        if kind == 'support_candidate' and episode.get('foot') == foot:
            support |= (pts >= episode['start_pts']) & (pts <= episode['end_pts_inclusive'])
    return support, lifted


class SourceTurnPath:
    """Signed direction and source facing cones, independent of turn count."""
    def __init__(self, problem, episode, warm):
        self.problem = problem
        self.ref = problem.reference
        self.state = problem.body.state
        self.episode = episode
        self.rows = active_episode_rows(self.ref.p['pts'], episode)
        self.times = self.ref.times[self.rows]
        self.base = self.ref.at(self.times)
        self.sign = int(episode['turn_direction_sign'])
        self.sign_provenance='source-visible orientation sequence'
        self.sign_scores=None
        if self.sign == 0:
            # Source establishes sustained turns. Where it leaves the
            # absolute sign ambiguous, compare both latent direction branches
            # on the same warm state. This is not a new source annotation.
            from .fullbody_state import matrices
            d=self.state.increments(warm,self.times)
            R=self.base['Q']@matrices(d[:,3:6])
            dt=np.diff(self.times)
            speed=_log(R[1:]@R[:-1].swapaxes(-1,-2))[:,2]/dt
            self.sign_scores={sign:float(dt@np.maximum(-sign*speed-.5,0.)**2/dt.sum()) for sign in (-1,1)}
            self.sign=min(self.sign_scores,key=self.sign_scores.get)
            self.sign_provenance='ambiguous source sign; latent continuation on common warm rotation'
        if self.sign not in (-1, 1) or len(self.times) < 3:
            raise ValueError('A source direction needs an ordered signed path')
        self.facing = tuple(episode.get('facing_observations', []))
        self.facing_ids = np.searchsorted(self.ref.p['pts'], [r['pts'] for r in self.facing]).astype(int)
        if len(self.facing_ids):
            if not np.array_equal(self.ref.p['pts'][self.facing_ids], [r['pts'] for r in self.facing]):
                raise ValueError('Facing observations need exact source PTS')
            self.facing_times = self.ref.times[self.facing_ids]
            self.facing_base = self.ref.at(self.facing_times)
            self.camera_centres = problem.camera.camera_centres(self.facing_ids)
            self.facing_angle = np.array([FACING_ANGLES[r['view']] for r in self.facing])
            self.facing_tolerance = np.deg2rad([r['tolerance_deg'] for r in self.facing])
        self.turn_scale = float(episode.get('direction_scale_radps', 4.))
        self.facing_scale = .55
        self.pulse_rows = np.empty(0,int)
        if action_policy(self.ref.source_document).get('structure_enabled'):
            pts=self.ref.p['pts'][self.rows]
            masks=[supported_phase_mask(self.ref.source_document,pts,foot)[0] for foot in (0,1)]
            same_foot=np.column_stack(masks)
            from .phase_ledger import phase_weights
            stable=phase_weights(self.ref,pts,default=0.,stable=1.,transition=0.)>0.
            same_foot &= stable[:,None]
            self.pulse_rows=np.flatnonzero(np.any(same_foot[:-2]&same_foot[1:-1]&same_foot[2:],axis=1))

    def facing_values(self, d):
        state = fk_state(self.ref, self.facing_base, d)
        anterior = (state['Q'] @ state['frames'][:, 9])[:, :2, 2]
        # Native SMPL-X rest anterior is +Z. A horizontal cone is relative to
        # the same exposure's optical centre. Pure camera roll is not yaw.
        view = self.camera_centres[:, :2] - state['P'][:, :2]
        view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
        right = np.stack((-view[:, 1], view[:, 0]), axis=-1)
        x = np.sum(anterior*view, axis=-1)
        y = np.sum(anterior*right, axis=-1)
        heading = np.arctan2(y, x)
        error = np.arctan2(np.sin(heading-self.facing_angle), np.cos(heading-self.facing_angle))
        outside = np.sign(error)*np.maximum(np.abs(error)-self.facing_tolerance, 0.)
        return outside / self.facing_scale, heading

    def evaluate(self, x, jacobian=False):
        d = self.state.increments(x, self.times)
        from .fullbody_state import matrices
        Q = self.base['Q'] @ matrices(d[:, 3:6])
        phi = _log(Q[1:] @ Q[:-1].swapaxes(-1, -2))
        dt = np.diff(self.times)
        omega = phi/dt[:, None]
        # Zero residual for all consistent speeds. A small angular ambiguity
        # allowance prevents near-stationary observations enforcing a turn.
        factor = np.sqrt(dt/dt.sum())/self.turn_scale
        reverse = -self.sign*omega[:, 2] - .5
        r = np.maximum(reverse, 0.)*factor
        J = None
        if jacobian:
            coeff = (inverse_right_jacobian(phi) @ Q[:-1])[:, 2]
            coeff *= ((reverse>0)*(-self.sign)*factor/dt)[:, None]
            tangent = right_jacobian(np.asarray(d)[:, 3:6])
            samples = np.arange(len(dt))
            J = csr_matrix((len(r), self.state.size))
            for offset, direction in ((0, -1), (1, 1)):
                local = np.zeros((len(dt), 1, 69))
                local[:, 0, 3:6] = direction*np.einsum('ni,nij->nj', coeff, tangent[samples+offset])
                J += d.chain(local, sample_indices=samples+offset)
        pts = self.ref.p['pts'][self.rows]
        named = [full_columns(block('rotation_direction_path', r, J,
                    dict(source_pts=pts[1:], scale=self.turn_scale, unit='rad/s')))]
        metrics = dict(path_source_pts=pts.tolist(), angular_velocity_world_radps=omega.tolist(),
                       selected_direction_sign=self.sign, direction_sign_provenance=self.sign_provenance,
                       warm_signed_branch_costs=self.sign_scores,
                       reverse_intervals=int(np.count_nonzero(reverse>0)),
                       integrated_signed_yaw_proxy_turns=float(phi[:,2].sum()/(2*np.pi)),
                       exact_revolutions_imposed=False, world_pivot_fixed=False,
                       greatest_direction_residual_pts=int(pts[1+np.argmax(r)]))
        if len(self.pulse_rows):
            pulse, details = self.pulse_block(d, Q, phi, dt, omega, pts, jacobian)
            named.append(pulse)
            metrics['same_support_turn_impulse']=details
        if len(self.facing_ids):
            fd = self.state.increments(x, self.facing_times)
            fr, heading = self.facing_values(fd)
            FJ = None
            if jacobian:
                step = 1e-6
                local = np.zeros((len(fd), 1, 69))
                for col in TORSO_COLUMNS:
                    plus, minus = fd.copy(), fd.copy()
                    plus[:, col] += step
                    minus[:, col] -= step
                    local[:, 0, col] = (self.facing_values(plus)[0]-self.facing_values(minus)[0])/(2*step)
                FJ = fd.chain(local)
            named.append(full_columns(block('source_facing_cone', fr, FJ,
                dict(source_pts=self.ref.p['pts'][self.facing_ids], scale=self.facing_scale, unit='rad'))))
            metrics.update(facing_rows=[dict(row, estimated_radians=float(value),
                scaled_residual=float(residual)) for row, value, residual in zip(self.facing, heading, fr)])
        from .weighted_blocks import assemble
        residual, matrix = assemble(named, jacobian)
        return dict(residual=residual, jac=matrix, weighted_blocks=named,
                    parts={b.name: float(b.residual@b.residual/2) for b in named}, metrics=metrics)


    def pulse_block(self,d,Q,phi,dt,omega,pts,jacobian):
        i=self.pulse_rows
        measure=.5*(dt[i]+dt[i+1])
        relative_clock=.02/measure
        difference=(omega[i+1]-omega[i])*relative_clock[:,None]
        magnitude=np.linalg.norm(difference,axis=1)
        # Broad finite allowance within one directly reviewed same-foot turn.
        # There is no speed target and no cross-phase acceleration stencil.
        allowance,scale=8.,7.
        weight=np.sqrt(measure/.02)/scale
        r=weight*np.maximum(magnitude-allowance,0.)
        J=None
        if jacobian:
            a=(inverse_right_jacobian(phi)@Q[:-1])/dt[:,None,None]
            radial=difference/np.maximum(magnitude[:,None],1e-12)
            radial*=((magnitude>allowance)*weight*relative_clock)[:,None]
            tangent=right_jacobian(np.asarray(d)[:,3:6])
            J=csr_matrix((len(i),self.state.size))
            for offset, A in ((0,a[i]),(1,-a[i]-a[i+1]),(2,a[i+1])):
                local=np.zeros((len(i),1,69))
                local[:,0,3:6]=np.einsum('ni,nij,njk->nk',radial,A,tangent[i+offset])
                J+=d.chain(local,sample_indices=i+offset)
        source=pts[i+1]
        named=full_columns(block('rotation_path_impulse',r,J,
            dict(source_pts=source,scale=scale,unit='rad/s change per 20 ms')))
        return named,dict(native_samples=len(i),nonzero_samples=int(np.count_nonzero(r)),
            worst_source_pts=int(source[np.argmax(r)]),max_change_radps=float(magnitude.max()),
            allowance_radps_per_20ms=allowance,scale_radps_per_20ms=scale)


class ProgrammeRootTrust:
    """Finite metric protection against unsupported global trajectory drift."""
    def __init__(self, problem, x):
        self.state = problem.body.state
        self.ref = problem.reference
        self.indices = np.unique(np.r_[np.arange(0, len(self.ref.times), 5), len(self.ref.times)-1])
        self.times = self.ref.times[self.indices]
        self.basis = self.state.basis(self.times)
        self.parent = self.position_controls(x).copy()
        from .skate_contact import trapezoid_time_weights
        self.weight = self.ref.binding.ratio*np.sqrt(trapezoid_time_weights(self.times))/.25

    def position_controls(self, x):
        return self.basis @ np.asarray(x)[:self.state.body_size].reshape(-1,69)[:, :3]

    def evaluate(self, x, jacobian=False):
        from scipy.sparse import diags, kron, hstack
        offset = self.position_controls(x)-self.parent
        r = (offset*self.weight[:, None]).ravel()
        J = None
        if jacobian:
            # This block uses only translation controls. No dense sampled
            # rotation chart is needed to differentiate linear P.
            t = diags(self.weight) @ self.basis
            local = np.zeros((3,69))
            local[:, :3] = np.eye(3)
            body = kron(t, csr_matrix(local), format='csr')
            J = hstack((body, csr_matrix((len(r), self.state.size-body.shape[1]))), format='csr')
        pts = self.ref.p['pts'][self.indices]
        named = full_columns(block('programme_parent_root_trust', r, J, dict(source_pts=pts, scale=.25, unit='m')))
        return dict(residual=r, jac=named.jacobian, weighted_blocks=[named],
            parts={named.name: float(r@r/2)},
            metrics={named.name: dict(samples=len(pts), reference='warm programme candidate',
                    max_change_m=float(np.max(np.linalg.norm(offset,axis=1))*self.ref.binding.ratio),
                    scale_m=.25, greatest_change_pts=int(pts[np.argmax(np.linalg.norm(offset,axis=1))]))})


def propose_turn_path(problem, x, factors):
    """One lifted-chart hypothesis from the source path rows. P stays exact."""
    from scipy.sparse import vstack
    blocks = [row for factor in factors for row in factor.evaluate(x[17:], True)['weighted_blocks']]
    if not blocks:
        return None
    J = vstack([b.jacobian for b in blocks], format='csr')
    columns = np.unique(J.indices)
    # Only root Q columns participate in the hypothesis. Shoulder/spine and
    # feet remain available to the actual joint solve.
    within_control = (columns-17)%69
    columns = columns[(columns>=17)&(within_control>=3)&(within_control<6)]
    if not len(columns):
        return None
    residual = np.concatenate([b.residual for b in blocks])
    step = lsmr(J[:, columns], -residual, damp=1.1, atol=1e-7, btol=1e-7)[0]
    limit = max(.55, np.max(np.abs(step)))
    proposed = x.copy()
    proposed[columns] += step*.55/limit
    return proposed


def is_action_block(name):
    return name.startswith('episode/') or name in {
        'programme_parent_root_trust','self_collision','signed_joint_relation','temporal_contact_mode'}


def record_action_factors(problem, value, output, *, jacobian):
    from .fullbody_round import write_json
    metrics = value['action_metrics']
    records, families = [], {}
    for b in value['weighted_blocks']:
        if not is_action_block(b.name):
            continue
        metadata = b.provenance if isinstance(b.provenance, dict) else {}
        pts = np.asarray(metadata.get('source_pts',[]), dtype=np.int64)
        r = b.residual
        k = int(np.argmax(abs(r))) if len(r) else None
        worst = None
        if k is not None and len(pts) and len(r)%len(pts)==0:
            worst = int(pts[k//(len(r)//len(pts))])
        cost = float(r@r/2)
        family = b.name.rsplit('/',1)[-1]
        row = dict(name=b.name, family=family, rows=len(r), nonzero_rows=int(np.count_nonzero(r)),
                   cost=cost, objective_weight=float(problem.objective_block_weights.get(b.name,1.)),
                   jacobian_nnz=int(b.jacobian.nnz) if jacobian else None,
                   source_exposures=int(len(np.unique(pts))), worst_residual_source_pts=worst,
                   source_first_last=None if not len(pts) else [int(pts.min()),int(pts.max())],
                   physical_scale=metadata.get('scale'),physical_unit=metadata.get('unit'),
                   finite_measure_weight=metadata.get('finite_weight'))
        records.append(row)
        entry = families.setdefault(family,dict(rows=0,nonzero_rows=0,cost=0.))
        for key in entry:
            entry[key] += row[key]
    relevant = {name:v for name,v in metrics.items()
        if name.startswith('episode/') or name in ('runner_support_diagnostics','mode_association',
            'self_collision','signed_joint_relation','programme_parent_root_trust')}
    relevant['new_source_skate_regions'] = [row for row in metrics.get('visible_runner_contours',[])
        if row['id'].startswith('s09_')]
    write_json(output,dict(actual_weighted_blocks=records,families=families,
        source_and_physics=relevant,actual_jacobian_measured=jacobian))


def action_direction_check(problem,x,first,output):
    """One centred check of the actual weighted sparse full-state derivatives."""
    from .absolute_motion_curve import RotationChartDomainError
    from .fullbody_round import write_json
    from .runner_interior import RunnerCoordinateDomainError
    groups = [b for b in first['weighted_blocks'] if is_action_block(b.name)
              and b.name!='programme_parent_root_trust']
    if not groups:
        return
    direction = -sum((b.jacobian.T@b.residual for b in groups), start=np.zeros(problem.size))
    active = getattr(problem,'active_parameter_indices',None)
    if active is not None:
        inactive = np.ones(len(direction),bool)
        inactive[active] = False
        direction[inactive] = 0.
    scale = np.linalg.norm(direction)
    if not scale:
        write_json(output,dict(reason='all included action directions stationary at this point'))
        return
    direction /= scale
    cache = (problem.cache_x,problem.cache,problem.body.cache_x,problem.body.cache)
    rejected_stencils=[]
    try:
        for step in (1e-5,1e-6,1e-7,1e-8):
            try:
                evaluated = [problem.evaluate(x+sign*step*direction,False) for sign in (1.,-1.)]
                break
            except (RotationChartDomainError,RunnerCoordinateDomainError) as error:
                rejected_stencils.append(dict(step=step,domain=type(error).__name__,detail=str(error)))
        else:
            write_json(output,dict(status='direction_check_outside_current_chart',same_global_state=True,
                rejected_stencils=rejected_stencils))
            return
        samples = [{b.name:b for b in v['weighted_blocks']} for v in evaluated]
        check = []
        for b in first['weighted_blocks']:
            plus,minus = (v[b.name] for v in samples)
            fd = (plus.residual-minus.residual)/(2*step)
            jv = b.jacobian@direction
            check.append(dict(name=b.name, rows=len(fd), analytic_cost_slope=float(b.residual@jv),
                central_cost_slope=float(b.residual@fd), derivative_l2_error=float(np.linalg.norm(fd-jv)),
                jacobian_direction_l2=float(np.linalg.norm(jv)),
                association_changed=plus.association!=minus.association))
    finally:
        problem.cache_x,problem.cache,problem.body.cache_x,problem.body.cache=cache
    write_json(output,dict(step=step,direction='negative gradient of actual weighted action blocks',
        rejected_stencils=rejected_stencils,
        same_global_state=True,source_and_legacy_blocks_included=True,blocks=check))


def turn_episode_image_metrics(problem, factors, observation_mask=None):
    """Protect the actual body rows of each turn from programme-wide dilution."""
    if not factors:
        return {}
    from .trajectory_probe import GROUPS
    body = problem.body
    frames = body.frames[body.of]
    joints = body.oj
    model_joints = getattr(body, 'model_oj', joints)
    projected = body.cache['uv'][body.of, body.layout.body_landmarks[model_joints]]
    pixels = np.linalg.norm(projected - body.obs['uv'][frames, joints], axis=-1)
    usable = np.isfinite(pixels)
    if observation_mask is not None:
        usable &= observation_mask[frames, joints]
    groups = {name: np.isin(joints, GROUPS[name]) for name in ('torso', 'lower_limbs')}
    metrics = {}
    for factor in factors:
        episode = factor.episode['episode_id']
        in_episode = np.isin(frames, factor.rows) & usable
        for name, group in groups.items():
            selected = pixels[in_episode & group]
            if selected.size:
                metrics[f'{episode}/{name}'] = dict(observations=int(selected.size),
                    pixel_p95=float(np.percentile(selected, 95)))
    return metrics


def turn_episode_image_checks(baseline, current):
    checks = {}
    for name, previous in baseline.items():
        before = previous['pixel_p95']
        trial = current.get(name)
        after = None if trial is None else trial['pixel_p95']
        limit = before + max(2., .1 * before)
        rejected = (trial is None or trial['observations'] != previous['observations'] or after > limit)
        checks[f'turn_image/{name}'] = dict(rejected=rejected, before=before, after=after,
            limit_px=limit, original_observations=previous['observations'])
    return checks


def choose_source_path_hypothesis(problem,x,factors,output,frozen_baseline=None):
    from .fullbody_round import write_json
    from .trajectory_probe import body_groups
    import json
    archived={} if frozen_baseline is None else json.loads(frozen_baseline.read_text(encoding='utf8'))
    fixed=archived.get('source',{})
    alternative=propose_turn_path(problem,x,factors)
    if alternative is None:
        return x
    # Compare explicit lifted-control hypotheses on the very same objective.
    # Source cones alone cannot justify a clear body or steel regression.
    def score(value):
        parts={b.name:float(b.residual@b.residual/2) for b in value['weighted_blocks']}
        path=sum(cost for name,cost in parts.items()
                 if name.endswith(('/rotation_direction_path','/source_facing_cone','/rotation_path_impulse')))
        clear_rows={r['id']:(float(np.percentile(r['outside_visible_band_px'],95)),
                    float(r['pixel_uncertainty_radius_px']))
                    for r in value['action_metrics'].get('visible_runner_contours',[])+
                             value['action_metrics'].get('visible_unassigned_skate_contours',[])}
        return dict(path=path,total=sum(parts.values()),torso=body_groups(problem,value)['torso']['pixel_p95'],
                    steel=clear_rows,turn_episodes=turn_episode_image_metrics(problem,factors))
    warm=score(problem.evaluate(x,False))
    turn_baseline={**warm['turn_episodes'],**archived.get('turn_episodes',{})}
    trials=[]
    selected=x
    for fraction in (1.,.5):
        trial=x+fraction*(alternative-x)
        try:
            metric=score(problem.evaluate(trial,False))
        except (ValueError,ArithmeticError) as error:
            trials.append(dict(fraction=fraction,accepted=False,domain_failure=str(error)))
            continue
        regressed=[]
        for name,(old,radius) in warm['steel'].items():
            maximum=old+.25*radius
            if name in fixed:
                value,uncertainty=fixed[name]
                maximum=min(maximum,value+.25*uncertainty)
            if metric['steel'][name][0]>maximum:
                regressed.append(name)
        episode_checks=turn_episode_image_checks(turn_baseline,metric['turn_episodes'])
        accepted=(not regressed and not any(v['rejected'] for v in episode_checks.values()) and
                  metric['path']<warm['path'] and
                  metric['total']<warm['total'] and metric['torso']<=warm['torso']+.75)
        trials.append(dict(fraction=fraction,accepted=accepted,path=metric['path'],
            total=metric['total'],torso=metric['torso'],source_rows_regressed=regressed,
            turn_episode_image_checks=episode_checks))
        if accepted:
            selected=trial
            break
    write_json(output/'source_path_hypotheses.json',dict(parent={k:v for k,v in warm.items() if k!='steel'},
        alternatives=trials,used_lifted_control_chart=True,root_position_changed=False,
        seed_selected=selected is not x,full_programme_comparison=True))
    return selected
