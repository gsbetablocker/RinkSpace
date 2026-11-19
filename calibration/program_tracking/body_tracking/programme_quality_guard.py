"""Shared programme source, structure and confirmed-flight trial guards."""
from dataclasses import dataclass

import numpy as np

from .continuous_physics import ConfirmedFlightFactors
from .programme_action_factors import turn_episode_image_checks, turn_episode_image_metrics
from .trajectory_probe import body_groups


@dataclass
class ProgrammeGuards:
    check: object
    source_checks: object
    baseline: dict
    torso_before: dict
    turn_before: dict
    joint_peaks: dict
    surface_peak: float | None
    flight_guard: object
    dense_flight_guard: object
    cavity_trial_guard: object


def build_programme_guards(problem, base, ref, mass, document, first, original,
                           guard_observations, *, use_saved_source_baseline=False):
    """Bind baseline limits once and reuse them in all trials of the stage."""
    step_policy=document.get('programme_structure',{}).get('solver_step',{})
    baseline={r['id']:(float(np.percentile(r['outside_visible_band_px'],95)),float(r['pixel_uncertainty_radius_px']))for r in first['action_metrics'].get('visible_runner_contours',[])+first['action_metrics'].get('visible_unassigned_skate_contours',[])}
    torso_before=body_groups(problem,first)['torso']
    if use_saved_source_baseline:
        baseline.update(original['source']);torso_before=original['torso']
    turn_before=turn_episode_image_metrics(problem,base.turn_factors,guard_observations)
    turn_before.update(original.get('turn_episodes',{}))
    structural_before=first.get('action_metrics',{})
    joint_peaks={v['name']:v['peak_dimensionless_violation'] for v in structural_before.get('anatomical_joint_domain',{}).get('joints',[])}
    surface_peak=structural_before.get('surface_intersection',{}).get('worst_signed_gap_m')
    flight_guard=None;dense_flight_guard=None
    cavity_trial_guard = None
    if step_policy.get('cavity_collar_nonincrease_guard', False):
        from .programme_guard_proposals import ProgrammeCavityTrialGuard
        cavity_trial_guard = ProgrammeCavityTrialGuard(problem, first)
    if step_policy.get('flight_nonincrease_guard',False):
        from .programme_flight_guard import ProgrammeFlightGuard
        from .weighted_blocks import signature
        definition=[ref.p['pts'],ref.p['time_base'],ref.body.J,ref.binding.ratio,ref.binding.x,
            mass.ids,mass.offsets,mass.mass,mass.inertia]
        for wrapped in sorted((f for f in base.factors if isinstance(getattr(f,'factor',None),ConfirmedFlightFactors)),key=lambda f:f.episode_id):
            factor=wrapped.factor
            if not factor.event_average:raise ValueError('Per-flight guard requires the same event-average physics')
            prefix='episode/'+wrapped.episode_id+'/'
            definition.extend((prefix,ref.p['pts'][factor.frames],factor.times,factor.gravity,
                factor.position_operator,factor.position_offset,factor.momentum_operator,
                problem.objective_block_weights[prefix+'flight_integrated_com'],
                problem.objective_block_weights[prefix+'flight_specific_momentum']))
        flight_guard=ProgrammeFlightGuard(first,previous=original.get('flight_guard'),
            definition_identity=signature(*definition))
        dense_factor=next((f for f in base.factors if type(f).__name__=='ModelCOMMotionFactor' and f.flight_rows),None)
        if dense_factor is not None:
            from .programme_flight_guard import ProgrammeDenseFlightGuard
            dense_definition=definition+[dense_factor.times,dense_factor.source_pts_float,
                dense_factor.gravity,np.array(dense_factor.flight_scale)]
            for episode,rows in sorted(dense_factor.flight_rows.items()):
                dense_definition.extend((episode,rows,dense_factor.flight_weights[episode]))
            dense_flight_guard=ProgrammeDenseFlightGuard(first,previous=original.get('dense_flight_guard'),
                definition_identity=signature(*dense_definition))
    if document.get('com_motion_policy',{}).get('confirmed_flight_acceleration_scale_mps2') is not None and dense_flight_guard is None:
        raise ValueError('Dense flight physics requires the original native-flight and new dense guards')
    def source_checks(rows):
        checks={}
        for row in rows:
            old,radius=baseline[row['id']];now=float(np.percentile(row['outside_visible_band_px'],95));checks[row['id']]=dict(rejected=now>old+.25*radius,before=old,after=now,original_radius_px=radius,model_foot=row.get('foot'))
        return checks
    def guard(value,relative_limit):
        checks=source_checks(value['action_metrics'].get('visible_runner_contours',[])+value['action_metrics'].get('visible_unassigned_skate_contours',[]))
        if problem.action_factors.whole_skate is not None:
            from .skate_run_policy import image_trial_quality
            checks['whole_skate_source']=image_trial_quality(whole_source_before,
                value['source_group_costs']['whole_skate_source'],relative_limit)
        torso=body_groups(problem,value,observation_mask=guard_observations)['torso'];checks['torso']=dict(rejected=torso['pixel_p95']>torso_before['pixel_p95']+max(2.,.1*torso_before['pixel_p95']),before=torso_before['pixel_p95'],after=torso['pixel_p95'],meaning='same source population across stages; finite engineering source guard, not 3D truth')
        for row in value.get('action_metrics',{}).get('anatomical_joint_domain',{}).get('joints',[]):
            if row['name'] in joint_peaks:
                before=joint_peaks[row['name']];now=row['peak_dimensionless_violation']
                checks['joint_peak_'+row['name']]=dict(rejected=now>before+1e-5,before=before,after=now,meaning='engineering domain severity; not clinical angle')
        if surface_peak is not None:
            now=value['action_metrics']['surface_intersection']['worst_signed_gap_m']
            checks['actual_surface_witness_peak']=dict(rejected=now<surface_peak-1e-6,before=surface_peak,after=now,meaning='same version oriented witness; whole-surface refresh follows accepted step')
        if flight_guard is not None:checks.update(flight_guard.checks(value))
        if dense_flight_guard is not None:checks.update(dense_flight_guard.checks(value))
        if cavity_trial_guard is not None:
            checks.update(cavity_trial_guard.checks(value))
        checks.update(turn_episode_image_checks(turn_before,
            turn_episode_image_metrics(problem,base.turn_factors,guard_observations)))
        return checks
    whole_source_before=first['source_group_costs'].get('whole_skate_source')
    return ProgrammeGuards(guard,source_checks,baseline,torso_before,turn_before,
                           joint_peaks,surface_peak,flight_guard,dense_flight_guard,
                           cavity_trial_guard)


class ScopedProgrammeGuard:
    """Frozen parent budgets per affected spline neighbourhood and foot."""

    def __init__(self, problem, base, spec, value, x, saved=None):
        from .programme_repair_factors import sampled_windows, ProgrammeComponentIce
        from .programme_joint_limits import ProgrammeJointLimits
        from .programme_com import ModelCOMMotionFactor
        from .weighted_blocks import signature

        self.problem = problem
        self.ref = problem.reference
        self.state = problem.body.state
        self.ice = next(f for f in base.factors if isinstance(f,ProgrammeComponentIce))
        self.joints = next(f for f in base.factors if isinstance(f,ProgrammeJointLimits))
        self.com = next(f for f in base.factors if isinstance(f,ModelCOMMotionFactor))
        self.policy = spec['guard']
        self.pts = self.ref.p['pts']
        self.scope = []
        self.physical_times = self.ice.times
        if spec.get('phase_objective_replay_registry'):
            clock = problem.physics_times
            local = sampled_windows(self.ref,self.state,clock,spec['focus_pts'])
            self.physical_times = np.union1d(self.physical_times,clock[local])
        self.objective_ice_indices = np.searchsorted(self.physical_times,self.ice.times)
        self.extra_ice_indices = np.flatnonzero(~np.isin(self.physical_times,self.ice.times))
        self.opening_physical = sampled_windows(self.ref,self.state,self.physical_times,[spec['opening_pts']])
        self.opening_native = sampled_windows(self.ref,self.state,self.ref.times,[spec['opening_pts']])
        for window in spec['focus_pts']:
            ts = self.ref.times
            local = sampled_windows(self.ref,self.state,ts,[window])
            mid = sampled_windows(self.ref,self.state,self.physical_times,[window])
            nearby = local[:-2]|local[1:-1]|local[2:]
            self.scope.append((window,local,mid,nearby))
        self.radii = np.array([self.ice.policy['tolerance_m'].get(k,self.ice.policy['tolerance_m']['solid'])
                              for k in self.ice.part_names])
        self.identity = signature(self.pts,self.state.knots,self.physical_times,
                                  self.joints.association,self.policy,spec['focus_pts'])
        self.anchor = self.snapshot(x,value)
        if saved is not None:
            if str(saved.pop('identity')) != self.identity or self.anchor.keys()!=saved.keys():
                raise ValueError('Exact resume changed scoped guard identity')
            for name,v in saved.items():
                if self.anchor[name].shape!=v.shape:
                    raise ValueError('Exact resume changed scoped guard layout: '+name)
            self.anchor = saved

    def checkpoint(self):
        return dict(identity=np.array(self.identity),**self.anchor)

    def opening_validation(self,x,value):
        row=self.snapshot(x,value)
        native=self.opening_native
        keep=native[:-2]|native[1:-1]|native[2:]
        return dict(root_peak_mps2=float(row['root'][self.opening_physical].max()),
                    com_peak_mps2=float(row['com'][keep].max()),
                    root_peak_limit_mps2=self.policy['opening_root_peak_mps2'],
                    com_peak_limit_mps2=self.policy['opening_com_peak_mps2'])

    def snapshot(self,x,value):
        from .absolute_motion_curve import _log
        from .absolute_motion_state import evaluate_curve

        body=self.problem.body
        row=body.frames[body.of]
        model_j=getattr(body,'model_oj',body.oj)
        projected=body.cache['uv'][body.of,body.layout.body_landmarks[model_j]]
        pixels=np.linalg.norm(projected-body.obs['uv'][row,body.oj],axis=1)
        k=self.state.current_curve(x[17:])
        rotation=evaluate_curve(k,self.ref.times)['rotations']
        angle=np.rad2deg(np.linalg.norm(_log(rotation[:-1].swapaxes(-1,-2)@rotation[1:]),axis=-1))
        root=k.position_derivative(self.physical_times,2)*self.ref.binding.ratio
        heights=self.ice.last_height.copy()
        if len(self.extra_ice_indices):
            from .programme_repair_factors import component_ice
            from .skate_physics import fk_state
            t=self.physical_times[self.extra_ice_indices]
            pose=fk_state(self.ref,self.ref.at(t),self.state.increments(x[17:],t))
            extra=component_ice(self.ref.binding,pose)[1]
            extended=np.empty((len(self.physical_times),*heights.shape[1:]))
            extended[self.objective_ice_indices]=heights
            extended[self.extra_ice_indices]=extra
            heights=extended
        whole=self.problem.action_factors.whole_skate
        silhouette=whole.last_distances.copy()
        directed=np.full((len(self.pts),2),np.nan)
        for row in value['action_metrics'].get('directed_boot_source',[]):
            i=np.searchsorted(self.pts,row['pts'])
            directed[i,row['foot']]=row['distance_px']
        return dict(pixel=pixels,ice=heights,
            anatomy=self.joints.last_values.copy(),
            cavity=np.maximum(0.,-value['contact']['margin_m'].reshape(-1,self.problem.cavity_rows_per_frame)[1:]),
            root=np.linalg.norm(root,axis=1),com=np.linalg.norm(self.com.last_native_acceleration,axis=1),
            angle=angle,silhouette=silhouette,directed=directed)

    def checks(self,x,value):
        cur=self.snapshot(x,value)
        before=self.anchor
        cfg=self.policy
        result={}
        body=self.problem.body
        row=body.frames[body.of]
        torso=np.isin(body.oj,(5,6,11,12))
        whole=self.problem.action_factors.whole_skate
        directed_sigma=np.full((len(self.pts),2),np.nan)
        for item in self.ref.source_document['whole_skate_source'].get('directed_boot_rows',[]):
            i=np.searchsorted(self.pts,item['pts'])
            directed_sigma[i,item['foot']]=item['sigma_px']

        def check(name, old, new, allowance, *, details=None):
            old,new,allowance=np.broadcast_arrays(old,new,allowance)
            excess=new-old-allowance
            i=np.unravel_index(int(np.argmax(excess)),excess.shape)
            result[name]=dict(rejected=bool(excess[i]>1e-9),before=float(old[i]),after=float(new[i]),
                allowance=float(allowance[i]),component=list(map(int,i)),**(details or {}))

        for window, native, physical, derivative in self.scope:
            prefix=f"local/{window[0]}-{window[1]}/"
            core=(self.pts>=window[0])&(self.pts<=window[1])
            for suffix,population in (('',core[row]),('_full_support',native[row])):
                for name,subset,quantile,budget in (
                    ('body_median',population,50,cfg['body_median_increase_px']),
                    ('body_p90',population,90,cfg['body_p90_increase_px']),
                    ('torso_p90',population&torso,90,cfg['torso_p90_increase_px']),
                ):
                    if subset.any():
                        check(prefix+name+suffix,np.percentile(before['pixel'][subset],quantile),
                              np.percentile(cur['pixel'][subset],quantile),budget)
            for foot in (0,1):
                q=cfg['maximum_new_trusted_source_sigmas']
                good=native&np.isfinite(before['silhouette'][:,foot])
                if good.any():
                    check(prefix+f'whole_skate_foot_{foot}',before['silhouette'][good,foot],
                          cur['silhouette'][good,foot],q*whole.packet['sigma_px'][good,foot])
                good=native&np.isfinite(before['directed'][:,foot])
                if good.any():
                    check(prefix+f'directed_skate_foot_{foot}',before['directed'][good,foot],
                          cur['directed'][good,foot],q*directed_sigma[good,foot])
            if physical.any():
                old=np.maximum(-before['ice'][physical]-self.radii,0.)
                new=np.maximum(-cur['ice'][physical]-self.radii,0.)
                check(prefix+'component_ice',old.max(0),new.max(0),cfg['component_height_numerical_m'],
                      details=dict(part_names=self.ice.part_names))
                check(prefix+'root_acceleration',before['root'][physical].max(),
                      cur['root'][physical].max(),max(2.,before['root'][physical].max()*(cfg['movement_budget_ratio']-1.)))
            a,b=before['anatomy'][native],cur['anatomy'][native]
            check(prefix+'joint_component_peak',a.max(0),b.max(0),cfg['anatomy_peak_budget'])
            violation_increase=np.maximum(b-a,0.).mean(axis=0)
            check(prefix+'joint_component_neighbourhood',0.,violation_increase,cfg['anatomy_new_component_budget'])
            check(prefix+'cavity',before['cavity'][native].max(0),cur['cavity'][native].max(0),
                  cfg['cavity_peak_budget_m'])
            step=native[:-1]|native[1:]
            if step.any():
                check(prefix+'local_rotation_step',before['angle'][step].max(0),cur['angle'][step].max(0),
                      cfg['world_angular_step_budget_deg'])
            if derivative.any():
                a,b=before['com'][derivative],cur['com'][derivative]
                check(prefix+'com_acceleration',a.max(),b.max(),max(2.,a.max()*(cfg['movement_budget_ratio']-1.)))
        return result
