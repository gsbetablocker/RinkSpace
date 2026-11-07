"""Explicit, bounded run monitoring; a policy stop is not convergence."""
from dataclasses import asdict, dataclass
from copy import deepcopy
import math
from numbers import Integral
import numpy as np


def jacobian_reserve_seconds(cached_x,cached_value,requested_x,uncached_seconds):
    """An exact already-assembled J consumes no new full-J wall-time reserve."""
    if (cached_x is not None and cached_value is not None
            and cached_value.get('jac')is not None and np.array_equal(cached_x,requested_x)):
        return 0.
    return float(uncached_seconds)


_REASON_CATEGORIES = dict(wall_budget='budget', accepted_budget='budget', outer_budget='budget',
    violation_stagnation='stagnation', image_cost_increase='quality', pose_cost_increase='quality',
    quality_metric_limit='quality', quality_metric_unavailable='domain', penalty_growth_disabled='policy')


def is_quality_failure(reason):
    """Quality/domain failure is distinct from a budget, stagnation or policy stop."""
    return reason.get('category', _REASON_CATEGORIES.get(reason.get('code'))) in ('quality', 'domain')


_CANDIDATE_VALUE_FIELDS = ('parts', 'gate', 'image_cost', 'pose_cost', 'raw_data_parts',
    'source_motion_metrics', 'quality_metrics')


@dataclass(frozen=True)
class CandidateSnapshot:
    """One coherent state; value deliberately excludes Jacobians/world/cache."""
    x: object
    value: dict
    metrics: dict
    step: int


def _copy_candidate(candidate):
    if candidate is None:return None
    return CandidateSnapshot(candidate.x.copy(), deepcopy(candidate.value), deepcopy(candidate.metrics), candidate.step)


class CandidateRetention:
    """Retain the latest quality-normal initial/accepted/solver-return state.

    Call ``observe`` with a decision and measurements from that same state,
    including at step 0 and on normal solver return. A budget/stagnation stop
    does not reject the current state. A quality/domain failure only updates
    last_attempt. No eligible initial state means selected is None, not a
    fabricated fallback. Eligibility here establishes neither feasibility nor
    user/source acceptance. Re-evaluate selected.x to restore mutable caches.
    """
    def __init__(self):
        self._last_attempt = None;self._last_good = None
        self._quality_failed = False;self._quality_reasons = []

    @property
    def last_attempt(self):return _copy_candidate(self._last_attempt)

    @property
    def last_good(self):return _copy_candidate(self._last_good)

    @property
    def selected(self):return self.last_good

    def observe(self, x, value, *, step, decision, metrics=None):
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError('step must be a nonnegative integer')
        if self._last_attempt is not None and step < self._last_attempt.step:
            raise ValueError('step must be nondecreasing')
        self._last_attempt = CandidateSnapshot(x.copy(),
            {key: deepcopy(value[key]) for key in _CANDIDATE_VALUE_FIELDS if key in value},
            deepcopy({} if metrics is None else metrics), int(step))
        self._quality_reasons = deepcopy(decision.get('quality_reasons') or
            [reason for reason in decision.get('reasons', []) if is_quality_failure(reason)])
        self._quality_failed = bool(decision.get('quality_failed') or self._quality_reasons)
        if not self._quality_failed:self._last_good = _copy_candidate(self._last_attempt)
        return self.selected

    def record(self):
        attempt, retained = self._last_attempt, self._last_good
        return dict(available=retained is not None, selection='last_good' if retained is not None else 'unavailable',
            attempt_step=None if attempt is None else attempt.step,
            retained_step=None if retained is None else retained.step,
            attempt_metrics=None if attempt is None else deepcopy(attempt.metrics),
            retained_metrics=None if retained is None else deepcopy(retained.metrics),
            quality_failed=self._quality_failed, quality_reasons=deepcopy(self._quality_reasons))


@dataclass(frozen=True)
class RunPolicy:
    wall_seconds: float = 900
    max_accepted: int = 8
    max_outer: int = 1
    stagnation_window: int = 4
    min_violation_improvement_m: float = .0001
    # Numerical joint-objective progress, matching solve() ftol; not a
    # body/ice accuracy threshold. Legacy callers may omit joint_cost.
    min_joint_relative_improvement: float = 1e-7
    max_image_cost_increase: float = .05
    max_pose_cost_increase: float = .10
    allow_penalty_growth: bool = False

    def __post_init__(self):
        for name in ('wall_seconds', 'min_violation_improvement_m', 'min_joint_relative_improvement',
                     'max_image_cost_increase', 'max_pose_cost_increase'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:raise ValueError(name+' must be finite and nonnegative')
        for name in ('max_accepted', 'max_outer', 'stagnation_window'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (1 if name == 'stagnation_window' else 0):
                raise ValueError(name+' must be a nonnegative integer (stagnation_window >= 1)')


def _metric(value):
    if value is None:return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def image_trial_quality(baseline, current_cost, relative_limit):
    """Pure trial check against the existing initial-image relative limit.

    Missing/zero baselines cannot establish a relative guard. This helper does
    not observe a candidate, update a monitor or replace the actual objective.
    """
    def finite(value):
        if value is None or isinstance(value, bool):return None
        try:value = float(value)
        except (TypeError, ValueError, OverflowError):return None
        return value if math.isfinite(value) else None
    baseline, current_cost, relative_limit = map(finite, (baseline, current_cost, relative_limit))
    available = (baseline is not None and baseline > 0 and current_cost is not None and current_cost >= 0
                 and relative_limit is not None and relative_limit >= 0)
    maximum = baseline*(1+relative_limit) if available else None
    available = available and math.isfinite(maximum)
    rejected = bool(available and current_cost > maximum)
    return dict(status=('rejected' if rejected else 'allowed') if available else 'unavailable',
                available=bool(available), rejected=rejected, baseline=baseline, image_cost=current_cost,
                relative_limit=relative_limit, maximum_image_cost=maximum if available else None,
                relative_increase=(current_cost-baseline)/baseline if available else None)


def structural_progress_is_observable(active_state_columns, *, optimize_shared):
    """Return false for a proved rigid-body-only stage with fixed footwear.

    The 69-column body layout uses 0:3 for world translation, 3:6 for global
    orientation, and 6:69 for local joint rotations. Foot-cavity and collar
    rows are expressed in body/foot coordinates, so a common world rigid
    motion cannot change them. Local-pose or shared-shape stages retain the
    existing stop. This is deliberately conservative; it does not claim
    that every local joint can change every structural row.
    """
    if not isinstance(optimize_shared, bool):
        raise ValueError('optimize_shared must be bool')
    columns = tuple(active_state_columns)
    for column in columns:
        if isinstance(column, bool) or not isinstance(column, Integral) or not 0 <= column < 69:
            raise ValueError('active body columns must be integers in [0, 69)')
    return optimize_shared or any(column >= 6 for column in columns)


class Monitor:
    """Monitor coherent initial/accepted-state measurements, never trial fits.

    ``now`` is caller-supplied monotonic seconds. ``outer_completed`` counts
    completed outer solves, so max_outer=1 permits the first solve to run.
    Image/pose costs must be the same RAW component definitions at every call,
    independent of AL penalty and multiplier changes. Missing/nonfinite or
    zero-baseline costs cannot establish relative degradation. Additional live
    measurements are diagnostic unless the caller supplies an explicit maximum
    and matching unit in quality_limits. No default action threshold is implied.
    """
    def __init__(self, policy=None, *, now, quality_limits=None, violation_stagnation_enabled=True):
        self.policy = RunPolicy() if policy is None else policy
        if not isinstance(violation_stagnation_enabled, bool):
            raise ValueError('violation_stagnation_enabled must be bool')
        self.violation_stagnation_enabled = violation_stagnation_enabled
        self.start_now = float(now)
        if not math.isfinite(self.start_now):raise ValueError('now must be finite monotonic seconds')
        self.last_now = self.start_now
        self.baseline = None;self.latest = None
        self.checks = 0;self.accepted = 0;self.outer_completed = 0
        self.best_violation_m = None;self.best_at_accepted = None
        self._violation_history = [];self._joint_history = []
        self.progress_diagnostics = []
        self.stop_reasons = [];self.first_stop_now = None
        self.quality_reasons = [];self.quality_diagnostics = []
        self.quality_limits = {}
        for name, limit in (quality_limits or {}).items():
            if not isinstance(name, str) or not name or not isinstance(limit.get('unit'), str) or not limit['unit'].strip():
                raise ValueError('quality limits require a metric name and explicit unit')
            maximum = float(limit['maximum'])
            if not math.isfinite(maximum):raise ValueError('quality maximum must be finite')
            self.quality_limits[name] = dict(unit=limit['unit'], maximum=maximum)

    def reserve_evaluation(self, *, now, estimated_seconds):
        """Stop before a measured expensive call that cannot fit the budget."""
        estimate=float(estimated_seconds)
        if not math.isfinite(estimate) or estimate<0:raise ValueError('estimated_seconds must be finite and nonnegative')
        status=self.observe(now=now)
        remaining=self.policy.wall_seconds-(float(now)-self.start_now)
        if not status['stop'] and estimate>remaining:
            self.stop_reasons=[dict(code='evaluation_budget_reserve',category='budget',
                message='Estimated evaluation cannot finish within remaining solve budget.',
                estimated_seconds=estimate,remaining_seconds=remaining)]
            self.first_stop_now=float(now)
            status=self.observe(now=now)
        return status

    def observe(self, *, now, violation_m=None, image_cost=None, pose_cost=None, joint_cost=None,
                accepted=False, outer_completed=None, penalty_growth_requested=False,
                quality_metrics=None, quality_failures=None):
        now = float(now)
        if not math.isfinite(now) or now < self.last_now:raise ValueError('now must be finite and monotonic')
        if outer_completed is not None:
            if isinstance(outer_completed, bool) or not isinstance(outer_completed, Integral) or outer_completed < self.outer_completed:
                raise ValueError('outer_completed must be a nondecreasing integer')
            self.outer_completed = int(outer_completed)
        self.last_now = now;self.checks += 1
        if accepted:self.accepted += 1
        sample = dict(now=now, elapsed_seconds=now-self.start_now,
            accepted=self.accepted, outer_completed=self.outer_completed,
            violation_m=_metric(violation_m), image_cost=_metric(image_cost), pose_cost=_metric(pose_cost),
            joint_cost=_metric(joint_cost))
        measured = quality_metrics is not None or accepted or any(sample[k] is not None for k in ('violation_m', 'image_cost', 'pose_cost'))
        live = {}
        for name, metric in (quality_metrics or {}).items():
            if not isinstance(name, str) or not name or not isinstance(metric.get('unit'), str) or not metric['unit'].strip():
                raise ValueError('quality metrics require a metric name and explicit unit')
            value = metric.get('value')
            value = None if value is None else float(value)
            live[name] = dict(unit=metric['unit'], value=value if value is not None and math.isfinite(value) else None)
        if quality_metrics is not None:sample['quality_metrics'] = live
        self.latest = deepcopy(sample)
        initial = self.baseline is None and any(sample[k] is not None for k in ('violation_m', 'image_cost', 'pose_cost'))
        if initial:self.baseline = deepcopy(sample)
        if initial or accepted:
            value = sample['violation_m']
            if value is not None and (self.best_violation_m is None or value < self.best_violation_m):
                self.best_violation_m = value;self.best_at_accepted = self.accepted
            # A missing accepted measurement breaks the comparable window.
            self._violation_history.append((self.accepted, self.best_violation_m if value is not None else None))
            self._violation_history = self._violation_history[-(self.policy.stagnation_window+1):]
            self._joint_history.append((self.accepted,sample['joint_cost']))
            self._joint_history = self._joint_history[-(self.policy.stagnation_window+1):]
        reasons = []
        def stop(code, message, **details):
            reasons.append(dict(code=code, category=_REASON_CATEGORIES[code], message=message, **details))
        self.quality_diagnostics = [];self.progress_diagnostics = []
        p = self.policy
        if sample['elapsed_seconds'] >= p.wall_seconds:
            stop('wall_budget', 'Wall-time budget reached.', elapsed_seconds=sample['elapsed_seconds'], limit_seconds=p.wall_seconds)
        if self.accepted >= p.max_accepted:
            stop('accepted_budget', 'Accepted-step budget reached.', accepted=self.accepted, limit=p.max_accepted)
        if self.outer_completed >= p.max_outer:
            stop('outer_budget', 'Completed outer-solve budget reached.', outer_completed=self.outer_completed, limit=p.max_outer)
        history = self._violation_history
        if self.violation_stagnation_enabled and accepted and len(history) == p.stagnation_window+1 and all(v is not None for _, v in history):
            if history[-1][0]-history[0][0] == p.stagnation_window:
                improvement = history[0][1]-history[-1][1]
                if improvement < p.min_violation_improvement_m:
                    joint=self._joint_history
                    comparable=(len(joint)==len(history) and
                        [i for i,_ in joint]==[i for i,_ in history] and
                        all(value is not None for _,value in joint))
                    relative=((joint[0][1]-joint[-1][1])/max(joint[0][1],1e-30)) if comparable else None
                    if relative is not None and relative > p.min_joint_relative_improvement:
                        self.progress_diagnostics.append(dict(
                            code='structural_plateau_with_joint_progress',
                            accepted_window=p.stagnation_window,
                            joint_relative_improvement=relative,
                            meaning='A structural component plateau is not stagnation of the fitted joint objective.'))
                    else:
                        stop('violation_stagnation', 'Structural violation plateau with no comparable joint-objective progress.',
                            accepted_window=p.stagnation_window, improvement_m=improvement,
                            required_improvement_m=p.min_violation_improvement_m,
                            best_violation_m=self.best_violation_m,
                            joint_relative_improvement=relative)
        if self.baseline is not None:
            for kind, limit in (('image', p.max_image_cost_increase), ('pose', p.max_pose_cost_increase)):
                base = self.baseline[kind+'_cost'];value = sample[kind+'_cost']
                if base is None or base == 0:
                    self.quality_diagnostics.append(dict(code='relative_quality_unavailable', metric=kind+'_cost',
                        reason='missing_initial_cost' if base is None else 'zero_initial_cost', baseline=base, latest=value))
                if base is not None and base > 0 and value is not None and value > base*(1+limit):
                    stop(kind+'_cost_increase', 'Raw '+kind+' cost degraded relative to the initial state.',
                        baseline=base, latest=value, relative_increase=(value-base)/base, limit=limit)
        for name, metric in live.items():
            if name not in self.quality_limits:
                self.quality_diagnostics.append(dict(code='quality_metric_diagnostic_only', metric=name, **metric))
        if measured:
            for name, limit in self.quality_limits.items():
                metric = live.get(name)
                if metric is None or metric['value'] is None or metric['unit'] != limit['unit']:
                    stop('quality_metric_unavailable', 'An explicitly guarded quality measurement is unavailable or has a different unit.',
                        metric=name, expected_unit=limit['unit'], latest=deepcopy(metric), maximum=limit['maximum'])
                elif metric['value'] > limit['maximum']:
                    stop('quality_metric_limit', 'A live quality measurement exceeded its explicit maximum.',
                        metric=name, unit=limit['unit'], latest=metric['value'], maximum=limit['maximum'])
        for failure in quality_failures or ():
            failure = deepcopy(failure)
            if not isinstance(failure.get('code'), str) or not failure['code']:
                raise ValueError('quality failure requires an explicit code')
            failure.setdefault('category', 'quality')
            if failure['category'] not in ('quality', 'domain'):
                raise ValueError('quality failure category must be quality or domain')
            failure.setdefault('message', 'Caller reported a quality/domain failure.')
            reasons.append(failure)
        if penalty_growth_requested and not p.allow_penalty_growth:
            stop('penalty_growth_disabled', 'This run policy does not permit penalty growth.')
        current_quality = [reason for reason in reasons if is_quality_failure(reason)]
        if current_quality and not self.quality_reasons:self.quality_reasons = deepcopy(current_quality)
        if reasons and not self.stop_reasons:
            self.stop_reasons = reasons;self.first_stop_now = now
        return dict(stop=bool(self.stop_reasons), reasons=deepcopy(self.stop_reasons),
            quality_failed=bool(self.quality_reasons), quality_reasons=deepcopy(self.quality_reasons), convergence_claim=False)

    def record(self):
        return dict(policy=asdict(self.policy), start_now=self.start_now, last_now=self.last_now,
            checks=self.checks, accepted=self.accepted, outer_completed=self.outer_completed,
            baseline=deepcopy(self.baseline), latest=deepcopy(self.latest),
            best_violation_m=self.best_violation_m, best_at_accepted=self.best_at_accepted,
            violation_stagnation_enabled=self.violation_stagnation_enabled,
            comparable_violation_history=[dict(accepted=n, best_violation_m=v) for n, v in self._violation_history],
            comparable_joint_history=[dict(accepted=n,joint_cost=v) for n,v in self._joint_history],
            progress_diagnostics=deepcopy(self.progress_diagnostics),
            stopped_by_policy=bool(self.stop_reasons), first_stop_now=self.first_stop_now,
            reasons=deepcopy(self.stop_reasons), quality_failed=bool(self.quality_reasons),
            quality_reasons=deepcopy(self.quality_reasons), quality_limits=deepcopy(self.quality_limits),
            quality_diagnostics=deepcopy(self.quality_diagnostics), convergence_claim=False,
            meaning='Quality/domain failure rejects the attempted candidate; budget/stagnation alone does not. Neither establishes convergence nor source acceptance.')
