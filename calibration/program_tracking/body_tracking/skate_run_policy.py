"""Explicit, bounded run monitoring; a policy stop is not convergence."""
from dataclasses import asdict, dataclass
import math
from numbers import Integral


@dataclass(frozen=True)
class RunPolicy:
    wall_seconds: float = 900
    max_accepted: int = 8
    max_outer: int = 1
    stagnation_window: int = 4
    min_violation_improvement_m: float = .0001
    max_image_cost_increase: float = .05
    max_pose_cost_increase: float = .10
    allow_penalty_growth: bool = False

    def __post_init__(self):
        for name in ('wall_seconds', 'min_violation_improvement_m',
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


class Monitor:
    """Monitor coherent initial/accepted-state measurements, never trial fits.

    ``now`` is caller-supplied monotonic seconds. ``outer_completed`` counts
    completed outer solves, so max_outer=1 permits the first solve to run.
    Image/pose costs must be the same RAW component definitions at every call,
    independent of AL penalty and multiplier changes. Missing/nonfinite or
    zero-baseline costs cannot establish relative degradation.
    """
    def __init__(self, policy=None, *, now):
        self.policy = RunPolicy() if policy is None else policy
        self.start_now = float(now)
        if not math.isfinite(self.start_now):raise ValueError('now must be finite monotonic seconds')
        self.last_now = self.start_now
        self.baseline = None;self.latest = None
        self.checks = 0;self.accepted = 0;self.outer_completed = 0
        self.best_violation_m = None;self.best_at_accepted = None
        self._violation_history = []
        self.stop_reasons = [];self.first_stop_now = None

    def observe(self, *, now, violation_m=None, image_cost=None, pose_cost=None,
                accepted=False, outer_completed=None, penalty_growth_requested=False):
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
            violation_m=_metric(violation_m), image_cost=_metric(image_cost), pose_cost=_metric(pose_cost))
        self.latest = sample.copy()
        initial = self.baseline is None and any(sample[k] is not None for k in ('violation_m', 'image_cost', 'pose_cost'))
        if initial:self.baseline = sample.copy()
        if initial or accepted:
            value = sample['violation_m']
            if value is not None and (self.best_violation_m is None or value < self.best_violation_m):
                self.best_violation_m = value;self.best_at_accepted = self.accepted
            # A missing accepted measurement breaks the comparable window.
            self._violation_history.append((self.accepted, self.best_violation_m if value is not None else None))
            self._violation_history = self._violation_history[-(self.policy.stagnation_window+1):]
        reasons = []
        def stop(code, message, **details):reasons.append(dict(code=code, message=message, **details))
        p = self.policy
        if sample['elapsed_seconds'] >= p.wall_seconds:
            stop('wall_budget', 'Wall-time budget reached.', elapsed_seconds=sample['elapsed_seconds'], limit_seconds=p.wall_seconds)
        if self.accepted >= p.max_accepted:
            stop('accepted_budget', 'Accepted-step budget reached.', accepted=self.accepted, limit=p.max_accepted)
        if self.outer_completed >= p.max_outer:
            stop('outer_budget', 'Completed outer-solve budget reached.', outer_completed=self.outer_completed, limit=p.max_outer)
        history = self._violation_history
        if accepted and len(history) == p.stagnation_window+1 and all(v is not None for _, v in history):
            if history[-1][0]-history[0][0] == p.stagnation_window:
                improvement = history[0][1]-history[-1][1]
                if improvement < p.min_violation_improvement_m:
                    stop('violation_stagnation', 'Best physical violation did not improve enough over accepted steps.',
                        accepted_window=p.stagnation_window, improvement_m=improvement,
                        required_improvement_m=p.min_violation_improvement_m, best_violation_m=self.best_violation_m)
        if self.baseline is not None:
            for kind, limit in (('image', p.max_image_cost_increase), ('pose', p.max_pose_cost_increase)):
                base = self.baseline[kind+'_cost'];value = sample[kind+'_cost']
                if base is not None and base > 0 and value is not None and value > base*(1+limit):
                    stop(kind+'_cost_increase', 'Raw '+kind+' cost degraded relative to the initial state.',
                        baseline=base, latest=value, relative_increase=(value-base)/base, limit=limit)
        if penalty_growth_requested and not p.allow_penalty_growth:
            stop('penalty_growth_disabled', 'This run policy does not permit penalty growth.')
        if reasons and not self.stop_reasons:
            self.stop_reasons = reasons;self.first_stop_now = now
        return dict(stop=bool(self.stop_reasons), reasons=[r.copy() for r in self.stop_reasons], convergence_claim=False)

    def record(self):
        return dict(policy=asdict(self.policy), start_now=self.start_now, last_now=self.last_now,
            checks=self.checks, accepted=self.accepted, outer_completed=self.outer_completed,
            baseline=None if self.baseline is None else self.baseline.copy(),
            latest=None if self.latest is None else self.latest.copy(),
            best_violation_m=self.best_violation_m, best_at_accepted=self.best_at_accepted,
            comparable_violation_history=[dict(accepted=n, best_violation_m=v) for n, v in self._violation_history],
            stopped_by_policy=bool(self.stop_reasons), first_stop_now=self.first_stop_now,
            reasons=[r.copy() for r in self.stop_reasons], convergence_claim=False,
            meaning='A budget or quality stop retains a candidate; it establishes neither convergence nor source acceptance.')
