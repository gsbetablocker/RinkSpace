"""Exercise new guard-command ordering and drainage with tiny owned actors.

This is a transport test, not a replacement for the completed actual source
certification. It never loads an event source graph or video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class TinyGuardBundle:
    def __init__(self):
        self.profile_weights = None
        self.robust_phase = False
        self.motion_rows = 0

    def cache(self, q, jacobian):
        if jacobian:
            raise ValueError('The original finite extent guard requests values only')
        return q

    def validate_finite_extents(self, cache):
        if cache[4] > 1.:
            raise ValueError('Synthetic finite extent rejection')


class TinyGuardBlock:
    def __init__(self, name, start):
        self.name, self.start = name, start
        self.bundle = TinyGuardBundle()
        self.actor_output_capacity = dict(rows=4, nnz=0, motion_rows=0)

    def local(self, q):
        import numpy as np
        return np.r_[q[:4], q[self.start:self.start+4]]

    def evaluate(self, q, jacobian):
        if jacobian:
            raise ValueError('Residual-only fixture')
        return self.local(q)[4:].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.parallel_source_guard import install_parallel_source_guard
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    from z190_process_source_evaluation import RemoteSourceError
    import numpy as np

    installation = install_parallel_source_guard()
    names = ('main1', 'main7', 'secondary', 'low', 'main9', 'main4')
    blocks = tuple(TinyGuardBlock(name, 4+4*i) for i, name in enumerate(names))
    q = np.zeros(28)
    q[7::4] = np.log(1000.)
    actors = ResidualOnlySourceActors(blocks, task_root=ROOT/'calibration/solver_runtime/legacy',
        native_threads=3, q_columns=len(q))
    records = []
    try:
        actors.guard(q)
        assert len(actors.last_stats) == 6
        assert all(actor.get('finite_extent_guard_supported') for actor in actors.delegate._actors)
        first_bad = q.copy()
        first_bad[blocks[0].start+3] = np.log(50.)
        first_bad[blocks[2].start] = 2.
        later_bad = q.copy()
        later_bad[blocks[1].start] = 2.
        later_bad[blocks[3].start] = 2.
        for label, value, first_name, message in (
                ('first_focal_beats_later_finite', first_bad, names[0], 'Physical focal domain'),
                ('first_finite_beats_later_finite', later_bad, names[1], 'Synthetic finite extent rejection')):
            try:
                actors.guard(value)
            except RemoteSourceError as exc:
                assert str(exc).startswith(first_name + ': ValueError: ' + message)
                records.append(dict(case=label, original_order_first_error=True,
                    rejected_as_original_value_error=True, message=str(exc).splitlines()[0]))
            else:
                raise AssertionError('Guard must reject the injected original source failure')
            assert not actors.delegate._busy
            # All sent replies must be drained before this next ordinary r
            # command. A leftover guard reply would corrupt its rows metadata.
            results = list(actors.ordered_evaluations(q, False))
            assert [block.name for block, _ in results] == list(names)
            assert all(np.array_equal(value, block.evaluate(q, False)) for block, value in results)
            actors.guard(q)
            assert len(actors.last_stats) == 6
            assert all(block.bundle.profile_weights is None and not block.bundle.robust_phase for block in blocks)
        pids = [actor['process'].pid for actor in actors.delegate._actors]
    finally:
        actors.close()
    assert actors.closed and not actors.delegate._actors
    report = dict(schema='rinkspace.tiny-resident-guard-order-and-drain-check.v1', passed=True,
        installation=installation, records=records, native_threads_per_actor=3,
        actual_guard_capability_seen_in_ready=True, original_frozen_profiles_unchanged=True,
        ordinary_residual_after_each_failure_exact=True, valid_guard_after_each_failure_succeeds=True,
        owned_actors_closed=True, pids=pids, real_programme_source_graph_loaded=False,
        performance_or_localization_acceptance_claimed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(dict(passed=True, report=str(args.output.resolve()))), flush=True)


if __name__ == '__main__':
    main()
