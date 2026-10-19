"""Tiny transport proof for fresh-J residual actor lifetime, without event data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.check_parallel_source_guard_transport import TinyGuardBlock


class ParentFDPoolSentinel:
    def __init__(self):
        self.released = False

    def __getstate__(self):
        raise RuntimeError('The parent FD pool must never be transmitted to residual actors')

    def release_idle(self):
        self.released = True


class TinyJBlock(TinyGuardBlock):
    def __init__(self, name, start, fd_pool):
        super().__init__(name, start)
        import numpy as np
        self.bundle.motion_rows = 4
        self.bundle.profile_weights = np.array([[.25, .75], [.5, -0.]], dtype=np.float64)
        self.bundle.robust_phase = True
        self.actor_output_capacity['motion_rows'] = 4
        self._absolute_geometry_fd_pool = fd_pool
        self.jacobian_parent_pids = []

    def evaluate(self, q, jacobian):
        result = super().evaluate(q, False)
        if not jacobian:
            return result
        import numpy as np
        from scipy.sparse import csr_matrix
        if self._absolute_geometry_fd_pool.released:
            raise RuntimeError('J must use the original parent FD stage before its release')
        self.jacobian_parent_pids.append(os.getpid())
        return result, csr_matrix((np.ones(4), np.arange(self.start, self.start+4), np.arange(5)),
            shape=(4, len(q)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retain-idle-residual-actors', action='store_true')
    args = parser.parse_args()
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.parallel_source_guard import install_parallel_source_guard
    from calibration.solver_runtime.residual_actor_lifecycle import install_residual_actor_lifecycle
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    from z190_joint_source_solver import _frozen_profile_token
    import numpy as np

    install_parallel_source_guard()
    installation = install_residual_actor_lifecycle(report_directory=args.output.parent,
        retain_idle_residual_actors=args.retain_idle_residual_actors)
    names = ('main1', 'main7', 'secondary', 'low', 'main9', 'main4')
    fd_pool = ParentFDPoolSentinel()
    blocks = tuple(TinyJBlock(name, 4+4*i, fd_pool) for i, name in enumerate(names))
    q = np.zeros(28)
    q[7::4] = np.log(1000.)
    actors = ResidualOnlySourceActors(blocks, task_root=ROOT/'calibration/solver_runtime/legacy',
        native_threads=3, q_columns=len(q))
    tokens = [_frozen_profile_token(actors.profile_update)]
    profiles = actors.profile_update.checkpoint_arrays()
    try:
        actors.start()
        original_delegate = actors.delegate
        old_processes = [actor['process'] for actor in original_delegate._actors]
        old_pids = [process.pid for process in old_processes]
        tokens.append(_frozen_profile_token(actors.profile_update))
        results = list(actors.ordered_evaluations(q, True))
        if args.retain_idle_residual_actors:
            assert actors.delegate is original_delegate and original_delegate.started
            assert not original_delegate.closed and all(process.is_alive() for process in old_processes)
            assert [actor['process'].pid for actor in original_delegate._actors] == old_pids
        else:
            assert original_delegate.closed and not original_delegate._actors
            assert all(not process.is_alive() for process in old_processes)
            assert actors.delegate is not original_delegate and not actors.delegate.started
            assert not actors.delegate._actors
        assert actors.delegate.native_threads == 3 and actors.delegate.q_columns == len(q)
        assert not actors.delegate.output_jacobian
        assert not fd_pool.released
        assert [block for block, _ in results] == list(blocks)
        for block, (residual, matrix) in results:
            assert block.jacobian_parent_pids == [os.getpid()]
            assert block._absolute_geometry_fd_pool is fd_pool
            assert np.array_equal(residual, block.local(q)[4:])
            assert np.array_equal(matrix.data, np.ones(4))
            assert np.array_equal(matrix.indices, np.arange(block.start, block.start+4))
        tokens.append(_frozen_profile_token(actors.profile_update))
        # Mirror the real raw source's existing stage-end callback before a
        # later guard/r command asks the ordinary lazy actors to start again.
        fd_pool.release_idle()
        q = q.copy()
        q[4::4] = .25
        actors.guard(q)
        new_pids = [actor['process'].pid for actor in actors.delegate._actors]
        assert actors.delegate.started and len(new_pids) == 6
        assert (new_pids == old_pids) == args.retain_idle_residual_actors
        assert all(actor.get('finite_extent_guard_supported') for actor in actors.delegate._actors)
        residuals = list(actors.ordered_evaluations(q, False))
        assert [block.name for block, _ in residuals] == list(names)
        assert all(np.array_equal(value, block.local(q)[4:]) for block, value in residuals)
        tokens.append(_frozen_profile_token(actors.profile_update))
        after = actors.profile_update.checkpoint_arrays()
        assert profiles.keys() == after.keys()
        assert all(profiles[name].dtype == after[name].dtype and profiles[name].shape == after[name].shape
            and profiles[name].tobytes() == after[name].tobytes() for name in profiles)
        assert len(set(tokens)) == 1
        events = list(actors.residual_lifecycle_events)
    finally:
        actors.close()
    assert all(not process.is_alive() for process in old_processes)
    report = dict(schema='rinkspace.tiny-residual-fresh-J-lifetime-check.v1', passed=True,
        installation=installation, original_owned_actor_pids=old_pids,
        subsequent_residual_owned_actor_pids=new_pids, profile_tokens=tokens,
        retained_idle_residual_actors=args.retain_idle_residual_actors,
        nonempty_signed_zero_frozen_profiles_byte_exact=True,
        original_parent_J_blocks_and_FD_pool_identity_preserved=True,
        fresh_J_computed_only_in_parent_pid=os.getpid(),
        old_owned_processes_closed_before_J=not args.retain_idle_residual_actors,
        residual_workers_not_restarted_during_J=True,
        original_FD_stage_release_precedes_lazy_restart=not args.retain_idle_residual_actors,
        subsequent_changed_q_original_r_exact=True, subsequent_complete_guard_succeeds=True,
        owned_actors_closed=True, events=events, real_programme_graph_loaded=False,
        memory_threshold_or_performance_acceptance_claimed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(dict(passed=True, report=str(args.output.resolve()))), flush=True)


if __name__ == '__main__':
    main()
