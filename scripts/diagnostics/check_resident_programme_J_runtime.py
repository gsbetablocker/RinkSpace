"""Tiny Windows complete-J transport, resize, bank and drainage contracts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.check_parallel_source_guard_transport import TinyGuardBlock


class TinyMaterialGeometry:
    parameter_count = 2
    def __init__(self):
        self.bindings = {'a': dict(offset=.5), 'b': dict(offset=1.5)}
    def _q(self, q):
        value = np.asarray(q, float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError('Original tiny physical world')
        return value.copy()
    validate_parameters = _q
    def _rinkspace_original_material_value(self, binding, q):
        return np.array([[binding['offset']+q[0], q[1], 3.]])


class TinyCompleteJBlock(TinyGuardBlock):
    def __init__(self, name, start, geometry, pool):
        super().__init__(name, start)
        self.geometry = geometry
        self.groups = []
        self.bundle.abs_groups = self.groups
        self.bundle.motion_rows = 4
        self.bundle.profile_weights = np.array([[.25, .75], [.5, -0.]])
        self.bundle.robust_phase = True
        self.actor_output_capacity.update(nnz=4, motion_rows=4)
        self._absolute_geometry_fd_pool = pool
        self.parent_J_calls = []
        self.parent_replay_calls = 0
    def geometry_values(self, q):
        return np.asarray(q)[:2]
    def local(self, q):
        self.parent_replay_calls += 1
        return super().local(q)
    def evaluate(self, q, jacobian):
        residual = super().evaluate(q, False)
        if not jacobian:
            return residual
        if q[self.start] > 1.:
            raise ValueError('Original synthetic J failure')
        from scipy.sparse import csr_matrix
        from calibration.solver_runtime import absolute_material_value_runtime as material
        from calibration.solver_runtime.resident_programme_J_runtime import _OriginalGroupsInActor
        self.parent_J_calls.append(os.getpid())
        bank = material._BANK_SCOPE.get()
        if bank is not None:
            assert isinstance(self._absolute_geometry_fd_pool, _OriginalGroupsInActor)
            scope = dict(job=None, geometry=self.geometry, root_world_bytes=None,
                root_world_factory=lambda:self.geometry_values(q), shared_bank=bank,
                hits=0, original_value_calls=0, uncacheable_results=0, root_cache_resets=0)
            token = material._SCOPE.set(scope)
            try:
                value = material._material_wrapper(TinyMaterialGeometry._rinkspace_original_material_value)(
                    self.geometry, self.geometry.bindings['a'], self.geometry_values(q))
                assert scope['shared_hits'] == 1 and scope['original_value_calls'] == 0
            finally:
                material._SCOPE.reset(token)
        else:
            value = self.geometry._rinkspace_original_material_value(self.geometry.bindings['a'], self.geometry_values(q))
        data = np.array([value[0,0], -0., self.bundle.profile_weights[1,0], 1.])
        return residual, csr_matrix((data, np.arange(self.start,self.start+4), np.arange(5)), shape=(4,len(q)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime import absolute_material_value_runtime as material
    from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool, _PRIVATE_GEOMETRIES
    from calibration.solver_runtime.parallel_source_guard import install_parallel_source_guard
    from calibration.solver_runtime.residual_actor_lifecycle import install_residual_actor_lifecycle
    from calibration.solver_runtime.resident_programme_J_runtime import install_resident_programme_J_runtime
    from calibration.solver_runtime.source_actor_payload_runtime import install_source_actor_payload_runtime
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    from z190_joint_source_solver import _frozen_profile_token
    from z190_process_source_evaluation import RemoteSourceError

    install_parallel_source_guard()
    install_residual_actor_lifecycle(report_directory=args.output.parent/'lifecycle')
    material.install_absolute_material_value_runtime(report_directory=args.output.parent/'material')
    receipt = install_resident_programme_J_runtime(report_directory=args.output.parent/'resident_J')
    install_source_actor_payload_runtime(report_directory=args.output.parent/'payload')
    geometry = TinyMaterialGeometry()
    snapshot_path = args.output.parent/'tiny_original_material.pickle'
    payload = pickle.dumps(geometry, protocol=5)
    snapshot_path.write_bytes(payload)
    snapshot = dict(path=str(snapshot_path),sha256=hashlib.sha256(payload).hexdigest())
    _PRIVATE_GEOMETRIES[(snapshot['path'],snapshot['sha256'])]=(geometry,set())
    # Original bank fill on a tiny thread executor; only source actors spawn.
    pool=AbsoluteGeometryGroupPool(snapshot,workers=2,parent_geometry=geometry)
    pool.executor=ThreadPoolExecutor(max_workers=2)
    names=('main1','main7','secondary','low','main9','main4')
    blocks=tuple(TinyCompleteJBlock(name,4+4*i,geometry,pool) for i,name in enumerate(names))
    q=np.zeros(28);q[:2]=[.125,.25];q[7::4]=np.log(1000.)
    expected=[block.evaluate(q,True) for block in blocks]
    for block in blocks:block.parent_J_calls.clear()
    manager=ResidualOnlySourceActors(blocks,task_root=ROOT/'calibration/solver_runtime/legacy',native_threads=1,q_columns=len(q))
    before=_frozen_profile_token(manager.profile_update)
    try:
        manager.start()
        pids=[actor['process'].pid for actor in manager.delegate._actors]
        results=list(manager.ordered_evaluations(q,True))
        assert [block.name for block,_ in results]==list(names)
        for (_, (actual_r, actual_J)), (expected_r, expected_J) in zip(results,expected,strict=True):
            assert actual_r.tobytes()==expected_r.tobytes()
            assert actual_J.data.tobytes()==expected_J.data.tobytes()
            assert actual_J.indices.tobytes()==expected_J.indices.tobytes()
            assert actual_J.indptr.tobytes()==expected_J.indptr.tobytes()
        assert all(not block.parent_J_calls and block.parent_replay_calls>0 for block in blocks)
        assert pool.executor is None and not manager.delegate.output_jacobian
        assert [actor['process'].pid for actor in manager.delegate._actors]==pids
        assert _frozen_profile_token(manager.profile_update)==before
        bad=q.copy();bad[blocks[0].start]=2.;bad[blocks[2].start]=2.
        try:list(manager.ordered_evaluations(bad,True))
        except RemoteSourceError as error:
            assert str(error).startswith('main1: ValueError: Original synthetic J failure')
        else:raise AssertionError('Original J failure must be preserved')
        assert not manager.delegate._busy and not manager.delegate.output_jacobian
        after=list(manager.ordered_evaluations(q,False))
        assert all(value.tobytes()==block.evaluate(q,False).tobytes() for block,value in after)
        early=manager.ordered_evaluations(q,True);next(early);early.close()
        assert not manager.delegate._busy and not manager.delegate.output_jacobian
        assert [actor['process'].pid for actor in manager.delegate._actors]==pids
        manager.guard(q)
        assert _frozen_profile_token(manager.profile_update)==before
        assert not list((args.output.parent/'payload/transient_owned_payloads').glob('*.pickle'))
    finally:
        manager.close();pool.close()
    report=dict(schema='rinkspace.tiny-resident-full-J-Windows-transport.v1',passed=True,installation=receipt,
        original_guard_entry_capability_retained=True,original_file_payload_transport_used=True,
        full_J_and_signed_zero_CSR_bytes_exact=True,full_frozen_profiles_bytes_exact=True,
        existing_actor_pids_retained=pids,existing_FD_executor_released_before_actor_J=True,
        shared_material_bank_actual_hit_without_original_recomputation=True,
        first_original_order_failure_and_early_close_drained=True,subsequent_original_r_and_guard_valid=True,
        parent_local_replay_executed_without_parent_J=True,owned_actors_closed=True,
        complete_event_source_graph_loaded=False,performance_or_accuracy_accepted=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
