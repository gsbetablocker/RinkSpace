"""Resident original residual/profile actors beside parent-only Jacobians.

Each original independent source block owns one persistent actor with residual
and profile buffers only.  The parent retains its existing geometry FD route
for full J.  No admission reserve, J section, actor eviction, nested FD pool,
source selection or replacement source mathematics is introduced here.
"""
import copy
import json
import os

from z190_process_source_evaluation import PersistentSourceActors, ProcessBlockProfileUpdate
from z190_parallel_source_evaluation import ordered_block_evaluations


class ResidualOnlySourceActors:
    def __init__(self, blocks, *, task_root, native_threads=3, q_columns=None):
        self.blocks = tuple(blocks)
        self.actor_blocks = tuple(copy.copy(block) for block in self.blocks)
        for original, actor_block in zip(self.blocks, self.actor_blocks):
            if actor_block.bundle is not original.bundle:
                raise ValueError('Residual actor parent clone must share the original profile bundle')
            vars(actor_block).pop('_absolute_geometry_fd_pool', None)
        self.delegate = PersistentSourceActors(self.actor_blocks, task_root=task_root,
            native_threads=native_threads, q_columns=q_columns, output_jacobian=False)
        # The original checkpoint callback references the ORIGINAL parent
        # blocks. Actor clones share those same bundles, so SHM profile writes
        # update the profiles consumed by the parent J route as before.
        self.profile_update = ProcessBlockProfileUpdate(self)
        self.last_stats = []
        self.memory_samples = []
        self.runtime_report_path = None

    @property
    def started(self):
        return self.delegate.started

    @property
    def closed(self):
        return self.delegate.closed

    def _report(self, phase):
        if self.runtime_report_path is not None:
            self.runtime_report_path.write_text(json.dumps(dict(
                phase=phase, parent_pid=os.getpid(), resident_workers=len(self.delegate._actors),
                actor_pids=[a['process'].pid for a in self.delegate._actors],
                source_blocks=[b.name for b in self.blocks], native_threads=self.delegate.native_threads,
                output_jacobian=False, shared_output_bytes=sum(a['spec']['bytes'] for a in self.delegate._actors),
                memory_admission_enabled=False, full_J_route='original_parent_with_attached_FD_pool',
                residual_profile_route='original_persistent_actors', last_stats=self.last_stats,
                source_rows_scales_and_operators_changed=False, original_profile_callback_reused=True,
                accuracy_accepted=False), ensure_ascii=False, separators=(',', ':')), encoding='utf8')

    def start(self):
        if not self.started:
            # Pool attachment happens later in the driver. Remove only this
            # execution attribute again at actual pickle time; source arrays,
            # geometry, masks, sigma and bundles retain their original identity.
            for block in self.actor_blocks:
                vars(block).pop('_absolute_geometry_fd_pool', None)
            self.delegate.start()
            self._report('resident_residual_actors_started')
        return self

    def synchronize_profiles(self):
        self.start()
        self.delegate.synchronize_profiles()

    def update_profiles(self, q):
        self.start()
        self.delegate.update_profiles(q)
        self.last_stats = list(self.delegate.last_stats)
        self._report('original_profiles_updated')

    def ordered_evaluations(self, q, jacobian):
        self.last_stats = []
        if jacobian:
            # The source producer's original blocks retain their attached FD
            # pool. Residual actors are idle and never run nested J/FD commands.
            for block, result in ordered_block_evaluations(self.blocks, q, True, 1):
                self.last_stats.append(dict(block=block.name, jacobian=True,
                    rows=len(result[0]), execution_route='original_parent'))
                yield block, result
            return
        self.start()
        by_name = {block.name: block for block in self.blocks}
        iterator = self.delegate.ordered_evaluations(q, False)
        try:
            for actor_block, residual in iterator:
                # A caller may retain current r while the next source command
                # overwrites SHM. Return its own small array in original order.
                yield by_name[actor_block.name], residual.copy()
        finally:
            iterator.close()
            self.last_stats = list(self.delegate.last_stats)
            self._report('original_residuals_evaluated')

    def ordered_absolute_records(self, q, held_by_name):
        self.start()
        by_name = {block.name: block for block in self.blocks}
        iterator = self.delegate.ordered_absolute_records(q, held_by_name)
        try:
            for actor_block, records in iterator:
                yield by_name[actor_block.name], records
        finally:
            iterator.close()
            self.last_stats = list(self.delegate.last_stats)
            self._report('original_absolute_records_reviewed')

    def close(self):
        self.delegate.close()
        self._report('resident_residual_actors_closed')
