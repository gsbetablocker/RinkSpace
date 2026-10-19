"""Original actor scheduling with a bounded, idle-only native parent route.

An idle pool cannot make progress by waiting for the IPC allocations which
the next block itself requires.  Try the original actor admission first;
after releasing only idle owned actors, evaluate the original block in the
parent when the unchanged memory reserve permits it.  No source operator,
profile, q chart, row order or native backend is replaced here.
"""
from dataclasses import asdict
import json
import math
import time

from .bounded_source_actors import BoundedSourceActors
from z190_parallel_source_evaluation import ordered_block_evaluations


class IdleParentFallbackSourceActors(BoundedSourceActors):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_fallback_records = []
        self.parent_fallback_report_path = None

    def _require_idle(self):
        if self.active_names or self._starting_parent is not None or any(
                manager._busy for manager in self.pool.values()):
            raise RuntimeError('Native parent fallback requires all owned actors idle')

    def _release_idle_pool(self):
        self._require_idle()
        released = list(self.pool)
        for name in released:
            self._remove(name)
        return released

    def _parent_admission(self, block, mode, jacobian=False):
        """Reserve native growth, without allocations unique to actor IPC.

        Use the existing measured command-mode peak with its original 1.25
        margin, or the larger original caller output reserve.  Existing parent
        source arrays and retained earlier r/J already reduce the fresh system
        counters.  They are neither subtracted from the measured peak nor freed.
        An unmeasured first command fails explicitly instead of waiting forever.
        """
        from rinkspace.runtime_memory import read_system_memory, memory_pressure
        self._require_idle()
        if self.pool:
            raise RuntimeError('Parent admission requires released idle actor memory')
        self.sample('queued_parent_admission_' + block.name)
        evaluation_mode, measured, unused = self._mode_calibration(mode, jacobian)
        peak = self._block_peak(block, mode, jacobian) if measured else None
        system = read_system_memory()
        caller = self._caller_inflight_bytes([block], mode, jacobian)
        phys = max(caller, math.ceil(peak.rss_bytes * 1.25)) if peak else caller
        commit = max(caller, math.ceil(peak.private_bytes * 1.25)) if peak else caller
        allowed = bool(peak and peak.samples > 0 and peak.rss_bytes > 0 and
            peak.private_bytes > 0 and not memory_pressure(system,
                reserve_physical_bytes=self.reserve_bytes,
                reserve_commit_bytes=self.reserve_bytes,
                inflight_physical_bytes=phys, inflight_commit_bytes=commit))
        record = dict(phase='idle_parent_admission_' + block.name, block=block.name,
            mode=mode, jacobian=bool(jacobian), evaluation_mode=evaluation_mode,
            allowed=allowed, measured_command_peak=None if peak is None else asdict(peak),
            peak_margin=1.25, caller_output_reserve_bytes=caller,
            native_growth_physical_bytes=phys, native_growth_commit_bytes=commit,
            reserve_physical_bytes=self.reserve_bytes, reserve_commit_bytes=self.reserve_bytes,
            system=asdict(system), owned_actor_count=len(self.pool),
            busy_actor_count=len(self.active_names), actor_pickle_and_shared_output_allocated=False)
        self._event(record)
        return allowed, record

    def _record_parent(self, record):
        self.parent_fallback_records.append(record)
        self.parent_fallback_records = self.parent_fallback_records[-128:]
        if self.parent_fallback_report_path is not None:
            self.parent_fallback_report_path.write_text(json.dumps(dict(
                execution_route_only=True, reserve_bytes=self.reserve_bytes,
                records=self.parent_fallback_records), ensure_ascii=False,
                separators=(',', ':')), encoding='utf8')

    def _parent_command(self, block, mode, q, jacobian=False, held_by_name=None):
        allowed, budget = self._parent_admission(block, mode, jacobian)
        if not allowed:
            self.queued_blocks = [block.name]
            self.queue_reason = ('Neither actor IPC nor idle native parent evaluation fits '
                'the unchanged reserve, or no measured command peak is available')
            self._record_parent(dict(status='denied', admission=budget))
            raise MemoryError(self.queue_reason + ': ' + block.name)
        begin, cpu = time.perf_counter(), time.process_time()
        self.queued_blocks = [block.name]
        self.queue_reason = 'Executing original source in idle parent without actor IPC'
        self.sample('before_idle_parent_' + block.name)
        try:
            if mode == 'evaluate':
                # This is exactly CurrentProgrammeSource's serial evaluation.
                unused, result = next(ordered_block_evaluations((block,), q, jacobian, 1))
                residual = result[0] if jacobian else result
                if residual.shape != (int(block.actor_output_capacity['rows']),):
                    raise ValueError('Original source residual row inventory changed')
                if jacobian and result[1].nnz > int(block.actor_output_capacity['nnz']):
                    raise ValueError('Source CSR exceeded original layout capacity')
                checks = getattr(getattr(block.bundle, 'motion_kernel', None), 'latest_checks', None)
                if checks is not None:
                    checks.clear()
                stats = dict(rows=len(residual), jacobian=bool(jacobian))
                if jacobian:
                    stats.update(nnz=int(result[1].nnz), columns=result[1].shape[1])
            elif mode == 'review':
                # Identical source review and state restoration to the original
                # actor worker; HELD records never enter numerical evaluation.
                bundle = block.bundle
                groups, weights, robust = bundle.abs_groups, bundle.profile_weights, bundle.robust_phase
                try:
                    result = list(block.absolute_records(q))
                    result.extend(block.absolute_records(q, held_by_name[block.name]))
                finally:
                    bundle.abs_groups = groups
                    bundle.profile_weights, bundle.robust_phase = weights, robust
                stats = dict(record_count=len(result))
            else:
                raise ValueError('Idle native parent route supports evaluate and review only')
            self.sample('after_idle_parent_' + block.name)
            stats.update(block=block.name, status='ok', execution_route='idle_native_parent',
                wall_seconds=time.perf_counter()-begin, process_cpu_seconds=time.process_time()-cpu)
            self.last_stats.append(stats)
            self._record_parent(dict(status='ok', admission=budget, execution=stats))
            return result
        except Exception as exc:
            self._record_parent(dict(status='error', admission=budget,
                error_type=type(exc).__name__, error=str(exc)))
            raise
        finally:
            self.queued_blocks = []
            self.queue_reason = None

    def _ordered_commands(self, mode, q, jacobian=False, held_by_name=None):
        if self._busy:
            raise RuntimeError('One source evaluation may be active at a time')
        if self.closed:
            raise RuntimeError('Source actors have closed')
        self._busy = True
        self.last_stats = []
        index = 0
        try:
            while index < len(self.blocks):
                batch = []
                for block in self.blocks[index:index+self.requested_workers]:
                    if not self._ensure(block, {b.name for b in batch}, mode, jacobian):
                        break
                    if not self._can_execute(batch + [block], mode, jacobian):
                        break
                    batch.append(block)
                    if not self._mode_calibration(mode, jacobian)[1]:
                        break
                if batch:
                    submitted = yield from self._submit_group(batch, mode, q, jacobian, held_by_name)
                    if submitted:
                        index += len(batch)
                        continue
                # No command is outstanding after ordered receive/finally.
                # Release idle peers once and retry the original actor route.
                # Do not sleep indefinitely with a fully idle, denied pool.
                block = self.blocks[index]
                released = self._release_idle_pool()
                self._event(dict(phase='idle_admission_no_progress', block=block.name,
                    mode=mode, jacobian=bool(jacobian), released_idle_owned_blocks=released,
                    active_commands=0, source_rows_omitted=False))
                if self._ensure(block, {'idle_single_retry'}, mode, jacobian) and self._can_execute([block], mode, jacobian):
                    submitted = yield from self._submit_group([block], mode, q, jacobian, held_by_name)
                    if submitted:
                        index += 1
                        continue
                self._release_idle_pool()
                result = self._parent_command(block, mode, q, jacobian, held_by_name)
                yield block, result
                del result
                index += 1
            self.queued_blocks = []
            self.queue_reason = None
        finally:
            self._busy = False

    def ordered_evaluations(self, q, jacobian):
        yield from self._ordered_commands('evaluate', q, jacobian)

    def ordered_absolute_records(self, q, held_by_name):
        yield from self._ordered_commands('review', q, held_by_name=held_by_name)
