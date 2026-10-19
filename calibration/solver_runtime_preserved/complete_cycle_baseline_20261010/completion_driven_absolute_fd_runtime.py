"""Keep original absolute FD workers supplied while preserving output order.

Only independent job scheduling changes. All material owners, source literals,
original worker functions, FD stencils, returned spans and caller CSR remain.
Completed jobs can wait for an earlier job without occupying an active slot.
The additional output buffer is limited to this one family's existing FD
spans, never a motion graph or full programme J.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np


def completion_driven_ordered_groups(self, groups_and_cameras, world66, absolute, step, *, batched_projection=True):
    from calibration.solver_runtime import absolute_geometry_fd_pool_candidate as source
    if self.closed:
        raise RuntimeError('Geometry FD manager closed')
    force_parent_activation = bool(getattr(self, '_rinkspace_force_parent_activation_FD', False))
    active_generators = int(getattr(self, '_rinkspace_active_geometry_group_iterators', 0))
    if force_parent_activation and active_generators:
        raise RuntimeError('Actor-busy activation parent FD requires an idle geometry FD manager')
    iterator = iter(enumerate(groups_and_cameras))
    self.last_stats = []
    self.last_route_events = []
    active, ready = {}, {}
    submitted = next_output = completed = parent_batches = 0
    buffered_bytes = 0
    parent_only = force_parent_activation
    begin = time.perf_counter()
    profile = dict(world_sha256=hashlib.sha256(np.asarray(world66, float).tobytes()).hexdigest(),
                   original_worker_function=source.evaluate_absolute_geometry_worker.__module__ + '.' + source.evaluate_absolute_geometry_worker.__qualname__,
                   requested_workers=self.workers, original_max_active_jobs=self.max_inflight,
                   original_groups_per_job=self.groups_per_job,
                   input_prepare_seconds=0., any_future_wait_seconds=0., caller_consume_seconds=0.,
                   completed_output_buffer_peak_bytes=0, completed_batches_peak=0,
                   blocked_original_order_rounds=0, errors_deferred_in_original_order=True,
                   original_numerical_producers_and_FD_stencils_retained=True,
                   force_parent_activation=force_parent_activation,
                   execution_scope='actor_busy_activation_only' if force_parent_activation else 'normal_full_J',
                   active_generators_at_entry=active_generators)

    def next_batch():
        started = time.perf_counter()
        groups = []
        for _ in range(self.groups_per_job):
            item = next(iterator, None)
            if item is None:
                break
            ordinal, (actual_pts, rows, camera) = item
            groups.append(dict(ordinal=ordinal, actual_pts=int(actual_pts), rows=rows,
                               camera25=np.asarray(camera, float).copy(),
                               source_group_sha256=source._source_group_digest(rows)))
        profile['input_prepare_seconds'] += time.perf_counter() - started
        if not groups:
            return None
        return dict(material_owner=self.material_owner, world66=np.asarray(world66, float).copy(),
                    absolute=absolute, step=float(step), batched_projection=bool(batched_projection), groups=groups)

    def checked_results(batch, results):
        if len(results) != len(batch['groups']):
            raise ValueError('Original group batch omitted')
        for job, result in zip(batch['groups'], results):
            if (result['ordinal'] != job['ordinal'] or result['actual_pts'] != job['actual_pts']
                    or result['source_group_sha256'] != job['source_group_sha256']):
                raise ValueError('Original absolute source group order/identity changed')
            self.last_stats.append(result['stats'])
            started = time.perf_counter()
            yield result
            profile['caller_consume_seconds'] += time.perf_counter() - started

    self._rinkspace_active_geometry_group_iterators = active_generators + 1
    try:
        waiting = next_batch()
        while waiting is not None or active or ready:
            # Refill from completed jobs before waiting for original-order
            # delivery. A slow earlier batch must not leave ready workers idle.
            while waiting is not None and len(active) < self.max_inflight and not parent_only:
                if not self._headroom('before_batch_submit'):
                    break
                executor = self._executor()
                if executor is None:
                    break
                future = executor.submit(source.evaluate_absolute_geometry_worker, waiting)
                active[future] = (submitted, waiting)
                submitted += 1
                waiting = next_batch()

            while next_output in ready:
                batch, future, result_bytes = ready.pop(next_output)
                buffered_bytes -= result_bytes
                # Retrieve exceptions only here: identical first original
                # batch/group error priority, regardless of completion order.
                results = future.result()
                yield from checked_results(batch, results)
                next_output += 1
                del results, future, batch

            if not active:
                if ready:
                    raise RuntimeError('Original FD batch order has a missing result')
                if waiting is None:
                    break
                if self.owns_executor and self.executor is not None:
                    self.executor.shutdown(wait=True, cancel_futures=True)
                    self.executor = None
                    self.last_route_events.append(dict(
                        phase='actor_busy_activation_idle_pool_closed' if force_parent_activation else 'low_headroom_idle_pool_closed',
                        owned_pool_closed=True))
                elif self.executor is not None:
                    self.last_route_events.append(dict(
                        phase='actor_busy_activation_parent_fallback' if force_parent_activation else 'low_headroom_parent_fallback',
                        borrowed_executor_left_owned_by_caller=True))
                parent_only = True
                if self.parent_geometry is None:
                    if source._sha(self.material_owner['path']) != self.material_owner['sha256']:
                        raise ValueError('Private original material owner changed before parent fallback')
                    import pickle
                    self.parent_geometry = pickle.loads(Path(self.material_owner['path']).read_bytes())
                common = {key: value for key, value in waiting.items() if key != 'groups'}
                results = [source._evaluate_group_on_owner(dict(common, **group), self.parent_geometry,
                           self.parent_certificates, 'parent_actor_busy_activation' if force_parent_activation else 'parent_low_headroom')
                           for group in waiting['groups']]
                yield from checked_results(waiting, results)
                parent_batches += 1
                waiting = next_batch()
                continue

            started = time.perf_counter()
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            profile['any_future_wait_seconds'] += time.perf_counter() - started
            for future in sorted(done, key=lambda future: active[future][0]):
                ordinal, batch = active.pop(future)
                # Count a completed span once. Keep cancellation/worker errors
                # for original-order result(), including a later failed job.
                result_bytes = 0
                if not future.cancelled() and future.exception() is None:
                    result_bytes = sum(result['geo'].nbytes for result in future.result())
                ready[ordinal] = (batch, future, result_bytes)
                buffered_bytes += result_bytes
                completed += 1
            done.clear()
            del future
            if ready and next_output not in ready:
                profile['blocked_original_order_rounds'] += 1
            profile['completed_output_buffer_peak_bytes'] = max(profile['completed_output_buffer_peak_bytes'], buffered_bytes)
            profile['completed_batches_peak'] = max(profile['completed_batches_peak'], len(ready))
    finally:
        self._rinkspace_active_geometry_group_iterators -= 1
        profile.update(wall_seconds=time.perf_counter() - begin, submitted_batches=submitted,
                       completed_batches=completed, original_order_output_batches=next_output + parent_batches,
                       worker_output_batches=next_output, parent_output_batches=parent_batches,
                       delivered_groups=len(self.last_stats), active_futures_at_exit=len(active),
                       buffered_batches_at_exit=len(ready))
        self.last_completion_scheduling = profile
        receipt = getattr(type(self), '_rinkspace_completion_driven_FD_runtime', None)
        if receipt is not None and receipt['report_directory'] is not None:
            destination = Path(receipt['report_directory'])
            with (destination / 'absolute_FD_completion_scheduling.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(profile, ensure_ascii=False, separators=(',', ':')) + '\n')


def install_completion_driven_absolute_fd_runtime(*, report_directory=None):
    """Install only in a new caller-owned run, before any FD jobs are sent."""
    from . import absolute_geometry_fd_pool_candidate as source
    cls = source.AbsoluteGeometryGroupPool
    destination = None if report_directory is None else Path(report_directory).resolve()
    previous = getattr(cls, '_rinkspace_completion_driven_FD_runtime', None)
    if previous is not None:
        if previous['report_directory'] != (None if destination is None else str(destination)):
            raise RuntimeError('Completion-driven FD already installed for another diagnostic directory')
        return previous
    original = cls.ordered_groups
    if original.__module__ != source.__name__ or original.__name__ != 'ordered_groups':
        raise TypeError('Original source-owned FD scheduling function required')
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).resolve()
    receipt = dict(schema='rinkspace.completion-driven-original-absolute-FD-execution.v1',
                   original_ordered_groups_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
                   original_worker_function_sha256=hashlib.sha256(inspect.getsource(source.evaluate_absolute_geometry_worker).encode()).hexdigest(),
                   helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                   report_directory=None if destination is None else str(destination),
                   original_worker_count_max_active_jobs_groups_per_job_and_release_policy_retained=True,
                   original_source_literals_groups_ordinals_domain_stencils_and_numeric_producers_retained=True,
                   original_output_order_and_exception_priority_retained=True,
                   original_headroom_and_parent_fallback_retained=True,
                   activation_parent_execution_attribute='_rinkspace_force_parent_activation_FD',
                   activation_parent_execution_scope='explicit_actor_busy_activation_only',
                   activation_parent_execution_route='parent_actor_busy_activation',
                   activation_preserves_parent_geometry_certificates_grouping_projection_and_FD_stencils=True,
                   activation_parent_execution_does_not_fabricate_memory_pressure=True,
                   normal_full_J_remains_parallel_after_activation_attribute_restored=True,
                   active_generator_count_attribute='_rinkspace_active_geometry_group_iterators',
                   buffers_only_existing_absolute_FD_spans_within_current_family=True,
                   no_full_motion_graph_or_programme_J_copy=True,
                   registered_source_files_modified=False,
                   actual_same_q_full_J_byte_equivalence_pending=True,
                   actual_fresh_J_speedup_pending=True, accuracy_accepted=False)
    cls._rinkspace_original_ordered_groups = original
    cls.ordered_groups = completion_driven_ordered_groups
    cls._rinkspace_completion_driven_FD_runtime = receipt
    if destination is not None:
        (destination / 'absolute_FD_completion_runtime_receipt.json').write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return receipt
