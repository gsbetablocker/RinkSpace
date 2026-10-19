"""Unregistered candidate: small private absolute-geometry FD process jobs.

This schedules only existing per-frame material FD spans. The caller retains
the original bundle evaluation, native motion chunk boundaries, local CSR,
global scatter, masks/scales, profiles, ledger and normal accumulation. A
private static material owner is loaded once per worker; no programme bundle,
motion UV graph, q34304 or full-source CSR enters a process job.

Prefer borrowing the already-owned pattern process executor. The module does
not alter or register any live runtime module or install a source hook.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import multiprocessing
import os
from pathlib import Path
import pickle
import time

import numpy as np

_PRIVATE_GEOMETRIES = {}
_VERIFIED_ABSOLUTE = set()
_INITIALIZED = False


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_group_digest(rows):
    # Exact original source literals, including fields absent from a row.
    return hashlib.sha256(pickle.dumps(rows, protocol=5)).hexdigest()


def freeze_private_material_owner(geometry, groups, destination):
    """Save only a registered material owner; never construct a source graph.

    Invoke once from the actual already-prepared caller. Its live geometry is
    not mutated. Original source certificates and producer types are retained;
    unused row registration/companion ledgers and disposable caches are omitted
    from the worker snapshot. Both full programme owners and bundles are absent.
    """
    from . import install
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle, install_geometry_unpickle
    install_mappingproxy_pickle(); install_geometry_unpickle()
    rows=[row for _, group in groups for row in group]
    keys={geometry._key(row) for row in rows}
    if not keys or not keys.issubset(geometry.bindings):
        raise ValueError('Only existing registered source materials may enter the worker owner')
    private=copy.copy(geometry)
    private.__dict__=dict(vars(geometry))
    private.bindings={key:geometry.bindings[key] for key in keys}
    # This prefix is used by qualification/finite-board registration, not by
    # already-qualified source rebind/activity/fields used in the FD job.
    private.__dict__.pop('_qualification_prefix',None)
    private._scene_cache=OrderedDict(); private._value_cache=OrderedDict()
    private._successful_world_validation=OrderedDict()
    private.registration_ledger=[geometry.bindings[key]['contract'] for key in keys]
    if '_companion_alias_certificates' in vars(geometry):
        private._companion_alias_certificates={key:value for key,value in geometry._companion_alias_certificates.items() if key[0] in keys}
        private.companion_alias_ledger=list(private._companion_alias_certificates.values())
    material_view=vars(geometry).get('_source_ice_material_view')
    if material_view is not None:
        view=copy.copy(material_view);view.__dict__=dict(vars(material_view))
        # Keep the actual producer and its rules; disposable same-q features
        # are reconstructed by the original view in this worker.
        if 'cache' in vars(view):view.cache=OrderedDict()
        if 'cache_bytes' in vars(view):view.cache_bytes=0
        private._source_ice_material_view=view
    path=Path(destination).resolve()
    if path.exists():raise FileExistsError('Private material snapshots are immutable; use a fresh path')
    payload=pickle.dumps(private,protocol=5)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    return dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),serialized_bytes=len(payload),
        registered_material_count=len(keys),source_row_count=len(rows),
        world_parameter_count=int(geometry.parameter_count),
        original_geometry_type=type(geometry).__module__+'.'+type(geometry).__qualname__,
        original_bundle_or_motion_graph_transmitted=False,live_owner_modified=False)


def initialize_absolute_geometry_worker():
    """Light qualified package entry; identical existing runtime bootstrap."""
    global _INITIALIZED
    if _INITIALIZED:return
    from . import install
    install()
    from prepared_cache_helpers import install_mappingproxy_pickle, install_geometry_unpickle
    install_mappingproxy_pickle();install_geometry_unpickle()
    from structured_programme_source_candidate.runtime_backend_contract import restore_native_backend_contract
    restore_native_backend_contract(stage='absolute_geometry_fd_worker_initializer',native_threads=1)
    import cv2
    cv2.setNumThreads(1)
    _INITIALIZED=True


def _evaluate_one_absolute_geometry_group(job):
    """Run the caller's unchanged FD helper against a worker-private owner."""
    initialize_absolute_geometry_worker()
    snapshot=job['material_owner'];token=(snapshot['path'],snapshot['sha256'])
    if token not in _PRIVATE_GEOMETRIES:
        if _sha(snapshot['path'])!=snapshot['sha256']:
            raise ValueError('Private original material owner changed before worker load')
        geometry=pickle.loads(Path(snapshot['path']).read_bytes())
        _PRIVATE_GEOMETRIES[token]=(geometry,set())
    geometry,certificates=_PRIVATE_GEOMETRIES[token]
    absolute=job['absolute']
    identity=(getattr(absolute,'__module__',type(absolute).__module__),getattr(absolute,'__qualname__',type(absolute).__qualname__))
    if identity not in _VERIFIED_ABSOLUTE:
        from structured_programme_source_candidate.runtime_backend_contract import restore_native_backend_contract
        restore_native_backend_contract(stage='absolute_geometry_fd_after_original_operator_unpickle',native_threads=1)
        _VERIFIED_ABSOLUTE.add(identity)
    return _evaluate_group_on_owner(job,geometry,certificates,'private_worker')


def _evaluate_group_on_owner(job,geometry,certificates,route):
    """The same helper in worker or parent; no parent backend mutation."""
    snapshot=job['material_owner'];absolute=job['absolute']
    rows=job['rows']
    if _source_group_digest(rows)!=job['source_group_sha256']:
        raise ValueError('Actual original source group changed in worker transport')
    world=np.asarray(job['world66'],float);camera=np.asarray(job['camera25'],float)
    ng=int(geometry.parameter_count)
    if ng<1 or world.shape!=(ng,) or camera.shape!=(25,):
        raise ValueError('Current material world parameter count and actual camera25 required')
    from structured_programme_source_candidate.batched_absolute_geometry_fd import evaluate_geometry_group
    begin=time.perf_counter();cpu=time.process_time()
    geo=evaluate_geometry_group(geometry,world,camera,rows,absolute,float(job['step']),
        certificates=certificates,batched_projection=bool(job['batched_projection']))
    geo=np.asarray(geo,float)
    expected=sum(2*len(row['uv']) for row in rows)
    if geo.shape!=(expected,ng) or not np.isfinite(geo).all():raise ValueError('Original absolute geometry span/count changed')
    return dict(geo=geo,ordinal=job['ordinal'],actual_pts=job['actual_pts'],
        source_group_sha256=job['source_group_sha256'],source_row_ids=[row['id'] for row in rows],
        stats=dict(pid=__import__('os').getpid(),geometry_sha256=snapshot['sha256'],source_group_sha256=job['source_group_sha256'],
            source_rows=len(rows),scalar_rows=expected,world_parameter_count=ng,result_bytes=geo.nbytes,
            evaluate_seconds=time.perf_counter()-begin,evaluate_cpu_seconds=time.process_time()-cpu,
            batched_projection=bool(job['batched_projection']),FD_step=float(job['step']),
            physical_domain_FD_stencils=getattr(geometry,'_last_absolute_geometry_fd_domain_checks',[]),
            private_certificate_count=len(certificates),actual_route=route))


def evaluate_absolute_geometry_worker(batch):
    """Amortize IPC over original contiguous source groups, no arithmetic merge."""
    common={key:value for key,value in batch.items() if key!='groups'}
    results=[]
    for group in batch['groups']:
        job=dict(common,**group)
        results.append(_evaluate_one_absolute_geometry_group(job))
    return results


class AbsoluteGeometryGroupPool:
    """Ordered bounded jobs, optionally on the existing owned pattern pool.

    executor is caller-owned when supplied; this manager then starts no new
    process and never shuts that pool down. Memory admission remains with the
    existing process owner. Only at most max_inflight small jobs are pending.
    """
    def __init__(self,material_owner,*,executor=None,workers=8,max_inflight=None,groups_per_job=16,parent_geometry=None):
        self.material_owner=dict(material_owner);self.executor=executor
        self.workers=int(workers);self.max_inflight=self.workers if max_inflight is None else int(max_inflight)
        self.groups_per_job=int(groups_per_job)
        if min(self.workers,self.max_inflight,self.groups_per_job)<1:raise ValueError('Positive existing owned concurrency/group chunk required')
        self.owns_executor=executor is None;self.closed=False;self.last_stats=[]
        self.parent_geometry=parent_geometry;self.parent_certificates=set()
        self.reserve_bytes=int(os.environ.get('RINKSPACE_SOLVER_MEMORY_RESERVE_BYTES','0'))
        if self.reserve_bytes<0:raise ValueError('Explicit memory reserve must be nonnegative')
        self.last_route_events=[]

    def _headroom(self,phase):
        # The user-requested default keeps the bounded pool resident. Optional
        # engineering headroom is checked only when explicitly configured.
        if self.reserve_bytes==0:return True
        from rinkspace.runtime_memory import read_system_memory,memory_pressure
        system=read_system_memory()
        pressure=memory_pressure(system,reserve_physical_bytes=self.reserve_bytes,reserve_commit_bytes=self.reserve_bytes)
        if pressure:
            self.last_route_events.append(dict(phase=phase,physical_available_bytes=system.physical_available_bytes,
                commit_available_bytes=system.commit_available_bytes,engineering_reserve_bytes=self.reserve_bytes,
                pressure=True))
        return not pressure

    def _executor(self):
        if self.closed:raise RuntimeError('Geometry FD manager closed')
        if self.executor is None:
            if not self._headroom('before_first_process_pool_start'):return None
            self.executor=ProcessPoolExecutor(max_workers=self.workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_absolute_geometry_worker)
        return self.executor

    def ordered_groups(self,groups_and_cameras,world66,absolute,step,*,batched_projection=True):
        if self.closed:raise RuntimeError('Geometry FD manager closed')
        iterator=iter(enumerate(groups_and_cameras));pending=deque();self.last_stats=[];self.last_route_events=[]
        def next_batch():
            groups=[]
            for _ in range(self.groups_per_job):
                item=next(iterator,None)
                if item is None:break
                ordinal,(actual_pts,rows,camera)=item
                groups.append(dict(ordinal=ordinal,actual_pts=int(actual_pts),rows=rows,
                    camera25=np.asarray(camera,float).copy(),source_group_sha256=_source_group_digest(rows)))
            if not groups:return None
            batch=dict(material_owner=self.material_owner,world66=np.asarray(world66,float).copy(),absolute=absolute,
                step=float(step),batched_projection=bool(batched_projection),groups=groups)
            return batch
        def checked_results(batch,results):
            if len(results)!=len(batch['groups']):raise ValueError('Original group batch omitted')
            for job,result in zip(batch['groups'],results):
                if (result['ordinal']!=job['ordinal'] or result['actual_pts']!=job['actual_pts']
                    or result['source_group_sha256']!=job['source_group_sha256']):
                    raise ValueError('Original absolute source group order/identity changed')
                self.last_stats.append(result['stats']);yield result
        waiting=next_batch();parent_only=False
        while waiting is not None or pending:
            while waiting is not None and len(pending)<self.max_inflight and not parent_only:
                if not self._headroom('before_batch_submit'):break
                executor=self._executor()
                if executor is None:break
                pending.append((waiting,executor.submit(evaluate_absolute_geometry_worker,waiting)))
                waiting=next_batch()
            if not pending:
                if waiting is None:break
                # A low-headroom resident pool is now idle. Release only this
                # owned pool before parent work, rather than add parent arrays
                # while all eight process imports/caches remain resident.
                if self.owns_executor and self.executor is not None:
                    self.executor.shutdown(wait=True,cancel_futures=True);self.executor=None
                    self.last_route_events.append(dict(phase='low_headroom_idle_pool_closed',owned_pool_closed=True))
                elif self.executor is not None:
                    self.last_route_events.append(dict(phase='low_headroom_parent_fallback',borrowed_executor_left_owned_by_caller=True))
                parent_only=True
                if self.parent_geometry is None:
                    if _sha(self.material_owner['path'])!=self.material_owner['sha256']:
                        raise ValueError('Private original material owner changed before parent fallback')
                    self.parent_geometry=pickle.loads(Path(self.material_owner['path']).read_bytes())
                common={key:value for key,value in waiting.items() if key!='groups'}
                results=[_evaluate_group_on_owner(dict(common,**group),self.parent_geometry,self.parent_certificates,'parent_low_headroom')
                    for group in waiting['groups']]
                yield from checked_results(waiting,results)
                waiting=next_batch();continue
            batch,future=pending.popleft();results=future.result()
            yield from checked_results(batch,results)

    def release_idle(self):
        """Finish and release this owned pool without closing the manager.

        The caller uses this after consuming the current source stage. The
        next stage rebuilds the same requested worker count. A borrowed
        executor remains with its original owner.
        """
        if self.closed or not self.owns_executor or self.executor is None:
            return False
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor = None
        return True

    def close(self):
        if self.closed:return
        self.closed=True
        if self.owns_executor and self.executor is not None:self.executor.shutdown(wait=True,cancel_futures=True)
        self.executor=None

    async def aclose(self):
        """Release only this owned executor without blocking an async caller."""
        import asyncio
        await asyncio.to_thread(self.close)

    def report(self):
        return dict(schema='rinkspace.private-absolute-geometry-fd-group-pool-candidate.v1',
            material_owner=self.material_owner,workers=self.workers,max_inflight=self.max_inflight,
            groups_per_job=self.groups_per_job,
            engineering_headroom_reserve_bytes=self.reserve_bytes,
            resource_route_events=list(self.last_route_events),
            borrowed_existing_process_executor=not self.owns_executor,jobs=list(self.last_stats),
            original_source_group_order_and_geometry_columns_preserved=True,
            original_bundle_motion_profiles_masks_scales_CSR_and_normal_route_owned_by_caller=True,
            whole_source_graph_or_bundle_transmitted=False,
            numerical_equivalence_and_actual_speedup_not_yet_claimed=True)
